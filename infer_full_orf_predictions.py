#!/usr/bin/env python3
"""Infer directional top-50 ORF predictions over the full JUMP profile set."""

import argparse
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch_geometric.data import HeteroData

from model import create_model
from motive import load_graph_helper

TOP_K = 50
EXPECTED_COMPOUNDS = 115_790
EXPECTED_ORF_GENES = 12_602
EXPECTED_ROWS = TOP_K * (EXPECTED_COMPOUNDS + EXPECTED_ORF_GENES)
EXPECTED_PROFILE_COVERED_ANNOTATIONS = 160_426
FROZEN_PROFILE_KEYS = {"source_emb.0.weight", "target_emb.0.weight"}


@dataclass(frozen=True)
class Direction:
    checkpoint_id: str
    leave_out: str
    prediction_direction: str
    query_entity_type: str
    candidate_entity_type: str
    checkpoint_validation_holdout: str
    topk_dimension: int

    def checkpoint_path(self, root: Path) -> Path:
        return (
            root
            / "outputs/orf"
            / self.leave_out
            / "st_expanded/gin"
            / self.checkpoint_id
            / "weights.pt"
        )

    def results_path(self, root: Path) -> Path:
        return self.checkpoint_path(root).parent / "cartesian/test/results.parquet"


DIRECTIONS = (
    Direction(
        checkpoint_id="6d6658",
        leave_out="source",
        prediction_direction="top_compounds_for_each_gene",
        query_entity_type="gene",
        candidate_entity_type="compound",
        checkpoint_validation_holdout="compounds_held_out_from_training",
        topk_dimension=0,
    ),
    Direction(
        checkpoint_id="5e30b1",
        leave_out="target",
        prediction_direction="top_genes_for_each_compound",
        query_entity_type="compound",
        candidate_entity_type="gene",
        checkpoint_validation_holdout="genes_held_out_from_training",
        topk_dimension=1,
    ),
)

DICTIONARY_STRING = pa.dictionary(pa.int8(), pa.string())
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("gene_perturbation_type", DICTIONARY_STRING),
        pa.field("prediction_direction", DICTIONARY_STRING),
        pa.field("model_checkpoint_id", DICTIONARY_STRING),
        pa.field("checkpoint_validation_holdout", DICTIONARY_STRING),
        pa.field("model_architecture", DICTIONARY_STRING),
        pa.field("graph_configuration", DICTIONARY_STRING),
        pa.field("node_representation_strategy", DICTIONARY_STRING),
        pa.field("query_entity_type", DICTIONARY_STRING),
        pa.field("query_entity_identifier", pa.string()),
        pa.field("candidate_entity_type", DICTIONARY_STRING),
        pa.field("candidate_entity_identifier", pa.string()),
        pa.field("gene_symbol", pa.string()),
        pa.field("compound_inchikey14", pa.string()),
        pa.field("model_pair_score", pa.float32()),
        pa.field("candidate_rank_within_query", pa.int16()),
        pa.field("number_of_candidates_scored_for_query", pa.int32()),
        pa.field("is_connection_in_available_annotations", pa.bool_()),
    ]
)


def constant_array(value: str, size: int) -> pa.DictionaryArray:
    indices = pa.array(np.zeros(size, dtype=np.int8))
    return pa.DictionaryArray.from_arrays(indices, pa.array([value]))


def deterministic_topk(
    scores: torch.Tensor, k: int, dimension: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return query-major top-k scores and candidate indices with stable ties."""
    values, indices = torch.topk(scores, k + 1, dim=dimension)
    if dimension == 0:
        values = values.T
        indices = indices.T
    elif dimension != 1:
        raise ValueError("dimension must be 0 or 1")

    boundary_ties = values[:, k - 1] == values[:, k]
    top_values = values[:, :k].float().cpu().numpy()
    top_indices = indices[:, :k].cpu().numpy()

    for query in torch.where(boundary_ties)[0].cpu().tolist():
        row = scores[:, query] if dimension == 0 else scores[query]
        row = row.float().cpu().numpy()
        order = np.lexsort((np.arange(len(row)), -row))[:k]
        top_values[query] = row[order]
        top_indices[query] = order

    order = np.lexsort((top_indices, -top_values), axis=1)
    top_values = np.take_along_axis(top_values, order, axis=1)
    top_indices = np.take_along_axis(top_indices, order, axis=1)
    return top_values, top_indices


def remap_graph(
    root: Path,
    original: HeteroData,
    full_compounds: pd.DataFrame,
    full_genes: pd.DataFrame,
) -> HeteroData:
    """Place the original inference edges in the full profile node universe."""
    original_compounds = pd.Index(
        pd.read_parquet(root / "data/st_expanded/orf/source.parquet").index
    )
    original_genes = pd.Index(
        pd.read_parquet(root / "data/st_expanded/orf/target.parquet").index
    )
    compound_lookup = full_compounds.index.get_indexer(original_compounds)
    gene_lookup = full_genes.index.get_indexer(original_genes)
    if (compound_lookup < 0).any() or (gene_lookup < 0).any():
        raise ValueError("The original graph contains nodes absent from full profiles")

    lookups = {
        "source": torch.from_numpy(compound_lookup),
        "target": torch.from_numpy(gene_lookup),
    }
    data = HeteroData()
    data["source"].node_id = torch.arange(len(full_compounds))
    data["source"].x = torch.from_numpy(
        full_compounds.to_numpy(dtype=np.float32, copy=True)
    )
    data["target"].node_id = torch.arange(len(full_genes))
    data["target"].x = torch.from_numpy(
        full_genes.to_numpy(dtype=np.float32, copy=True)
    )
    for edge_type, edges in original.edge_index_dict.items():
        source_type, _, target_type = edge_type
        data[edge_type].edge_index = torch.stack(
            [lookups[source_type][edges[0]], lookups[target_type][edges[1]]]
        )

    if data.metadata() != original.metadata():
        raise ValueError(
            f"Graph metadata changed: {original.metadata()} -> {data.metadata()}"
        )
    return data


def load_full_model(
    root: Path,
    direction: Direction,
    full_compounds: pd.DataFrame,
    full_genes: pd.DataFrame,
    device: torch.device,
) -> tuple[torch.nn.Module, HeteroData, np.ndarray, np.ndarray]:
    """Load learned weights while replacing only frozen profile tables."""
    _, _, original = load_graph_helper(direction.leave_out, "orf", "st_expanded")
    data = remap_graph(root, original, full_compounds, full_genes)
    model = create_model(
        {
            "model": "gin",
            "initialization": "cp",
            "hidden_channels": 1024,
        },
        data,
    )
    checkpoint = torch.load(
        direction.checkpoint_path(root), map_location="cpu", weights_only=True
    )["model_state_dict"]
    for key, expected in (
        ("source_emb.0.weight", original["source"].x),
        ("target_emb.0.weight", original["target"].x),
    ):
        torch.testing.assert_close(checkpoint[key], expected, rtol=0, atol=0)

    learned_state = {
        k: v for k, v in checkpoint.items() if k not in FROZEN_PROFILE_KEYS
    }
    missing, unexpected = model.load_state_dict(learned_state, strict=False)
    if set(missing) != FROZEN_PROFILE_KEYS or unexpected:
        raise ValueError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )

    original_compounds = pd.read_parquet(root / "data/st_expanded/orf/source.parquet")
    original_genes = pd.read_parquet(root / "data/st_expanded/orf/target.parquet")
    compound_lookup = full_compounds.index.get_indexer(original_compounds.index)
    gene_lookup = full_genes.index.get_indexer(original_genes.index)

    del data["source"].x
    del data["target"].x
    model.eval().to(device)
    return model, data.to(device), compound_lookup, gene_lookup


@torch.inference_mode()
def score_full_matrix(model: torch.nn.Module, data: HeteroData) -> torch.Tensor:
    x_dict = {
        "source": model.source_emb(data["source"].node_id),
        "target": model.target_emb(data["target"].node_id),
    }
    embeddings = model.gnn(x_dict, data.edge_index_dict)
    scores = embeddings["source"] @ embeddings["target"].T
    if not torch.isfinite(scores).all():
        raise ValueError("Full score matrix contains non-finite values")
    return scores


def validate_score_parity(
    root: Path,
    direction: Direction,
    scores: torch.Tensor,
    compound_lookup: np.ndarray,
    gene_lookup: np.ndarray,
) -> None:
    results = pd.read_parquet(
        direction.results_path(root), columns=["source", "target", "logits"]
    )
    positions = np.linspace(0, len(results) - 1, 10_000, dtype=np.int64)
    batch = results.iloc[positions]
    source = torch.as_tensor(compound_lookup[batch["source"]], device=scores.device)
    target = torch.as_tensor(gene_lookup[batch["target"]], device=scores.device)
    actual = scores[source, target].float().cpu()
    expected = torch.from_numpy(batch["logits"].to_numpy(dtype=np.float32))
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-3)


def validate_benchmark_topk_membership(
    root: Path,
    direction: Direction,
    scores: torch.Tensor,
    compound_lookup: np.ndarray,
    gene_lookup: np.ndarray,
) -> None:
    """Require full-batch inference to recover stored benchmark top-k pairs."""
    results = pd.read_parquet(
        direction.results_path(root), columns=["source", "target", "logits"]
    )
    benchmark_sources = np.sort(results["source"].unique())
    benchmark_targets = np.sort(results["target"].unique())
    source_ids = torch.as_tensor(
        compound_lookup[benchmark_sources], device=scores.device
    )
    target_ids = torch.as_tensor(gene_lookup[benchmark_targets], device=scores.device)
    benchmark_scores = scores[source_ids][:, target_ids]
    _, top_indices = deterministic_topk(
        benchmark_scores, TOP_K, direction.topk_dimension
    )

    if direction.query_entity_type == "gene":
        predicted_sources = benchmark_sources[top_indices.reshape(-1)]
        predicted_targets = np.repeat(benchmark_targets, TOP_K)
        query_column, candidate_column = "target", "source"
    else:
        predicted_sources = np.repeat(benchmark_sources, TOP_K)
        predicted_targets = benchmark_targets[top_indices.reshape(-1)]
        query_column, candidate_column = "source", "target"

    expected = (
        results.sort_values(
            [query_column, "logits", candidate_column],
            ascending=[True, False, True],
        )
        .groupby(query_column, sort=False)
        .head(TOP_K)
    )
    key_width = len(gene_lookup)
    expected_keys = np.sort(
        expected["source"].to_numpy(dtype=np.int64) * key_width
        + expected["target"].to_numpy(dtype=np.int64)
    )
    predicted_keys = np.sort(
        predicted_sources.astype(np.int64) * key_width
        + predicted_targets.astype(np.int64)
    )
    if not np.array_equal(predicted_keys, expected_keys):
        missing = np.setdiff1d(expected_keys, predicted_keys)
        extra = np.setdiff1d(predicted_keys, expected_keys)
        raise ValueError(
            "Benchmark top-k membership changed: "
            f"missing={len(missing)}, extra={len(extra)}"
        )


def annotation_keys(root: Path, compounds: pd.Index, genes: pd.Index) -> np.ndarray:
    annotations = pd.read_parquet(
        root / "inputs/annotations/compound_gene.parquet",
        columns=["inchikey", "target"],
    ).dropna()
    source = compounds.get_indexer(annotations["inchikey"].str[:14])
    target = genes.get_indexer(annotations["target"])
    present = (source >= 0) & (target >= 0)
    keys = np.unique(source[present].astype(np.int64) * len(genes) + target[present])
    if len(keys) != EXPECTED_PROFILE_COVERED_ANNOTATIONS:
        raise ValueError(
            f"Unexpected number of profile-covered annotations: {len(keys):,}"
        )
    return keys


def annotation_flags(pair_keys: np.ndarray, known_keys: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(known_keys, pair_keys)
    valid = positions < len(known_keys)
    flags = np.zeros(len(pair_keys), dtype=bool)
    flags[valid] = known_keys[positions[valid]] == pair_keys[valid]
    return flags


def write_direction(
    writer: pq.ParquetWriter,
    direction: Direction,
    values: np.ndarray,
    indices: np.ndarray,
    compounds: np.ndarray,
    genes: np.ndarray,
    known_keys: np.ndarray,
    query_chunk_size: int = 10_000,
) -> None:
    num_queries, k = indices.shape
    for start in range(0, num_queries, query_chunk_size):
        stop = min(start + query_chunk_size, num_queries)
        size = (stop - start) * k
        ranks = np.tile(np.arange(1, k + 1, dtype=np.int16), stop - start)
        candidates = indices[start:stop].reshape(-1)

        if direction.query_entity_type == "gene":
            query_positions = np.repeat(np.arange(start, stop), k)
            gene_symbols = genes[query_positions]
            compound_ids = compounds[candidates]
            query_ids = gene_symbols
            candidate_ids = compound_ids
            pair_keys = candidates.astype(np.int64) * len(genes) + query_positions
            candidate_count = len(compounds)
        else:
            query_positions = np.repeat(np.arange(start, stop), k)
            compound_ids = compounds[query_positions]
            gene_symbols = genes[candidates]
            query_ids = compound_ids
            candidate_ids = gene_symbols
            pair_keys = query_positions.astype(np.int64) * len(genes) + candidates
            candidate_count = len(genes)

        arrays = [
            constant_array("orf_overexpression", size),
            constant_array(direction.prediction_direction, size),
            constant_array(direction.checkpoint_id, size),
            constant_array(direction.checkpoint_validation_holdout, size),
            constant_array("graph_isomorphism_network", size),
            constant_array(
                "compound_gene_edges_plus_compound_and_gene_similarity_edges", size
            ),
            constant_array("cell_painting_profile_features", size),
            constant_array(direction.query_entity_type, size),
            pa.array(query_ids),
            constant_array(direction.candidate_entity_type, size),
            pa.array(candidate_ids),
            pa.array(gene_symbols),
            pa.array(compound_ids),
            pa.array(values[start:stop].reshape(-1), type=pa.float32()),
            pa.array(ranks),
            pa.array(np.full(size, candidate_count, dtype=np.int32)),
            pa.array(annotation_flags(pair_keys, known_keys)),
        ]
        writer.write_table(pa.Table.from_arrays(arrays, schema=OUTPUT_SCHEMA))


def validate_output(root: Path, path: Path) -> None:
    con = duckdb.connect()
    quoted = "'" + str(path).replace("'", "''") + "'"
    summary = con.execute(
        f"""
        SELECT prediction_direction, model_checkpoint_id,
               count(*) AS rows, count(DISTINCT query_entity_identifier) AS queries,
               min(candidate_rank_within_query) AS min_rank,
               max(candidate_rank_within_query) AS max_rank,
               count(*) FILTER (WHERE is_connection_in_available_annotations)
                   AS known_rows
        FROM read_parquet({quoted})
        GROUP BY ALL ORDER BY 1
        """
    ).fetchall()
    expected = {
        "top_compounds_for_each_gene": ("6d6658", EXPECTED_ORF_GENES * TOP_K),
        "top_genes_for_each_compound": ("5e30b1", EXPECTED_COMPOUNDS * TOP_K),
    }
    if sum(row[2] for row in summary) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} rows, found {summary}")
    for direction, checkpoint, rows, queries, low, high, _ in summary:
        expected_checkpoint, expected_rows = expected[direction]
        if (checkpoint, rows, low, high) != (
            expected_checkpoint,
            expected_rows,
            1,
            TOP_K,
        ):
            raise ValueError(f"Invalid direction summary: {summary}")
        expected_queries = expected_rows // TOP_K
        if queries != expected_queries:
            raise ValueError(f"Expected {expected_queries} queries, found {queries}")

    bad = con.execute(
        f"""
        SELECT
          count(*) FILTER (WHERE query_entity_identifier IS NULL
                              OR candidate_entity_identifier IS NULL
                              OR model_pair_score IS NULL
                              OR NOT isfinite(model_pair_score)) AS invalid_rows,
          count(*) - count(DISTINCT (
              prediction_direction, query_entity_identifier,
              candidate_entity_identifier
          )) AS duplicate_rows,
          count(*) FILTER (
              WHERE query_entity_type = 'gene'
                AND (query_entity_identifier != gene_symbol
                  OR candidate_entity_identifier != compound_inchikey14
                  OR number_of_candidates_scored_for_query != 115790)
          ) AS gene_direction_errors,
          count(*) FILTER (
              WHERE query_entity_type = 'compound'
                AND (query_entity_identifier != compound_inchikey14
                  OR candidate_entity_identifier != gene_symbol
                  OR number_of_candidates_scored_for_query != 12602)
          ) AS compound_direction_errors
        FROM read_parquet({quoted})
        """
    ).fetchone()
    bad_groups = con.execute(
        f"""
        SELECT count(*) FROM (
          SELECT prediction_direction, query_entity_identifier
          FROM read_parquet({quoted})
          GROUP BY ALL
          HAVING count(*) != 50 OR min(candidate_rank_within_query) != 1
             OR max(candidate_rank_within_query) != 50
             OR count(DISTINCT candidate_rank_within_query) != 50
        )
        """
    ).fetchone()[0]
    if bad != (0, 0, 0, 0) or bad_groups:
        raise ValueError(f"Output invariant failure: rows={bad}, groups={bad_groups}")

    annotation_path = root / "inputs/annotations/compound_gene.parquet"
    annotation_quoted = "'" + str(annotation_path).replace("'", "''") + "'"
    annotation_mismatches = con.execute(
        f"""
        WITH annotations AS (
          SELECT DISTINCT left(inchikey, 14) AS compound_inchikey14,
                          target AS gene_symbol
          FROM read_parquet({annotation_quoted})
          WHERE inchikey IS NOT NULL AND target IS NOT NULL
        ), checked AS (
          SELECT p.is_connection_in_available_annotations AS observed,
                 a.compound_inchikey14 IS NOT NULL AS expected
          FROM read_parquet({quoted}) AS p
          LEFT JOIN annotations AS a USING (compound_inchikey14, gene_symbol)
        )
        SELECT count(*) FILTER (WHERE observed != expected),
               count(*) FILTER (WHERE observed),
               count(*) FILTER (WHERE NOT observed)
        FROM checked
        """
    ).fetchone()
    if annotation_mismatches[0] or not all(annotation_mismatches[1:]):
        raise ValueError(f"Annotation flag validation failed: {annotation_mismatches}")
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to outputs/prediction_exports under ROOT",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    os.chdir(root)
    output = args.output or (
        root
        / "outputs/prediction_exports"
        / "motive_full_jump_orf_top50_directional_predictions.parquet"
    )
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    compounds = pd.read_parquet(root / "data/all_source.parquet")
    genes = pd.read_parquet(root / "data/orf_all_target.parquet")
    if compounds.shape != (EXPECTED_COMPOUNDS, 737):
        raise ValueError(f"Unexpected compound profile shape: {compounds.shape}")
    if genes.shape != (EXPECTED_ORF_GENES, 722):
        raise ValueError(f"Unexpected ORF profile shape: {genes.shape}")
    if not compounds.index.is_unique or not genes.index.is_unique:
        raise ValueError("Full profile identifiers must be unique")
    if not compounds.index.is_monotonic_increasing:
        raise ValueError("Compound identifiers must be sorted for deterministic ties")
    if not genes.index.is_monotonic_increasing:
        raise ValueError("Gene identifiers must be sorted for deterministic ties")

    compound_ids = compounds.index.astype(str).to_numpy()
    gene_ids = genes.index.astype(str).to_numpy()
    known_keys = annotation_keys(root, compounds.index, genes.index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Full inference requires a CUDA GPU")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False

    with tempfile.NamedTemporaryFile(
        prefix=output.stem + ".",
        suffix=".partial.parquet",
        dir=output.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)

    try:
        with pq.ParquetWriter(
            temporary_path, OUTPUT_SCHEMA, compression="zstd", use_dictionary=True
        ) as writer:
            for direction in DIRECTIONS:
                print(f"Loading checkpoint {direction.checkpoint_id}")
                model, data, source_lookup, target_lookup = load_full_model(
                    root, direction, compounds, genes, device
                )
                scores = score_full_matrix(model, data)
                validate_score_parity(
                    root, direction, scores, source_lookup, target_lookup
                )
                validate_benchmark_topk_membership(
                    root, direction, scores, source_lookup, target_lookup
                )
                values, indices = deterministic_topk(
                    scores, TOP_K, direction.topk_dimension
                )
                write_direction(
                    writer,
                    direction,
                    values,
                    indices,
                    compound_ids,
                    gene_ids,
                    known_keys,
                )
                del model, data, scores, values, indices
                torch.cuda.empty_cache()

        validate_output(root, temporary_path)
        os.replace(temporary_path, output)
        output.chmod(0o644)
    except BaseException:
        print(f"Incomplete output retained for inspection: {temporary_path}")
        raise

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"Wrote {EXPECTED_ROWS:,} rows to {output}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
