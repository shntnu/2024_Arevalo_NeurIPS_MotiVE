#!/usr/bin/env python3
"""Infer directional top-50 ORF predictions over all JUMP profiles."""

import hashlib
import os
import sys
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from model import create_model
from motive import load_graph_helper

TOP_K, COMPOUNDS, GENES = 50, 115_790, 12_602
ROWS, ANNOTATIONS = TOP_K * (COMPOUNDS + GENES), 160_426
FROZEN = ("source_emb.0.weight", "target_emb.0.weight")
DEFAULT_OUTPUT = "outputs/prediction_exports/motive_full_jump_orf_top50_directional_predictions.parquet"
# One checkpoint was validated on held-out compounds; the other on held-out genes.
# fmt: off
RUNS = (
    {"checkpoint": "6d6658", "leave_out": "source", "direction": "top_compounds_for_each_gene", "query": "gene", "validation": "compounds_held_out_from_training", "dimension": 0},
    {"checkpoint": "5e30b1", "leave_out": "target", "direction": "top_genes_for_each_compound", "query": "compound", "validation": "genes_held_out_from_training", "dimension": 1},
)
# fmt: on


def run_dir(root, run):
    return (
        root / "outputs/orf" / run["leave_out"] / "st_expanded/gin" / run["checkpoint"]
    )


def topk(scores, k, dimension):
    """Return query-major top-k scores, breaking ties by candidate position."""
    values, indices = torch.topk(scores, k + 1, dim=dimension)
    if dimension == 0:
        values, indices = values.T, indices.T
    elif dimension != 1:
        raise ValueError("dimension must be 0 or 1")
    boundary_ties = values[:, k - 1] == values[:, k]
    values, indices = values[:, :k].float().cpu().numpy(), indices[:, :k].cpu().numpy()
    for query in torch.where(boundary_ties)[0].cpu().tolist():
        row = scores[:, query] if dimension == 0 else scores[query]
        row = row.float().cpu().numpy()
        order = np.lexsort((np.arange(len(row)), -row))[:k]
        values[query], indices[query] = row[order], order
    order = np.lexsort((indices, -values), axis=1)
    return np.take_along_axis(values, order, axis=1), np.take_along_axis(
        indices, order, axis=1
    )


def load_profiles(root):
    profiles = {
        "source": pd.read_parquet(root / "data/all_source.parquet"),
        "target": pd.read_parquet(root / "data/orf_all_target.parquet"),
    }
    for node, expected in (("source", (COMPOUNDS, 737)), ("target", (GENES, 722))):
        frame = profiles[node]
        if frame.shape != expected:
            raise ValueError(f"Unexpected {node} profile shape: {frame.shape}")
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{node} identifiers must be unique and sorted")
    return profiles


def load_model(root, run, profiles, device):
    """Load learned weights and substitute the full frozen profile tables."""
    _, _, original = load_graph_helper(run["leave_out"], "orf", "st_expanded")
    data, mappings = HeteroData(), {}
    for node in ("source", "target"):
        old_ids = pd.read_parquet(root / f"data/st_expanded/orf/{node}.parquet").index
        mappings[node] = profiles[node].index.get_indexer(old_ids)
        if (mappings[node] < 0).any():
            raise ValueError(f"Original {node} nodes are absent from full profiles")
        data[node].node_id = torch.arange(len(profiles[node]))
        data[node].x = torch.from_numpy(
            profiles[node].to_numpy(dtype=np.float32, copy=True)
        )
    maps = {node: torch.from_numpy(ids) for node, ids in mappings.items()}
    for edge_type, edges in original.edge_index_dict.items():
        source, _, target = edge_type
        data[edge_type].edge_index = torch.stack(
            [maps[source][edges[0]], maps[target][edges[1]]]
        )
    if data.metadata() != original.metadata():
        raise ValueError("Graph metadata changed while expanding the node universe")

    model = create_model(
        {"model": "gin", "initialization": "cp", "hidden_channels": 1024}, data
    )
    checkpoint = torch.load(
        run_dir(root, run) / "weights.pt", map_location="cpu", weights_only=True
    )["model_state_dict"]
    for node, key in zip(("source", "target"), FROZEN, strict=True):
        torch.testing.assert_close(checkpoint[key], original[node].x, rtol=0, atol=0)
    learned = {key: value for key, value in checkpoint.items() if key not in FROZEN}
    missing, unexpected = model.load_state_dict(learned, strict=False)
    if set(missing) != set(FROZEN) or unexpected:
        raise ValueError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    del data["source"].x, data["target"].x
    return model.eval().to(device), data.to(device), mappings


def check_benchmark(root, run, scores, mappings):
    """Require raw-score parity and exact stored benchmark top-50 membership."""
    results = pd.read_parquet(
        run_dir(root, run) / "cartesian/test/results.parquet",
        columns=["source", "target", "logits"],
    )
    sample = results.iloc[np.linspace(0, len(results) - 1, 10_000, dtype=np.int64)]
    source = torch.as_tensor(mappings["source"][sample["source"]], device=scores.device)
    target = torch.as_tensor(mappings["target"][sample["target"]], device=scores.device)
    torch.testing.assert_close(
        scores[source, target].float().cpu(),
        torch.from_numpy(sample["logits"].to_numpy(dtype=np.float32)),
        rtol=0,
        atol=1e-3,
    )
    old_sources = np.sort(results["source"].unique())
    old_targets = np.sort(results["target"].unique())
    source = torch.as_tensor(mappings["source"][old_sources], device=scores.device)
    target = torch.as_tensor(mappings["target"][old_targets], device=scores.device)
    _, indices = topk(scores[source][:, target], TOP_K, run["dimension"])
    width = len(mappings["target"])
    if run["query"] == "gene":
        predicted = old_sources[indices.ravel()] * width + np.repeat(old_targets, TOP_K)
        query, candidate = "target", "source"
    else:
        predicted = np.repeat(old_sources, TOP_K) * width + old_targets[indices.ravel()]
        query, candidate = "source", "target"
    expected = (
        results.sort_values([query, "logits", candidate], ascending=[True, False, True])
        .groupby(query, sort=False)
        .head(TOP_K)
    )
    expected = expected["source"].to_numpy(dtype=np.int64) * width + expected[
        "target"
    ].to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(predicted), np.sort(expected)):
        raise ValueError("Benchmark top-50 membership changed")


def known_keys(root, profiles):
    annotations = pd.read_parquet(
        root / "inputs/annotations/compound_gene.parquet",
        columns=["inchikey", "target"],
    ).dropna()
    compounds = profiles["source"].index.get_indexer(annotations["inchikey"].str[:14])
    genes = profiles["target"].index.get_indexer(annotations["target"])
    present = (compounds >= 0) & (genes >= 0)
    keys = np.unique(compounds[present].astype(np.int64) * GENES + genes[present])
    if len(keys) != ANNOTATIONS:
        raise ValueError(f"Expected {ANNOTATIONS:,} annotations, got {len(keys):,}")
    return keys


def known_flags(pair_keys, known):
    positions = np.searchsorted(known, pair_keys)
    valid = positions < len(known)
    flags = np.zeros(len(pair_keys), dtype=bool)
    flags[valid] = known[positions[valid]] == pair_keys[valid]
    return flags


def output_table(run, values, indices, identifiers, known):
    query_count, k = indices.shape
    queries = np.repeat(np.arange(query_count, dtype=np.int64), k)
    candidates = indices.ravel()
    if run["query"] == "gene":
        genes = identifiers["target"][queries]
        compounds = identifiers["source"][candidates]
        query_ids, candidate_ids = genes, compounds
        pair_keys = candidates * GENES + queries
        candidate_count, candidate_type = COMPOUNDS, "compound"
    else:
        compounds = identifiers["source"][queries]
        genes = identifiers["target"][candidates]
        query_ids, candidate_ids = compounds, genes
        pair_keys = queries * GENES + candidates
        candidate_count, candidate_type = GENES, "gene"
    size = len(queries)
    # fmt: off
    return pd.DataFrame({
        "gene_perturbation_type": "orf_overexpression",
        "prediction_direction": run["direction"],
        "model_checkpoint_id": run["checkpoint"],
        "checkpoint_validation_holdout": run["validation"],
        "model_architecture": "graph_isomorphism_network",
        "graph_configuration": "compound_gene_edges_plus_compound_and_gene_similarity_edges",
        "node_representation_strategy": "cell_painting_profile_features",
        "query_entity_type": run["query"],
        "query_entity_identifier": query_ids,
        "candidate_entity_type": candidate_type,
        "candidate_entity_identifier": candidate_ids,
        "gene_symbol": genes,
        "compound_inchikey14": compounds,
        "model_pair_score": values.ravel().astype(np.float32, copy=False),
        "candidate_rank_within_query": np.tile(np.arange(1, k + 1, dtype=np.int16), query_count),
        "number_of_candidates_scored_for_query": np.full(size, candidate_count, dtype=np.int32),
        "is_connection_in_available_annotations": known_flags(pair_keys, known),
    })
    # fmt: on


def validate(root, path):
    stats = (
        duckdb.connect()
        .execute(
            """
        WITH p AS (SELECT * FROM read_parquet(?)),
        annotations AS (
          SELECT DISTINCT left(inchikey, 14) compound_inchikey14, target gene_symbol
          FROM read_parquet(?) WHERE inchikey IS NOT NULL AND target IS NOT NULL
        ), checked AS (
          SELECT p.*, annotations.compound_inchikey14 IS NOT NULL annotated
          FROM p LEFT JOIN annotations USING (compound_inchikey14, gene_symbol)
        ), bad_groups AS (
          SELECT 1 FROM p GROUP BY prediction_direction, query_entity_identifier
          HAVING count(*) != 50 OR min(candidate_rank_within_query) != 1
            OR max(candidate_rank_within_query) != 50
            OR count(DISTINCT candidate_rank_within_query) != 50
        )
        SELECT count(*),
          count(*) FILTER (WHERE prediction_direction = 'top_compounds_for_each_gene'
            AND model_checkpoint_id = '6d6658'
            AND checkpoint_validation_holdout = 'compounds_held_out_from_training'),
          count(*) FILTER (WHERE prediction_direction = 'top_genes_for_each_compound'
            AND model_checkpoint_id = '5e30b1'
            AND checkpoint_validation_holdout = 'genes_held_out_from_training'),
          count(DISTINCT query_entity_identifier) FILTER
            (WHERE prediction_direction = 'top_compounds_for_each_gene'),
          count(DISTINCT query_entity_identifier) FILTER
            (WHERE prediction_direction = 'top_genes_for_each_compound'),
          count(DISTINCT (prediction_direction, query_entity_identifier,
                          candidate_entity_identifier)),
          min(candidate_rank_within_query), max(candidate_rank_within_query),
          count(*) FILTER (WHERE query_entity_identifier IS NULL
            OR candidate_entity_identifier IS NULL OR gene_symbol IS NULL
            OR compound_inchikey14 IS NULL OR model_pair_score IS NULL
            OR NOT isfinite(model_pair_score)
            OR gene_perturbation_type != 'orf_overexpression'
            OR model_architecture != 'graph_isomorphism_network'
            OR graph_configuration != 'compound_gene_edges_plus_compound_and_gene_similarity_edges'
            OR node_representation_strategy != 'cell_painting_profile_features'
            OR (query_entity_type = 'gene' AND (candidate_entity_type != 'compound'
              OR query_entity_identifier != gene_symbol
              OR candidate_entity_identifier != compound_inchikey14
              OR number_of_candidates_scored_for_query != 115790))
            OR (query_entity_type = 'compound' AND (candidate_entity_type != 'gene'
              OR query_entity_identifier != compound_inchikey14
              OR candidate_entity_identifier != gene_symbol
              OR number_of_candidates_scored_for_query != 12602))),
          (SELECT count(*) FROM bad_groups),
          count(*) FILTER (WHERE is_connection_in_available_annotations != annotated),
          count(*) FILTER (WHERE is_connection_in_available_annotations),
          count(*) FILTER (WHERE NOT is_connection_in_available_annotations)
        FROM checked
        """,
            [str(path), str(root / "inputs/annotations/compound_gene.parquet")],
        )
        .fetchone()
    )
    expected = (ROWS, GENES * TOP_K, COMPOUNDS * TOP_K, GENES, COMPOUNDS, ROWS, 1, TOP_K, 0, 0, 0)  # fmt: skip
    if stats[:-2] != expected or not all(stats[-2:]):
        raise ValueError(f"Output invariant failure: {stats}")
    print(stats)


@torch.inference_mode()
def main():
    if len(sys.argv) > 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} [OUTPUT.parquet]")
    root = Path.cwd()
    output = Path(sys.argv[1] if len(sys.argv) == 2 else DEFAULT_OUTPUT)
    output = (root / output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles(root)
    identifiers = {
        node: frame.index.astype(str).to_numpy() for node, frame in profiles.items()
    }
    known, device, tables = known_keys(root, profiles), torch.device("cuda"), []
    if not torch.cuda.is_available():
        raise RuntimeError("Full inference requires a CUDA GPU")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False

    for run in RUNS:
        print(f"Loading checkpoint {run['checkpoint']}")
        model, data, mappings = load_model(root, run, profiles, device)
        nodes = {
            "source": model.source_emb(data["source"].node_id),
            "target": model.target_emb(data["target"].node_id),
        }
        embeddings = model.gnn(nodes, data.edge_index_dict)
        scores = embeddings["source"] @ embeddings["target"].T
        if not torch.isfinite(scores).all():
            raise ValueError("Full score matrix contains non-finite values")
        check_benchmark(root, run, scores, mappings)
        values, indices = topk(scores, TOP_K, run["dimension"])
        tables.append(output_table(run, values, indices, identifiers, known))
        del model, data, embeddings, scores, values, indices
        torch.cuda.empty_cache()

    predictions = pd.concat(tables, ignore_index=True)
    with tempfile.NamedTemporaryFile(
        prefix=output.stem + ".",
        suffix=".partial.parquet",
        dir=output.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        predictions.to_parquet(
            temporary_path, compression="zstd", index=False, use_dictionary=True
        )
        validate(root, temporary_path)
        temporary_path.chmod(0o644)
        os.link(temporary_path, output)
        temporary_path.unlink()
    except BaseException:
        print(f"Incomplete output retained for inspection: {temporary_path}")
        raise
    with output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print(f"Wrote {ROWS:,} rows to {output}\nSHA-256: {digest}")


if __name__ == "__main__":
    main()
