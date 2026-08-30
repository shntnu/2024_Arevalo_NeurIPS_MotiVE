#!/usr/bin/env python3
"""Infer directional top-50 CRISPR predictions over all JUMP profiles."""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from infer_full_orf_predictions import FROZEN, known_flags, topk
from model import create_model
from motive import load_graph_helper

TOP_K, COMPOUNDS, GENES = 50, 115_790, 7_977
ROWS, ANNOTATIONS = TOP_K * (COMPOUNDS + GENES), 143_633
DEFAULT_OUTPUT = "outputs/prediction_exports/motive_full_jump_crispr_top50_directional_predictions.parquet"
EXPECTED_CONFIG = {
    "graph_type": "st_expanded",
    "target_type": "crispr",
    "neg_ratio": 100,
    "model": "gin",
    "num_epochs": 200,
    "eval_freq": 5,
    "initialization": "cp",
    "hidden_channels": 1024,
    "learning_rate": 7.23340981522738e-05,
    "weight_decay": 0.0040421026533938,
    "output_path": "outputs",
}
# Each direction uses the checkpoint whose candidates were held out from training.
# fmt: off
RUNS = (
    {"checkpoint": "ff7c02", "leave_out": "source", "direction": "top_compounds_for_each_gene", "query": "gene", "validation": "compounds_held_out_from_training", "dimension": 0},
    {"checkpoint": "be48db", "leave_out": "target", "direction": "top_genes_for_each_compound", "query": "compound", "validation": "genes_held_out_from_training", "dimension": 1},
)
SCHEMA = (
    ("gene_perturbation_type", "VARCHAR"),
    ("prediction_direction", "VARCHAR"),
    ("model_checkpoint_id", "VARCHAR"),
    ("checkpoint_validation_holdout", "VARCHAR"),
    ("model_architecture", "VARCHAR"),
    ("graph_configuration", "VARCHAR"),
    ("node_representation_strategy", "VARCHAR"),
    ("query_entity_type", "VARCHAR"),
    ("query_entity_identifier", "VARCHAR"),
    ("candidate_entity_type", "VARCHAR"),
    ("candidate_entity_identifier", "VARCHAR"),
    ("gene_symbol", "VARCHAR"),
    ("compound_inchikey14", "VARCHAR"),
    ("model_pair_score", "FLOAT"),
    ("candidate_rank_within_query", "SMALLINT"),
    ("number_of_candidates_scored_for_query", "INTEGER"),
    ("is_connection_in_available_annotations", "BOOLEAN"),
)
# fmt: on


def run_dir(root, run):
    return (
        root
        / "outputs/crispr"
        / run["leave_out"]
        / "st_expanded/gin"
        / run["checkpoint"]
    )


def load_profiles(root):
    profiles = {
        "source": pd.read_parquet(root / "data/all_source.parquet"),
        "target": pd.read_parquet(root / "data/crispr_all_target.parquet"),
    }
    for node, expected in (("source", (COMPOUNDS, 737)), ("target", (GENES, 259))):
        frame = profiles[node]
        if frame.shape != expected:
            raise ValueError(f"Unexpected {node} profile shape: {frame.shape}")
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{node} identifiers must be unique and sorted")
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"{node} profiles contain non-finite values")
    return profiles


def load_model(root, run, profiles, device):
    """Load learned weights and substitute the full frozen profile tables."""
    _, _, original = load_graph_helper(run["leave_out"], "crispr", "st_expanded")
    data, mappings = HeteroData(), {}
    for node in ("source", "target"):
        old = pd.read_parquet(root / f"data/st_expanded/crispr/{node}.parquet")
        if not profiles[node].columns.equals(old.columns):
            raise ValueError(f"{node} feature columns differ from the checkpoint graph")
        mappings[node] = profiles[node].index.get_indexer(old.index)
        if (mappings[node] < 0).any():
            raise ValueError(f"Original {node} nodes are absent from full profiles")
        if not np.array_equal(
            profiles[node].iloc[mappings[node]].to_numpy(), old.to_numpy()
        ):
            raise ValueError(f"Original {node} profiles changed in the full table")
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

    expected = EXPECTED_CONFIG | {
        "leave_out": run["leave_out"],
        "hash": run["checkpoint"],
    }
    config = json.loads((run_dir(root, run) / "config.json").read_text())
    if config != expected:
        wrong = {
            key: config.get(key)
            for key, value in expected.items()
            if config.get(key) != value
        }
        extra = sorted(config.keys() - expected.keys())
        raise ValueError(f"Checkpoint config mismatch: wrong={wrong}, extra={extra}")

    model = create_model(config, data)
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


def check_benchmark(root, run, embeddings, mappings):
    """Require raw-score parity and exact stored benchmark top-50 membership."""
    results = pd.read_parquet(
        run_dir(root, run) / "cartesian/test/results.parquet",
        columns=["source", "target", "logits"],
    )
    device = embeddings["source"].device
    source = torch.as_tensor(mappings["source"][results["source"]], device=device)
    target = torch.as_tensor(mappings["target"][results["target"]], device=device)
    scores = (
        (embeddings["source"][source] * embeddings["target"][target])
        .sum(1)
        .float()
        .cpu()
    )
    torch.testing.assert_close(
        scores,
        torch.from_numpy(results["logits"].to_numpy(dtype=np.float32)),
        rtol=0,
        atol=1e-3,
    )

    width = len(mappings["target"])
    if run["query"] == "gene":
        query, candidate = "target", "source"
    else:
        query, candidate = "source", "target"
    predicted = (
        results.assign(_score=scores.numpy())
        .sort_values([query, "_score", candidate], ascending=[True, False, True])
        .groupby(query, sort=False)
        .head(TOP_K)
    )
    predicted = predicted["source"].to_numpy(dtype=np.int64) * width + predicted[
        "target"
    ].to_numpy(dtype=np.int64)
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
        "gene_perturbation_type": "crispr_gene_perturbation",
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


def validate(root, path, identifiers):
    connection = duckdb.connect()
    schema = tuple(
        (row[0], row[1])
        for row in connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
        ).fetchall()
    )
    if schema != SCHEMA:
        raise ValueError(f"Output schema mismatch: {schema}")

    expected_queries = {
        "top_compounds_for_each_gene": set(identifiers["target"]),
        "top_genes_for_each_compound": set(identifiers["source"]),
    }
    expected_candidates = {
        "top_compounds_for_each_gene": set(identifiers["source"]),
        "top_genes_for_each_compound": set(identifiers["target"]),
    }
    for direction, queries in expected_queries.items():
        observed_queries = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT query_entity_identifier FROM read_parquet(?) "
                "WHERE prediction_direction = ?",
                [str(path), direction],
            ).fetchall()
        }
        observed_candidates = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT candidate_entity_identifier FROM read_parquet(?) "
                "WHERE prediction_direction = ?",
                [str(path), direction],
            ).fetchall()
        }
        if observed_queries != queries:
            raise ValueError(f"Unexpected query coverage for {direction}")
        if not observed_candidates <= expected_candidates[direction]:
            raise ValueError(f"Unknown candidates in {direction}")

    stats = connection.execute(
        """
        WITH p AS (SELECT * FROM read_parquet(?)),
        annotations AS (
          SELECT DISTINCT left(inchikey, 14) compound_inchikey14,
            target gene_symbol
          FROM read_parquet(?)
          WHERE inchikey IS NOT NULL AND target IS NOT NULL
        ), checked AS (
          SELECT p.*, annotations.compound_inchikey14 IS NOT NULL annotated,
            lag(model_pair_score) OVER (
              PARTITION BY prediction_direction, query_entity_identifier
              ORDER BY candidate_rank_within_query) previous_score,
            lag(candidate_entity_identifier) OVER (
              PARTITION BY prediction_direction, query_entity_identifier
              ORDER BY candidate_rank_within_query) previous_candidate
          FROM p LEFT JOIN annotations USING (compound_inchikey14, gene_symbol)
        ), bad_groups AS (
          SELECT 1 FROM p GROUP BY prediction_direction, query_entity_identifier
          HAVING count(*) != 50
            OR count(DISTINCT candidate_rank_within_query) != 50
            OR count(DISTINCT candidate_entity_identifier) != 50
            OR min(candidate_rank_within_query) != 1
            OR max(candidate_rank_within_query) != 50
        )
        SELECT count(*),
          count(*) FILTER (WHERE prediction_direction = 'top_compounds_for_each_gene'
            AND model_checkpoint_id = 'ff7c02'
            AND checkpoint_validation_holdout = 'compounds_held_out_from_training'),
          count(*) FILTER (WHERE prediction_direction = 'top_genes_for_each_compound'
            AND model_checkpoint_id = 'be48db'
            AND checkpoint_validation_holdout = 'genes_held_out_from_training'),
          count(DISTINCT (prediction_direction, query_entity_identifier,
                          candidate_entity_identifier)),
          count(*) FILTER (WHERE query_entity_identifier IS NULL
            OR candidate_entity_identifier IS NULL OR gene_symbol IS NULL
            OR compound_inchikey14 IS NULL OR model_pair_score IS NULL
            OR NOT isfinite(model_pair_score)
            OR gene_perturbation_type IS DISTINCT FROM 'crispr_gene_perturbation'
            OR model_architecture IS DISTINCT FROM 'graph_isomorphism_network'
            OR graph_configuration IS DISTINCT FROM 'compound_gene_edges_plus_compound_and_gene_similarity_edges'
            OR node_representation_strategy IS DISTINCT FROM 'cell_painting_profile_features'
            OR (prediction_direction = 'top_compounds_for_each_gene' AND
              (query_entity_type IS DISTINCT FROM 'gene'
              OR candidate_entity_type IS DISTINCT FROM 'compound'
              OR query_entity_identifier IS DISTINCT FROM gene_symbol
              OR candidate_entity_identifier IS DISTINCT FROM compound_inchikey14
              OR number_of_candidates_scored_for_query IS DISTINCT FROM 115790))
            OR (prediction_direction = 'top_genes_for_each_compound' AND
              (query_entity_type IS DISTINCT FROM 'compound'
              OR candidate_entity_type IS DISTINCT FROM 'gene'
              OR query_entity_identifier IS DISTINCT FROM compound_inchikey14
              OR candidate_entity_identifier IS DISTINCT FROM gene_symbol
              OR number_of_candidates_scored_for_query IS DISTINCT FROM 7977))
            OR prediction_direction IS NULL
            OR prediction_direction NOT IN ('top_compounds_for_each_gene',
                                             'top_genes_for_each_compound')
            OR is_connection_in_available_annotations IS DISTINCT FROM annotated
            OR (candidate_rank_within_query > 1 AND
              (model_pair_score > previous_score
              OR (model_pair_score = previous_score AND
                  candidate_entity_identifier < previous_candidate)))),
          (SELECT count(*) FROM bad_groups),
          count(*) FILTER (WHERE is_connection_in_available_annotations)
        FROM checked
        """,
        [str(path), str(root / "inputs/annotations/compound_gene.parquet")],
    ).fetchone()
    expected = (ROWS, GENES * TOP_K, COMPOUNDS * TOP_K, ROWS, 0, 0)
    if stats[:-1] != expected:
        raise ValueError(f"Output invariant failure: {stats}")
    print(f"Validated rows and annotations: {stats}")


@torch.inference_mode()
def main():
    if len(sys.argv) > 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} [OUTPUT.parquet]")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    root = Path.cwd()
    output = Path(sys.argv[1] if len(sys.argv) == 2 else DEFAULT_OUTPUT)
    output = (root / output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("Full inference requires a CUDA GPU")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False

    profiles = load_profiles(root)
    identifiers = {
        node: frame.index.astype(str).to_numpy() for node, frame in profiles.items()
    }
    known, device, tables = known_keys(root, profiles), torch.device("cuda"), []
    for run in RUNS:
        print(f"Loading checkpoint {run['checkpoint']}")
        model, data, mappings = load_model(root, run, profiles, device)
        nodes = {
            "source": model.source_emb(data["source"].node_id),
            "target": model.target_emb(data["target"].node_id),
        }
        embeddings = model.gnn(nodes, data.edge_index_dict)
        check_benchmark(root, run, embeddings, mappings)
        scores = embeddings["source"] @ embeddings["target"].T
        if not torch.isfinite(scores).all():
            raise ValueError("Full score matrix contains non-finite values")
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
        validate(root, temporary_path, identifiers)
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
