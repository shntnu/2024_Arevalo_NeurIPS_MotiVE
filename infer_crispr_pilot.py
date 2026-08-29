#!/usr/bin/env python3
"""Validate one CRISPR checkpoint on a bounded full-profile inference slice."""

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from infer_full_orf_predictions import FROZEN, topk
from model import create_model
from motive import load_graph_helper

COMPOUNDS, GENES, QUERIES, TOP_K = 115_790, 7_977, 100, 50
RUN = Path("outputs/crispr/target/st_expanded/gin/be48db")
OUTPUT = RUN / "pilot_top50_genes_for_100_compounds.parquet"
EXPECTED_CONFIG = {
    "target_type": "crispr",
    "leave_out": "target",
    "graph_type": "st_expanded",
    "model": "gin",
    "initialization": "cp",
    "hidden_channels": 1024,
    "neg_ratio": 100,
    "num_epochs": 200,
    "eval_freq": 5,
    "learning_rate": 7.23340981522738e-05,
    "weight_decay": 0.0040421026533938,
}


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


def load_model(root, profiles, device):
    _, _, original = load_graph_helper("target", "crispr", "st_expanded")
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

    config = json.loads((root / RUN / "config.json").read_text())
    wrong = {
        key: config.get(key)
        for key, value in EXPECTED_CONFIG.items()
        if config.get(key) != value
    }
    if wrong:
        raise ValueError(
            f"Checkpoint config differs from the registered study: {wrong}"
        )
    model = create_model(config, data)
    checkpoint = torch.load(
        root / RUN / "weights.pt", map_location="cpu", weights_only=True
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


def check_benchmark(root, embeddings, mappings):
    results = pd.read_parquet(
        root / RUN / "cartesian/test/results.parquet",
        columns=["source", "target", "logits"],
    )
    sample = results.iloc[np.linspace(0, len(results) - 1, 10_000, dtype=np.int64)]
    device = embeddings["source"].device
    source = torch.as_tensor(mappings["source"][sample["source"]], device=device)
    target = torch.as_tensor(mappings["target"][sample["target"]], device=device)
    scores = (embeddings["source"][source] * embeddings["target"][target]).sum(1)
    torch.testing.assert_close(
        scores.float().cpu(),
        torch.from_numpy(sample["logits"].to_numpy(dtype=np.float32)),
        rtol=0,
        atol=1e-3,
    )
    old_sources = np.sort(results["source"].unique())
    old_targets = np.sort(results["target"].unique())
    source = torch.as_tensor(mappings["source"][old_sources], device=device)
    target = torch.as_tensor(mappings["target"][old_targets], device=device)
    scores = embeddings["source"][source] @ embeddings["target"][target].T
    _, indices = topk(scores, TOP_K, 1)
    width = len(mappings["target"])
    predicted = np.repeat(old_sources, TOP_K) * width + old_targets[indices.ravel()]
    expected = (
        results.sort_values(
            ["source", "logits", "target"], ascending=[True, False, True]
        )
        .groupby("source", sort=False)
        .head(TOP_K)
    )
    expected = expected["source"].to_numpy(dtype=np.int64) * width + expected[
        "target"
    ].to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(predicted), np.sort(expected)):
        raise ValueError("Benchmark top-50 membership changed")


def output_table(root, profiles, values, indices, query_positions):
    compounds = profiles["source"].index.astype(str).to_numpy()[query_positions]
    genes = profiles["target"].index.astype(str).to_numpy()[indices.ravel()]
    compounds = np.repeat(compounds, TOP_K)
    annotations = pd.read_parquet(
        root / "inputs/annotations/compound_gene.parquet",
        columns=["inchikey", "target"],
    ).dropna()
    known = set(zip(annotations["inchikey"].str[:14], annotations["target"]))
    flags = np.fromiter(
        (
            (compound, gene) in known
            for compound, gene in zip(compounds, genes, strict=True)
        ),
        dtype=bool,
        count=len(genes),
    )
    size = len(genes)
    return pd.DataFrame(
        {
            "gene_perturbation_type": "crispr_gene_perturbation",
            "prediction_direction": "top_genes_for_each_compound",
            "model_checkpoint_id": "be48db",
            "checkpoint_validation_holdout": "genes_held_out_from_training",
            "model_architecture": "graph_isomorphism_network",
            "graph_configuration": "compound_gene_edges_plus_compound_and_gene_similarity_edges",
            "node_representation_strategy": "cell_painting_profile_features",
            "query_entity_type": "compound",
            "query_entity_identifier": compounds,
            "candidate_entity_type": "gene",
            "candidate_entity_identifier": genes,
            "gene_symbol": genes,
            "compound_inchikey14": compounds,
            "model_pair_score": values.ravel().astype(np.float32, copy=False),
            "candidate_rank_within_query": np.tile(
                np.arange(1, TOP_K + 1, dtype=np.int16), QUERIES
            ),
            "number_of_candidates_scored_for_query": np.full(
                size, GENES, dtype=np.int32
            ),
            "is_connection_in_available_annotations": flags,
        }
    )


def validate(root, path, expected_queries):
    query_ids = set(
        pd.read_parquet(path, columns=["query_entity_identifier"])[
            "query_entity_identifier"
        ]
    )
    if query_ids != set(expected_queries):
        raise ValueError("Output contains unexpected query identifiers")
    stats = (
        duckdb.connect()
        .execute(
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
                  PARTITION BY query_entity_identifier
                  ORDER BY candidate_rank_within_query) previous_score
              FROM p LEFT JOIN annotations USING (compound_inchikey14, gene_symbol)
            ), bad_groups AS (
              SELECT 1 FROM p GROUP BY query_entity_identifier
              HAVING count(*) != 50
                OR count(DISTINCT candidate_rank_within_query) != 50
                OR count(DISTINCT candidate_entity_identifier) != 50
                OR min(candidate_rank_within_query) != 1
                OR max(candidate_rank_within_query) != 50
            )
            SELECT count(*), count(DISTINCT query_entity_identifier),
              count(*) FILTER (WHERE query_entity_identifier IS NULL
                OR candidate_entity_identifier IS NULL
                OR gene_symbol IS NULL OR compound_inchikey14 IS NULL
                OR model_pair_score IS NULL OR NOT isfinite(model_pair_score)
                OR gene_perturbation_type IS DISTINCT FROM 'crispr_gene_perturbation'
                OR prediction_direction IS DISTINCT FROM 'top_genes_for_each_compound'
                OR model_checkpoint_id IS DISTINCT FROM 'be48db'
                OR checkpoint_validation_holdout IS DISTINCT FROM 'genes_held_out_from_training'
                OR model_architecture IS DISTINCT FROM 'graph_isomorphism_network'
                OR graph_configuration IS DISTINCT FROM 'compound_gene_edges_plus_compound_and_gene_similarity_edges'
                OR node_representation_strategy IS DISTINCT FROM 'cell_painting_profile_features'
                OR query_entity_type IS DISTINCT FROM 'compound'
                OR candidate_entity_type IS DISTINCT FROM 'gene'
                OR query_entity_identifier IS DISTINCT FROM compound_inchikey14
                OR candidate_entity_identifier IS DISTINCT FROM gene_symbol
                OR number_of_candidates_scored_for_query IS DISTINCT FROM 7977
                OR is_connection_in_available_annotations IS DISTINCT FROM annotated
                OR (candidate_rank_within_query > 1
                  AND model_pair_score > previous_score)),
              (SELECT count(*) FROM bad_groups)
            FROM checked
            """,
            [str(path), str(root / "inputs/annotations/compound_gene.parquet")],
        )
        .fetchone()
    )
    if stats != (QUERIES * TOP_K, QUERIES, 0, 0):
        raise ValueError(f"Output invariant failure: {stats}")


@torch.inference_mode()
def main():
    root = Path.cwd()
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("Pilot inference requires a CUDA GPU")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    profiles = load_profiles(root)
    model, data, mappings = load_model(root, profiles, torch.device("cuda"))
    nodes = {
        "source": model.source_emb(data["source"].node_id),
        "target": model.target_emb(data["target"].node_id),
    }
    embeddings = model.gnn(nodes, data.edge_index_dict)
    check_benchmark(root, embeddings, mappings)
    query_positions = np.linspace(0, COMPOUNDS - 1, QUERIES, dtype=np.int64)
    scores = (
        embeddings["source"][torch.as_tensor(query_positions, device="cuda")]
        @ embeddings["target"].T
    )
    if not torch.isfinite(scores).all():
        raise ValueError("Pilot score matrix contains non-finite values")
    values, indices = topk(scores, TOP_K, 1)
    predictions = output_table(root, profiles, values, indices, query_positions)
    expected_queries = profiles["source"].index.astype(str).to_numpy()[query_positions]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.parquet")
    if temporary.exists():
        raise FileExistsError(f"Refusing to replace existing output: {temporary}")
    predictions.to_parquet(temporary, compression="zstd", index=False)
    validate(root, temporary, expected_queries)
    temporary.rename(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"Wrote {len(predictions):,} validated rows to {output}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
