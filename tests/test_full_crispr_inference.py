import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

import infer_full_crispr_predictions as predictions
from infer_full_crispr_predictions import GENES, RUNS, SCHEMA, output_table


def test_directional_artifact_contract():
    identifiers = {
        "source": np.array(["AAA", "BBB", "CCC"]),
        "target": np.array(["GENE1", "GENE2"]),
    }
    known = np.array([2 * GENES])
    values = np.array([[9.0, 8.0], [7.0, 6.0]], dtype=np.float32)
    indices = np.array([[2, 1], [0, 2]])

    compounds = output_table(RUNS[0], values, indices, identifiers, known)
    assert tuple(compounds.columns) == tuple(name for name, _ in SCHEMA)
    assert compounds["query_entity_identifier"].tolist() == [
        "GENE1",
        "GENE1",
        "GENE2",
        "GENE2",
    ]
    assert compounds["candidate_entity_identifier"].tolist() == [
        "CCC",
        "BBB",
        "AAA",
        "CCC",
    ]
    assert compounds["is_connection_in_available_annotations"].tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert compounds["number_of_candidates_scored_for_query"].unique() == [115_790]

    gene_indices = np.array([[1, 0], [0, 1]])
    genes = output_table(RUNS[1], values, gene_indices, identifiers, known)
    assert genes["query_entity_type"].unique() == ["compound"]
    assert genes["candidate_entity_type"].unique() == ["gene"]
    assert genes["model_checkpoint_id"].unique() == ["be48db"]
    assert genes["query_entity_identifier"].tolist() == ["AAA", "AAA", "BBB", "BBB"]
    assert genes["candidate_entity_identifier"].tolist() == [
        "GENE2",
        "GENE1",
        "GENE1",
        "GENE2",
    ]
    assert not genes["is_connection_in_available_annotations"].any()
    assert genes["number_of_candidates_scored_for_query"].unique() == [7_977]
    assert genes["candidate_rank_within_query"].tolist() == [1, 2, 1, 2]


def test_benchmark_membership_uses_pairwise_scores(monkeypatch):
    embeddings = {
        "source": torch.tensor([[2.0, 0.0], [9.0, 9.0], [0.0, 2.0], [8.0, 8.0]]),
        "target": torch.tensor(
            [[1.0, 1.0], [0.0, 1.0], [7.0, 7.0], [1.0, 0.0], [2.0, 2.0]]
        ),
    }
    mappings = {"source": np.array([2, 0]), "target": np.array([4, 1, 3])}
    rows = []
    for source in range(2):
        for target in range(3):
            score = embeddings["source"][mappings["source"][source]].dot(
                embeddings["target"][mappings["target"][target]]
            )
            rows.append((source, target, score.item()))
    benchmark = pd.DataFrame(rows, columns=["source", "target", "logits"])
    monkeypatch.setattr(
        predictions.pd, "read_parquet", lambda *_args, **_kwargs: benchmark
    )
    monkeypatch.setattr(predictions, "TOP_K", 1)
    monkeypatch.setattr(
        predictions, "topk", lambda *_args: pytest.fail("unexpected matrix ranking")
    )
    predictions.check_benchmark(Path("."), RUNS[1], embeddings, mappings)
