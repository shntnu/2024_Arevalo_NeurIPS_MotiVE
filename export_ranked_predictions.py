#!/usr/bin/env python3
"""Export bidirectional top-50 MotiVE predictions to one Parquet file."""

import json
import sys
from pathlib import Path

import duckdb

TOP_K = 50
EXPECTED_RUNS = 36
EXPECTED_ROWS = 5_466_600

SPLITS = {
    "random": "connections_randomly_held_out_from_training",
    "source": "compounds_held_out_from_training",
    "target": "genes_held_out_from_training",
}
GRAPHS = {
    "bipartite": "compound_gene_edges_only",
    "st_expanded": "compound_gene_edges_plus_compound_and_gene_similarity_edges",
}
MODELS = {
    "gnn": "graphsage",
    "gin": "graph_isomorphism_network",
    "mlp": "multilayer_perceptron",
    "bilinear": "bilinear_scoring_model",
}


def sql_string(value: Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def select_sql(metadata: list[str], paths: list[str], direction: str) -> str:
    query_column, candidate_column = (
        ("target", "source") if direction == "gene" else ("source", "target")
    )
    return (
        f"""
        WITH scored AS (
            SELECT
                p.source,
                p.target,
                p.score,
                count(*) OVER (PARTITION BY p.{query_column}) AS candidate_count,
                count(*) OVER (PARTITION BY p.{query_column}, p.score) AS tie_count,
                a.source IS NOT NULL AS is_annotated
            FROM read_parquet(?) AS p
            LEFT JOIN (
                SELECT DISTINCT source, target FROM read_parquet(?)
            ) AS a USING (source, target)
        ), ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY {query_column}
                    ORDER BY score DESC, {candidate_column}
                ) AS candidate_rank
            FROM scored
        )
        SELECT
            ? AS gene_perturbation_type,
            ? AS evaluation_split_strategy,
            ? AS graph_configuration,
            ? AS model_architecture,
            ? AS node_representation_strategy,
            ? AS model_run_id,
            '{direction}' AS query_entity_type,
            CASE WHEN '{direction}' = 'gene'
                THEN CAST(t.Metadata_Symbol AS VARCHAR)
                ELSE CAST(s.Metadata_InChIKey AS VARCHAR)
            END AS query_entity_identifier,
            CASE WHEN '{direction}' = 'gene' THEN 'compound' ELSE 'gene' END
                AS candidate_entity_type,
            CASE WHEN '{direction}' = 'gene'
                THEN CAST(s.Metadata_InChIKey AS VARCHAR)
                ELSE CAST(t.Metadata_Symbol AS VARCHAR)
            END AS candidate_entity_identifier,
            CAST(t.Metadata_Symbol AS VARCHAR) AS gene_symbol,
            CAST(s.Metadata_InChIKey AS VARCHAR) AS compound_inchikey14,
            r.score AS model_ranking_score,
            r.candidate_rank AS candidate_rank_within_query,
            r.candidate_count AS number_of_candidates_scored_for_query,
            r.tie_count AS number_of_candidates_tied_at_score_within_query,
            r.is_annotated AS is_connection_in_benchmark_annotations
        FROM ranked AS r
        JOIN read_parquet(?) AS s ON r.source = s."0"
        JOIN read_parquet(?) AS t ON r.target = t."0"
        WHERE r.candidate_rank <= {TOP_K}
    """,
        paths[:2] + metadata + paths[2:],
    )


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    output = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else root
        / "outputs/prediction_exports/motive_top50_bidirectional_predictions.parquet"
    )
    configs = []
    for path in sorted((root / "outputs/orf").glob("*/*/*/*/config.json")):
        config = json.loads(path.read_text())
        result = path.parent / "cartesian/test/results.parquet"
        if (
            result.exists()
            and config.get("leave_out") in SPLITS
            and config.get("graph_type") in GRAPHS
            and config.get("model") in MODELS
        ):
            configs.append((path, config, result))
    assert len(configs) == EXPECTED_RUNS, (
        f"expected {EXPECTED_RUNS} runs, found {len(configs)}"
    )

    con = duckdb.connect()
    first = True
    for path, config, result in configs:
        graph = config["graph_type"]
        target = config["target_type"]
        split = config["leave_out"]
        node_strategy = (
            "cell_painting_profile_features"
            if config["model"] in {"mlp", "bilinear"}
            or config.get("initialization") == "cp"
            else "learned_node_embeddings"
        )
        metadata = [
            "orf_overexpression",
            SPLITS[split],
            GRAPHS[graph],
            MODELS[config["model"]],
            node_strategy,
            path.parent.name,
        ]
        paths = [
            str(result),
            str(root / f"data/{graph}/{target}/{split}/s_t_labels.parquet"),
            str(root / f"data/{graph}/{target}/source_map.parquet"),
            str(root / f"data/{graph}/{target}/target_map.parquet"),
        ]
        for direction in ("gene", "compound"):
            query, parameters = select_sql(metadata, paths, direction)
            command = (
                "CREATE TABLE predictions AS " if first else "INSERT INTO predictions "
            )
            con.execute(command + query, parameters)
            first = False

    row_count = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
    assert row_count == EXPECTED_ROWS, (
        f"expected {EXPECTED_ROWS:,} rows, found {row_count:,}"
    )
    assert not con.execute("""
        SELECT EXISTS (
            SELECT 1
            FROM predictions
            GROUP BY model_run_id, query_entity_type, query_entity_identifier
            HAVING count(*) != 50 OR min(candidate_rank_within_query) != 1
                OR max(candidate_rank_within_query) != 50
        )
    """).fetchone()[0]
    assert not con.execute("""
        SELECT EXISTS (
            SELECT 1
            FROM predictions
            GROUP BY model_run_id, query_entity_type, gene_symbol, compound_inchikey14
            HAVING count(*) > 1
        )
    """).fetchone()[0]
    assert not con.execute("""
        SELECT EXISTS (
            SELECT 1 FROM predictions
            WHERE query_entity_identifier IS NULL
               OR candidate_entity_identifier IS NULL
               OR model_ranking_score IS NULL
        )
    """).fetchone()[0]
    assert con.execute("""
        SELECT bool_or(is_connection_in_benchmark_annotations)
           AND bool_or(NOT is_connection_in_benchmark_annotations)
        FROM predictions
    """).fetchone()[0]

    output.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"COPY predictions TO {sql_string(output)} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    print(
        con.execute("""
        SELECT evaluation_split_strategy, query_entity_type,
               count(*) AS rows,
               count(*) FILTER (WHERE is_connection_in_benchmark_annotations)
                   AS annotated_rows
        FROM predictions
        GROUP BY ALL
        ORDER BY 1, 2
    """)
        .fetchdf()
        .to_string(index=False)
    )
    print(f"\nWrote {row_count:,} rows to {output}")


if __name__ == "__main__":
    main()
