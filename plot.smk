import plot


rule waterfall:
    input:
        config="{output_path}/config.json",
        preds="{output_path}/{infer_mode}/{subset}/results.parquet",
        # plot.waterfall reads this annotation table directly. Declaring it as
        # an input makes the inputs/ dependency explicit so the DAG fetches it
        # (via download_from_s3) instead of failing late when only data/ exists.
        annotations="inputs/annotations/compound_gene.parquet",
    output:
        "{output_path}/{infer_mode}/{subset}/analysis/waterfall.pdf",
    run:
        plot.waterfall(input.config, input.preds, *output)


rule heatmap:
    input:
        "{output_path}/config.json",
        "{output_path}/{infer_mode}/{subset}/results.parquet",
    output:
        "{output_path}/{infer_mode}/{subset}/analysis/heatmap.png",
    run:
        plot.heatmap(*input, *output)


rule umap:
    input:
        "{output_path}/config.json",
        "{output_path}/weights.pt",
    output:
        "{output_path}/umap.parquet",
    run:
        plot.umap(*input, *output)


rule scatter:
    input:
        "{output_path}/umap.parquet",
    output:
        "{output_path}/scatter.png",
    run:
        plot.scatter(*input, *output)


rule bipartite_target_knn_baseline:
    input:
        "{output_path}/{infer_mode}/{subset}/results.parquet",
    output:
        "{output_path}/{infer_mode}/{subset}/analysis/bipartite_target_knn_baseline.pdf",
    run:
        plot.bipartite_target_knn_baseline(*input, *output)
