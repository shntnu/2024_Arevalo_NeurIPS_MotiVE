# MOTI$`\mathcal{VE}`$

Source code and documentation for ["MOTI$`\mathcal{V}\mathcal{E}`$: A
Drug-Target Interaction Graph For Inductive Link
Prediction"](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fdb3fa770c2e0ecbb4b7dc7083ef5be9-Abstract-Datasets_and_Benchmarks_Track.html).

See the [Wiki](https://github.com/carpenter-singh-lab/motive/wiki) for full documentation, operational details and other information.

## Installation

We recommend using [uv](https://docs.astral.sh/uv/) for
environment management. The following commands clone the repository, create the environment, and install the required packages.

```bash
git clone https://github.com/carpenter-singh-lab/motive.git
git checkout motivev2
uv sync
source .venv/bin/activate
```

## Download data

The MOTI$`\mathcal{VE}`$ dataset files are available in the [Cell Painting Gallery viewer](https://cellpainting-gallery.s3.amazonaws.com/index.html#cpg0034-arevalo-su-motive/broad/workspace/publication_data/2024_MOTIVE).
We provide two options for programmatic access. Both will populate the working directory with the necessary gene-compound relationships, node features, and metadata. For more information about the directory contents, refer to the [Wiki page](https://github.com/carpenter-singh-lab/motive/wiki).

### Using aws-cli

The following command will download `inputs` and `data` folders:

```bash
aws s3 sync --no-sign-request s3://cellpainting-gallery/cpg0034-arevalo-su-motive/broad/workspace/publication_data/2024_MOTIVE .
```
### Run the snakemake pipeline
Alternatively, you can also run the [Snakemake](https://snakemake.readthedocs.io/en/v7.32.3/) pipeline included in this repo which downloads the necessary `inputs` and generates the `data` files.

```bash
snakemake -c1
```
With `1` being the number of cores you want to use.

## Train
Run the following command to train a model on the MOTI$`\mathcal{VE}`$ dataset. The config file should indicate the graph type (optimized configs are only provided for the `bipartite` and `st_expanded` graph structures), gene type, data split, and model. An example is provided below.

`snakemake -s train.smk --configfile gnn.json --config output_path=outputs/`

The training will produce a `results.parquet` file in the `outputs/` folder with the predicted scores for each source target pair in the test set.

|   source |   target |    score |    logits | y_pred   | y_true   |
|---------:|---------:|---------:|----------:|:---------|:---------|
|        4 |      172 | 0.374103 | -0.514653 | False    | True     |
|        5 |     1501 | 0.603371 |  0.419531 | True     | True     |
|        6 |      797 | 0.402376 | -0.395574 | False    | True     |
|        7 |      179 | 0.538556 |  0.154529 | True     | True     |
|        7 |      651 | 0.570341 |  0.283244 | True     | True     |


## Explore params

```bash
snakemake -s explore.smk --configfile gnn.json --config output_path=optimize num_search=10
```
