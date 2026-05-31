# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Code for the NeurIPS 2024 paper "MOTIVE: A Drug-Target Interaction Graph For Inductive Link Prediction". It builds a heterogeneous graph from JUMP Cell Painting data and trains link-prediction models that predict compound-gene (drug-target) interactions, with an emphasis on *inductive* generalization to unseen compounds or genes (cold splits).

**You are on the `motivev2` branch** (version 0.2.0), which has diverged substantially from `main` (the frozen NeurIPS-paper state). If you read docs, issues, or the Wiki that describe a `run_training.py` script, a conda `environment.yml`, or Hits@500 model selection, those describe `main` and are stale here. The differences are summarized at the end.

## Environment and commands

Dependencies are managed with **uv** (not conda), pinned in `pyproject.toml` + `uv.lock`: torch 2.5.1 / CUDA 12.4 wheels (via the `[tool.uv] find-links` pyg index), torch-geometric 2.6.1 and the pyg sparse-op stack, snakemake 8, Python 3.12.

```bash
uv sync                       # create .venv and install locked deps
source .venv/bin/activate     # or prefix commands with `uv run`
```

A Nix `flake.nix` + `.envrc` provide a dev shell (C runtime libs the wheels link against, plus `uv`, `duckdb`). `direnv allow` auto-enters it; or `nix develop`. The shell runs `uv sync` on entry. The flake is cross-platform: **x86_64-linux** gets CUDA and torch from nixpkgs (the GPU servers - spirit, oppy, karkinos); **macOS** skips CUDA/nix-torch and lets uv install CPU/MPS torch wheels (so a Mac can edit and smoke-test, but real training wants a GPU box).

Build the dataset (downloads `inputs/` from S3, generates `data/`). Alternatively `aws s3 sync` the prebuilt `data/` and `inputs/` (see README):

```bash
snakemake -c1   # default Snakefile; -cN for N cores
```

Train + infer + evaluate + plot one model (the whole pipeline for one config):

```bash
snakemake -s train.smk --configfile gnn.json --config output_path=outputs/
```

Random hyperparameter search:

```bash
snakemake -s explore.smk --configfile gnn.json --config output_path=optimize num_search=10
```

Run the tests / formatters (dev deps, not enforced in CI):

```bash
pytest tests/             # currently just tests/test_bpr.py
# pre-commit only runs `trailing-whitespace`; ruff (via python-lsp-ruff), yapf, and snakefmt are available but not hooked
```

## Configuration: one config dict, expanded into the Snakemake DAG

There is **no longer a `configs/` directory of per-split JSON files** (that was `main`). A single template lives at the repo root - `gnn.json` - and you override fields on the command line with `--config key=value`. The config holds every axis of a run:

```json
{ "leave_out": "target", "graph_type": "bipartite", "target_type": "orf",
  "neg_ratio": 100, "model": "gnn", "num_epochs": 200, "eval_freq": 5,
  "initialization": "cp", "hidden_channels": 1024,
  "learning_rate": 7.23e-05, "weight_decay": 0.0040 }
```

`train.smk` computes `config["hash"] = hashname(config)` (6-hex md5 of the config) at parse time and uses **every config key as a Snakemake wildcard**, so outputs land in a human-readable, content-addressed tree (not the bare md5 dir `main` used):

```
{output_path}/{target_type}/{leave_out}/{graph_type}/{model}/{hash}/
  config.json, weights, runs/ (tensorboard),
  {sampled,cartesian}/{train,valid,test}/results.parquet, metrics/*.npy, analysis/*,
  umap.parquet, scatter.png
```

The three axes you vary (`graph_type`, `leave_out`, `target_type`) are unchanged in meaning from `main` - see the next two sections. NOTE the config key is **`leave_out`** here (it was `data_split` on `main`) and the model key is **`model`** (was `model_name`).

## The two axes you configure

- **graph_type** (`bipartite`, `s_expanded`, `t_expanded`, `st_expanded`): which similarity edges augment the core compound-gene bipartite graph. `s_expanded` adds compound-compound (`source-similar-source`) edges, `t_expanded` adds gene-gene (`target-similar-target`), `st_expanded` adds both.
- **leave_out** (`random`, `source`, `target`): `random` is transductive; `source` and `target` are the inductive "cold" splits that hold out entire compounds or genes from train.

`target_type` (`orf` or `crispr`) selects which JUMP gene-perturbation modality supplies target node features. `neg_ratio` is the per-batch negative:positive sampling ratio (now a config field, was hardcoded on `main`).

## Vocabulary mapping

The code is deliberately generic ("source"/"target") so it can be reused; the MOTIVE instantiation is:

- **source** = compound, **target** = gene (ORF or CRISPR perturbation).
- **`binds` edge** (`source-binds-target`) = the compound-gene interaction being predicted.
- **node `.x`** = Cell Painting morphological profile features (or learned embeddings, see initialization below).

## Architecture

**Snakemake is the orchestrator (four `.smk` files).** Everything runs through Snakemake, not a hand-run training script:
- `Snakefile` (+ `rules/jump.smk`): the **data pipeline**. Downloads raw parquet from S3, builds node features and the three label tables (`s_s_labels`, `s_t_labels`, `t_t_labels`), merges per graph type, and splits into train/valid/test. `rule all` materializes every `data/{graph_type}/{tgt_type}/{split}/s_t_labels.parquet`.
- `train.smk` (includes `plot.smk`): the **train -> infer -> metrics -> plot** pipeline for one config. Rules: `train`, `infer_sampled`, `infer_cartesian`, per-metric `.npy` rules collated by `metrics`, `register_tensorboard`, and the plot rules.
- `explore.smk`: random **hyperparameter search** - generates a `param_search` table and one config per sample, runs each, and renders TensorBoard contour plots over (learning_rate, weight_decay).

**Python entry points the Snakemake rules call (`mworkflow.py`):** `init(config_path)` reads the config JSON, builds loaders via `get_loaders(leave_out, target_type, graph_type, neg_ratio)`, and creates the model with `create_model(config, train_data)`. `train()` wraps `train_loop`; `infer_sampled()` and `infer_cartesian()` load the checkpoint and run `run_test`. **Two inference modes**: *sampled* uses the `LinkNeighborLoader` (neighbor-sampled subgraphs, like training); *cartesian* uses `get_cartesian_loader` to score **all** source-target pairs for complete ranking tables.

**Data pipeline internals (`motive/jump.py`, `split.py`, `store_splits.py`):** cold splits hold out nodes by binning a degree-like column into deciles (`split_per_column_value`) and propagate held-out nodes into same-type edges (`split_same_type`) with explicit leakage assertions. Same scheme as `main`.

**message vs supervision edges:** the train set is partitioned - ~60% "message" edges (GNN message passing / connectivity) and the rest "supervision" edges (labeled positives). `motive/base.py::load_graph_helper` encodes the escalation: train sees only message edges as structure, valid additionally sees train edges, test additionally sees valid edges.

**Graph loading (`motive/base.py`):** parquet -> PyG `HeteroData`, edges made undirected, wrapped in a `LinkNeighborLoader` (4-hop full neighbor sampling, `subgraph_type="bidirectional"`). `get_loaders` returns train (bsz 512, shuffle) / valid / test (bsz 8192) loaders; `get_cartesian_loader` is the all-pairs variant for cartesian inference.

**Negative sampling (`motive/sample_negatives.py`):** `SampleNegatives` is a PyG transform applied *per batch* at load time. It samples non-edges on GPU, restricts the sampleable node set by split (`select_nodes_to_sample`) so cold-split test nodes don't leak into train negatives, and - new on this branch - builds the **BPR pairing** for the batch (`bpr_indices`, `bpr_weights`) via `motive/bpr.py::create_positive_negative_pairs` (per source node, all positive x negative pairs, weighted by `1/count`).

**Models (`model.py`):** `create_model(config, data)` dispatches on `config["model"]`:
- `gnn`/`gat`/`gin`: a 2-layer homogeneous GNN (`GNN`=GraphSAGE, `GAT`=GATv2, `GIN`) made heterogeneous via `to_hetero`, wrapped in `GraphSAGE_Embs`. The `initialization` field picks `embs` (learned `nn.Embedding`) vs `cp` (`GraphSAGE_CP`: embeddings frozen-initialized from Cell Painting features, then a trainable linear projection). A dot-product `Classifier` produces edge logits.
- `mlp`, `bilinear`: operate directly on node features, no message passing.
- `cosine`: parameter-free baseline. Has no trainable parameters, so `train_loop` sets `optimizer = None` and skips backward - watch for this when touching the training loop.

**Training (`train.py`):** `train_loop(model, model_path, config, train_loader, val_loader)` optimizes a **weighted BPR loss** (`-logsigmoid(pos - neg) * bpr_weights`, indexed by `data.bpr_indices`), not plain BCE. **Model selection checkpoints on minimum validation loss** (`best_loss`), evaluated every `eval_freq` epochs - this replaced `main`'s Hits@500 criterion. `SEED = 2024313`. `run_test` produces the `results.parquet` table (source, target, score, logits, y_pred, y_true).

**Evaluation - two modules, don't confuse them:** `utils/evaluate.py` holds the config-driven `Evaluator` class (AUC, Hits@K, Precision@K, F1, BCELoss, mAP) used inside the training loop. Top-level `evaluate.py` is the **Snakemake metric layer**: it reads `results.parquet` and writes one `.npy` per metric (acc, roc_auc, hits_at_500, precision_at_500, f1, mrr, bce, plus per-node `mAP`, phenotypic activity, and success@K, with random baselines), then `collate`s them into `metrics.parquet` and registers to TensorBoard. mAP is computed via `copairs` (the lab's retrieval-metric library).

**Plotting (`plot/`):** `waterfall.py` (oncoPrint-style top-K predicted interactions per compound), `heatmap.py`, `knn_baseline.py` (GNN vs a kNN-on-CP-features baseline), `projection.py` (UMAP of model hidden activations, colored by node type and seen/unseen), `exploration.py` (HP-sweep contour plots). `plot/base.py::to_numpy` renders figures for TensorBoard.

**Other top-level scripts** are analysis/ablation utilities, mostly run by hand or by `explore.smk`: `predict_all.py` (full-graph one-pass inference), `connected.py` (connectivity / external-DB annotation of predictions), `filter_rank_compounds.py`, `morph_sim_annotate.py`, `spectral_clust.py`, `compute_all_metrics_dict.py`, `query.sql`. The `new_profiles/` directory holds variant Cell Painting profile-construction scripts (centering by modality/microscope, phenotypic-activity filtering, single-source) for feature-engineering ablations.

## What changed from `main` (quick reference)

- conda `environment.yml` (torch 2.1.2) -> **uv** `pyproject.toml`/`uv.lock` (torch 2.5.1), plus a Nix flake.
- single `run_training.py` per config -> **Snakemake** pipelines (`train.smk`/`explore.smk`/`plot.smk` + data `Snakefile`).
- `PathLocator` md5-dir outputs -> human-readable wildcard tree keyed by the same hash.
- BCE loss + **Hits@500** model selection -> weighted **BPR loss** + **min validation loss** selection (`motive/bpr.py`).
- single sampled inference -> **sampled + cartesian** (all-pairs) inference.
- config keys renamed: `data_split` -> `leave_out`, `model_name` -> `model`; `neg_ratio` now configurable; added kNN/random baselines, a test suite (`tests/test_bpr.py`), and extensive plotting.

## Reproducibility notes

Seeds are intentional and split across files: `train.py::SEED = 2024313` (training) and `motive/split.py::SEED = [2023, 7, 12]` (data splits). The full operational documentation (data dictionary, directory contents) is in the [project Wiki](https://github.com/carpenter-singh-lab/motive/wiki), not in this repo.
