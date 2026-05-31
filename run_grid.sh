#!/usr/bin/env bash
# Run the MOTIVE benchmark grid: one full train.smk pipeline (train -> infer ->
# metrics -> plots) per config. Each config is keyed by its own md5 hash, so
# reruns skip already-completed configs (Snakemake won't redo existing outputs).
#
# Run it inside the pixi env, e.g. on a GPU server:
#   nix develop -c pixi run bash run_grid.sh
#
# Defaults to the full paper grid. Override any axis (space-separated) or the
# epoch count via env vars. Quick smoke:
#   EPOCHS=3 GRAPHS=bipartite SPLITS=random TARGETS=orf MODELS="gnn:cp cosine:-" \
#     bash run_grid.sh
#
# Note: hyperparameters (hidden_channels/lr/wd/neg_ratio) come from gnn.json and
# are held constant across the grid - fine for a sweep/sanity pass, but the paper
# used per-config tuned values (see explore.smk).
set -uo pipefail

OUT="${OUT:-outputs}"
EPOCHS="${EPOCHS:-200}"
CORES="${CORES:-4}"
read -ra GRAPHS  <<< "${GRAPHS:-bipartite st_expanded}"
read -ra SPLITS  <<< "${SPLITS:-random source target}"
read -ra TARGETS <<< "${TARGETS:-orf crispr}"
# model:init pairs; init "-" means the model takes no initialization flag.
read -ra MODELS  <<< "${MODELS:-gnn:cp gnn:embs gat:cp gat:embs gin:cp gin:embs mlp:- bilinear:- cosine:-}"

n=0 fail=0 failed=()
for tgt in "${TARGETS[@]}"; do
  for split in "${SPLITS[@]}"; do
    for graph in "${GRAPHS[@]}"; do
      for mi in "${MODELS[@]}"; do
        model="${mi%%:*}"; init="${mi##*:}"
        n=$((n + 1))
        label="$tgt/$split/$graph/$model${init:+:$init}"
        cfg=(model="$model" graph_type="$graph" leave_out="$split" target_type="$tgt" num_epochs="$EPOCHS")
        [ "$init" != "-" ] && cfg+=(initialization="$init")
        echo ">>> [$n] $label"
        if ! snakemake -s train.smk --configfile gnn.json \
             --config output_path="$OUT" "${cfg[@]}" --cores "$CORES"; then
          echo "!!! FAILED: $label"
          fail=$((fail + 1)); failed+=("$label")
        fi
      done
    done
  done
done

echo "=== grid done: $n configs run, $fail failed ==="
[ "$fail" -gt 0 ] && printf '  - %s\n' "${failed[@]}"
exit 0
