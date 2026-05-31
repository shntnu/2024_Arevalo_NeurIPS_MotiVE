#!/usr/bin/env bash
# Run the MOTIVE benchmark grid: one train -> infer -> metrics pipeline per
# config. Each config is keyed by its own md5 hash, so reruns skip completed
# configs (Snakemake won't redo existing outputs).
#
# Run it inside the pixi env, e.g. on a GPU server:
#   nix develop -c pixi run bash run_grid.sh
#
# Defaults to the metrics_only Snakemake target (no per-config plots - you make
# the interpretive plots for a chosen model, not all ~100 grid cells). Override
# any axis (space-separated), the epoch count, or the target via env vars.
# Quick smoke:
#   EPOCHS=6 GRAPHS=bipartite SPLITS=random TARGETS=orf MODELS="gnn:cp mlp:-" \
#     bash run_grid.sh
#
# NOTE: hyperparameters come from gnn.json (GNN-tuned). Per-model overrides
# below keep the non-GNN baselines runnable, but a real benchmark should tune
# HPs per config (see explore.smk). cosine is intentionally absent: it needs
# equal source/target feature dims, and MOTIVE's are 737 vs 722.
set -uo pipefail

OUT="${OUT:-outputs}"
EPOCHS="${EPOCHS:-200}"
CORES="${CORES:-4}"
SNAKE_TARGET="${TARGET:-metrics_only}"
read -ra GRAPHS  <<< "${GRAPHS:-bipartite st_expanded}"
read -ra SPLITS  <<< "${SPLITS:-random source target}"
read -ra TARGETS <<< "${TARGETS:-orf crispr}"
# model:init pairs; init "-" means the model takes no initialization flag.
read -ra MODELS  <<< "${MODELS:-gnn:cp gnn:embs gat:cp gat:embs gin:cp gin:embs mlp:- bilinear:-}"

# Per-model HP overrides (gnn.json is GNN-tuned: hidden 1024, neg_ratio 100):
#  - mlp: hidden 1024 OOMs in its bilinear head; use a small hidden.
#  - bilinear: runs nn.Bilinear on raw features (737x722), so backward memory
#    scales with the batch's negative count; cap neg_ratio. Tune for real runs.
model_overrides() {
  case "$1" in
    mlp)      echo "hidden_channels=64" ;;
    bilinear) echo "neg_ratio=20" ;;
    *)        echo "" ;;
  esac
}

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
        read -ra extra <<< "$(model_overrides "$model")"
        echo ">>> [$n] $label  ${extra[*]:-}"
        if ! snakemake -s train.smk "$SNAKE_TARGET" --configfile gnn.json \
             --config output_path="$OUT" "${cfg[@]}" "${extra[@]}" --cores "$CORES"; then
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
