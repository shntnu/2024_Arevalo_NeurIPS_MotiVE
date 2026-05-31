#!/usr/bin/env bash
# Run the MOTIVE benchmark grid: one train -> infer -> metrics pipeline per
# config. Each config is keyed by its own md5 hash, so reruns skip completed
# configs (Snakemake won't redo existing outputs).
#
# Run it inside the pixi env, e.g. on a GPU server:
#   nix develop -c pixi run bash run_grid.sh
#
# These models are tiny (a 200-epoch GNN uses ~7 GB GPU, ~0% util - the work is
# CPU-bound: neighbor sampling, negative sampling, mAP nulls). So on a big box,
# run many configs at once, round-robin them across the GPUs, and thread-cap each:
#   GPUS=4 JOBS=16 THREADS=16 nix develop -c pixi run bash run_grid.sh
# JOBS>1 adds --nolock (safe: configs write to disjoint hash dirs) and logs each
# config under logs/. GPUS round-robins CUDA_VISIBLE_DEVICES so configs spread
# over the GPUs (one GPU can't hold many of these at ~7 GB each). Defaults
# (JOBS=1, GPUS=1) run sequentially on GPU 0 with normal locking.
#
# Defaults to the metrics_only target (no per-config plots). Override any axis
# (space-separated), epochs, or target via env vars. Quick smoke:
#   EPOCHS=6 GRAPHS=bipartite SPLITS=random TARGETS=orf MODELS="gnn:cp mlp:-" \
#     bash run_grid.sh
#
# NOTE: HPs come from gnn.json (GNN-tuned); per-model overrides below keep the
# baselines runnable, but a real benchmark tunes HPs per config (explore.smk).
# cosine is absent: it needs equal src/tgt feature dims; MOTIVE's are 737 vs 722.
set -uo pipefail

OUT="${OUT:-outputs}"
EPOCHS="${EPOCHS:-200}"
CORES="${CORES:-2}"
JOBS="${JOBS:-1}"          # how many configs to run concurrently
GPUS="${GPUS:-1}"          # number of GPUs to round-robin configs across
THREADS="${THREADS:-}"     # per-config CPU thread cap (OMP/MKL); empty = unset
SNAKE_TARGET="${TARGET:-metrics_only}"
read -ra GRAPHS  <<< "${GRAPHS:-bipartite st_expanded}"
read -ra SPLITS  <<< "${SPLITS:-random source target}"
read -ra TARGETS <<< "${TARGETS:-orf crispr}"
read -ra MODELS  <<< "${MODELS:-gnn:cp gnn:embs gat:cp gat:embs gin:cp gin:embs mlp:- bilinear:-}"

# --rerun-incomplete: recover configs left half-written by a killed run.
snake_flags=(--rerun-incomplete)
[ "$JOBS" -gt 1 ] && snake_flags+=(--nolock)
RESULTS="$(mktemp)"
mkdir -p logs

# Per-model HP overrides (gnn.json is GNN-tuned: hidden 1024, neg_ratio 100):
#  - mlp: hidden 1024 OOMs in its bilinear head; use a small hidden.
#  - bilinear: nn.Bilinear on raw features (737x722); backward memory scales
#    with the batch negative count, so cap neg_ratio. Tune for real runs.
model_overrides() {
  case "$1" in
    mlp)      echo "hidden_channels=64" ;;
    bilinear) echo "neg_ratio=20" ;;
    *)        echo "" ;;
  esac
}

run_one() {
  local gpu="$1" label="$2"; shift 2
  local log="logs/${label//\//_}.log"
  local pfx=(env CUDA_VISIBLE_DEVICES="$gpu")
  [ -n "$THREADS" ] && pfx+=(OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS")
  if "${pfx[@]}" snakemake -s train.smk "$SNAKE_TARGET" --configfile gnn.json \
       --config "$@" --cores "$CORES" "${snake_flags[@]}" >"$log" 2>&1; then
    echo "OK    [gpu$gpu] $label" | tee -a "$RESULTS"
  else
    echo "FAIL  [gpu$gpu] $label  (tail: $(tail -n1 "$log"))" | tee -a "$RESULTS"
  fi
}

n=0
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
        gpu=$(( (n - 1) % GPUS ))
        echo ">>> [$n] launch $label  [gpu$gpu]  ${extra[*]:-}"
        run_one "$gpu" "$label" output_path="$OUT" "${cfg[@]}" "${extra[@]}" &
        # throttle to JOBS concurrent
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
      done
    done
  done
done
wait

ok=$(grep -c '^OK'   "$RESULTS" || true)
fail=$(grep -c '^FAIL' "$RESULTS" || true)
echo "=== grid done: $n configs, $ok ok, $fail failed ==="
grep '^FAIL' "$RESULTS" || true
rm -f "$RESULTS"
exit 0
