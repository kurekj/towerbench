#!/bin/bash
# GPU worker: walks a shared job list (one line: dataset protocol window model seed, see gen_jobs.py)
# and claims jobs with an atomic mkdir lock, so any number of workers (one per GPU) share one queue.
# Finished jobs are skipped.  Usage:  bash scripts/worker.sh <gpu_id> <jobs_file>
set -u
GPU=$1; JOBS=$2
ROOT=${TOWERBENCH_ROOT:-$HOME/towerbench}
export CUDA_VISIBLE_DEVICES=$GPU TOKENIZERS_PARALLELISM=false
LOCKS=$ROOT/results/.locks; mkdir -p "$LOCKS" "$ROOT/logs/runs"
while read -r D P W M S; do
  [ -z "$D" ] && continue
  TAG=$P; [ "$P" = temporal ] && TAG=${P}_w$W
  [ -f "$ROOT/results/$D/$TAG/$M/seed$S/metrics.json" ] && continue
  mkdir "$LOCKS/${D}_${TAG}_${M}_s$S" 2>/dev/null || continue
  echo "[$(date +%F_%T)] GPU$GPU START $D $TAG $M seed$S"
  towerbench run "$D" "$M" --seed "$S" --protocol "$P" --window "$W" > "$ROOT/logs/runs/${D}_${TAG}_${M}_s$S.log" 2>&1 \
    && echo "[$(date +%F_%T)] GPU$GPU DONE  $D $TAG $M seed$S" \
    || { echo "[$(date +%F_%T)] GPU$GPU FAILED $D $TAG $M seed$S"; rmdir "$LOCKS/${D}_${TAG}_${M}_s$S"; }
done < "$JOBS"
echo "[$(date +%F_%T)] GPU$GPU WORKER DONE"
