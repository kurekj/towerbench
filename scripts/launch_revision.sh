#!/bin/bash
# Launch the revision (v1.1.0) job queues on deep3 on top of the finished v1.0.0 sweep:
# two workers per GPU on the GPU queue (sensitivity grids, LightGCN, noise runs) and one worker
# on the CPU queue (EASE lambda grid on Inside Airbnb, solved on the CPU).  Every worker skips
# finished jobs, so the script can be re-run after an interruption.
set -u
source ~/miniforge3/etc/profile.d/conda.sh; conda activate towerbench
cd ~/towerbench/towerbench_pkg
export TOKENIZERS_PARALLELISM=false
mkdir -p ~/towerbench/logs/runs
python scripts/gen_jobs_revision.py > ~/towerbench/jobs_rev_gpu.txt
python scripts/gen_jobs_revision.py --queue cpu > ~/towerbench/jobs_rev_cpu.txt
echo "[$(date +%F_%T)] revision jobs: gpu=$(wc -l < ~/towerbench/jobs_rev_gpu.txt) cpu=$(wc -l < ~/towerbench/jobs_rev_cpu.txt)"
ACT="source ~/miniforge3/etc/profile.d/conda.sh; conda activate towerbench; cd ~/towerbench/towerbench_pkg;"
for g in 0 1; do
  for k in a b; do
    tmux has-session -t rev_gpu${g}${k} 2>/dev/null && continue
    tmux new-session -d -s rev_gpu${g}${k} "$ACT bash scripts/worker.sh $g ~/towerbench/jobs_rev_gpu.txt > ~/towerbench/logs/rev_worker_gpu${g}${k}.log 2>&1"
  done
done
tmux has-session -t rev_cpu 2>/dev/null || \
  tmux new-session -d -s rev_cpu "$ACT bash scripts/worker.sh 0 ~/towerbench/jobs_rev_cpu.txt > ~/towerbench/logs/rev_worker_cpu.log 2>&1"
echo "[$(date +%F_%T)] workers:"; tmux ls
