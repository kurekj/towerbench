#!/bin/bash
# Runs unattended on deep3 (tmux session rev_finish): waits until every revision worker session has
# exited, then regenerates the paper assets, the window/seed statistics and the marked manuscript.
# Log: ~/towerbench/logs/finish_revision.log
set -u
source ~/miniforge3/etc/profile.d/conda.sh; conda activate towerbench
cd ~/towerbench/towerbench_pkg
echo "[$(date +%F_%T)] waiting for workers: $(tmux ls 2>/dev/null | grep -E '^rev_(gpu|cpu|lam)' | cut -d: -f1 | tr '\n' ' ')"
while tmux ls 2>/dev/null | grep -qE '^rev_(gpu|cpu|lam)'; do sleep 300; done
echo "[$(date +%F_%T)] all workers finished: done=$(grep -h DONE ~/towerbench/logs/rev_worker_*.log | wc -l) failed=$(grep -h FAILED ~/towerbench/logs/rev_worker_*.log | wc -l)"
grep -h FAILED ~/towerbench/logs/rev_worker_*.log
python scripts/make_paper_assets.py && echo "[$(date +%F_%T)] assets done"
python scripts/revision_stats.py > ~/towerbench/logs/revision_stats_final.log 2>&1 && echo "[$(date +%F_%T)] stats done"
cd ~/towerbench/paper_assets && pdflatex -interaction=nonstopmode towerbench-R1-marked.tex > r1.log 2>&1 && pdflatex -interaction=nonstopmode towerbench-R1-marked.tex > r1.log 2>&1 && echo "[$(date +%F_%T)] pdf done: $(pdfinfo towerbench-R1-marked.pdf | grep Pages)"
echo "[$(date +%F_%T)] FINISHED"
