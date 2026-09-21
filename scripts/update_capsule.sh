#!/bin/bash
# Refresh results_capsule/ from the deep3 results tree: every metrics.json (never per_user.npz or
# test_topk.npy), the test-read ledgers, the job lists, summary.json and revision_stats.json.
# Run from the repository root on the laptop:  bash scripts/update_capsule.sh [ssh-host]
set -eu
HOST=${1:-deep3-vpn}
cd "$(dirname "$0")/.."
mkdir -p results_capsule/results
ssh -o BatchMode=yes "$HOST" 'cd ~/towerbench/results && find . -name metrics.json -o -name test_read_ledger.jsonl | tar cf - -T -' | tar xf - -C results_capsule/results
ssh -o BatchMode=yes "$HOST" 'cd ~/towerbench && tar cf - jobs.txt jobs_rev_gpu.txt jobs_rev_cpu.txt jobs_rev_lam_extra.txt jobs_rev_lam2.txt jobs_rev_noise.txt jobs_rev_gowalla_tt.txt jobs_rev_probe.txt paper_assets/summary.json paper_assets/revision_stats.json' | tar xf - -C results_capsule
mv -f results_capsule/paper_assets/summary.json results_capsule/summary.json
mv -f results_capsule/paper_assets/revision_stats.json results_capsule/revision_stats.json
rmdir results_capsule/paper_assets
echo "capsule: $(find results_capsule/results -name metrics.json | wc -l) result files, $(ls results_capsule/jobs*.txt | wc -l) job lists"
