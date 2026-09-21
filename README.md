# towerbench

[![ci](https://github.com/kurekj/towerbench/actions/workflows/ci.yml/badge.svg)](https://github.com/kurekj/towerbench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Frozen-protocol, statistically rigorous benchmarking of two-tower retrieval models against
autoencoder and sequential recommenders on public implicit-feedback data with item side information.**

`towerbench` packages the evaluation regime of an industrial two-tower study as reusable software:
full-catalogue ranking, one shared metric implementation, chronological splits with rolling-origin
replication, paired per-user bootstrap intervals against a reference model, a ledger of every test
read, and one configuration switch per two-tower training technique. Seven public datasets ship
with reproducible loaders; the full sweep runs unattended on any number of GPUs.

## Features

* **Protocols** — `loo` (per-user temporal leave-last-out) and `temporal` (frozen chronological
  train/validation/test windows, rolling origin `--window 0..4`), both ranking the full catalogue.
  The split rules live in `towerbench/eval/protocol.py`, the metric definitions in `towerbench/eval/metrics.py`.
* **Statistics** — bootstrap CIs over users, paired bootstrap of per-user differences on identical
  users, several seeds, pooling across windows, Holm correction. Per-user metric vectors are stored,
  so any comparison is recomputable offline. Two optional variants surface the fine print of the
  pooled interval: `paired_bootstrap_seeds` resamples seeds as well as users, and
  `pooled_window_variants` recomputes a pooled rolling-origin interval with a cluster bootstrap over
  users, a block bootstrap over windows and a random-effects combination of the windows.
* **Models** — popularity, EASE (closed form, GPU), Mult-VAE, SASRec, LightGCN, and a two-tower model whose
  techniques are single switches: logQ-corrected in-batch softmax, mixed negative sampling, MixGCF
  hard negatives, DCN-v2 cross layers, ID dropout, multi-head attention pooling, frozen
  sentence-transformer text embeddings, geo / price / category side features.
* **Datasets** — MovieLens-1M, Amazon Reviews 2023 (three categories with item text and price),
  Gowalla (geo), Inside Airbnb (ten cities: text, geo, price, room type), Steam. All downloaded from
  their original hosts by one script; no registration or API key needed.
* **Audit trail** — every result file records split constants, fit diagnostics, wall-clock time and
  device; every test read is appended to `test_read_ledger.jsonl`. The ledger is written by the
  `towerbench run` entry point (`towerbench.run.run`), which is the only code path that reads the
  test split; it is a harness convention, not a hard guarantee: a caller who uses the Python API
  directly (`Split.positives("test")`) bypasses it.
* **Training-noise switch** — `towerbench run ... --noise 0.1` replaces 10 % of the training events by
  events on uniformly random items (validation and test untouched) to check whether a verdict survives
  unreliable feedback.

## Installation

```bash
git clone https://github.com/kurekj/towerbench && cd towerbench
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or .../whl/cpu
pip install -e .                                                       # or: pip install -r requirements-lock.txt
python -m pytest -q tests                                              # CPU, synthetic data, ~10 s
```

Python 3.10+; Linux, macOS and Windows. A GPU is optional but recommended for the two-tower and
SASRec models. Data, results and logs live under `~/towerbench` (override with `TOWERBENCH_ROOT`).

Pinned environments of the reported sweep: `requirements-lock.txt` (pip), `environment.yml` (conda)
and `Dockerfile` (PyTorch 2.11 / CUDA 12.8 image):

```bash
docker build -t towerbench . && docker run --gpus all -it -v $HOME/towerbench:/root/towerbench towerbench bash
```

EASE is solved on the GPU up to `gpu_max_items` items (default 60 000, `configs/models/ease.yaml`);
larger catalogues (Inside Airbnb, 105k items, whose item-item matrix is 44 GB in fp32) are solved
and scored on the CPU, which takes about 90 min on a 112-core host.

## Quick start (MovieLens-1M, a few minutes on one GPU)

```bash
bash scripts/download_raw.sh                 # public data (MovieLens is 30 MB; the rest ~5 GB)
towerbench prepare ml-1m                     # k-core, ids, leave-last-out marks, dataset card
towerbench embed ml-1m                       # frozen bge-base item-text embeddings (GPU)
towerbench run ml-1m ease                    # closed-form baseline, seconds
towerbench run ml-1m tt_feat --seed 0        # two-tower with text + category features
towerbench run ml-1m tt_feat --protocol temporal --window 1
towerbench report ml-1m --ref ease           # paired-bootstrap table (markdown + json)
towerbench status                            # everything finished so far
```

Each run writes `results/<dataset>/<protocol>/<model>/seed<k>/{metrics.json, per_user.npz,
test_topk.npy}`. `metrics.json` holds the mean and 95 % bootstrap interval of every metric, the
validation scores, catalogue coverage and popularity bias, fit diagnostics and the split constants.

## Reproducing the sweep

```bash
bash scripts/download_raw.sh
for d in ml-1m amazon-instruments amazon-videogames amazon-software gowalla airbnb steam; do
  towerbench prepare $d && towerbench embed $d
done
python scripts/gen_jobs.py > jobs.txt          # 1 974 jobs: 11 model configs x seeds x (loo + 5 windows) x 7 datasets
bash scripts/worker.sh 0 jobs.txt &            # one worker per GPU; workers share the list through atomic locks
bash scripts/worker.sh 1 jobs.txt &
towerbench report <dataset> --ref ease         # paired-bootstrap tables once the jobs are in
```

Model configurations are the YAML files in `configs/models/` (one switch per two-tower technique);
per-dataset protocol constants are in `configs/datasets.yaml`. Exact package versions are in
`requirements-lock.txt`.

The revision of the paper (v1.1.0) adds, on top of the sweep above, an EASE regularisation grid,
a two-tower temperature x width grid, LightGCN on every dataset, a 300-epoch LightGCN check
(`configs/models/lightgcn_e300.yaml`) and the training-noise runs (750 experiments, about 100 GPU-hours):

```bash
python scripts/gen_sensitivity_configs.py                 # writes the ease_lam* and tt_*_t*_d* configs
python scripts/gen_jobs_revision.py > jobs_rev_gpu.txt    # LightGCN, EASE lambda grid, two-tower tau x dim grid, noise runs
python scripts/gen_jobs_revision.py --queue cpu > jobs_rev_cpu.txt   # 4 EASE runs on the CPU (Airbnb)
bash scripts/worker.sh 0 jobs_rev_gpu.txt &
python scripts/revision_stats.py                          # window-aggregation and seed-variance checks
bash scripts/update_capsule.sh                            # copy every metrics.json into results_capsule/
python scripts/make_paper_assets.py                       # tables (LaTeX), figures, summary.json
```

`results_capsule/` holds every `metrics.json` of the sweep reported in the paper (1 974 files: means,
bootstrap intervals, fit diagnostics, split constants, timings), the job list and the test-read
ledgers. The paired per-user comparisons additionally need the `per_user.npz` vectors, which the sweep
writes next to each `metrics.json` and which are regenerated by re-running the jobs. The full sweep
took 103 GPU-hours on two NVIDIA L40S (two workers per GPU, about 26 h wall-clock).

## Datasets

| Dataset | Source | Side information |
|---|---|---|
| MovieLens-1M | <https://grouplens.org/datasets/movielens/1m/> | title + genres (text), genre |
| Amazon Reviews 2023 (Musical Instruments, Video Games, Software) | <https://amazon-reviews-2023.github.io/> | title + description (text), price, category |
| Gowalla check-ins | <https://snap.stanford.edu/data/loc-gowalla.html> | latitude / longitude |
| Inside Airbnb (10 cities) | <https://insideairbnb.com/get-the-data/> | description + amenities (text), geo, price, room type |
| Steam | <https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data> | title + genres + tags (text), price, genre |

Per-dataset protocol constants (`k`-core, window lengths) are in `configs/datasets.yaml`.

## Maintenance

`towerbench` is maintained by Jarosław Kurek (Warsaw University of Life Sciences, SGGW; e-mail in
`pyproject.toml`). Continuous integration runs the linter and the unit tests on Python 3.10, 3.11 and
3.12 for every push and pull request. Releases follow semantic versioning and are tagged on GitHub;
a change of any protocol constant, split rule or metric definition is a new minor version, so a
result is always reproducible from the tag it cites (`v1.0.0`: the sweep of the SoftwareX paper;
`v1.1.0`: the revision's additions). Bug reports, dataset loaders and new models are welcome as
GitHub issues and pull requests.

## Extending

A dataset is one loader function returning `(events, items)` registered in `towerbench/data/loaders.py`
plus one entry in `configs/datasets.yaml`; a model is one subclass of `towerbench.models.base.Recommender`
with `fit()` and `score()` registered in `towerbench/run.py` plus one YAML under `configs/models/`;
a new statistic operates on the persisted `per_user.npz` vectors (see `towerbench/eval/stats.py`).

## Citation

If you use `towerbench`, please cite the SoftwareX article (under review):

```
M. Bieniek, B. Kanabus, A. Kozłowski, J. Kurek, towerbench: A frozen-protocol benchmark harness for two-tower retrieval and autoencoder
recommenders with paired bootstrap statistics on public implicit-feedback data, SoftwareX (2026).
```

## License

MIT — see [LICENSE](LICENSE).
