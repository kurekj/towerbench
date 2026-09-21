"""Emit the job lists of the revision (v1.1.0) on top of the published 1,974-experiment sweep.
One line: dataset protocol window model seed [noise].  Ordered so that the cheapest and most
informative blocks land first.

  --queue gpu   (default) sensitivity grids, LightGCN, training-noise runs
  --queue cpu   EASE regularisation grid on Inside Airbnb (105k items: solved on the CPU)

Blocks:
  S1  EASE lambda grid, leave-last-out + 5 rolling windows, six GPU-sized datasets
  S2  two-tower tau x dim grid (TT-ID and TT-Feat), leave-last-out, 3 seeds, three datasets
  G1  LightGCN, leave-last-out + 5 windows, 5 seeds, seven datasets (seeds 0-2 first)
  N1  training-noise switch 10% / 20%, leave-last-out, two datasets, five model families
"""
import argparse

DATASETS = ["ml-1m", "amazon-instruments", "gowalla", "amazon-videogames", "amazon-software", "airbnb", "steam"]
GPU_EASE = [d for d in DATASETS if d != "airbnb"]
SWEEP_DS = ["ml-1m", "amazon-software", "gowalla", "airbnb"]
NOISE_DS = ["ml-1m", "amazon-software"]
NOISE_MODELS = ["ease", "multvae", "tt_id", "tt_feat", "lightgcn"]
DETERMINISTIC = {"pop", "ease"}
EASE_GRID = [f"ease_lam{lam:g}" for lam in (2, 5, 10, 20, 50, 200, 1000, 2000)]
TT_GRID = [f"{fam}_t{tau:g}_d{dim}" for fam in ("tt_id", "tt_feat") for tau in (0.02, 0.05, 0.1)
           for dim in (64, 128, 256) if not (tau == 0.05 and dim == 128)]
WINDOWS = [0, 1, 2, 3, 4]


def gpu_jobs():
    for d in GPU_EASE:                                     # S1 (seconds per run)
        for m in EASE_GRID:
            yield d, "loo", 0, m, 0
    for d in SWEEP_DS:                                     # S2
        for s in (0, 1, 2):
            for m in TT_GRID:
                yield d, "loo", 0, m, s
    for d in GPU_EASE:                                     # S1, rolling windows
        for w in WINDOWS:
            for m in EASE_GRID:
                yield d, "temporal", w, m, 0
    for s in (0, 1, 2):                                    # G1, leave-last-out
        for d in DATASETS:
            yield d, "loo", 0, "lightgcn", s
    for d in NOISE_DS:                                     # N1
        for noise in ("0.1", "0.2"):
            for m in NOISE_MODELS:
                for s in (0, 1, 2):
                    if m in DETERMINISTIC and s != 0:
                        continue
                    yield d, "loo", 0, m, s, noise
    for s in (0, 1, 2):                                    # G1, rolling windows
        for d in DATASETS:
            for w in WINDOWS:
                yield d, "temporal", w, "lightgcn", s
    for s in (3, 4):                                       # G1, remaining seeds
        for d in DATASETS:
            yield d, "loo", 0, "lightgcn", s
            for w in WINDOWS:
                yield d, "temporal", w, "lightgcn", s


def cpu_jobs():
    for m in EASE_GRID:
        yield "airbnb", "loo", 0, m, 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", choices=["gpu", "cpu"], default="gpu")
    a = ap.parse_args()
    for job in (gpu_jobs() if a.queue == "gpu" else cpu_jobs()):
        print(*job)
