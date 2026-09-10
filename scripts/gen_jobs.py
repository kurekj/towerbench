"""Emit the sweep job list (one line: dataset protocol window model seed), ordered so that partial
results are usable early: (1) leave-last-out, seeds 0-2, every dataset; (2) rolling-origin temporal
windows, seeds 0-2, every dataset; (3) seeds 3-4 of both protocols."""
import sys

DATASETS = sys.argv[1:] or ["ml-1m", "amazon-instruments", "gowalla", "amazon-videogames",
                            "amazon-software", "airbnb", "steam"]
MODELS = ["pop", "ease", "multvae", "sasrec", "tt_id", "tt_id_nologq", "tt_id_mns", "tt_id_mixgcf",
          "tt_feat", "tt_feat_dcn", "tt_feat_all"]
DETERMINISTIC = {"pop", "ease"}
PROTOCOLS = (("loo", [0]), ("temporal", [0, 1, 2, 3, 4]))
for seed_block in ([0, 1, 2], [3, 4]):
    for proto, windows in PROTOCOLS:
        for d in DATASETS:
            for w in windows:
                for m in MODELS:
                    for s in seed_block:
                        if m in DETERMINISTIC and s != 0:
                            continue
                        print(d, proto, w, m, s)
