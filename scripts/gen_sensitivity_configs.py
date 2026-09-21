"""Write the configuration files of the revision's sensitivity analysis into configs/models/.

EASE regularisation grid:      ease_lam{2,5,10,20,50,200,1000,2000}       (ease.yaml is lambda = 500)
Two-tower temperature x width: tt_{id,feat}_t{0.02,0.05,0.1}_d{64,128,256}
                               (tt_id.yaml / tt_feat.yaml are tau = 0.05, dim = 128)
Every derived file differs from its base configuration in the swept keys only.
"""
from pathlib import Path

import yaml

CFG = Path(__file__).resolve().parent.parent / "configs" / "models"

EASE_LAMBDAS = [2.0, 5.0, 10.0, 20.0, 50.0, 200.0, 1000.0, 2000.0]
TAUS = [0.02, 0.05, 0.1]
DIMS = [64, 128, 256]


def main():
    written = []
    base = yaml.safe_load((CFG / "ease.yaml").read_text())
    for lam in EASE_LAMBDAS:
        cfg = {**base, "lambda": lam}
        p = CFG / f"ease_lam{lam:g}.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        written.append(p.name)
    for fam in ("tt_id", "tt_feat"):
        base = yaml.safe_load((CFG / f"{fam}.yaml").read_text())
        for tau in TAUS:
            for dim in DIMS:
                if tau == base["tau"] and dim == base["dim"]:
                    continue  # that is the base configuration itself
                cfg = {**base, "tau": tau, "dim": dim}
                p = CFG / f"{fam}_t{tau:g}_d{dim}.yaml"
                p.write_text(yaml.safe_dump(cfg, sort_keys=False))
                written.append(p.name)
    print(f"{len(written)} configuration files:", ", ".join(written))


if __name__ == "__main__":
    main()
