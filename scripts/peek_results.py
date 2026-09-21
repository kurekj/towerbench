"""Print one line per finished experiment matching a glob under results/ (default: everything).
Usage: python scripts/peek_results.py '*/loo/ease*'  """
import json
import sys

from towerbench.paths import RESULTS

pat = sys.argv[1] if len(sys.argv) > 1 else "*/*/*"
for p in sorted(RESULTS.glob(pat + "/seed*/metrics.json")):
    m = json.loads(p.read_text())
    t = m["test"]["ndcg@10"]
    print(f"{m['dataset']:<20} {m['protocol']:<14} {m['model_cfg']:<20} seed{m['seed']} "
          f"NDCG@10={t['mean']:.4f} [{t['lo']:.4f},{t['hi']:.4f}] val={m['val']['ndcg@10']:.4f} "
          f"n={m['n_test_users']} ep={m['fit'].get('epochs', 0)} {m['seconds']:.0f}s")
