"""AuditKIT before/after comparison of the MGSM capability probe (pre-computed outputs, no GPU). Non-fatal."""
import auditkit as ak
from common import *

def run_for(name):
    raw = read_jsonl(RES / f"cap_raw_{name}.jsonl")
    s = [ak.Sample(input=f"{r['lang']}-{r['i']}", target=r["gold"], actual_output=r["pred"]) for r in raw]
    return ak.evaluate(s, "precomputed", ["exact_match"])

if not (RES / "cap_raw_base.jsonl").exists():
    raise SystemExit("base capability outputs missing")
base = run_for("base")
out, sums = {"base": {"headline": getattr(base, "headline", None)}}, {}
for p in sorted(RES.glob("cap_raw_*.jsonl")):
    n = p.stem.replace("cap_raw_", "")
    if n == "base":
        continue
    try:
        run = run_for(n)
        sums[n] = str(ak.compare(base, run).summary())
        out[n] = {"headline": getattr(run, "headline", None)}
        print(f"\n=== base -> {n} ===\n{sums[n]}")
    except Exception as e:
        sums[n] = f"AuditKIT compare failed: {type(e).__name__}: {e}"
        print(n, sums[n])
json.dump({"headlines": out, "summaries": sums}, open(RES / "auditkit_compare.json", "w"), indent=1, default=str)
