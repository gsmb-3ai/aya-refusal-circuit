"""Stage 3b: run CircuitKIT on the same model and cross-check against our hook-based layers.

IMPORTANT (unverified): the CircuitKIT docs I could read only show built-in tasks (e.g. "ioi").
Defining a CUSTOM refusal task (your contrastive prompts) is described in its User Guide:
  https://circuitkit.lexsi.ai/guides/   (Pipeline, custom data, selectors, tasks)
Set TASK below once you have read that page. With TASK="ioi" this script only proves the
library loads Tiny Aya and runs discovery (a smoke test); it does NOT locate refusal.

Run in a FRESH notebook: CircuitKIT pins transformer-lens 3.8.0 / transformers 5.
CircuitKIT must be imported BEFORE transformer_lens.
"""
import argparse, glob, json
import circuitkit as ck  # noqa: keep first
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="ioi")
ap.add_argument("--n", type=int, default=32)
args = ap.parse_args()

model = ck.load_model(MODEL_ID)                 # documented: also accepts local dirs / Hub ids
circuit = ck.discover(model, args.task, n_examples=args.n)   # documented signature
print("circuit object:", type(circuit))

# Cross-check: CircuitKIT writes *_scores.json files with `layer_suggestions` / `safety_units`.
hits = glob.glob("**/*_scores.json", recursive=True)
print("score files:", hits)
mine = json.load(open(RES / "circuit_layers.json")) if (RES / "circuit_layers.json").exists() else None
for f in hits:
    d = json.load(open(f))
    print(f, "keys:", list(d)[:12])
    sug = d.get("layer_suggestions")
    if mine and isinstance(sug, list) and all(isinstance(x, int) for x in sug):
        a, b = set(sug), set(mine["importance_layers"])
        print("overlap with hook-based layers:", sorted(a & b), "| jaccard", len(a & b) / max(1, len(a | b)))
        json.dump({"circuitkit_layers": sorted(a), "hook_layers": sorted(b)}, open(RES / "circuitkit_crosscheck.json", "w"))
