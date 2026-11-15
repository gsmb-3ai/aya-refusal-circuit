"""Quick sanity view: per arm and language, refusal rate by judge vs keyword list, plus a few reply openings.
Large judge/keyword gaps mean the labels cannot be trusted for that arm/language until checked by hand."""
import argparse
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default=None, help="print reply openings for this arm")
ap.add_argument("--langs", nargs="*", default=["es", "hi", "ar", "bn"])
ap.add_argument("--k", type=int, default=3)
args = ap.parse_args()
GEN, FLAGS = RES / "gen", RES / "flags"
arms = sorted(p.stem for p in FLAGS.glob("*.jsonl"))
print(f"{'arm':26s} " + " ".join(f"{l:>13s}" for l in ALL_LANGS) + "     (judge% / keyword%, harmful prompts)")
for a in arms:
    fl = [r for r in read_jsonl(FLAGS / f"{a}.jsonl") if r["kind"] == "harmful"]
    cells, warn = [], []
    for l in ALL_LANGS:
        x = [r for r in fl if r["lang"] == l]
        j, k = 100 * np.mean([r["judge"] for r in x]), 100 * np.mean([r["lex"] for r in x])
        cells.append(f"{j:5.0f}/{k:<5.0f}")
        if abs(j - k) > 25:
            warn.append(l)
    print(f"{a:26s} " + " ".join(f"{c:>13s}" for c in cells) + (f"   <-- labelers disagree: {warn}" if warn else ""))
if args.arm:
    g = read_jsonl(GEN / f"{args.arm}.jsonl")
    fl = {(r["id"], r["lang"], r["kind"]): r for r in read_jsonl(FLAGS / f"{args.arm}.jsonl")}
    for l in args.langs:
        print(f"\n--- {args.arm} / {l}: first {args.k} harmful prompts, reply openings ---")
        for r in [r for r in g if r["lang"] == l and r["kind"] == "harmful"][:args.k]:
            f = fl[(r["id"], r["lang"], r["kind"])]
            print(f"judge={int(f['judge'])} kw={int(f['lex'])} | {r['response'][:110]!r}")
