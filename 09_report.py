"""Stage 6: aggregate everything into results/summary.{json,md}, figures and a Space gallery."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

ARM_ORDER = ["base", "full", "allld", "circuit", "random"]
arms = [a for a in ARM_ORDER if (FLAGS / f"{a}.jsonl").exists()]
arms += sorted(p.stem for p in FLAGS.glob("steer_a*.jsonl"))
use = os.environ.get("LABEL", "judge")   # judge | lex
flags = {a: read_jsonl(FLAGS / f"{a}.jsonl") for a in arms}


def vals(a, kind, langs, prefix=False):
    return [int(f[use]) for f in flags[a] if f["lang"] in langs and
            (f["kind"].startswith(kind) if prefix else f["kind"] == kind)]


def stat(v):
    m, lo, hi = mean_ci(v)
    return {"mean": m, "lo": lo, "hi": hi, "n": len(v)}


summary = {}
for a in arms:
    tj = RES / f"train_{a}.json"
    cj = RES / f"cap_{a}.json"
    summary[a] = {
        "refusal_harmful_seen_langs": stat(vals(a, "harmful", TRAIN_LANGS)),
        "refusal_harmful_heldout_langs": stat(vals(a, "harmful", HELDOUT_LANGS)),
        "refusal_harmful_english": stat(vals(a, "harmful", ["en"])),
        "jailbreak_not_refused_seen": stat([1 - x for x in vals(a, "jb_", TRAIN_LANGS, True)]),
        "jailbreak_not_refused_heldout": stat([1 - x for x in vals(a, "jb_", HELDOUT_LANGS, True)]),
        "over_refusal_benign": stat(vals(a, "benign", ALL_LANGS)),
        "over_refusal_borderline": stat(vals(a, "border", ALL_LANGS)),
        "capability_mgsm_lite": json.load(open(cj))["mean"] if cj.exists() else None,
        "params_changed_pct": json.load(open(tj))["params_changed_pct"] if tj.exists() else 0.0,
        "per_lang_refusal": {l: float(np.mean(vals(a, "harmful", [l]))) for l in ALL_LANGS},
    }
json.dump(summary, open(RES / "summary.json", "w"), indent=1)

rows = [("Refusal on harmful, seen langs (higher=safer)", "refusal_harmful_seen_langs", 1),
        ("Refusal on harmful, HELD-OUT langs", "refusal_harmful_heldout_langs", 1),
        ("Jailbreak not-refused, seen langs (lower=safer)", "jailbreak_not_refused_seen", 1),
        ("Jailbreak not-refused, held-out langs", "jailbreak_not_refused_heldout", 1),
        ("Over-refusal, benign (lower=better)", "over_refusal_benign", 1),
        ("Over-refusal, borderline-safe", "over_refusal_borderline", 1),
        ("Capability: MGSM-lite accuracy", "capability_mgsm_lite", 0),
        ("Params changed (%)", "params_changed_pct", 0)]
md = ["| Metric | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
for label, key, ci in rows:
    cells = []
    for a in arms:
        v = summary[a][key]
        if v is None: cells.append("n/a")
        elif ci: cells.append(f"{100*v['mean']:.1f} [{100*v['lo']:.1f}, {100*v['hi']:.1f}]")
        elif key == "params_changed_pct": cells.append(f"{v:.3f}")
        else: cells.append(f"{100*v:.1f}")
    md.append(f"| {label} | " + " | ".join(cells) + " |")
open(RES / "summary.md", "w").write("\n".join(md) + "\n\n(values in %, brackets = 95% bootstrap CI; label source: " + use + ")\n")
print("\n".join(md))

fig, ax = plt.subplots(figsize=(11, 4))
w = 0.8 / max(1, len(arms))
for k, a in enumerate(arms):
    ax.bar(np.arange(len(ALL_LANGS)) + k * w, [100 * summary[a]["per_lang_refusal"][l] for l in ALL_LANGS], w, label=a)
ax.set_xticks(np.arange(len(ALL_LANGS)) + 0.4 - w / 2); ax.set_xticklabels(ALL_LANGS)
ax.set_ylabel("% harmful prompts refused"); ax.set_title("Refusal by language (last four = held-out)"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(RES / "refusal_by_lang.png", dpi=150); plt.close(fig)

# Space gallery: harmful prompts show refusal flags only; completions appear ONLY for benign/borderline prompts.
rows_p = read_jsonl(DATA / "prompts.jsonl")
en = {r["id"]: r["text"] for r in rows_p if r["lang"] == "en"}
hid = [i for i in dict.fromkeys(f["id"] for f in flags[arms[0]] if f["kind"] == "harmful")][:15]
gal = {"harmful": {i: {"text_en": en[i], "refused": {a: {f["lang"]: bool(f[use]) for f in flags[a]
                                                         if f["id"] == i and f["kind"] == "harmful"} for a in arms}} for i in hid},
       "benign": []}
for a in arms:
    p = GEN / f"{a}.jsonl"
    if p.exists():
        g = [r for r in read_jsonl(p) if r["kind"] in ("benign", "border")]
        for r in g[:60:6]:
            gal["benign"].append({"arm": a, "lang": r["lang"], "prompt": r["prompt"], "response": r["response"]})
json.dump(gal, open(RES / "gallery.json", "w"), ensure_ascii=False, indent=1)
