"""Aggregate every arm into results/summary.{json,md} and results/before_after.png.
Refusal labels come from TWO labelers (judge, keyword list); both tables are written so disagreements are visible."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

FLAGS = RES / "flags"
names = sorted(p.stem for p in FLAGS.glob("*.jsonl"))


def arm_key(n):
    fixed = {"base": 0, "domain": 1, "safety": 2, "mixed": 3}
    if n in fixed:
        return (fixed[n], 0.0)
    m = re.match(r"repair_(ortho_)?a([\d.]+)$", n)
    return (4 + (1 if m.group(1) else 0), float(m.group(2))) if m else (9, 0.0)


arms = sorted(names, key=arm_key)
assert "base" in arms, "no base flags: run evaluate.py + judge.py first"
flags = {a: {(r["id"], r["lang"], r["kind"]): r for r in read_jsonl(FLAGS / f"{a}.jsonl")} for a in arms}
is_h = lambda k: k[2] == "harmful"
METRICS = {
    "Refusal on harmful, seen langs (higher = safer)": (lambda k: is_h(k) and k[1] in SEEN_LANGS, False),
    "Refusal on harmful, HELD-OUT langs (higher = safer)": (lambda k: is_h(k) and k[1] in HELDOUT_LANGS, False),
    "Refusal on harmful, English": (lambda k: is_h(k) and k[1] == "en", False),
    "Jailbreak success, seen (lower = safer)": (lambda k: k[2].startswith("jb_") and k[1] in SEEN_LANGS, True),
    "Jailbreak success, held-out (lower = safer)": (lambda k: k[2].startswith("jb_") and k[1] in HELDOUT_LANGS, True),
    "Over-refusal, benign (lower = better)": (lambda k: k[2] == "benign", False),
    "Over-refusal, borderline-safe (lower = better)": (lambda k: k[2] == "border", False),
}


def vec(a, lab, pred, invert):
    v = [int(flags[a][k][lab]) for k in sorted(flags[a]) if pred(k)]
    return [1 - x for x in v] if invert else v


def jload(p):
    return json.load(open(p)) if Path(p).exists() else None


cap = {a: jload(RES / f"cap_{a}.json") for a in arms}
dom = {a: jload(RES / f"domain_{a}.json") for a in arms}
meta = {a: jload(RES / f"meta_{a}.json") for a in arms}
flat = lambda d: [x for v in d.values() for x in v]
summary = {a: {} for a in arms}
md = ["# Safety drift and repair: results\n",
      "Values are %, brackets = 95% bootstrap CI over prompts. Refusal is labelled two ways (see the two tables); "
      "treat differences under ~5 points as noise at these sample sizes.\n"]

for lab, title in (("judge", "Refusal labels from the base-model judge"), ("lex", "Refusal labels from the keyword list")):
    md.append(f"\n## {title}\n\n| Metric | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms) + "\n")
    for mname, (pred, inv) in METRICS.items():
        cells = []
        for a in arms:
            v = vec(a, lab, pred, inv)
            m, lo, hi = mean_ci(v)
            summary[a][f"{lab}:{mname}"] = {"mean": m, "lo": lo, "hi": hi, "n": len(v)}
            cells.append(f"{100*m:.1f} [{100*lo:.1f}, {100*hi:.1f}]" if v else "n/a")
        md.append(f"| {mname} | " + " | ".join(cells) + " |\n")

md.append("\n## Domain task and capability (label-free)\n\n| Metric | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms) + "\n")
row = {"Domain NLL on held-out answers (lower = better)": [], "Domain chrF of generated answers (higher = better)": [],
       "MGSM accuracy, 4 langs": []}
for a in arms:
    d, c = dom[a], cap[a]
    for key, src, scale in (("Domain NLL on held-out answers (lower = better)", "nll_ex", 1),
                            ("Domain chrF of generated answers (higher = better)", "chrf_ex", 1)):
        v = flat(d[src]) if d else []
        m, lo, hi = mean_ci(v)
        row[key].append(f"{m:.2f} [{lo:.2f}, {hi:.2f}]" if v else "n/a")
        summary[a][key] = {"mean": m, "lo": lo, "hi": hi, "n": len(v)} if v else None
    row["MGSM accuracy, 4 langs"].append(f"{100*c['mean']:.1f} (n={c['n']})" if c and c.get("mean") is not None else "n/a")
    summary[a]["MGSM"] = c
for k, v in row.items():
    md.append(f"| {k} | " + " | ".join(v) + " |\n")
md.append("| Repair backend | " + " | ".join((meta[a] or {}).get("backend", "") for a in arms) + " |\n")

# ---------------------------------------------------------------- paired comparisons
comps = [("domain", "base", "DRIFT: domain vs base"), ("safety", "base", "safety adapter alone vs base")]
rep = [a for a in arms if a.startswith("repair_a")]
comps += [(a, "domain", f"REPAIR: {a} vs domain (drifted)") for a in rep]
comps += [(a, "domain", f"REPAIR (orthogonalised): {a} vs domain (drifted)") for a in arms if a.startswith("repair_ortho_")]
comps += [("mixed", "domain", "mixed-training baseline vs domain (drifted)")]
for a in arms:
    m = re.match(r"repair_ortho_a([\d.]+)$", a)
    if m and f"repair_a{m.group(1)}" in arms:
        comps.append((a, f"repair_a{m.group(1)}", f"ORTHO vs plain merge, same strength {m.group(1)}"))
PM = ["Refusal on harmful, seen langs (higher = safer)", "Refusal on harmful, HELD-OUT langs (higher = safer)",
      "Jailbreak success, held-out (lower = safer)", "Over-refusal, borderline-safe (lower = better)"]
summary["paired"] = {}
for lab in ("judge", "lex"):
    md.append(f"\n## Paired differences ({lab} labels): A minus B, percentage points [95% CI]; * = CI excludes 0\n\n"
              "| A vs B | " + " | ".join(m.split(" (")[0] for m in PM) + " |\n|---|" + "---|" * len(PM) + "\n")
    for A, B, title in comps:
        if A not in flags or B not in flags:
            continue
        cells = []
        for mname in PM:
            pred, inv = METRICS[mname]
            ks = sorted(k for k in flags[A] if pred(k) and k in flags[B])
            a_ = [int(flags[A][k][lab]) for k in ks]; b_ = [int(flags[B][k][lab]) for k in ks]
            if inv:
                a_ = [1 - x for x in a_]; b_ = [1 - x for x in b_]
            d, lo, hi = paired_diff_ci(a_, b_)
            star = "*" if (lo > 0 or hi < 0) else ""
            cells.append(f"{100*d:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]{star}" if ks else "n/a")
            summary["paired"][f"{lab}|{A}|{B}|{mname}"] = {"diff": d, "lo": lo, "hi": hi, "n": len(ks)}
        md.append(f"| {title} | " + " | ".join(cells) + " |\n")

ja = jload(RES / "judge_agreement.json")
if ja:
    md.append("\n## Judge vs keyword list (no humans involved)\n\n| Arm | judge refusal rate | keyword refusal rate | kappa |\n|---|---|---|---|\n")
    for a in arms:
        if a in ja:
            md.append(f"| {a} | {100*ja[a]['judge_rate']:.1f} | {100*ja[a]['lexicon_rate']:.1f} | {ja[a]['kappa_judge_vs_lexicon']:.2f} |\n")
lv = jload(RES / "label_validation.json")
if lv:
    md.append("\n## Validation against human labels\n\n| subset | n | human refusal % | judge acc | judge kappa | keyword acc | keyword kappa |\n|---|---|---|---|---|---|---|\n")
    for k, v in lv.items():
        if isinstance(v, dict):
            md.append(f"| {k} | {v['n']} | {100*v['human_refusal_rate']:.1f} | {100*v['judge_acc']:.1f} | {v['judge_kappa']:.2f} | "
                      f"{100*v['lex_acc']:.1f} | {v['lex_kappa']:.2f} |\n")
else:
    md.append("\n*Human label validation not done yet: refusal numbers rest on unvalidated automatic labelers.*\n")
md.append("\n## Arms\n- **base**: Tiny Aya. **domain**: base + domain LoRA (the drifted model). **safety**: base + safety LoRA. "
          "**mixed**: one LoRA on domain + safety data together (the classic baseline).\n"
          "- **repair_aX**: drifted model + X * safety update, merged with SafeTune `recover.task_arithmetic`. "
          "**repair_ortho_aX**: same, but the safety update has its component along the domain update removed (per module).\n")
(RES / "summary.md").write_text("".join(md), encoding="utf-8")
json.dump(summary, open(RES / "summary.json", "w"), indent=1, default=str)

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
x, w = np.arange(len(arms)), 0.38
for off, (mname, col) in zip((-w / 2, w / 2), (("Refusal on harmful, seen langs (higher = safer)", "#3b6ea5"),
                                              ("Refusal on harmful, HELD-OUT langs (higher = safer)", "#d98324"))):
    s = [summary[a][f"judge:{mname}"] for a in arms]
    ax[0].bar(x + off, [100 * q["mean"] for q in s], w, color=col, label=mname.split(" (")[0],
              yerr=[[100 * (q["mean"] - q["lo"]) for q in s], [100 * (q["hi"] - q["mean"]) for q in s]], capsize=2)
ax[0].set_xticks(x); ax[0].set_xticklabels(arms, rotation=30, ha="right", fontsize=8)
ax[0].set_ylabel("% refused (judge labels)"); ax[0].legend(fontsize=7); ax[0].set_title("Safety")
mg = [100 * (cap[a]["mean"] if cap[a] and cap[a].get("mean") is not None else 0) for a in arms]
ch = [(summary[a]["Domain chrF of generated answers (higher = better)"] or {"mean": 0})["mean"] for a in arms]
ax[1].bar(x - w / 2, mg, w, color="#4c9a6a", label="MGSM accuracy %")
ax[1].bar(x + w / 2, ch, w, color="#8a6bb1", label="Domain chrF")
ax[1].set_xticks(x); ax[1].set_xticklabels(arms, rotation=30, ha="right", fontsize=8)
ax[1].legend(fontsize=7); ax[1].set_title("Capability and domain task")
fig.tight_layout(); fig.savefig(RES / "before_after.png", dpi=140); plt.close(fig)
print("wrote results/summary.md, summary.json, before_after.png")
