"""Human validation of the two refusal labelers.
  python validate_labels.py --export 120      -> results/to_label.csv   (fill the `human` column: 1 = the reply refuses/declines, 0 = it answers)
  python validate_labels.py --score results/to_label.csv               -> accuracy / kappa of judge and keyword list vs YOUR labels
Rows you cannot read: leave `human` empty. Half of the rows are cases where the two labelers disagree."""
import argparse, csv, math
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--export", type=int, default=0)
ap.add_argument("--score", default=None)
args = ap.parse_args()
GEN, FLAGS = RES / "gen", RES / "flags"


def load_all():
    flags, gen = {}, {}
    for p in sorted(FLAGS.glob("*.jsonl")):
        for r in read_jsonl(p):
            flags[(p.stem, r["id"], r["lang"], r["kind"])] = r
    for p in sorted(GEN.glob("*.jsonl")):
        for r in read_jsonl(p):
            gen[(p.stem, r["id"], r["lang"], r["kind"])] = r
    return flags, gen


if args.export:
    flags, gen = load_all()
    en = {r["id"]: r["text"] for r in read_jsonl(DATA / "eval_prompts.jsonl") if r["lang"] == "en"}
    rng = random.Random(SEED)
    per_lang = math.ceil(args.export / len(ALL_LANGS))
    out = []
    for lang in ALL_LANGS:
        keys = [k for k in flags if k[2] == lang and k[3] == "harmful" and k in gen]
        dis = [k for k in keys if flags[k]["judge"] != flags[k]["lex"]]
        rest = [k for k in keys if k not in set(dis)]
        rng.shuffle(dis); rng.shuffle(rest)
        out += (dis[:per_lang // 2] + rest[:per_lang - min(len(dis), per_lang // 2)])[:per_lang]
    rng.shuffle(out)
    with open(RES / "to_label.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "id", "lang", "kind", "english_request", "response_head", "human"])
        for (arm, i, lang, kind) in out:
            w.writerow([arm, i, lang, kind, en.get(i, "")[:160], gen[(arm, i, lang, kind)]["response"][:300].replace("\n", " "), ""])
    print(f"wrote {RES / 'to_label.csv'} with {len(out)} rows. Fill the `human` column (1 refuses / 0 answers).")
    raise SystemExit

flags, _ = load_all()
H, J, L, G = [], [], [], []
for r in csv.DictReader(open(args.score, encoding="utf-8")):
    if r["human"].strip() not in ("0", "1"):
        continue
    k = (r["arm"], r["id"], r["lang"], r["kind"])
    if k not in flags:
        continue
    H.append(int(r["human"])); J.append(int(flags[k]["judge"])); L.append(int(flags[k]["lex"])); G.append(r["lang"])
H, J, L, G = map(np.array, (H, J, L, G))
res = {"n": int(len(H))}


def stats(mask, name):
    if mask.sum() < 8:
        return
    h, j, l = H[mask], J[mask], L[mask]
    res[name] = {"n": int(mask.sum()), "human_refusal_rate": float(h.mean()),
                 "judge_acc": float((h == j).mean()), "judge_kappa": cohen_kappa(h, j),
                 "lex_acc": float((h == l).mean()), "lex_kappa": cohen_kappa(h, l)}


stats(np.ones(len(H), bool), "all")
stats(np.isin(G, SEEN_LANGS), "seen_langs")
stats(np.isin(G, HELDOUT_LANGS), "heldout_langs")
for lg in ALL_LANGS:
    stats(G == lg, lg)
if "all" in res:
    res["better_labeler"] = "judge" if res["all"]["judge_kappa"] >= res["all"]["lex_kappa"] else "lex"
json.dump(res, open(RES / "label_validation.json", "w"), indent=1)
print(f"{'subset':14s} {'n':>4s} {'human%':>7s} {'judge acc':>10s} {'judge k':>8s} {'lex acc':>8s} {'lex k':>7s}")
for k, v in res.items():
    if isinstance(v, dict):
        print(f"{k:14s} {v['n']:4d} {100*v['human_refusal_rate']:7.1f} {100*v['judge_acc']:10.1f} {v['judge_kappa']:8.2f} "
              f"{100*v['lex_acc']:8.1f} {v['lex_kappa']:7.2f}")
print("better labeler overall:", res.get("better_labeler"))
