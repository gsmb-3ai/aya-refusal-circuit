"""Generate responses and measure domain-task + capability metrics for ONE model variant.
Refusal labels are produced afterwards by judge.py (this script never labels anything).

Variants:   --adapter PATH                       a single LoRA adapter merged into the base (base: no flag)
            --domain PATH [--safety PATH]        drifted model (= base + domain adapter), optionally repaired with
              --alpha A [--ortho]                the safety adapter via SafeTune task arithmetic, strength A
"""
import argparse
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--adapter", default=None)
ap.add_argument("--domain", default=None)
ap.add_argument("--safety", default=None)
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--ortho", action="store_true")
ap.add_argument("--no_safetune", action="store_true")
ap.add_argument("--n_harm", type=int, default=100)
ap.add_argument("--n_jb", type=int, default=20)
ap.add_argument("--n_over", type=int, default=40)
ap.add_argument("--n_dom", type=int, default=40)
ap.add_argument("--n_dom_gen", type=int, default=24)
ap.add_argument("--n_cap", type=int, default=40)
ap.add_argument("--no_cap", action="store_true")
args = ap.parse_args()
seed_all()
GEN = RES / "gen"
GEN.mkdir(parents=True, exist_ok=True)

backend = "peft_merge" if args.adapter else "base"
if args.domain:
    from repair import build_repaired
    model, tok = load_model()
    backend = build_repaired(model, args.domain, args.safety, args.alpha, args.ortho, not args.no_safetune)
    model.eval()
else:
    model, tok = load_model(adapter=args.adapter)
print(f"[{args.name}] backend: {backend}", flush=True)

rows = read_jsonl(DATA / "eval_prompts.jsonl")


def pick(kind, lang, n):
    cand = [r for r in rows if r["kind"] == kind and r["lang"] == lang and r["split"] == "test"]
    random.Random(SEED).shuffle(cand)
    return cand[:n]


items = []   # (id, lang, kind, prompt)
for lang in ALL_LANGS:
    h = pick("harmful", lang, args.n_harm)
    items += [(r["id"], lang, "harmful", r["text"]) for r in h]
    for r in h[:args.n_jb]:
        for tname, tpl in JB.items():
            items.append((r["id"], lang, f"jb_{tname}", tpl.format(p=r["text"])))
    items += [(r["id"], lang, "benign", r["text"]) for r in pick("benign", lang, args.n_over)]
    items += [(r["id"], lang, "border", r["text"]) for r in pick("border", lang, args.n_over // 2)]
print("safety prompts:", len(items), flush=True)
outs = generate_batch(model, tok, [i[3] for i in items], max_new_tokens=64, bs=16)
write_jsonl(GEN / f"{args.name}.jsonl", [{"id": i[0], "lang": i[1], "kind": i[2], "prompt": i[3], "response": o}
                                         for i, o in zip(items, outs)])

# ---------------- domain task: held-out answer NLL (lower = better) + chrF of generated answers (higher = better)
dom = read_jsonl(DATA / "domain_test.jsonl")
nll_ex, chrf_ex = {}, {}
for lang in ["hi", "bn", "ta"]:
    sub = [r for r in dom if r["lang"] == lang]
    nll_ex[lang] = [v for v in (answer_nll(model, tok, r["instruction"], r["output"]) for r in sub[:args.n_dom])
                    if v is not None]
    g = generate_batch(model, tok, [r["instruction"] for r in sub[:args.n_dom_gen]], max_new_tokens=80, bs=8)
    chrf_ex[lang] = [chrf(o, r["output"]) for o, r in zip(g, sub[:args.n_dom_gen])]
flat = lambda d: [x for v in d.values() for x in v]
dom_res = {"nll_ex": nll_ex, "chrf_ex": chrf_ex,
           "nll_mean": float(np.mean(flat(nll_ex))) if flat(nll_ex) else None,
           "chrf_mean": float(np.mean(flat(chrf_ex))) if flat(chrf_ex) else None}
json.dump(dom_res, open(RES / f"domain_{args.name}.json", "w"), indent=1)
print("domain:", {k: v for k, v in dom_res.items() if k.endswith("mean")}, flush=True)

# ---------------- capability: MGSM (multilingual grade-school math), 4 languages x n_cap questions
if not args.no_cap:
    from datasets import load_dataset
    per, raw = {}, []
    for lang in ["en", "es", "bn", "sw"]:
        try:
            d = load_dataset("juletxara/mgsm", lang, split="test")
        except Exception as e:
            print("[warn] MGSM", lang, repr(e)[:120]); continue
        qs = [r["question"] for r in d][:args.n_cap]
        gold = [str(int(r["answer_number"])) for r in d][:args.n_cap]
        pr = [q + "\n\nThink step by step briefly, then finish with 'Answer: <number>'." for q in qs]
        o = generate_batch(model, tok, pr, max_new_tokens=192, bs=8)
        ok = [extract_number(x) == g for x, g in zip(o, gold)]
        per[lang] = float(np.mean(ok))
        raw += [{"lang": lang, "i": i, "gold": g, "pred": extract_number(x) or ""}
                for i, (x, g) in enumerate(zip(o, gold))]
    cap = {"per_lang": per, "mean": float(np.mean(list(per.values()))) if per else None,
           "n": len(raw)}
    json.dump(cap, open(RES / f"cap_{args.name}.json", "w"), indent=1)
    write_jsonl(RES / f"cap_raw_{args.name}.jsonl", raw)
    print("capability:", cap, flush=True)

json.dump({"name": args.name, "backend": backend, "args": vars(args)},
          open(RES / f"meta_{args.name}.json", "w"), indent=1)
print("wrote", GEN / f"{args.name}.jsonl")
