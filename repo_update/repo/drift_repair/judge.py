"""Label every response refused / not refused. Two labelers are stored side by side:
  judge : the BASE model (never the model under test) answering Yes/No on the reply text only
  lex   : multilingual keyword list
Only labels go to results/flags (publishable). results/gen holds raw text (private)."""
import argparse
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--arms", nargs="*", default=[])
ap.add_argument("--force", action="store_true")
args = ap.parse_args()
GEN, FLAGS = RES / "gen", RES / "flags"
FLAGS.mkdir(parents=True, exist_ok=True)
arms = args.arms or sorted(p.stem for p in GEN.glob("*.jsonl"))
if not args.force:
    arms = [a for a in arms if not (FLAGS / f"{a}.jsonl").exists()]
if not arms:
    print("nothing to judge"); raise SystemExit
model, tok = load_model()
agree = json.load(open(RES / "judge_agreement.json")) if (RES / "judge_agreement.json").exists() else {}
for arm in arms:
    g = read_jsonl(GEN / f"{arm}.jsonl")
    sc = judge_refusal(model, tok, [r["response"] for r in g])
    out = [{"id": r["id"], "lang": r["lang"], "kind": r["kind"], "lex": bool(lexicon_refusal(r["response"])),
            "judge": bool(s > 0), "score": s} for r, s in zip(g, sc)]
    write_jsonl(FLAGS / f"{arm}.jsonl", out)
    j, l = [int(o["judge"]) for o in out], [int(o["lex"]) for o in out]
    agree[arm] = {"n": len(out), "judge_rate": float(np.mean(j)), "lexicon_rate": float(np.mean(l)),
                  "kappa_judge_vs_lexicon": cohen_kappa(j, l)}
    print(arm, agree[arm], flush=True)
json.dump(agree, open(RES / "judge_agreement.json", "w"), indent=1)
