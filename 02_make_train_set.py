"""Stage 2: build the safety training set, then clean + dedup it with CuratorKIT.

 - harmful TRAIN prompts (train languages)  -> short, varied refusals (translated templates)
 - benign + borderline TRAIN prompts        -> the BASE model's own answers (self-distillation, so
                                               helpfulness is preserved and over-refusal is discouraged)
Held-out languages are never used here.
"""
import random
from collections import Counter
from common import *

seed_all()
rng = random.Random(SEED)
rows = read_jsonl(DATA / "prompts.jsonl")
refusals = json.load(open(DATA / "refusals.json", encoding="utf-8"))
train = [r for r in rows if r["split"] == "train" and r["lang"] in TRAIN_LANGS]
harm = [r for r in train if r["kind"] == "harmful"]
helpful = [r for r in train if r["kind"] in ("benign", "border")]

model, tok = load_model()
answers = generate_batch(model, tok, [r["text"] for r in helpful], max_new_tokens=200, bs=16)
del model
torch.cuda.empty_cache()

examples = []
for r, a in zip(helpful, answers):
    if len(a) < 10 or lexicon_refusal(a):
        continue  # never train on refusals to benign prompts
    examples.append({"instruction": r["text"], "input": "", "output": a, "lang": r["lang"], "kind": r["kind"]})
for r in harm:
    examples.append({"instruction": r["text"], "input": "", "output": rng.choice(refusals[r["lang"]]),
                     "lang": r["lang"], "kind": "harmful"})
rng.shuffle(examples)
write_jsonl(DATA / "train_raw.jsonl", examples)
meta = {e["instruction"]: {"lang": e["lang"], "kind": e["kind"]} for e in examples}
print("raw examples:", len(examples))


def curate():
    """CuratorKIT: clean + MinHash dedup, export Alpaca format. Falls back to plain python dedup."""
    out_dir = DATA / "curated"
    try:
        from curatorkit import Curator, CuratorConfig
        result = Curator(CuratorConfig(dataset=str(DATA / "train_raw.jsonl"), dedup="minhash", clean=True,
                                       export_formats=["alpaca"], output_dir=str(out_dir))).run()
        result.print_summary()
        files = sorted(out_dir.glob("*alpaca*.jsonl"))
        if not files:
            raise RuntimeError(f"no alpaca export found in {out_dir}: {list(out_dir.iterdir())}")
        kept = []
        for f in files:
            kept += read_jsonl(f)
        print("CuratorKIT kept", len(kept), "of", len(examples), "-> manifest in", out_dir)
        return kept, True
    except Exception as e:
        print("[warn] CuratorKIT step failed, using fallback dedup. Reason:", repr(e))
        seen, kept = set(), []
        for e_ in examples:
            k = e_["instruction"].lower()
            if k not in seen:
                seen.add(k); kept.append(e_)
        return kept, False


kept, used_ck = curate()
final = []
for r in kept:
    m = meta.get(r.get("instruction"))
    if m and r.get("output"):
        final.append({"instruction": r["instruction"], "output": r["output"], **m})
write_jsonl(DATA / "train_curated.jsonl", final)
json.dump({"used_curatorkit": used_ck, "raw": len(examples), "final": len(final)},
          open(RES / "curation.json", "w"), indent=1)
print(Counter((r["lang"], r["kind"]) for r in final))
