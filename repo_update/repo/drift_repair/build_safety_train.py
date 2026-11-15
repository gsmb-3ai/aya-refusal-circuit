"""Create safety fine-tuning examples."""
import random, json
from common import *

seed_all()
rng = random.Random(SEED)

rows = read_jsonl(DATA / "eval_prompts.jsonl")
refusals = json.load(open(DATA / "refusals.json", encoding="utf-8"))

# use discovery split of seen languages only
train_langs = ["en", "es", "hi", "ar"]
harm = [r for r in rows if r["split"] == "discovery" and r["kind"] == "harmful" and r["lang"] in train_langs]
help_ = [r for r in rows if r["split"] == "discovery" and r["kind"] in ("benign", "border") and r["lang"] in train_langs]

model, tok = load_model()
answers = generate_batch(model, tok, [r["text"] for r in help_], max_new_tokens=120, bs=8)
del model
torch.cuda.empty_cache()

examples = []
for r, a in zip(help_, answers):
    if len(a) < 8 or lexicon_refusal(a):
        continue
    examples.append({"instruction": r["text"], "output": a, "lang": r["lang"], "kind": r["kind"]})

for r in harm:
    examples.append({"instruction": r["text"],
                     "output": rng.choice(refusals.get(r["lang"], refusals["en"])),
                     "lang": r["lang"], "kind": "harmful"})

rng.shuffle(examples)
write_jsonl(DATA / "safety_train_raw.jsonl", examples)   # CuratorKIT cleans this -> safety_train.jsonl
print("safety train examples (raw):", len(examples))
print(Counter((e["lang"], e["kind"]) for e in examples))