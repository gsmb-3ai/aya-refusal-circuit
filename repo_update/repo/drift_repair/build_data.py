"""Build domain-task data + safety evaluation prompts."""
import csv, sys, random, re, json
from pathlib import Path
import torch
from common import *

seed_all()
rng = random.Random(SEED)

# -------------------- English seed pools --------------------
def load_english_pools():
    from datasets import load_dataset
    pools = {"harmful": [], "benign": [], "border": [], "domain": []}

    def attempt(name, fn):
        try:
            fn()
        except Exception as e:
            print(f"[warn] could not load {name}: {e}")

    def jbb():
        d = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
        pools["harmful"] += [r["Goal"] for r in d["harmful"]]
        pools["benign"]  += [r["Goal"] for r in d["benign"]]

    def adv():
        d = load_dataset("walledai/AdvBench", split="train")
        pools["harmful"] += [r["prompt"] for r in d]

    def xs():
        d = load_dataset("walledai/XSTest", split="test")
        pools["border"] += [r["prompt"] for r in d if r["label"] == "safe"]

    def alp():
        d = load_dataset("tatsu-lab/alpaca", split="train")
        for r in d:
            if not r["input"] and 20 < len(r["instruction"]) < 180 and 30 < len(r["output"]) < 600:
                pools["benign"].append(r["instruction"])
                pools["domain"].append({"instruction": r["instruction"], "output": r["output"]})

    attempt("JBB-Behaviors", jbb)
    attempt("AdvBench", adv)
    attempt("XSTest", xs)
    attempt("Alpaca", alp)

    for k in ("harmful", "benign", "border"):
        seen, uniq = set(), []
        for t in pools[k]:
            t = re.sub(r"\s+", " ", t).strip()
            if t and t.lower() not in seen:
                seen.add(t.lower()); uniq.append(t)
        rng.shuffle(uniq)
        pools[k] = uniq[:200 if k != "border" else 100]
        print(k, len(pools[k]))

    # domain pool: short Q&A, never overlapping any evaluation prompt; 150 items for training, 40 held out for testing
    norm = lambda x: re.sub(r"\s+", " ", x).strip().lower()
    eval_texts = {norm(x) for k in ("harmful", "benign", "border") for x in pools[k]}
    dom = [d for d in pools["domain"] if norm(d["instruction"]) not in eval_texts]
    rng.shuffle(dom)
    pools["domain"] = dom[:190]
    print("domain", len(pools["domain"]), "(150 train + 40 test per language)")
    return pools

def make_translator():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    name = "facebook/nllb-200-distilled-600M"
    ntok = AutoTokenizer.from_pretrained(name, src_lang="eng_Latn")
    try:
        mt = AutoModelForSeq2SeqLM.from_pretrained(name, dtype=torch.float16)
    except TypeError:
        mt = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=torch.float16)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mt = mt.to(dev).eval()

    @torch.no_grad()
    def translate(texts, tgt, bs=24):
        out = []
        for i in range(0, len(texts), bs):
            enc = ntok(texts[i:i+bs], return_tensors="pt", padding=True,
                       truncation=True, max_length=180).to(dev)
            gen = mt.generate(**enc, forced_bos_token_id=ntok.convert_tokens_to_ids(tgt),
                              max_new_tokens=200, num_beams=2)
            out += ntok.batch_decode(gen, skip_special_tokens=True)
        return out
    return translate

def romanize_hindi(text):
    from indic_transliteration import sanscript
    return sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS).lower()

def build():
    pools = load_english_pools()
    translate = make_translator()

    # ---- safety evaluation prompts (all languages) ----
    eval_rows = []
    for kind in ("harmful", "benign", "border"):
        for i, t in enumerate(pools[kind]):
            eid = f"{kind}-{i:04d}"
            eval_rows.append({"id": eid, "kind": kind, "lang": "en", "text": t,
                              "split": "test" if i >= 40 else "discovery"})
    en_texts = [r["text"] for r in eval_rows if r["lang"] == "en"]
    ids = [r["id"] for r in eval_rows if r["lang"] == "en"]
    kinds = [r["kind"] for r in eval_rows if r["lang"] == "en"]
    splits = [r["split"] for r in eval_rows if r["lang"] == "en"]

    for lang, code in NLLB.items():
        print("translating eval ->", lang)
        tr = translate(en_texts, code)
        for i, t in enumerate(tr):
            eval_rows.append({"id": ids[i], "kind": kinds[i], "lang": lang,
                              "text": re.sub(r"\s+", " ", t).strip(), "split": splits[i]})
        if lang == "hi":
            for i, t in enumerate(tr):
                eval_rows.append({"id": ids[i], "kind": kinds[i], "lang": "hi_latn",
                                  "text": re.sub(r"\s+", " ", romanize_hindi(t)).strip(),
                                  "split": splits[i]})

    write_jsonl(DATA / "eval_prompts.jsonl", eval_rows)
    print("eval prompts:", len(eval_rows))

    # ---- domain task data (hi / bn / ta): 150 English items for training, 40 held out for testing ----
    domain_en = pools["domain"]
    train_rows, test_rows = [], []
    for lang in ["hi", "bn", "ta"]:
        print("translating domain ->", lang)
        tr_i = translate([d["instruction"] for d in domain_en], NLLB[lang])
        tr_o = translate([d["output"] for d in domain_en], NLLB[lang])
        for i, (ii, oo) in enumerate(zip(tr_i, tr_o)):
            row = {"instruction": re.sub(r"\s+", " ", ii).strip(), "output": re.sub(r"\s+", " ", oo).strip(),
                   "lang": lang, "kind": "domain"}
            (train_rows if i < 150 else test_rows).append(row)
    rng.shuffle(train_rows)
    write_jsonl(DATA / "domain_train_raw.jsonl", train_rows)     # CuratorKIT cleans this -> domain_train.jsonl
    write_jsonl(DATA / "domain_test.jsonl", test_rows)
    print("domain train (raw):", len(train_rows), "| domain test:", len(test_rows))

    # ---- safety training data (refusals for harmful, self-distill for benign) ----
    REFUSAL_TEMPLATES_EN = [
        "I'm sorry, but I can't help with that request.",
        "I can't assist with this, because it could cause harm to people.",
        "Sorry, I won't provide that. If you're worried about safety, I'm happy to help with a safer alternative.",
        "That's something I have to decline, as it could be used to hurt others.",
        "I'm not able to help with that. Is there something else I can do for you?",
        "I must refuse this request because it involves potential harm.",
    ]
    refusals = {"en": REFUSAL_TEMPLATES_EN}
    for lang, code in NLLB.items():
        refusals[lang] = [re.sub(r"\s+", " ", x).strip()
                          for x in translate(REFUSAL_TEMPLATES_EN, code)]
    json.dump(refusals, open(DATA / "refusals.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # We will generate the actual safety training set later (needs base model answers)
    print("done building data")

if __name__ == "__main__":
    build()