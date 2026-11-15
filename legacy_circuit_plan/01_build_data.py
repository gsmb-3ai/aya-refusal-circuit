"""Stage 1: build the multilingual prompt set from PUBLIC benchmarks (prompts only, no harmful completions).

Sources (accept the terms on each HF page first; some are gated):
  JailbreakBench/JBB-Behaviors  (harmful + matched benign)
  walledai/AdvBench             (harmful)
  walledai/XSTest               (borderline-but-safe, for over-refusal)
  tatsu-lab/alpaca              (ordinary harmless prompts)
Fallback: put your own  data/seed_en.csv  with columns  kind,text  (kind in harmful|benign|border).
Verify dataset ids / columns / licences yourself and cite them in your write-up.
"""
import csv, sys, random, re
import torch
from common import *

seed_all()
rng = random.Random(SEED)
CAPS = {"harmful": 240, "benign": 240, "border": 120}
# (discovery, train, test) sizes per kind; discovery is used ONLY to locate the circuit
SPLITS = {"harmful": (40, 120, 80), "benign": (40, 120, 80), "border": (0, 40, 80)}
REFUSAL_TEMPLATES_EN = [
    "I'm sorry, but I can't help with that request.",
    "I can't assist with this, because it could cause harm to people.",
    "Sorry, I won't provide that. If you're worried about safety, I'm happy to help with a safer alternative.",
    "That's something I have to decline, as it could be used to hurt others.",
    "I'm not able to help with that. Is there something else I can do for you?",
    "I must refuse this request because it involves potential harm.",
]


def clean(t):
    return re.sub(r"\s+", " ", t).strip()


def load_english():
    from datasets import load_dataset
    pools = {"harmful": [], "benign": [], "border": []}

    def attempt(name, fn):
        try:
            fn()
        except Exception as e:
            print(f"[warn] could not load {name}: {e}")

    def jbb():
        d = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
        pools["harmful"] += [r["Goal"] for r in d["harmful"]]
        pools["benign"] += [r["Goal"] for r in d["benign"]]

    def adv():
        d = load_dataset("walledai/AdvBench", split="train")
        pools["harmful"] += [r["prompt"] for r in d]

    def xs():
        d = load_dataset("walledai/XSTest", split="test")
        pools["border"] += [r["prompt"] for r in d if r["label"] == "safe"]

    def alp():
        d = load_dataset("tatsu-lab/alpaca", split="train")
        pools["benign"] += [r["instruction"] for r in d if not r["input"] and 20 < len(r["instruction"]) < 200][:3000]

    attempt("JBB-Behaviors", jbb); attempt("AdvBench", adv)
    attempt("XSTest", xs); attempt("Alpaca", alp)
    seed = DATA / "seed_en.csv"
    if seed.exists():
        for r in csv.DictReader(open(seed, encoding="utf-8")):
            pools[r["kind"]].append(r["text"])
    for k, v in pools.items():
        seen, uniq = set(), []
        for t in map(clean, v):
            if t and t.lower() not in seen:
                seen.add(t.lower()); uniq.append(t)
        rng.shuffle(uniq)
        pools[k] = uniq[:CAPS[k]]
        print(k, len(pools[k]))
    assert all(len(v) > 60 for v in pools.values()), "Too few prompts loaded; see warnings / use seed_en.csv"
    return pools


def split_pool(items, kind):
    a, b, c = SPLITS[kind]
    return {"discovery": items[:a], "train": items[a:a + b], "test": items[a + b:a + b + c]}


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
    def translate(texts, tgt, bs=32):
        out = []
        for i in range(0, len(texts), bs):
            enc = ntok(texts[i:i + bs], return_tensors="pt", padding=True,
                       truncation=True, max_length=200).to(dev)
            gen = mt.generate(**enc, forced_bos_token_id=ntok.convert_tokens_to_ids(tgt),
                              max_new_tokens=220, num_beams=2)
            out += ntok.batch_decode(gen, skip_special_tokens=True)
        return out
    return translate


def romanize_hindi(text):
    from indic_transliteration import sanscript
    return sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS).lower()


def build():
    pools = load_english()
    items = []  # (kind, idx, split, english text)
    for kind, lst in pools.items():
        for split, texts in split_pool(lst, kind).items():
            for t in texts:
                items.append((kind, split, t))
    ids = {}
    rows = []
    for n, (kind, split, t) in enumerate(items):
        ids[n] = f"{kind}-{n:04d}"
        rows.append({"id": ids[n], "kind": kind, "split": split, "lang": "en", "text": t})
    translate = make_translator()
    en_texts = [t for _, _, t in items]
    hi_cache = None
    for lang, code in NLLB.items():
        print("translating ->", lang)
        tr = translate(en_texts, code)
        if lang == "hi":
            hi_cache = tr
        for n, t in enumerate(tr):
            rows.append({"id": ids[n], "kind": items[n][0], "split": items[n][1], "lang": lang, "text": clean(t)})
    for n, t in enumerate(hi_cache):  # romanized Hindi (a real, common way people type Hindi)
        rows.append({"id": ids[n], "kind": items[n][0], "split": items[n][1], "lang": "hi_latn",
                     "text": clean(romanize_hindi(t))})
    write_jsonl(DATA / "prompts.jsonl", rows)
    ref = {"en": REFUSAL_TEMPLATES_EN}
    for lang, code in NLLB.items():
        ref[lang] = [clean(x) for x in translate(REFUSAL_TEMPLATES_EN, code)]
    json.dump(ref, open(DATA / "refusals.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # hand-check sample: 10 random items per language. Fill the 'ok' column with 1/0.
    with open(DATA / "handcheck_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["lang", "id", "english", "translation", "ok"])
        en = {r["id"]: r["text"] for r in rows if r["lang"] == "en"}
        for lang in ALL_LANGS[1:]:
            for r in rng.sample([r for r in rows if r["lang"] == lang], 10):
                w.writerow([lang, r["id"], en[r["id"]], r["text"], ""])
    print("wrote", len(rows), "rows. Now fill data/handcheck_sample.csv and run: python 01_build_data.py --stats")


def stats():
    import collections
    ok = collections.defaultdict(list)
    for r in csv.DictReader(open(DATA / "handcheck_sample.csv", encoding="utf-8")):
        if r["ok"] in ("0", "1"):
            ok[r["lang"]].append(int(r["ok"]))
    for l, v in ok.items():
        print(f"{l}: translation pass-rate {sum(v)}/{len(v)}")


if __name__ == "__main__":
    stats() if "--stats" in sys.argv else build()
