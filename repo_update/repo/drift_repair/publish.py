"""Publish PUBLIC artifacts to NEW repos (<user>/aya-drift-<tag>-*). Raw completions (results/gen) are never uploaded."""
import shutil, tempfile
from huggingface_hub import HfApi
from common import *

api = HfApi(token=HF_TOKEN)
DS, MD, LG = f"{PUB}-prompts", f"{PUB}-adapters", f"{PUB}-logs"
summary_md = (RES / "summary.md").read_text(encoding="utf-8") if (RES / "summary.md").exists() else "(summary not built yet)"


def upload(repo, rtype, folder, card):
    api.create_repo(repo, repo_type=rtype, exist_ok=True, private=False)
    (Path(folder) / "README.md").write_text(card, encoding="utf-8")
    api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type=rtype)
    print("published:", repo, flush=True)


tmp = Path(tempfile.mkdtemp())
d = tmp / "ds"; d.mkdir()
for f in ["eval_prompts.jsonl", "domain_train.jsonl", "domain_test.jsonl", "safety_train.jsonl", "refusals.json"]:
    if (DATA / f).exists():
        shutil.copy(DATA / f, d / f)
for f in RES.glob("curation_*.json"):
    shutil.copy(f, d / f.name)
for mf in (DATA / "curated").glob("*/manifest.json"):
    shutil.copy(mf, d / f"curatorkit_manifest_{mf.parent.name}.json")
upload(DS, "dataset", d, f"""---
license: other
language: [en, es, hi, ar, bn, ta, sw]
tags: [safety, multilingual, red-teaming, refusal, safety-drift]
---
# Prompts and training data: safety drift after domain fine-tuning (Tiny Aya)
`eval_prompts.jsonl`: harmful / benign / borderline-safe prompts (JailbreakBench JBB-Behaviors, AdvBench, XSTest, Alpaca),
machine-translated with NLLB-200 into es, hi, ar, bn, ta, sw (+ romanized Hindi `hi_latn`). Splits: discovery (safety training source) / test.
`domain_train.jsonl`, `domain_test.jsonl`: short instruction/answer pairs in hi, bn, ta (Alpaca, machine-translated; test items are held out).
`safety_train.jsonl`: translated refusals for harmful prompts + the base model's own answers for benign prompts. Train and domain files were
cleaned and de-duplicated with CuratorKIT (see the manifests). Prompts only: no harmful completions. Respect the source datasets' licences.
""")
m = tmp / "models"; m.mkdir()
for arm in ["domain", "safety", "mixed"]:
    src = CKPT / arm
    if src.exists():
        (m / arm).mkdir()
        for f in ["adapter_config.json", "adapter_model.safetensors", "train_meta.json"]:
            if (src / f).exists():
                shutil.copy(src / f, m / arm / f)
upload(MD, "model", m, f"""---
base_model: {MODEL_ID}
library_name: peft
tags: [safety, lora, safety-drift]
license: other
---
# LoRA adapters: domain drift, safety, mixed
`domain` is a domain-task adapter and **produces a model with weakened refusal behaviour** (the point of the experiment): research use only,
never deploy it. `safety` restores refusals; `mixed` is the domain+safety joint-training baseline. Repaired models are built by merging
`domain` and `safety` (SafeTune `recover.task_arithmetic`), see the logs repo.

{summary_md}
""")
l = tmp / "logs"; l.mkdir()
FLAGS = RES / "flags"
if FLAGS.exists():
    shutil.copytree(FLAGS, l / "flags")
for f in RES.glob("*"):
    if f.is_file() and (f.suffix in (".json", ".md", ".png") or f.name.startswith("cap_raw_")):
        shutil.copy(f, l / f.name)      # to_label.csv (reply openings) is deliberately not copied
upload(LG, "dataset", l, "---\nlicense: other\n---\n# Logs\nPer-prompt refusal labels (judge and keyword list), summaries, paired comparisons, "
       "label validation, figures. Raw model completions are intentionally not published.\n")
print("\nLinks:\n ", "\n  ".join(f"https://huggingface.co/{'datasets/' if r in (DS, LG) else ''}{r}" for r in (DS, MD, LG)))
