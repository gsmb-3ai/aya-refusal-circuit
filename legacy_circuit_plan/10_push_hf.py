"""Stage 7: publish PUBLIC artifacts (dataset, adapters, logs, Space). Never uploads results/gen (raw completions)."""
import shutil, tempfile
from huggingface_hub import HfApi
from common import *

api = HfApi(token=HF_TOKEN)
DS, MD, LG, SP = (f"{HF_USER}/aya-multilingual-refusal-prompts", f"{HF_USER}/aya-refusal-circuit-adapters",
                  f"{HF_USER}/aya-refusal-circuit-logs", f"{HF_USER}/aya-refusal-circuit-demo")
summary_md = (RES / "summary.md").read_text() if (RES / "summary.md").exists() else "(run 09_report.py first)"


def upload(repo, rtype, folder, card, **kw):
    api.create_repo(repo, repo_type=rtype, exist_ok=True, private=False, **kw)
    folder = Path(folder)
    (folder / "README.md").write_text(card, encoding="utf-8")
    api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type=rtype)
    print("published", repo)


tmp = Path(tempfile.mkdtemp())
# 1) dataset: prompts + refusal targets + curated train set (no harmful completions anywhere)
d = tmp / "ds"; d.mkdir()
for f in ["prompts.jsonl", "refusals.json", "train_curated.jsonl"]:
    shutil.copy(DATA / f, d / f)
if (DATA / "curated" / "manifest.json").exists():
    shutil.copy(DATA / "curated" / "manifest.json", d / "curatorkit_manifest.json")
upload(DS, "dataset", d, f"""---
language: [en, es, hi, ar, bn, ta, sw]
license: other
tags: [safety, multilingual, red-teaming, refusal]
---
# Multilingual refusal prompt set (8 language variants)
Prompts only. Harmful/benign/borderline prompts come from public benchmarks (JailbreakBench JBB-Behaviors,
AdvBench, XSTest, Alpaca) machine-translated with NLLB-200 (translation hand-checked on a small sample; see logs).
`hi_latn` = romanized Hindi. Splits: discovery (locate circuit), train, test. Train languages: {TRAIN_LANGS};
held-out: {HELDOUT_LANGS}. `train_curated.jsonl` was cleaned/deduplicated with CuratorKIT; helpful answers are
self-distilled from the base model; refusals are translated templates. Respect the licences of the source datasets.
Intended use: safety evaluation and research. Not a complete safety benchmark.
""")
# 2) adapters
m = tmp / "models"; m.mkdir()
for arm in ["full", "allld", "circuit", "random"]:
    if (CKPT / arm).exists():
        shutil.copytree(CKPT / arm, m / arm)
        if (RES / f"train_{arm}.json").exists(): shutil.copy(RES / f"train_{arm}.json", m / arm / "train_info.json")
upload(MD, "model", m, f"""---
base_model: {MODEL_ID}
library_name: peft
tags: [safety, lora, multilingual, refusal]
---
# LoRA adapters: closing the multilingual safety gap
Folders = arms: `full` (all modules, all layers), `allld` (o_proj+down_proj, all layers),
`circuit` (o_proj+down_proj only in the located refusal layers), `random` (control, random layers).
Base model: {MODEL_ID}. Results (before/after):

{summary_md}
Research artifacts, not certified for deployment. Follow the base model's and Lexsi libraries' licences.
""")
# 3) logs (labels, circuit-discovery results, figures) - NOT raw generations
l = tmp / "logs"; l.mkdir()
shutil.copytree(FLAGS, l / "flags")
for f in RES.glob("*"):
    if f.is_file() and f.suffix in (".json", ".md", ".png", ".txt"):
        shutil.copy(f, l / f.name)
upload(LG, "dataset", l, "# Logs\nRefusal labels per prompt id, circuit discovery results, summaries and figures. Raw completions are intentionally not published.\n")
# 4) Space (CPU, cached results)
s = tmp / "space"; s.mkdir()
for f in ["app.py", "requirements.txt"]:
    shutil.copy(ROOT / "app" / f, s / f)
for f in ["summary.json", "summary.md", "gallery.json", "refusal_by_lang.png", "fig_direction_cosine.png", "fig_separation.png", "fig_layer_contrib.png"]:
    if (RES / f).exists(): shutil.copy(RES / f, s / f)
upload(SP, "space", s, "---\ntitle: Aya refusal circuit\nsdk: gradio\napp_file: app.py\npinned: false\n---\n# Multilingual safety gap explorer\n", space_sdk="gradio")
