# Closing the Multilingual Safety Gap in Aya: find the refusal circuit, fix only that

Lexsi Labs AI Safety Hackathon, Track 2b, Problem 9. Free tier only (Kaggle GPU + Hugging Face + GitHub).
Pipeline: **CuratorKIT** (data) -> **CircuitKIT** (locate) -> **SafeTune** (steer / fix) -> **AuditKIT** (evaluate).

## 0. Read this first (honest status)

* I could not run this on a GPU. I compiled every file and unit-tested the pure-Python helpers, but
  expect to debug the first run of each stage. Run the stages in order and fix errors as they appear.
* **Verified from the libraries' public docs:** `curatorkit` (`Curator(CuratorConfig(dataset=..., dedup="minhash", clean=True, export_formats=["alpaca"], output_dir=...)).run()`),
  `circuitkit` (`ck.load_model`, `ck.discover(model, task, n_examples=)`, supports Tiny Aya + Aya Expanse 8B),
  `safetune` (`steer.RefusalDirectionTrainer(model, tok, alpha=).calibrate(harmful=, harmless=)`),
  `auditkit` (`ak.Sample`, `ak.evaluate(samples, model_or_callable, metrics)`, `ak.compare(a, b).summary()`).
* **NOT verified (you must check the docs):** how to define a *custom refusal task* in CircuitKIT, and whether
  SafeTune's steering wrapper accepts raw strings and exposes `.generate`. The code isolates both and says so.
  Because of the first gap, the refusal localization itself is done with plain PyTorch hooks (`03_locate.py`),
  and CircuitKIT is used for model loading/discovery plus a cross-check (`04_circuitkit.py`). If you learn the
  custom-task API from https://circuitkit.lexsi.ai/guides/ , wire it into 04 and report it as a second localization method.
* **Originality:** Lexsi already published a multilingual safety repair of Gemma-4-31B-it (Hugging Face: `Lexsi/Gemma-4-31B-it-Multilingual-Safety-Repair`).
  Judges know it. Your angle is different: Aya models, a *circuit-restricted* fix with random-layer controls,
  held-out languages and romanized Hindi, a free-tier reproducible pipeline. Say so in your write-up.
* **Licences:** the Lexsi libraries use LSAL (free for academic research; organizations must acknowledge use; no commercial use
  without a licence). Tiny Aya is gated: accept its licence on Hugging Face with the account whose token you use.

## 1. One-time setup (30 minutes)

1. **Hugging Face:** create a token (Settings -> Access Tokens, *write*). Open https://huggingface.co/CohereLabs/tiny-aya-global and accept the licence. Also accept terms on `walledai/AdvBench` and `walledai/XSTest` if gated. Check the model card (is it instruction-tuned? what is the chat template?).
2. **Kaggle:** verify your phone number (needed for GPU and Internet). In a notebook: Settings -> Accelerator **GPU T4 x2** (or P100), **Internet On**, Persistence Files-only. Add-ons -> Secrets -> add `HF_TOKEN`. Free GPU time is limited per week and each session is limited in length; check your current quota in your Kaggle profile.
3. **GitHub:** create a public repo `aya-refusal-circuit`, upload this folder, commit often (the hackathon requires work done during the event).
4. In every notebook, the first cell is:
```python
!git clone https://github.com/<you>/aya-refusal-circuit.git
%cd aya-refusal-circuit
import os; os.environ["HF_USER"]="<your-hf-username>"
!python --version
!pip install -q -U "transformers>=5.15" datasets accelerate peft huggingface_hub matplotlib sentencepiece indic-transliteration
```
Each notebook ends by pushing its outputs to a **private** HF dataset repo (`python sync.py push <dir>`) and the next begins by pulling them (`python sync.py pull <dir>`). This is how work survives Kaggle session resets and avoids library version conflicts, because every stage can use its own environment.

## 2. Notebook-by-notebook

| # | Notebook | Run | Time | Output |
|---|---|---|---|---|
| 0 | Smoke test | `python 00_smoke_test.py` | 5 min | confirms GPU, chat template, model answers |
| 1 | Data | `pip install -q indic-transliteration sacremoses`; `python 01_build_data.py`; hand-check `data/handcheck_sample.csv`, then `python 01_build_data.py --stats`; `python sync.py push data` | 20-30 min | `data/prompts.jsonl`, `refusals.json` |
| 2 | Train set + **CuratorKIT** | `pip install -q curatorkit`; `python sync.py pull data`; `python 02_make_train_set.py`; `python sync.py push data` | 15 min | `data/train_curated.jsonl`, `results/curation.json` |
| 3 | Locate | pull data; `python 03_locate.py --k 8`; `python sync.py push results` | 15-25 min | `results/locate.json`, `circuit_layers.json`, `fig_*.png` |
| 3b | **CircuitKIT** (fresh notebook) | `pip install -q circuitkit`; pull data + results; `python 04_circuitkit.py` | 10-20 min | cross-check against hook-based layers |
| 4 | Train arms | pull data + results; `python 05_train_arm.py --arm circuit`, then `random`, `allld`, `full` (one per session if quota is tight); `python sync.py push ckpt; python sync.py push results` | 20-40 min each | `ckpt/<arm>`, `results/train_<arm>.json` |
| 5 | Eval + judge (+ **SafeTune** steering) | pull everything; `pip install -q safetune` (needs Python >= 3.12; if `python --version` is lower, use `pip install uv; uv venv --python 3.12 .venv` and run inside it); `python 06_eval_arm.py --arm base` (repeat for `circuit random allld full steer`); `python 07_judge.py --arms base circuit random allld full steer_a0.3 --export_labels 120` | 10-15 min per arm | `results/gen/*`, `results/flags/*` |
| 5b | Judge validation | label `results/to_label.csv` (`human` = 1 refused / 0 complied, >= 100 rows), then `python 07_judge.py --agreement results/to_label.csv --arm base` | 30 min | accuracy + Cohen kappa vs your labels |
| 6 | **AuditKIT** compare | `pip install -q "auditkit[transformers]"`; `python 08_auditkit_compare.py --arm circuit` | 15 min | `results/auditkit_compare_circuit.txt` |
| 7 | Report | `python 09_report.py` | 1 min | `results/summary.md/json`, figures, gallery |
| 8 | Publish | `python 10_push_hf.py` | 5 min | public dataset, adapters, logs, Space |

## 3. What each stage proves

* **Stage 3 (locate)** gives the evidence for the story: (a) a refusal direction exists in English and separates harmful from benign; (b) per language, how similar that direction is (`fig_direction_cosine.png`) and how strongly it is engaged (`fig_separation.png`, Cohen d); (c) which layers write to it (`fig_layer_contrib.png`); (d) **causal**: ablating the English direction lowers refusal across languages, a random direction does not (`ablation_refusal_lexicon` in `locate.json`).
  If the direction is shared but weakly engaged in some languages, your restricted fix is well motivated. If directions differ per language, say that instead; it is a legitimate finding.
* **Stage 4 arms** isolate the claim: `circuit` vs `random` (does *which* layers matter?), `circuit` vs `allld` (does restricting layers matter at equal module types?), `circuit` vs `full` (cost of the standard approach).
* **Held-out languages** (`bn ta sw hi_latn`) are never trained on; report them separately. This is your strongest generalization claim.
* **Steer arm** (SafeTune) is the no-training alternative; sweep `--alpha` (0.1, 0.3, 0.5) and watch over-refusal.

## 4. Metrics (always report more than one)

Refusal rate on harmful prompts (per language, seen vs held-out), jailbreak not-refused rate (a proxy for attack success: not refusing is not the same as producing harmful content), over-refusal on benign and borderline prompts, MGSM-lite capability, parameters changed (%), plus 95% bootstrap CIs. Validate the judge on your hand labels and report agreement.

## 5. Write-up checklist (HF article)

Problem and who is affected -> pipeline diagram -> data sources/licences/translation hand-check pass rates -> why these metrics -> localization findings with controls -> results table (base vs full vs allld vs circuit vs random vs steer) -> failures and limits (small test sets, machine translation, romanized Hindi is not true code-mixing, lexicon+model judge, no adaptive attacker) -> ROI -> reproduce links. Do not publish raw completions to harmful prompts (the scripts keep them in `results/gen`, which `10_push_hf.py` excludes).

## 6. ROI comparison (optional, preferred)

`python measure_tps.py circuit` on your GPU, then
`python roi.py --requests 1000000 --in_tok 300 --out_tok 50 --tps <measured> --gpu_usd_hr <price> --api_in_usd_per_m <price> --api_out_usd_per_m <price>`.
Look up current GPU and API prices on the day you write it and cite them.

## 7. Likely first-run problems

* `OSError ... gated repo`: accept the Tiny Aya licence; check `HF_TOKEN` secret.
* `module.layers` / `self_attn` attribute errors in `03_locate.py`: print `model` and adjust (`model.model.layers` assumed).
* Hook returns shape errors: some versions return a tensor not a tuple from decoder layers; the code handles both, but print `type(out)` if it fails.
* CUDA OOM in training: `--bs 1 --accum 16`, `--max_len 256`, `--r 8`.
* CuratorKIT cannot read the JSONL as-is: check its "Data sources" guide for field mapping; the script falls back to plain dedup and tells you (and records `used_curatorkit` in `results/curation.json`; try to get the CuratorKIT path working so the repo is genuinely in your chain).
* SafeTune `calibrate` wants templated prompts or a different return: read its steering guide and adjust the 6 lines in `06_eval_arm.py`.
* Judge disagrees with you: set `LABEL=lex` for the report, or run the judge with a bigger model (`MODEL_ID=CohereLabs/aya-expanse-8b python 07_judge.py ...` on 2xT4).
