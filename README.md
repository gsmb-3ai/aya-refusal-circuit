# Repairing Safety Drift in Tiny Aya with a Weight-Space Safety Update

Entry for the **Lexsi AI Safety & Alignment Hackathon**, Track 2 (sub-track 2b, problem 9). Model: [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global). Built and run on a free Kaggle T4 GPU.

- **Write-up (Hugging Face article):** https://huggingface.co/blog/GSMB/aya-safety-drift-repair
- **Adapters:** [GSMB/aya-drift-v3-adapters](https://huggingface.co/GSMB/aya-drift-v3-adapters)
- **Datasets:** [prompts](https://huggingface.co/datasets/GSMB/aya-drift-v3-prompts) · [logs](https://huggingface.co/datasets/GSMB/aya-drift-v3-logs)
- **Collections:** https://huggingface.co/GSMB/collections

> **Responsible-use note.** The `domain` adapter yields a model with measurably weakened refusals. It is released only to reproduce the drift measurement. Do not deploy it.

## What this repository contains

| Part | Status |
|---|---|
| `notebooks/aya-drift-repair-v3-kaggle.ipynb` | **The submitted experiment.** All reported results come from this notebook. |
| `00_smoke_test.py` to `10_push_hf.py`, `common.py`, `sync.py`, `roi.py`, `measure_tps.py`, `app.py` | **Earlier exploratory plan** (find the refusal circuit, fix only that). Kept for transparency. **Not used for any reported result and not validated end to end.** |

Why two parts: we first planned a circuit-localisation fix (CuratorKIT → CircuitKIT → SafeTune → AuditKIT). CircuitKIT ran out of memory on the free T4, and a layer-restricted LoRA run was invalid (see Known failures). We then pursued the drift-repair experiment below, which ran correctly.

## The experiment in brief
**Question.** Fine-tuning can quietly weaken refusals, especially in languages nobody checked. Can the drift be reversed by adding a separately trained safety adapter's update to the drifted model, without retraining it?

**Method.** Train a LoRA domain adapter (translated Hindi/Bengali/Tamil Q&A) to create the drifted model; train a LoRA safety adapter (English, Spanish, Hindi, Arabic); repair with `drifted + α × safety update` for α in {0.5, 1.0, 1.5}, plus an orthogonalised variant; compare with a mixed-training baseline. Evaluate in eight languages, four of them (Bengali, Tamil, Swahili, romanised Hindi) held out of safety training.

**Lexsi repos chained:** [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT) (clean and de-duplicate training data) → [SafeTune](https://github.com/Lexsi-Labs/SafeTune) (`recover.task_arithmetic` builds every repaired model) → [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) (before/after maths comparison). [CircuitKIT](https://github.com/Lexsi-Labs/CircuitKIT) was attempted and could not be run on a T4.

## Key results
Keyword-labelled refusal, percent. Full tables, intervals and caveats are in the article.

| | base | domain (drifted) | repair α 1.0 | safety only | mixed |
|---|---|---|---|---|---|
| Refusal, seen languages | 94.2 | 87.9 | 95.0 | 79.6* | 70.0 |
| Refusal, held-out languages | 65.8 | 52.1 | 63.7 | 72.9 | 54.2 |
| Jailbreak success, held-out (lower is better) | 51.2 | 66.2 | 50.0 | 38.8 | 56.2 |
| Domain chrF | 28.7 | 34.6 | 34.3 | 28.6 | 33.3 |
| MGSM accuracy | 56.2 | 47.5 | 48.8 | 59.4 | 49.4 |

\*The keyword list under-counts this cell because it misses some trained refusal phrases.

- Domain fine-tuning lowered held-out refusal by 13.8 points (95% CI 9.2 to 18.8) and raised held-out jailbreak success by 15.0 points (7.5 to 23.8).
- Repair at α 0.5 to 1.0 brought refusal and jailbreak resistance back to about base level, kept the domain gains and showed no measurable over-refusal.
- The repair is a trade-off: the safety adapter alone is safer but has no domain skill. Lost maths ability was not recovered. Removing the overlap between the updates made no difference. Mixed training did not close the gap.

## Reproduce
1. Create a Kaggle notebook with a **GPU T4**, Internet on, and a Kaggle secret `HF_TOKEN` (Hugging Face token with write access).
2. Accept the [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) license on Hugging Face.
3. Import `notebooks/aya-drift-repair-v3-kaggle.ipynb` and run it (Save Version, then Save & Run All). Expect about two hours. Progress is backed up to a private Hub repo, so reruns resume.
4. At step 11 the notebook prints 104 replies for labelling. The `LABELS` string in the notebook is the one used for the reported numbers. For an independent check, replace it with your own labels.
5. The notebook begins with a self-test that checks the merge arithmetic against PEFT's sequential merge. Set `PUBLISH = False` (default) to avoid uploading anything.

## Known failures and limitations
- **Labels.** The 104-reply validation set was labelled with Claude's assistance and reviewed by the author; it is not independent human annotation. Only 12 of the 97 usable rows were "answered", so false positives are not measured. A small LLM judge agreed with these labels only at chance level (kappa 0.19), so the keyword list (kappa 0.69) is the primary labeler.
- One seed; α chosen on the test set (optimistic); machine-translated prompts; 60 harmful prompts per language and 160 maths questions.
- **Gap-aware layer location (earlier plan).** A normalised layer-gap score pointed to late layers (35, 34, 26, 27, 33, 25, 32, 31, 28, 29), but the restricted-LoRA run that followed is **invalid**: training loss and gradient norm were exactly 0 because reply tokens were truncated and every label was masked. No circuit-based fix is claimed.
- **CircuitKIT** ran out of GPU memory on the T4, so no circuit analysis was completed.
- Earlier drift-repair versions (v1, v2) had merge and measurement bugs and are superseded.
- This is not a circuit or neuron localisation study. Tiny Aya Global only.

## Licenses
- Base model and derived adapters: CC-BY-NC-4.0 (non-commercial), following [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global).
- Translation model: [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M), CC-BY-NC-4.0.
- Source datasets and their licenses are listed in the article.
- Code in this repository: [Code in this repository: all rights reserved; contact the author for reuse.].

## Acknowledgements
Lexsi Labs for the hackathon and the CuratorKIT, SafeTune, AuditKIT and CircuitKIT toolkits; Cohere for Tiny Aya; the JailbreakBench, AdvBench, XSTest, MGSM and Alpaca authors.
