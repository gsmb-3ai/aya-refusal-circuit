"""Stage 5c: AuditKIT before/after comparison of the capability probe (MGSM-lite) for one adapter arm.

Uses AuditKIT's documented spine:  ak.Sample -> ak.evaluate(samples, model, metrics) -> ak.compare(a, b).summary()
`model` may be any list[str] -> list[str] callable (documented), so our local LoRA model plugs straight in.
AuditKIT also ships red-team probes (prompt injection / jailbreak / encoding / over-refusal) via RedTeamRunner
and `auditkit redteam`; see its docs and add them as an extra robustness section if you have time.
"""
import argparse
import auditkit as ak
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["full", "allld", "circuit", "random"])
ap.add_argument("--n", type=int, default=40)
args = ap.parse_args()
seed_all()

from datasets import load_dataset
samples = []
for lang in ["en", "es", "bn", "sw"]:
    d = load_dataset("juletxara/mgsm", lang, split="test")
    for r in list(d)[:args.n]:
        q = r["question"] + "\n\nThink step by step briefly, then finish with 'Answer: <number>'."
        samples.append(ak.Sample(input=q, target=str(int(r["answer_number"]))))

model, tok = load_model()


def as_callable(m):
    def f(prompts):
        outs = generate_batch(m, tok, list(prompts), max_new_tokens=192, bs=8)
        return [extract_number(o) or "" for o in outs]
    return f


base_res = ak.evaluate(samples, as_callable(model), ["exact_match"])
from peft import PeftModel
tuned = PeftModel.from_pretrained(model, str(CKPT / args.arm)).eval()
tuned_res = ak.evaluate(samples, as_callable(tuned), ["exact_match"])
summary = ak.compare(base_res, tuned_res).summary()
print(summary)
open(RES / f"auditkit_compare_{args.arm}.txt", "w").write(str(summary))
