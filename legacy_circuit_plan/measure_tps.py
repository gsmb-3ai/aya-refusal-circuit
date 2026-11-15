"""Measure batched generation throughput of an arm on your GPU (feeds roi.py)."""
import sys, time
from common import *
arm = sys.argv[1] if len(sys.argv) > 1 else "base"
model, tok = load_model(adapter=None if arm == "base" else CKPT / arm)
prompts = [r["text"] for r in read_jsonl(DATA / "prompts.jsonl") if r["lang"] == "en" and r["kind"] == "benign"][:32]
generate_batch(model, tok, prompts[:4], max_new_tokens=8, bs=4)  # warmup
t = time.time(); outs = generate_batch(model, tok, prompts, max_new_tokens=128, bs=32); dt = time.time() - t
n = sum(len(tok(o, add_special_tokens=False)["input_ids"]) for o in outs)
print(f"{arm}: {n} generated tokens in {dt:.1f}s = {n / dt:.0f} tokens/sec (batch 32)")
