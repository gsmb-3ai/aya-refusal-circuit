"""Train a LoRA adapter (domain / safety / mixed)."""
import argparse, math, time, json, random
from pathlib import Path
import torch
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["domain", "safety", "mixed"])
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--bs", type=int, default=1)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--max_len", type=int, default=512)
ap.add_argument("--r", type=int, default=16)
args = ap.parse_args()
seed_all()

from peft import LoraConfig, get_peft_model

model, tok = load_model()
dev = next(model.parameters()).device
base_params = sum(p.numel() for p in model.parameters())
model.config.use_cache = False

# allld-style: only o_proj + down_proj, all layers
cfg = LoraConfig(
    target_modules=["o_proj", "down_proj"],
    r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, cfg)
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"arm={args.arm} | trainable={n_train:,} ({100*n_train/base_params:.4f}%)")

if args.arm == "domain":
    data = read_jsonl(DATA / "domain_train.jsonl")
elif args.arm == "safety":
    data = read_jsonl(DATA / "safety_train.jsonl")
else:  # mixed
    data = read_jsonl(DATA / "domain_train.jsonl") + read_jsonl(DATA / "safety_train.jsonl")
    random.Random(SEED).shuffle(data)

def encode(r):
    p = tok(chat_text(tok, r["instruction"]), add_special_tokens=False)["input_ids"]
    o = tok(r["output"], add_special_tokens=False)["input_ids"]
    if r.get("kind") == "harmful":
        o = o + [tok.eos_token_id]
    room = args.max_len - len(p)
    if room < 12:
        return None
    o = o[:room]
    return p + o, [-100] * len(p) + o

encoded = [e for e in (encode(r) for r in data) if e is not None]
print(f"usable examples: {len(encoded)}/{len(data)}")
assert encoded, "all examples dropped – raise max_len"

def collate(batch):
    m = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), m), tok.pad_token_id, dtype=torch.long)
    lab = torch.full((len(batch), m), -100, dtype=torch.long)
    att = torch.zeros((len(batch), m), dtype=torch.long)
    for k, (i, l) in enumerate(batch):
        ids[k, :len(i)] = torch.tensor(i)
        lab[k, :len(l)] = torch.tensor(l)
        att[k, :len(i)] = 1
    return ids.to(dev), lab.to(dev), att.to(dev)

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda", enabled=CUDA)
model.train()

step, losses = 0, []
t0 = time.time()
while step < args.steps:
    random.Random(SEED + step).shuffle(encoded)
    for b in range(0, len(encoded), args.bs * args.accum):
        if step >= args.steps:
            break
        chunk = encoded[b:b + args.bs * args.accum]
        n_mb = max(1, math.ceil(len(chunk) / args.bs))
        tot = 0.0
        for k in range(0, len(chunk), args.bs):
            ids, lab, att = collate(chunk[k:k + args.bs])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=CUDA):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / n_mb
            scaler.scale(loss).backward()
            tot += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        step += 1
        losses.append(tot)
        if step % 10 == 0 or step == 1:
            print(f"step {step}/{args.steps}  loss={tot:.4f}  elapsed={time.time()-t0:.0f}s")

out_dir = CKPT / args.arm
out_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(out_dir))
tok.save_pretrained(str(out_dir))
json.dump({"arm": args.arm, "steps": args.steps, "final_loss": losses[-1] if losses else None,
           "trainable_pct": 100 * n_train / base_params},
          open(out_dir / "train_meta.json", "w"), indent=1)
print("saved", out_dir)