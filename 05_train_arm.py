"""Stage 4: train one fix arm with LoRA on the curated safety set (same data/steps for every arm).

 full    : LoRA on all attention+MLP projections, all layers      (standard safety fine-tune baseline)
 allld   : LoRA on o_proj+down_proj only, all layers             (control: same modules, no layer restriction)
 circuit : LoRA on o_proj+down_proj ONLY in the located layers    (the targeted fix)
 random  : same as circuit but k random layers disjoint from it   (control: does location matter?)
CircuitKIT also ships its own circuit-restricted LoRA ("Fine-tune" in its docs); you can compare against it.
"""
import argparse, math, time
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["full", "allld", "circuit", "random"])
ap.add_argument("--select", default="importance", choices=["importance", "gap"])
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--max_len", type=int, default=320)
ap.add_argument("--r", type=int, default=16)
args = ap.parse_args()
seed_all()

from peft import LoraConfig, get_peft_model

model, tok = load_model(device_map={"": 0})
base_params = sum(p.numel() for p in model.parameters())
model.config.use_cache = False
mods_all = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
cl = json.load(open(RES / "circuit_layers.json")) if args.arm in ("circuit", "random") else None
if args.arm == "full":
    cfg = LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05, target_modules=mods_all, task_type="CAUSAL_LM")
    layers = "all"
elif args.arm == "allld":
    cfg = LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05,
                     target_modules=["o_proj", "down_proj"], task_type="CAUSAL_LM")
    layers = "all"
else:
    layers = cl["random_layers"] if args.arm == "random" else cl[f"{args.select}_layers"]
    cfg = LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05, target_modules=["o_proj", "down_proj"],
                     layers_to_transform=layers, layers_pattern="layers", task_type="CAUSAL_LM")
print("arm:", args.arm, "| layers:", layers)
model = get_peft_model(model, cfg)
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable params: {n_train:,} = {100 * n_train / base_params:.4f}% of base")

data = read_jsonl(DATA / "train_curated.jsonl")


def encode(r):
    p = tok(chat_text(tok, r["instruction"]), add_special_tokens=False)["input_ids"]
    o = tok(r["output"], add_special_tokens=False)["input_ids"]
    if r["kind"] == "harmful":      # refusals are complete; helpful answers may be truncated -> no EOS
        o = o + [tok.eos_token_id]
    ids, lab = (p + o)[:args.max_len], ([-100] * len(p) + o)[:args.max_len]
    return ids, lab


enc = [e for e in map(encode, data) if any(x != -100 for x in e[1])]


def collate(batch):
    m = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), m), tok.pad_token_id)
    lab = torch.full((len(batch), m), -100)
    att = torch.zeros((len(batch), m), dtype=torch.long)
    for k, (i, l) in enumerate(batch):
        ids[k, :len(i)], lab[k, :len(l)], att[k, :len(i)] = torch.tensor(i), torch.tensor(l), 1
    return ids.cuda(), lab.cuda(), att.cuda()


params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
steps_total = math.ceil(len(enc) / (args.bs * args.accum)) * args.epochs
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 10) * max(0.0, 1 - s / max(1, steps_total)))
scaler = torch.amp.GradScaler("cuda")
model.train()
t0, step, log = time.time(), 0, []
for ep in range(args.epochs):
    random.Random(SEED + ep).shuffle(enc)
    for b in range(0, len(enc), args.bs * args.accum):
        chunk = enc[b:b + args.bs * args.accum]
        tot = 0.0
        for k in range(0, len(chunk), args.bs):
            ids, lab, att = collate(chunk[k:k + args.bs])
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / math.ceil(len(chunk) / args.bs)
            scaler.scale(loss).backward(); tot += loss.item()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
        log.append(tot)
        if step % 5 == 0:
            print(f"ep {ep} step {step}/{steps_total} loss {tot:.4f}")
model.save_pretrained(CKPT / args.arm)
json.dump({"arm": args.arm, "layers": layers, "trainable_params": n_train, "base_params": base_params,
           "params_changed_pct": 100 * n_train / base_params, "steps": step, "final_loss": log[-1],
           "minutes": (time.time() - t0) / 60, "examples": len(enc)},
          open(RES / f"train_{args.arm}.json", "w"), indent=1)
print("saved adapter to", CKPT / args.arm)
