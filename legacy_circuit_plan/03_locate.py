"""Stage 3: LOCATE where refusal lives, and why it under-fires outside English.

Method (plain PyTorch hooks, so it works on Tiny Aya regardless of library support):
 1. last-token residual-stream activations on discovery prompts (harmful vs benign) per language
 2. refusal direction per layer = mean(harmful) - mean(benign); compare each language to English
 3. pick the layer where the English direction separates best (Cohen's d)
 4. direct contribution of each layer's attention / MLP output to that direction
 5. CAUSAL check: ablate the English direction everywhere -> refusal should drop; random-direction control
 6. choose k layers for the restricted fine-tune (+ a disjoint random-layer control set)
Only TRAIN languages are used for selecting layers; held-out languages are reported, never used.
"""
import argparse
import numpy as np, torch, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=8, help="number of layers in the circuit")
ap.add_argument("--n_abl", type=int, default=30, help="prompts per language for the ablation check")
args = ap.parse_args()
seed_all()

rows = read_jsonl(DATA / "prompts.jsonl")


def texts(split, kind, lang):
    return [r["text"] for r in rows if r["split"] == split and r["kind"] == kind and r["lang"] == lang]


model, tok = load_model()
layers = model.model.layers
L = len(layers)


@torch.no_grad()
def collect(prompts, bs=8):
    cur = {}
    hooks = []

    def mk(name, i):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            cur[(name, i)] = o[:, -1, :].float().cpu()
        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.self_attn.register_forward_hook(mk("attn", i)))
        hooks.append(layer.mlp.register_forward_hook(mk("mlp", i)))
    R, A, M = [], [], []
    tx = [chat_text(tok, p) for p in prompts]
    dev = model.device
    for s in range(0, len(tx), bs):
        enc = tok(tx[s:s + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        R.append(torch.stack([h[:, -1, :].float().cpu() for h in out.hidden_states[1:L + 1]], 1))
        A.append(torch.stack([cur[("attn", i)] for i in range(L)], 1))
        M.append(torch.stack([cur[("mlp", i)] for i in range(L)], 1))
    for h in hooks:
        h.remove()
    return torch.cat(R), torch.cat(A), torch.cat(M)


print("collecting activations ...")
acts = {(l, k): collect(texts("discovery", k, l)) for l in ALL_LANGS for k in ("harmful", "benign")}

dirs = {l: acts[(l, "harmful")][0].mean(0) - acts[(l, "benign")][0].mean(0) for l in ALL_LANGS}   # [L, D]
u_en = dirs["en"] / dirs["en"].norm(dim=-1, keepdim=True)


def cohen_d(a, b):
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-8
    return float((a.mean() - b.mean()) / pooled)


cos, sep, gap = {}, {}, {}
for l in ALL_LANGS:
    h, b = acts[(l, "harmful")][0], acts[(l, "benign")][0]
    cos[l] = F.cosine_similarity(dirs[l], dirs["en"], dim=-1).tolist()
    ph, pb = (h * u_en[None]).sum(-1), (b * u_en[None]).sum(-1)       # [N, L]
    sep[l] = [cohen_d(ph[:, i].numpy(), pb[:, i].numpy()) for i in range(L)]
    gap[l] = [float(ph[:, i].mean() - pb[:, i].mean()) for i in range(L)]

lo, hi = int(0.25 * L), int(0.75 * L)
best = lo + int(np.argmax(sep["en"][lo:hi]))
u = u_en[best]
print(f"best layer (English separation): {best}/{L}")

dla = {}  # direct contribution of attn/mlp outputs at layer i to the refusal direction at `best`
for l in ALL_LANGS:
    d = {}
    for name, idx in (("attn", 1), ("mlp", 2)):
        h, b = acts[(l, "harmful")][idx], acts[(l, "benign")][idx]
        d[name] = [float((h[:, i] @ u).mean() - (b[:, i] @ u).mean()) if i <= best else 0.0 for i in range(L)]
    dla[l] = d
total = {l: np.array(dla[l]["attn"]) + np.array(dla[l]["mlp"]) for l in ALL_LANGS}
importance = np.clip(total["en"], 0, None)
weak = [l for l in TRAIN_LANGS if l != "en"]
gap_score = np.clip(total["en"] - np.mean([total[l] for l in weak], 0), 0, None)
top = lambda s: sorted(np.argsort(-s)[:args.k].tolist())
imp_layers, gap_layers = top(importance), top(gap_score)
rest = [i for i in range(L) if i not in imp_layers]
rand_layers = sorted(random.Random(SEED).sample(rest, args.k))
print("importance layers:", imp_layers, "| gap layers:", gap_layers, "| random control:", rand_layers)


# ---------------- causal check: ablate the English refusal direction everywhere
class Ablate:
    def __init__(self, vec):
        self.vec, self.h = vec, []

    def __enter__(self):
        def hook(mod, inp, out):
            x = out[0] if isinstance(out, (tuple, list)) else out
            v = self.vec.to(x.device, x.dtype)
            x2 = x - (x @ v).unsqueeze(-1) * v
            return (x2,) + tuple(out[1:]) if isinstance(out, (tuple, list)) else x2
        self.h = [ly.register_forward_hook(hook) for ly in layers]
        return self

    def __exit__(self, *a):
        for h in self.h:
            h.remove()


def refusal_by_lang():
    res = {}
    for l in ALL_LANGS:
        outs = generate_batch(model, tok, texts("test", "harmful", l)[:args.n_abl], max_new_tokens=48, bs=16)
        res[l] = float(np.mean([lexicon_refusal(o) for o in outs]))
    return res


print("ablation check ...")
abl = {"none": refusal_by_lang()}
with Ablate(u):
    abl["ablate_english_direction"] = refusal_by_lang()
rand_runs = []
for s in range(3):
    g = torch.Generator().manual_seed(s)
    v = torch.randn(u.shape[0], generator=g); v /= v.norm()
    with Ablate(v):
        rand_runs.append(refusal_by_lang())
abl["ablate_random_direction_mean"] = {l: float(np.mean([r[l] for r in rand_runs])) for l in ALL_LANGS}

out = {"n_layers": L, "best_layer": best, "k": args.k, "cos_to_english": cos, "separation_d": sep,
       "projection_gap": gap, "dla": dla, "importance_layers": imp_layers, "gap_layers": gap_layers,
       "random_layers": rand_layers, "ablation_refusal_lexicon": abl}
json.dump(out, open(RES / "locate.json", "w"), indent=1)
json.dump({"best_layer": best, "importance_layers": imp_layers, "gap_layers": gap_layers,
           "random_layers": rand_layers, "n_layers": L}, open(RES / "circuit_layers.json", "w"), indent=1)
torch.save({"u_en": u_en, "dirs": dirs, "best": best}, CKPT / "refusal_dirs.pt")

# ---------------- figures
fig, ax = plt.subplots(figsize=(10, 3.5))
im = ax.imshow([cos[l] for l in ALL_LANGS], aspect="auto", vmin=0, vmax=1, cmap="viridis")
ax.set_yticks(range(len(ALL_LANGS))); ax.set_yticklabels(ALL_LANGS); ax.set_xlabel("layer")
ax.set_title("Cosine of each language's refusal direction with English"); fig.colorbar(im)
fig.tight_layout(); fig.savefig(RES / "fig_direction_cosine.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3.5))
for l in ALL_LANGS:
    ax.plot(sep[l], label=l)
ax.axvline(best, color="k", ls="--"); ax.legend(ncol=4, fontsize=7)
ax.set_xlabel("layer"); ax.set_ylabel("Cohen d along English direction")
ax.set_title("How strongly each language engages the English refusal direction")
fig.tight_layout(); fig.savefig(RES / "fig_separation.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3.5))
w = 0.4
ax.bar(np.arange(L) - w / 2, total["en"], w, label="English")
ax.bar(np.arange(L) + w / 2, np.mean([total[l] for l in weak], 0), w, label="train langs (non-en) mean")
for i in imp_layers:
    ax.axvspan(i - .5, i + .5, color="orange", alpha=.2)
ax.set_xlabel("layer"); ax.set_ylabel("write to refusal direction"); ax.legend()
fig.tight_layout(); fig.savefig(RES / "fig_layer_contrib.png", dpi=150); plt.close(fig)
print("done. See results/locate.json and results/fig_*.png")
