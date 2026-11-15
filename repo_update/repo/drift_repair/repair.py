"""Weight-space safety repair of a drifted model.

Operates on the weight UPDATES (dW = alpha/r * B @ A) of the two LoRA adapters, never on A and B separately.
 drifted model  = base + dW_domain
 repaired model = drifted + alpha * dW_safety          (SafeTune: safetune.recover.task_arithmetic)
 ortho repair   = drifted + alpha * (dW_safety minus its component along dW_domain, per module)
"""
import json, torch
from pathlib import Path
from safetensors.torch import load_file


class LoraDeltas:
    """Lazy view of a PEFT LoRA adapter folder: delta(module) = scaling * B @ A, computed on demand."""

    def __init__(self, adapter_dir):
        d = Path(adapter_dir)
        cfg = json.load(open(d / "adapter_config.json"))
        self.scaling = cfg["lora_alpha"] / cfg["r"]
        sd = load_file(str(d / "adapter_model.safetensors"))
        self.A, self.B = {}, {}
        for k, v in sd.items():
            if ".lora_A." in k:
                mod = k.split(".lora_A.")[0].replace("base_model.model.", "", 1)
                self.A[mod] = v
                self.B[mod] = sd[k.replace(".lora_A.", ".lora_B.")]

    def modules(self):
        return sorted(self.A)

    def delta(self, mod, device="cpu"):
        A, B = self.A[mod].to(device).float(), self.B[mod].to(device).float()
        return self.scaling * (B @ A)


def ortho_delta(saf, dom, mod, device="cpu"):
    """Safety update with its component along the domain update (same module) removed."""
    s = saf.delta(mod, device)
    if mod in dom.A:
        d = dom.delta(mod, device)
        den = (d * d).sum().clamp_min(1e-12)
        s = s - ((s * d).sum() / den) * d
    return s


@torch.no_grad()
def merge_adapter_into(model, deltas, scale=1.0):
    for m in deltas.modules():
        w = model.get_submodule(m).weight
        w.copy_((w.float() + scale * deltas.delta(m, w.device)).to(w.dtype))
    return model


class _LazySD:
    """Mapping with only the keys safetune.recover.task_arithmetic looks up; values are built on demand."""

    def __init__(self, keys, getter):
        self.keys, self.getter = set(keys), getter

    def __contains__(self, k):
        return k in self.keys

    def __getitem__(self, k):
        return self.getter(k)


class _SDHolder(torch.nn.Module):
    def __init__(self, sd):
        super().__init__()
        self._lazy = sd

    def state_dict(self, *a, **k):
        return self._lazy


@torch.no_grad()
def build_repaired(model, domain_dir, safety_dir=None, alpha=1.0, ortho=False, use_safetune=True):
    """Turn `model` (the base) into the drifted model, then optionally repair it. Returns the backend name used."""
    dom = LoraDeltas(domain_dir)
    saf = LoraDeltas(safety_dir) if safety_dir else None
    base_cpu = {}
    if saf is not None:   # keep the pre-drift base weights of the modules the safety adapter touches
        base_cpu = {m: model.get_submodule(m).weight.detach().to("cpu", copy=True) for m in saf.modules()}
    merge_adapter_into(model, dom, 1.0)
    if saf is None:
        return "domain_only"

    def sdelta(m):
        return ortho_delta(saf, dom, m) if ortho else saf.delta(m)

    if use_safetune:
        try:
            import safetune
            keys = [m + ".weight" for m in saf.modules()]
            base_v = _SDHolder(_LazySD(keys, lambda k: base_cpu[k[:-7]].float()))
            aligned_v = _SDHolder(_LazySD(keys, lambda k: base_cpu[k[:-7]].float() + sdelta(k[:-7])))
            safetune.recover.task_arithmetic(model, base=base_v, aligned=aligned_v, alpha=alpha)
            return "safetune.recover.task_arithmetic"
        except (ImportError, AttributeError, TypeError) as e:     # raised before any weight is touched
            print("[warn] SafeTune task_arithmetic unavailable, using the built-in equivalent:", repr(e)[:200])
    for m in saf.modules():
        w = model.get_submodule(m).weight
        w.copy_((w.float() + alpha * sdelta(m).to(w.device)).to(w.dtype))
    return "builtin_task_arithmetic"


def selftest():
    """CPU check on a tiny random Llama: the repaired model must equal PEFT's own sequential merge (scale linear, not squared),
    ortho must remove the overlap, and domain-only must equal a plain PEFT merge. Raises AssertionError on failure."""
    import copy, tempfile
    from transformers import LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    tmp = Path(tempfile.mkdtemp())
    cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=4)
    torch.manual_seed(0)
    base = LlamaForCausalLM(cfg).eval()

    def make(name, seed):
        torch.manual_seed(seed)
        pm = get_peft_model(copy.deepcopy(base), LoraConfig(r=4, lora_alpha=8, target_modules=["o_proj", "down_proj"]))
        with torch.no_grad():
            for n, p in pm.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 0.2)
        pm.save_pretrained(tmp / name)
        return str(tmp / name)

    dom_dir, saf_dir = make("dom", 1), make("saf", 2)
    x = torch.randint(0, 100, (2, 9))

    def peft_seq(scale):
        m = PeftModel.from_pretrained(copy.deepcopy(base), dom_dir).merge_and_unload()
        if scale:
            sm = PeftModel.from_pretrained(m, saf_dir)
            with torch.no_grad():
                for n, p in sm.named_parameters():
                    if "lora_B" in n:
                        p.mul_(scale)        # scaling B alone scales dW linearly
            m = sm.merge_and_unload()
        return m.eval()

    with torch.no_grad():
        for alpha in (0.5, 1.0, 1.5):
            ref = peft_seq(alpha)(x).logits
            for st in (True, False):
                m = copy.deepcopy(base)
                be = build_repaired(m, dom_dir, saf_dir, alpha, False, st)
                ok = torch.allclose(ref, m.eval()(x).logits, atol=1e-5)
                print(f"alpha={alpha} backend={be:34s} matches PEFT sequential merge: {ok}")
                assert ok, "repair arithmetic does not match PEFT"
        m = copy.deepcopy(base)
        build_repaired(m, dom_dir)
        assert torch.allclose(peft_seq(0)(x).logits, m.eval()(x).logits, atol=1e-5), "domain-only merge mismatch"
        dom, saf = LoraDeltas(dom_dir), LoraDeltas(saf_dir)
        mx = 0.0
        for mod in saf.modules():
            o, d = ortho_delta(saf, dom, mod), dom.delta(mod)
            mx = max(mx, abs(float((o * d).sum() / (o.norm() * d.norm()))))
        print(f"max |cos(ortho safety update, domain update)| = {mx:.1e}")
        assert mx < 1e-4, "orthogonalisation failed"
    print("SELFTEST PASSED")
