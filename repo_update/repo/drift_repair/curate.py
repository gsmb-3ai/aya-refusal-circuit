"""CuratorKIT stage: clean + MinHash-deduplicate a raw instruction/output file, keep the lang/kind metadata.
Falls back to plain exact de-duplication if CuratorKIT fails (the fallback is recorded in the manifest)."""
import argparse, shutil
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, help="domain_train | safety_train")
args = ap.parse_args()

raw = read_jsonl(DATA / f"{args.name}_raw.jsonl")
norm = lambda t: re.sub(r"\s+", " ", t or "").strip().lower()
meta = {norm(r["instruction"]): {"lang": r["lang"], "kind": r["kind"]} for r in raw}
rows = [{"instruction": r["instruction"], "input": "", "output": r["output"], "lang": r["lang"], "kind": r["kind"]}
        for r in raw]
tmp_in = DATA / f"{args.name}_ck_in.jsonl"
write_jsonl(tmp_in, rows)
out_dir = DATA / "curated" / args.name
used, final, note = False, [], ""
try:
    from curatorkit import Curator, CuratorConfig
    shutil.rmtree(out_dir, ignore_errors=True)
    Curator(CuratorConfig(dataset=str(tmp_in), dedup="minhash", clean=True, export_formats=["alpaca"],
                          output_dir=str(out_dir))).run().print_summary()
    kept = []
    for f in sorted(out_dir.glob("*alpaca*.jsonl")):
        kept += read_jsonl(f)
    for r in kept:
        m = meta.get(norm(r.get("instruction")))
        if m and r.get("output"):
            final.append({"instruction": r["instruction"], "output": r["output"], **m})
    used = len(final) >= 0.7 * len(raw)
    if not used:
        note = f"CuratorKIT kept only {len(final)}/{len(raw)} (<70%): using the plain-dedup fallback instead"
        print("[warn]", note)
except Exception as e:
    note = f"CuratorKIT failed: {e!r}"
    print("[warn]", note)
if not used:
    seen, final = set(), []
    for r in raw:
        k = norm(r["instruction"])
        if k not in seen and r.get("output"):
            seen.add(k); final.append({"instruction": r["instruction"], "output": r["output"],
                                       "lang": r["lang"], "kind": r["kind"]})
write_jsonl(DATA / f"{args.name}.jsonl", final)
cnt = Counter(f"{r['lang']}/{r['kind']}" for r in final)
json.dump({"name": args.name, "used_curatorkit": used, "raw": len(raw), "final": len(final), "note": note,
           "by_lang_kind": dict(cnt)}, open(RES / f"curation_{args.name}.json", "w"), indent=1)
print(f"{args.name}: raw {len(raw)} -> final {len(final)} | CuratorKIT used: {used}")
print(dict(cnt))
