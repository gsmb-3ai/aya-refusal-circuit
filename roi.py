"""ROI: self-hosted small model vs closed API. All inputs are YOUR measured/looked-up numbers.
  python roi.py --requests 1000000 --in_tok 300 --out_tok 50 --tps 900 --gpu_usd_hr 0.60 \
                --api_in_usd_per_m 0.15 --api_out_usd_per_m 0.60
tps = measured aggregate tokens/sec (see measure_tps.py). Look up prices on the day you write it up and cite them.
"""
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--requests", type=float, required=True)
ap.add_argument("--in_tok", type=float, required=True)
ap.add_argument("--out_tok", type=float, required=True)
ap.add_argument("--tps", type=float, required=True, help="measured generated tokens/sec on your GPU (batched)")
ap.add_argument("--prefill_mult", type=float, default=8.0, help="prefill is ~N x faster than decode per token")
ap.add_argument("--gpu_usd_hr", type=float, required=True)
ap.add_argument("--util", type=float, default=0.5, help="fraction of GPU time actually busy")
ap.add_argument("--api_in_usd_per_m", type=float, required=True)
ap.add_argument("--api_out_usd_per_m", type=float, required=True)
a = ap.parse_args()
sec = a.requests * (a.out_tok / a.tps + a.in_tok / (a.tps * a.prefill_mult))
hrs = sec / 3600 / a.util
self_cost = hrs * a.gpu_usd_hr
api_cost = a.requests * (a.in_tok * a.api_in_usd_per_m + a.out_tok * a.api_out_usd_per_m) / 1e6
print(f"GPU hours (at {a.util:.0%} util): {hrs:,.1f} -> self-hosted ${self_cost:,.2f}")
print(f"API: ${api_cost:,.2f}   | ratio API/self: {api_cost / max(self_cost, 1e-9):.2f}x")
print("State assumptions (utilization, batch size, engineering time, quality gap) in your write-up.")
