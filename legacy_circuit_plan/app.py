import json, gradio as gr, pandas as pd

S = json.load(open("summary.json"))
G = json.load(open("gallery.json"))
arms = list(S)
ids = list(G["harmful"])


def explore(pid):
    d = G["harmful"][pid]
    langs = list(next(iter(d["refused"].values())))
    df = pd.DataFrame({a: ["refused" if d["refused"][a].get(l) else "NOT refused" for l in langs] for a in arms}, index=langs)
    return d["text_en"], df.reset_index().rename(columns={"index": "language"})


def benign(arm):
    return pd.DataFrame([r for r in G["benign"] if r["arm"] == arm])[["lang", "prompt", "response"]]


with gr.Blocks(title="Multilingual safety gap") as demo:
    gr.Markdown("# Where does refusal live, and can we fix a multilingual safety gap by touching only that circuit?\n"
                "Cached results from our evaluation (research demo, not a safety guarantee). "
                "Harmful completions are never shown; only whether the model refused.")
    with gr.Tab("Gap explorer"):
        dd = gr.Dropdown(ids, value=ids[0], label="Benchmark request id")
        req = gr.Textbox(label="Request (English, from a public benchmark)", interactive=False)
        tbl = gr.Dataframe(label="Same request in 8 language variants: refused?")
        dd.change(explore, dd, [req, tbl]); demo.load(explore, dd, [req, tbl])
    with gr.Tab("Helpfulness check"):
        a = gr.Dropdown(arms, value=arms[0], label="Model arm")
        t = gr.Dataframe(); a.change(benign, a, t); demo.load(benign, a, t)
    with gr.Tab("Results"):
        gr.Markdown(open("summary.md").read())
        for f in ["refusal_by_lang.png", "fig_direction_cosine.png", "fig_separation.png", "fig_layer_contrib.png"]:
            try: gr.Image(f, label=f)
            except Exception: pass
demo.launch()
