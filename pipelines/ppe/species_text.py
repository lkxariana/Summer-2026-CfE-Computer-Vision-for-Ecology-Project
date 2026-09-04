import argparse
import json
import torch


def main():
    """BioCLIP-2 text embeddings for plant taxa, as {names, embeddings}. Prompt = the raw
    binomial string — the exact format species_embeddings_v2.pt was built with (verified:
    cosine 1.0 against its rows), so new taxa land in the same text space e98 trained on."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True, help="JSON list of binomials, or modelled_universe.json (plants[])")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    spec = json.load(open(args.names))
    names = spec["plants"] if isinstance(spec, dict) else spec
    names = [n["label"] if isinstance(n, dict) else n for n in names]
    print(f"n={len(names)} | ex: {names[:3]}", flush=True)

    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
    tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")
    model = model.to(args.device).eval()
    emb = torch.zeros(len(names), 768)
    with torch.inference_mode():
        for s in range(0, len(names), 256):
            emb[s:s + 256] = model.encode_text(tok(names[s:s + 256]).to(args.device)).float().cpu()
    torch.save({"embeddings": emb, "names": names, "encoder": "bioclip2_v2_text"}, args.out)
    print(f"wrote {args.out} shape={tuple(emb.shape)}", flush=True)


if __name__ == "__main__":
    main()
