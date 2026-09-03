import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipelines.config import load_config, resolve

HIGHER = {o: ("Animalia", "Arthropoda", "Insecta") for o in
          ["Hymenoptera", "Lepidoptera", "Diptera", "Coleoptera", "Hemiptera", "Odonata",
           "Orthoptera", "Neuroptera", "Mantodea", "Thysanoptera"]}
HIGHER["Apodiformes"] = HIGHER["Passeriformes"] = ("Animalia", "Chordata", "Aves")


def linnaean(order, family, latin, common=""):
    k, p, c = HIGHER.get(order, ("Animalia", "", ""))
    ranks = [x for x in [k, p, c, order, family, latin] if isinstance(x, str) and x]
    s = " ".join(ranks)
    return f"{s} {common}" if common else (s or latin)


def occ_prompts(cfg):
    taxon_ids = np.load(cfg["paths"]["sdm_data"] / "pollinator_occ.npz")["taxon_ids"]
    meta = json.load(open(resolve(cfg, "geo_prior_meta")))
    id2name = {int(m["taxon_id"]): ((m.get("latin_name") or ""), (m.get("common_name") or "")) for m in meta}
    lock = pd.read_csv(resolve(cfg, "locked_species"))
    tid2globi = dict(zip(lock["train_taxon_id"].astype(int), lock["globi_name"]))
    g = pd.read_csv(resolve(cfg, "globi"),
                    usecols=["sourceTaxonName", "sourceTaxonFamilyName", "sourceTaxonOrderName"]).dropna().drop_duplicates("sourceTaxonName")
    name2of = {r.sourceTaxonName: (r.sourceTaxonOrderName, r.sourceTaxonFamilyName) for r in g.itertuples()}
    prompts = []
    for tid in taxon_ids:
        latin, common = id2name.get(int(tid), ("", ""))
        gname = tid2globi.get(int(tid), latin)
        order, family = name2of.get(gname, ("", ""))
        prompts.append(linnaean(order, family, latin or gname, common))
    return prompts, {"taxon_ids": torch.from_numpy(taxon_ids)}


def zeroshot_prompts(path):
    df = pd.read_csv(path)
    prompts = [linnaean(r.sourceTaxonOrderName, r.sourceTaxonFamilyName, r.sourceTaxonName) for r in df.itertuples()]
    return prompts, {"names": df["sourceTaxonName"].tolist(),
                     "order": df["sourceTaxonOrderName"].tolist(), "family": df["sourceTaxonFamilyName"].tolist()}


def main():
    """BioCLIP-2 text embeddings (the LE-SINR conditioner): --set occ = one row per
    pollinator_occ.npz species in sidx order; --set zeroshot = rows from a GloBI-named CSV."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["occ", "zeroshot"], required=True)
    ap.add_argument("--csv", default=None, help="zeroshot species CSV (default <sdm_data>/groupB_zeroshot_species.csv)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = load_config()
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    sd = cfg["paths"]["sdm_data"]
    if args.set == "occ":
        prompts, extra = occ_prompts(cfg)
        out = sd / "pollinator_species_text.pt"
    else:
        prompts, extra = zeroshot_prompts(args.csv or sd / "groupB_zeroshot_species.csv")
        out = sd / "zeroshot_text.pt"
    print(f"n={len(prompts)} | ex: {prompts[:3]}", flush=True)

    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
    tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")
    model = model.to(dev).eval()
    emb = torch.zeros(len(prompts), 768)
    with torch.inference_mode():
        for s in range(0, len(prompts), 256):
            emb[s:s + 256] = model.encode_text(tok(prompts[s:s + 256]).to(dev)).float().cpu()
    torch.save({"embeddings": emb, "names": prompts, "encoder": "bioclip2_text", **extra}, out)
    print(f"wrote {out} shape={tuple(emb.shape)}", flush=True)


if __name__ == "__main__":
    main()
