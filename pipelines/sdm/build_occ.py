import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipelines.config import load_config, resolve

LAT = (24.0, 50.0)
LON = (-125.0, -66.0)


def main():
    """Pollinator occurrence npz. --source inat: geo-prior extract filtered to the locked taxa,
    sidx over the full sorted locked list (zero-obs taxa keep a slot; taxon_ids key).
    --source gbif: the name-keyed GBIF extract (~25k species; names key instead of taxon_ids)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["inat", "gbif"], default="inat")
    ap.add_argument("--species-csv", default=None, help="override locked list; needs a train_taxon_id column")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config()

    if args.source == "gbif":
        df = pd.read_csv(resolve(cfg, "gbif_obs"))
        df = df[df.lat.between(*LAT) & df.lon.between(*LON) & df.doy.between(1, 366)]
        names = np.sort(df.pollinator_species.unique()).astype(str)
        sidx = df.pollinator_species.map({n: i for i, n in enumerate(names)}).values
        out = Path(args.out) if args.out else cfg["paths"]["sdm_data"] / "pollinator_occ_gbif.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, lat=df.lat.values.astype(np.float32), lon=df.lon.values.astype(np.float32),
                 doy=df.doy.values.astype(np.int16), year=df.year.values.astype(np.int16),
                 sidx=sidx.astype(np.int32), names=names)
        print(f"[done] {out}: {len(df):,} obs, {len(names)} species", flush=True)
        return

    lock = pd.read_csv(args.species_csv or resolve(cfg, "locked_species"))
    tids = set(lock["train_taxon_id"].astype(int))
    print(f"[species] {len(tids)} locked taxon ids", flush=True)

    chunks = []
    for i, ch in enumerate(pd.read_csv(resolve(cfg, "geo_prior"), chunksize=2_000_000,
                                       usecols=["latitude", "longitude", "taxon_id", "observed_on"])):
        m = (ch.taxon_id.isin(tids) & ch.latitude.between(*LAT) & ch.longitude.between(*LON))
        sub = ch[m]
        if len(sub):
            dt = pd.to_datetime(sub.observed_on, errors="coerce")
            sub = sub.assign(doy=dt.dt.dayofyear, year=dt.dt.year).dropna(subset=["doy"])
            chunks.append(sub[["latitude", "longitude", "taxon_id", "doy", "year"]])
        print(f"  chunk {i}: kept {sum(len(c) for c in chunks):,}", flush=True)
    df = pd.concat(chunks)
    taxon_ids = np.sort(np.fromiter(tids, np.int32))
    tid2sidx = {int(t): i for i, t in enumerate(taxon_ids)}
    out = Path(args.out) if args.out else cfg["paths"]["sdm_data"] / "pollinator_occ.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, lat=df.latitude.values.astype(np.float32), lon=df.longitude.values.astype(np.float32),
             doy=df.doy.values.astype(np.int16), year=df.year.values.astype(np.int16),
             sidx=df.taxon_id.map(tid2sidx).values.astype(np.int32), taxon_ids=taxon_ids)
    print(f"[done] {out}: {len(df):,} obs, {len(taxon_ids)} species", flush=True)


if __name__ == "__main__":
    main()
