"""Stage D — assemble iNat weekly flowering curves for the Step-4 plug-in.

Takes the iNat probe predictions and adds, per (cell, species), the weekly
flowering curve normalized to sum to 1 over the weeks that exist for that cell
-- the drop-in replacement for the observed plant timing histogram in Part-1
Step 4.

Flowering signal: inat_p_flowering (iNat 4-class probe, class 2 "flowering").
Output <OUT>/data/flowering_curves_<cells>.parquet.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUTDATA = Path(os.environ.get('PPE_OUT_DIR', HERE.parent / 'Part2_PhenoField_Outputs')) / 'data'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', choices=['used', 'all'], default='used')
    args = ap.parse_args()

    inat = pd.read_parquet(OUTDATA / f'predictions_inat_{args.cells}.parquet')
    # inat_p_flowering == p_flowering (class 2); keep the 4 class cols + flowering
    keep = ['cell_idx', 'centroid_lat', 'centroid_lon', 'species', 'week', 'doy',
            'inat_p_no_phenology', 'inat_p_budding', 'inat_p_flowering', 'inat_p_fruiting']
    m = inat[[c for c in keep if c in inat.columns]].copy()
    m = m.sort_values(['cell_idx', 'species', 'week']).reset_index(drop=True)

    # normalized weekly timing distribution (sum to 1 per cell x species)
    s = m.groupby(['cell_idx', 'species'])['inat_p_flowering'].transform('sum')
    m['inat_norm'] = np.where(s > 0, m['inat_p_flowering'] / s, np.nan)

    out = OUTDATA / f'flowering_curves_{args.cells}.parquet'
    m.to_parquet(out, index=False)
    print(f"saved {out}  ({len(m):,} rows, {m.cell_idx.nunique()} cells, "
          f"{m.species.nunique()} species)")


if __name__ == '__main__':
    main()
