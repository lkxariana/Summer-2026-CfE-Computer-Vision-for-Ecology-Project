"""
05_geocode_hotspots.py -- Step 7, Part 5

Reverse-geocodes each of the top 100 hotspot bins to a US state, same
approach as Step 4 (geopy.Nominatim with a 1 req/sec rate limiter).

Note: an early attempt to install geopy with `pip install geopy --user
--break-system-packages` failed with "no such option" -- the pip version on
crow predates PEP 668 enforcement and does not recognize that flag. Plain
`pip install geopy --user` worked without issue.

Result: all 100 bins resolved to a named state -- no "Unknown (no match)"
entries, even for bins whose center falls just offshore near a coastline
(e.g. San Diego). Nominatim apparently snaps near-coastal points to the
nearest land-based administrative boundary rather than failing, so the
coastal-bin-center artifact noted in Step 4's README (bins falling on
coastal water get manually labeled with the adjacent state) did not require
manual intervention here.

State distribution (n=100): Virginia 25, Massachusetts 24, Vermont 19,
California 11, Minnesota 7, Maryland 4, Illinois 2, Texas 2, North Carolina
2, Florida 2, Washington 1, Indiana 1. The Northeast/Mid-Atlantic cluster
(VA+MA+VT = 68) and the California/desert cluster (11) account for the
large majority of top pairs -- driven respectively by Northeast woodland
spring-ephemeral plants (Aquilegia canadensis, Sanguinaria canadensis,
Caltha palustris, Lysimachia borealis) and Southern California chaparral/
desert specialists (Malosma laurina, Larrea tridentata).
"""

import time
import pandas as pd
from pathlib import Path

BIN_SIZE = 0.5
DATA_DIR = Path("/scratch/ariana.l")


def geocode_hotspots(jaccard_results, bin_size=BIN_SIZE):
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    geolocator = Nominatim(user_agent="cfe2026_step7_jaccard")
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

    states = []
    for _, row in jaccard_results.iterrows():
        # Bin center (+ half bin size), consistent with hotspot semantics elsewhere in this project
        lat_c = row["hotspot_lat"] + bin_size / 2
        lon_c = row["hotspot_lon"] + bin_size / 2
        try:
            loc = reverse((lat_c, lon_c), exactly_one=True, language="en")
            if loc:
                state = loc.raw.get("address", {}).get("state", "Unknown")
            else:
                state = "Unknown (no match - likely water)"
        except Exception:
            state = "Unknown (geocode error)"
        states.append(state)

    jaccard_results["state"] = states
    print(jaccard_results["state"].value_counts())
    return jaccard_results


if __name__ == "__main__":
    jaccard_results = pd.read_csv(DATA_DIR / "top100_jaccard_pairs.csv")
    jaccard_results = geocode_hotspots(jaccard_results)
    jaccard_results.to_csv(DATA_DIR / "top100_jaccard_pairs.csv", index=False)
    print("Saved with state column added.")
