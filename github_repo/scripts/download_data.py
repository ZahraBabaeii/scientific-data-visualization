"""Step 0 - download the raw inputs into data/.

Every file comes from a public GitHub repository and is pinned to the commit used for the analysis,
so re-running the pipeline reproduces the same numbers. Set PIN = False to fetch the latest versions instead.

Usage:  python scripts/download_data.py
"""

import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PIN = True

# (repository, pinned commit, path in repository, local path under data/)
FILES = [
    ("owid/energy-data", "f8de88a4f6f9cab44f1cf606047cbd6c4414042c", "owid-energy-data.csv", "owid_energy/owid-energy-data.csv"),
    (
        "owid/energy-data",
        "f8de88a4f6f9cab44f1cf606047cbd6c4414042c",
        "owid-energy-codebook.csv",
        "owid_energy/owid-energy-codebook.csv",
    ),
    (
        "owid/covid-19-data",
        "dc00e7296b17ab5548092c9e388b6fac6751f095",
        "scripts/input/wb/income_groups.csv",
        "other/income_groups.csv",
    ),
    (
        "owid/covid-19-data",
        "dc00e7296b17ab5548092c9e388b6fac6751f095",
        "scripts/input/owid/continents.csv",
        "other/continents.csv",
    ),
    ("datasets/country-codes", "2d84034a0cae19ea5444b78c2131304a5c7f4bb9", "data/country-codes.csv", "other/country-codes.csv"),
    (
        "nvkelso/natural-earth-vector",
        "ca96624a56bd078437bca8184e78163e5039ad19",
        "geojson/ne_110m_admin_0_countries.geojson",
        "ne/ne_110m_admin_0_countries.geojson",
    ),
]


def fetch(url, target, tries=4):
    """Download with a short back-off; GitHub's raw host sometimes answers 429 (rate limit)."""
    for attempt in range(1, tries + 1):
        try:
            urllib.request.urlretrieve(url, target)
            return
        except Exception as err:  # network problems, rate limits, moved files
            if attempt == tries:
                sys.exit(f"could not download {url}: {err}\nRe-run later; files already present are skipped.")
            wait = 15 * attempt
            print(f"  {err}; retrying in {wait} s")
            time.sleep(wait)


def main():
    for repo, commit, path, local in FILES:
        ref = commit if PIN else "HEAD"
        url = f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"
        target = DATA / local
        if target.exists() and target.stat().st_size > 0:
            print(f"present, skipped: {target.relative_to(ROOT)}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        fetch(url, target)
        print(f"  -> {target.relative_to(ROOT)} ({target.stat().st_size / 1e6:.1f} MB)")
    print("done")


if __name__ == "__main__":
    main()
