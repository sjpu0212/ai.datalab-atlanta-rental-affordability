"""
Step 2: score every scraped rental with the SiteLine LIHTC location scorer.

Reads the newest scrape in data/redfin_rentals/, scores each property's latitude/longitude with SiteLine
(https://github.com/kmcalist682336/SitelineLIHTCStartUp, out of 46 points under Georgia's 2024-2025 QAP), and
writes the result to data/redfin_rentals_lihtc_scores.csv (overwritten each run).

SiteLine is NOT part of this repository. Install it once, next to this project:
    git clone https://github.com/kmcalist682336/SitelineLIHTCStartUp.git ../SitelineLIHTCStartUp
    cd ../SitelineLIHTCStartUp && pip install -e ".[geo]"
If you cloned it somewhere else, set the SITELINE_REPO environment variable to that folder.

Usage:
    python siteline_scorer.py
    python siteline_scorer.py --input data/redfin_rentals/redfin_atlanta_rentals_2026-11-05_1830.csv
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RENTALS_DIR = DATA_DIR / "redfin_rentals"
VERSION_PREFIX = "redfin_atlanta_rentals"
OUTPUT_CSV = DATA_DIR / "redfin_rentals_lihtc_scores.csv"

SITELINE_REPO = Path(os.environ.get("SITELINE_REPO", ROOT.parent / "SitelineLIHTCStartUp")).expanduser()

LAT_COL = "latitude"
LON_COL = "longitude"
TENANCY = "family"                    # only affects section XIII (schools): "family" or e.g. "elderly"
KEEP_ALL_DETAIL_COLUMNS = False       # True = also keep the ~60 "nearest amenity / school / award" columns

# Columns kept from SiteLine's output by default (scores, context and the census tract join key)
CORE_COLUMNS = [
    "tract_geoid", "county", "pool", "municipality", "in_qct", "usda_food_desert", "in_scope",
    "score_XI", "score_XII", "score_XIII", "score_XV", "score_XVI", "score_XVIII", "score_XIX", "score_XXII",
    "block_winner", "block_score", "total_score", "n_flags",
]


# ---------------------------------------------------------------------------
# Import SiteLine
# ---------------------------------------------------------------------------
def import_siteline():
    try:
        import siteline
        return siteline
    except ModuleNotFoundError as e:
        if e.name != "siteline":
            # siteline WAS found, but one of ITS dependencies is missing.
            raise ModuleNotFoundError(
                f"siteline was found, but it needs the package '{e.name}', which is not installed in this "
                f"Python environment. Run:\n    pip install shapely scipy numpy pandas geopandas matplotlib",
                name=e.name,
            ) from e
        if SITELINE_REPO.is_dir():
            sys.path.insert(0, str(SITELINE_REPO))
            import siteline
            return siteline
        raise ModuleNotFoundError(
            "Could not find the 'siteline' package. Clone "
            "https://github.com/kmcalist682336/SitelineLIHTCStartUp and run `pip install -e \".[geo]\"` inside "
            f"it, or set SITELINE_REPO to the cloned folder (currently '{SITELINE_REPO}').",
            name="siteline",
        ) from e


# ---------------------------------------------------------------------------
# Load the newest scrape
# ---------------------------------------------------------------------------
def latest_rentals_path():
    files = sorted(RENTALS_DIR.glob(f"{VERSION_PREFIX}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No scraped datasets found in {RENTALS_DIR}. Run redfin_scraper.py first.")
    return files[-1]


def load_rentals(path):
    df = pd.read_csv(path, dtype={"Postal Code": str})
    before = len(df)
    df = df.dropna(subset=["Address"]).reset_index(drop=True)      # drops site-level rows with no street address
    if len(df) < before:
        print(f"Dropped {before - len(df)} non-listing rows (no street address).")
    print(f"Loaded {path.name} ({len(df)} listings)")
    return df


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_rentals(df, siteline):
    """
    Return df with LIHTC columns added: tract_geoid, county, pool, one score per QAP section
    (score_XI ... score_XXII), block_winner, block_score and total_score (out of 46).
    Rows with missing coordinates, or outside SiteLine's 11-county region, get empty scores.
    """
    df = df.copy().reset_index(drop=True)

    lat = pd.to_numeric(df[LAT_COL], errors="coerce")
    lon = pd.to_numeric(df[LON_COL], errors="coerce")
    valid = lat.notna() & lon.notna()
    if (~valid).sum():
        print(f"Skipping {int((~valid).sum())} rows with missing latitude/longitude.")

    to_score = df.loc[valid].copy()
    to_score[LAT_COL] = lat[valid]
    to_score[LON_COL] = lon[valid]
    to_score["_row_id"] = to_score.index          # lets us put results back on the right rows

    print(f"Scoring {len(to_score):,} properties...")
    kwargs = dict(lat_col=LAT_COL, lon_col=LON_COL, tenancy=TENANCY)
    try:
        scored = siteline.score_sites(to_score, **kwargs)
    except Exception as e:
        # One bad row shouldn't lose the whole run: retry one row at a time and skip the failures.
        print(f"Batch scoring failed ({type(e).__name__}: {e}). Retrying row by row...")
        parts, failed = [], []
        for i in to_score.index:
            try:
                parts.append(siteline.score_sites(to_score.loc[[i]], progress=False, **kwargs))
            except Exception as row_error:
                failed.append((df.loc[i, "Address"], str(row_error)))
        if failed:
            print(f"  {len(failed)} rows could not be scored, for example: {failed[:3]}")
        if not parts:
            raise RuntimeError("No rows could be scored.") from e
        scored = pd.concat(parts, ignore_index=True)

    scored = scored.set_index("_row_id")

    # score_sites() also returns its own lat/lon columns and repeats the input columns; keep only what is new
    if KEEP_ALL_DETAIL_COLUMNS:
        new_cols = [c for c in scored.columns if c not in df.columns and c not in ("lat", "lon")]
    else:
        new_cols = [c for c in CORE_COLUMNS if c in scored.columns]

    result = df.join(scored[new_cols])
    for c in CORE_COLUMNS:                         # make sure the columns exist even if nothing was in scope
        if c not in result.columns:
            result[c] = pd.NA
    return result


def print_summary(df_scored):
    in_scope = df_scored["in_scope"].eq(True)
    print("\n--- LIHTC scoring summary ---")
    print(f"Properties: {len(df_scored):,} | scored: {int(in_scope.sum()):,} | "
          f"outside the 11-county region or no coordinates: {int((~in_scope).sum()):,}")
    if in_scope.any():
        print(pd.to_numeric(df_scored.loc[in_scope, "total_score"]).describe().round(2).to_string())
        print(f"\nDistinct census tracts: {df_scored.loc[in_scope, 'tract_geoid'].nunique():,}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Score scraped Redfin rentals with the SiteLine LIHTC scorer.")
    p.add_argument("--input", type=Path, default=None, help="scraped CSV to score (default: newest in data/redfin_rentals)")
    args = p.parse_args(argv)

    siteline = import_siteline()
    df = load_rentals(args.input or latest_rentals_path())
    df_scored = score_rentals(df, siteline)
    print_summary(df_scored)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_scored.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df_scored):,} rows to {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
