"""
Step 3: attach census-tract median household income to every scored rental.

Takes data/redfin_rentals_lihtc_scores.csv (output of siteline_scorer.py, which already carries each property's
11-digit census tract GEOID in `tract_geoid`) and joins it to a LOCAL census income table. Writes the final
dataset to data/atlanta_rentals_final.csv.

The income table lives in data/census_income/. To update it, just drop a new download into that folder
(see data/census_income/README.md for what to download). The script picks the most recent one automatically:
the newest ACS year found in the file name (e.g. ACSDT5Y2024...), then the newest file date. Files can be the
downloaded .zip as-is or the extracted ...-Data.csv. Other files in the folder are ignored.

Added columns:
    tract_name                     e.g. "Census Tract 16, Fulton County, Georgia"
    median_household_income        ACS table B19013, dollars (empty where the Census suppresses it)
    median_household_income_moe    margin of error
    income_top_coded               True where the Census only reports "250,000+" (the true median is higher)
    income_source_file             which file in data/census_income/ was used
    acs_year                       ACS 5-year vintage parsed from the file name (empty if not in the name)

Usage:
    python income.py
    python income.py --file data/census_income/ACSDT5Y2023.B19013-Data.csv    # force a specific file
"""

import argparse
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INCOME_DIR = DATA_DIR / "census_income"
INPUT_CSV = DATA_DIR / "redfin_rentals_lihtc_scores.csv"
OUTPUT_CSV = DATA_DIR / "atlanta_rentals_final.csv"

INCOME_VAR = "B19013_001E"             # median household income (estimate)
MOE_VAR = "B19013_001M"                # median household income (margin of error)
MISSING_TOKENS = {"", "-", "--", "(x)", "n", "null", "nan", "n/a", "**", "***", "*****"}
SKIP_NAME_PARTS = ("metadata", "notes", "readme")


# ---------------------------------------------------------------------------
# Reading the downloaded table
# ---------------------------------------------------------------------------
def clean_number(value):
    """
    Census cell -> (number, top_coded).
    '109,505' -> 109505 | '250,000+' -> 250000 (top coded) | '2,500-' -> 2500 | '-', '**', '(X)', '-666666666' -> NaN
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float("nan"), False
    text = str(value).strip().replace(",", "").replace("$", "")
    if text.lower() in MISSING_TOKENS:
        return float("nan"), False
    top_coded = text.endswith("+")
    try:
        number = float(text.rstrip("+-"))
    except ValueError:
        return float("nan"), False
    if number < 0:                                   # Census sentinel values such as -666666666
        return float("nan"), False
    return number, top_coded


def _find_column(columns, exact, keywords):
    """First column named `exact`, else the first whose header contains every keyword (case-insensitive)."""
    lowered = {c.strip().lower(): c for c in columns}
    if exact.lower() in lowered:
        return lowered[exact.lower()]
    for c in columns:
        text = c.lower()
        if all(k in text for k in keywords):
            return c
    return None


def parse_income_table(df):
    """Turn a raw census table (all strings) into tract_geoid / income columns, or None if it doesn't fit."""
    id_col = next((c for c in df.columns if c.strip().upper() in ("GEO_ID", "GEOID", "GEOID10", "GEOID20", "GEO_ID2")), None)
    est_col = _find_column(df.columns, INCOME_VAR, ("estimate", "median household income"))
    if id_col is None or est_col is None:
        return None
    moe_col = _find_column(df.columns, MOE_VAR, ("margin of error", "median household income"))
    name_col = next((c for c in df.columns if c.strip().upper() in ("NAME", "GEOGRAPHIC AREA NAME")), None)

    # '1400000US13121001600' or '13121001600' -> the trailing 11 digits; the label row ('Geography') and non-tract rows drop out
    geoid = df[id_col].astype(str).str.extract(r"(\d{11})$")[0]
    keep = geoid.notna()
    if not keep.any():
        return None

    est = df.loc[keep, est_col].map(clean_number)
    out = pd.DataFrame({
        "tract_geoid": geoid[keep],
        "tract_name": df.loc[keep, name_col] if name_col else pd.NA,
        "median_household_income": est.map(lambda t: t[0]),
        "income_top_coded": est.map(lambda t: t[1]),
        "median_household_income_moe": (df.loc[keep, moe_col].map(lambda v: clean_number(v)[0]) if moe_col else pd.NA),
    })
    return out.drop_duplicates(subset="tract_geoid").reset_index(drop=True)


def read_income_file(path):
    """Read one candidate file (.csv, or a .zip holding the table's -Data.csv). Returns a parsed table or None."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            members = [n for n in z.namelist()
                       if n.lower().endswith(".csv") and not any(p in n.lower() for p in SKIP_NAME_PARTS)]
            members.sort(key=lambda n: "-data" not in n.lower())          # the -Data.csv first
            for name in members:
                with z.open(name) as f:
                    table = parse_income_table(pd.read_csv(f, dtype=str, encoding="utf-8-sig"))
                if table is not None:
                    return table
        return None
    return parse_income_table(pd.read_csv(path, dtype=str, encoding="utf-8-sig"))


def acs_year_from_name(path):
    """ACS vintage from the file name, e.g. ACSDT5Y2024.B19013-Data.csv -> 2024 (None if there isn't one)."""
    match = re.search(r"ACS\w*?(20\d\d)", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def candidate_files():
    """Files in data/census_income/ ranked newest first: by ACS year in the name, then by modification time."""
    if not INCOME_DIR.is_dir():
        return []
    files = [p for p in INCOME_DIR.iterdir()
             if p.is_file() and p.suffix.lower() in (".csv", ".zip")
             and not any(part in p.name.lower() for part in SKIP_NAME_PARTS)]
    return sorted(files, key=lambda p: (acs_year_from_name(p) or 0, p.stat().st_mtime), reverse=True)


def load_latest_income(forced=None):
    """Return (income_table, source_path). Tries files newest first and uses the first one that reads as a tract income table."""
    files = [forced] if forced else candidate_files()
    if not files:
        raise FileNotFoundError(
            f"No census income files found in {INCOME_DIR}. Download the Georgia tract table B19013 from "
            f"data.census.gov and put it there (see {INCOME_DIR / 'README.md'})."
        )
    for path in files:
        try:
            table = read_income_file(path)
        except Exception as e:                                       # unreadable file: report and try the next one
            print(f"  Skipping {path.name}: {type(e).__name__}: {e}")
            continue
        if table is None:
            print(f"  Skipping {path.name}: no tract GEO_ID / B19013 estimate columns found.")
            continue
        others = [p.name for p in files if p != path]
        print(f"Using census income file: {path.name} ({len(table):,} tracts)")
        if others:
            print(f"  (also in the folder, not used: {', '.join(others)})")
        return table, path
    raise ValueError(f"None of the files in {INCOME_DIR} looked like a tract-level B19013 income table.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Add census-tract median household income to the scored rentals.")
    p.add_argument("--file", type=Path, default=None, help="use this income file instead of the newest in data/census_income")
    args = p.parse_args(argv)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"{INPUT_CSV} not found. Run siteline_scorer.py first.")
    scored = pd.read_csv(INPUT_CSV, dtype={"Postal Code": str, "tract_geoid": str})

    income, source = load_latest_income(args.file)
    income["income_source_file"] = source.name
    income["acs_year"] = acs_year_from_name(source)

    final = scored.merge(income, on="tract_geoid", how="left", validate="many_to_one")

    has_tract = final["tract_geoid"].notna()
    has_income = final["median_household_income"].notna()
    print("\n--- Final dataset summary ---")
    print(f"Properties: {len(final):,}")
    print(f"With a census tract: {int(has_tract.sum()):,} | with tract income: {int(has_income.sum()):,}")
    if (has_tract & ~has_income).any():
        print(f"Tract has no income value (suppressed, or tract not in the file): {int((has_tract & ~has_income).sum()):,} properties")
    if (~has_tract).any():
        print(f"No tract (outside SiteLine's 11-county region or no coordinates): {int((~has_tract).sum()):,}")
    if final["income_top_coded"].eq(True).any():
        print(f"Top-coded incomes (reported only as '250,000+'): {int(final['income_top_coded'].eq(True).sum()):,} properties")

    # A big gap between tracts in the scored data and tracts in the income file usually means the file is for the
    # wrong state, or uses older (pre-2020) tract boundaries, which are in ACS releases up to 2019.
    tracts_scored = set(final.loc[has_tract, "tract_geoid"])
    if tracts_scored:
        matched = len(tracts_scored & set(income["tract_geoid"]))
        if matched / len(tracts_scored) < 0.9:
            print(f"\nWARNING: only {matched} of {len(tracts_scored)} scored tracts were found in {source.name}. "
                  f"Check that the file covers all Georgia tracts and is from the 2020 ACS 5-year release or later.")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(final):,} rows to {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
