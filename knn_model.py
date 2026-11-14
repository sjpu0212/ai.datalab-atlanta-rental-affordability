"""
Step 4: predict a rental's monthly price with K-Nearest Neighbors, and measure how well that works.

HOW THE MODEL WORKS
    For a rental, find the K most similar listings in data/atlanta_rentals_final.csv and combine their rents.
    "Similar" means close in all of these (each is a weighted distance component; see RentKNN):
        1. location        latitude and longitude, converted to kilometres
        2. SiteLine score  every section score: XI, XII, XIII, XV, XVI, XVIII, XIX, XXII (each divided by its
                           maximum points, so every section runs 0-1)
        3. postal code     same ZIP = closer; different ZIP = farther
        4. unit size       bedrooms and floor area (square feet on a log scale; see BEDS_SCALE and
                           SQFT_LOG_SCALE for how much one bedroom or 50% more floor area counts)
        5. property type   apartment, home, ... (from the listing URL)
    How much each group matters (its weight; 0 = ignored) and K are chosen by cross-validation.

BEDS AND SQUARE FEET
    Redfin gives a range for buildings with several unit types (for example 1-2 beds, 1,150-1,350 sq ft) and its rent
    is the STARTING price, i.e. the cheapest unit. `--unit-basis min` (default) therefore uses the smallest unit of
    each listing, which is the one the rent belongs to; `max` and `mid` (the middle of the range) are there so you can
    check that choice with cross-validation. Missing beds/sqft are filled with typical values (square feet from the
    typical size of that many bedrooms), so listings without them still work.

HOW THE NEIGHBORS' RENTS ARE COMBINED (the `aggregate` setting, also chosen by cross-validation)
    mean      plain average of the K rents
    median    the middle rent; one extreme neighbor cannot move it
    trimmed   average after dropping the highest and lowest 20% of the neighbors
    filtered  drop neighbors whose rent is unusually far from the neighbors' median (judged in percent terms, using
              the median absolute deviation; `outlier_z` is how many robust standard deviations counts as "unusual"),
              then average the rest. This is the "remove neighbors priced way above (or below) the rest" idea.

WHICH RENTS ARE USED
    Listings with rents far outside the market are removed first. By default the rent range comes from the data:
    Tukey's IQR fences on log rent (--iqr-k, default 1.5). Or fix it: --min-rent 600 --max-rent 4500.

WHY MEDIAN INCOME IS NOT A PREDICTOR
    Tract income is already reflected in where a property is (latitude/longitude/ZIP), and income is what we later
    compare predicted rent against to decide affordability, so using it to predict rent would be circular.
    (SiteLine's total_score and block_score are left out too: they are sums of the section scores.)
    Affordability, in `predict` mode: annual rent / tract median household income above THRESHOLD (default 30%)
    means the rent is not affordable for the typical household.

HOW THE MODEL IS EVALUATED (python knn_model.py train)
    Nested, grouped cross-validation, so every number is on properties the model has not seen:
      * OUTER folds = testing sets. Each fold's properties are held out completely.
      * INNER folds = validation sets, made from the remaining training data only, used to choose K, the weights
        and the averaging method. The testing fold is never used for that choice.
      * Listings at the same building location are kept in the same fold (--group-by building). Otherwise a
        unit's twin in the same building would sit in the training data and make the model look better than it
        is. --group-by tract is stricter still (whole neighbourhoods held out).
    Models are scored on the same folds, so you can see what each ingredient adds:
        mean_baseline         always predicts the average rent
        location_only         latitude/longitude, plain average
        location_zip          + postal code, plain average
        location_zip_robust   same, but the averaging method is chosen among median / trimmed / filtered
        full                  + SiteLine section scores, plain average
        full_robust           same, but the averaging method is chosen among median / trimmed / filtered
        units_robust          location + postal code + bedrooms/size/type (no SiteLine scores), robust averaging
        all_robust            everything: location, postal code, SiteLine scores, bedrooms/size/type
    (The two unit models are skipped when the data has no beds/sqft yet; rescrape with the current scraper.)
    The comparisons that matter: do unit features help (units_robust vs location_zip_robust), and do the
    LIHTC scores still add anything once unit features are in (all_robust vs units_robust)?

USAGE
    python knn_model.py train                        # cross-validate, then tune and save final settings
    python knn_model.py train --no-cv                # skip the evaluation, only (re)tune the final settings
    python knn_model.py train --unit-basis mid       # use the middle of beds/sqft ranges instead of the smallest unit
    python knn_model.py predict --property "Castlegate Collier Hills"     # a name or address in the dataset
    python knn_model.py predict --lat 33.7660 --lon -84.3491 --zip 30307 --beds 2 --sqft 950 --type apartment

    WHAT `predict` USES FOR THE UNIT
      * A property found in the dataset: its own beds, square feet and type (the --unit-basis end of any range).
      * Otherwise (coordinates or a new address): only what you give with --beds, --sqft, --type. Anything you leave
        out is filled with a typical value (square feet: the typical size for that many bedrooms), and `predict`
        prints exactly what it assumed. --beds / --sqft / --type always override the dataset values.
      * If the tuned model gave unit features zero weight, they are ignored and `predict` says so.
    `predict` needs SiteLine installed (see siteline_scorer.py) to score the new location. Other addresses are
    geocoded (U.S. Census geocoder, then OpenStreetMap Nominatim), which needs internet. Latitude/longitude always
    works and is the most reliable.

OUTPUT (data/model/)
    cv_metrics.csv           per-fold test scores for every model, with the settings chosen in each fold
    cv_predictions.csv       each property's out-of-fold predicted rent, from every model
    cv_actual_vs_predicted.png
    best_params.json         the K, weights, averaging method, rent range and unit basis used by `predict`
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FINAL_CSV = DATA_DIR / "atlanta_rentals_final.csv"
MODEL_DIR = DATA_DIR / "model"
PARAMS_JSON = MODEL_DIR / "best_params.json"

TARGET = "Rental Price"
LAT_COL, LON_COL, ZIP_COL = "latitude", "longitude", "Postal Code"

# Every SiteLine section score, with its maximum points (used to put each on a 0-1 scale).
# score_XIX only applies outside the 4 core metro counties; where it is empty it is treated as 0.
SCORE_MAX = {"score_XI": 20, "score_XII": 6, "score_XIII": 3, "score_XV": 10,
             "score_XVI": 10, "score_XVIII": 5, "score_XIX": 10, "score_XXII": 2}
SCORE_COLUMNS = list(SCORE_MAX)
BEDS_COL, SQFT_COL, TYPE_COL = "beds", "sqft", "property_type"     # built by load_training_data from the *_min / *_max columns
UNIT_COLUMNS = [BEDS_COL, SQFT_COL, TYPE_COL]
FEATURE_COLUMNS = [LAT_COL, LON_COL, ZIP_COL, *SCORE_COLUMNS, *UNIT_COLUMNS]
UNIT_BASES = ["min", "max", "mid"]        # which end of a beds / sqft range to use (--unit-basis)
BEDS_SCALE = 1.0                          # a difference of 1 bedroom counts as 1.0 in the neighbour search
SQFT_LOG_SCALE = 0.4                      # a 50% difference in floor area (log 1.5 = 0.405) counts as 1.0
MIN_UNIT_COVERAGE = 0.5                   # unit features are used only if at least this share of listings has beds and sqft

MIN_RENT, MAX_RENT = 300, 10_000          # hard sanity limits: listings outside this range are always treated as data errors
IQR_K = 1.5                               # --min-rent / --max-rent "auto": Tukey fences at Q1 - k*IQR and Q3 + k*IQR, on log rent
KM_PER_DEG_LAT = 111.2
LOC_SCALE_KM = 5.0                        # 5 km of distance counts as 1.0 in the neighbour search
AFFORDABILITY_THRESHOLD = 0.30            # rent above 30% of household income = not affordable

# How neighbour rents are combined
TRIM_FRACTION = 0.2                       # "trimmed": drop this share of neighbours from each end (rounded down)
MIN_LOG_SPREAD = 0.05                     # "filtered": never treat spreads below ~5% as meaningful (avoids dropping everything when rents tie)
OUTLIER_Z_GRID = [1.5, 2.0, 3.0]          # "filtered": how many robust standard deviations from the median counts as an outlier
AGGREGATES = ["mean", "median", "trimmed", "filtered"]

# Settings searched by cross-validation
K_GRID = [3, 5, 8, 12, 20, 30]            # number of neighbours
W_SCORE_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]  # weight of the SiteLine scores (0 = ignored)
W_ZIP_GRID = [0.0, 0.25, 0.5, 1.0]        # weight of the postal code (0 = ignored)
W_UNIT_GRID = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]   # weight of bedrooms + floor area (0 = ignored)
W_TYPE_GRID = [0.0, 1.0]                  # weight of the property type (0 = ignored)
OUTER_FOLDS, INNER_FOLDS = 5, 3
SEED = 42

# Averaging options each model may choose from: (aggregate, outlier_z)
PLAIN_OPTIONS = [("mean", None)]
ROBUST_OPTIONS = [("median", None), ("trimmed", None), *[("filtered", z) for z in OUTLIER_Z_GRID]]
ALL_OPTIONS = PLAIN_OPTIONS + ROBUST_OPTIONS

MODEL_SPECS = {
    "location_only": {"w_score": [0.0], "w_zip": [0.0], "options": PLAIN_OPTIONS},
    "location_zip": {"w_score": [0.0], "w_zip": W_ZIP_GRID, "options": PLAIN_OPTIONS},
    "location_zip_robust": {"w_score": [0.0], "w_zip": W_ZIP_GRID, "options": ROBUST_OPTIONS},
    "full": {"w_score": W_SCORE_GRID, "w_zip": W_ZIP_GRID, "options": PLAIN_OPTIONS},
    "full_robust": {"w_score": W_SCORE_GRID, "w_zip": W_ZIP_GRID, "options": ROBUST_OPTIONS},
    "units_robust": {"w_score": [0.0], "w_zip": W_ZIP_GRID, "w_unit": W_UNIT_GRID, "w_type": W_TYPE_GRID,
                     "options": ROBUST_OPTIONS},
    "all_robust": {"w_score": W_SCORE_GRID, "w_zip": W_ZIP_GRID, "w_unit": W_UNIT_GRID, "w_type": W_TYPE_GRID,
                   "options": ROBUST_OPTIONS},
}
UNIT_MODELS = {"units_robust", "all_robust"}          # models that need beds / sqft in the data
MODEL_ORDER = ["mean_baseline", "location_only", "location_zip", "location_zip_robust", "full", "full_robust",
               "units_robust", "all_robust"]


# ---------------------------------------------------------------------------
# Combining the neighbours' rents
# ---------------------------------------------------------------------------
def aggregate_neighbors(Y, mode="mean", outlier_z=2.0):
    """
    Combine each row's neighbour rents into one prediction.
    Y: (n_rows, k) rents of the k nearest listings. Returns (prediction (n_rows,), used (n_rows, k) boolean).
    `used` marks the neighbours that contributed to a mean; for "median" every neighbour counts as used.
    """
    used = np.ones(Y.shape, dtype=bool)
    if mode == "mean":
        return Y.mean(axis=1), used
    if mode == "median":
        return np.median(Y, axis=1), used
    if mode == "trimmed":
        k = Y.shape[1]
        cut = int(np.floor(k * TRIM_FRACTION))
        if cut > 0:
            order = np.argsort(Y, axis=1, kind="stable")
            rows = np.arange(Y.shape[0])[:, None]
            used[rows, order[:, :cut]] = False
            used[rows, order[:, k - cut:]] = False
    elif mode == "filtered":
        # Work in log-rent so "unusual" means unusual in percent terms ($4,000 among $1,500 rents is far; $1,700 is not).
        logy = np.log(np.maximum(Y, 1.0))
        center = np.median(logy, axis=1, keepdims=True)
        spread = np.maximum(1.4826 * np.median(np.abs(logy - center), axis=1, keepdims=True), MIN_LOG_SPREAD)
        used = np.abs(logy - center) <= outlier_z * spread          # the median neighbour always survives
    else:
        raise ValueError(f"unknown aggregate '{mode}' (use one of {AGGREGATES})")
    return (Y * used).sum(axis=1) / used.sum(axis=1), used


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class RentKNN(RegressorMixin, BaseEstimator):
    """
    K-nearest-neighbours rent estimator on [location | SiteLine scores | postal code | unit size | property type].

    Distance = Euclidean over a feature vector built from:
        location     (east-west km, north-south km) / loc_scale_km, times w_loc
        scores       each section score / its max points, times w_score
        postal code  one-hot (1 for the property's ZIP), times w_zip
        unit size    bedrooms / BEDS_SCALE and log(square feet) / SQFT_LOG_SCALE, times w_unit
        type         one-hot property type (apartment, home, ...), times w_type
    so a weight of 0 switches that group off, and larger weights make that group matter more.
    Missing beds are filled with the training median; missing square feet with the training median for that many
    bedrooms (overall median if that is unknown); missing type counts as its own "unknown" type.
    The neighbours' rents are then combined as `aggregate` says (mean, median, trimmed or filtered).
    """

    def __init__(self, n_neighbors=8, w_score=0.5, w_zip=0.5, w_loc=1.0, loc_scale_km=LOC_SCALE_KM,
                 aggregate="mean", outlier_z=2.0, w_unit=0.0, w_type=0.0):
        self.n_neighbors = n_neighbors
        self.w_score = w_score
        self.w_zip = w_zip
        self.w_loc = w_loc
        self.loc_scale_km = loc_scale_km
        self.aggregate = aggregate
        self.outlier_z = outlier_z
        self.w_unit = w_unit
        self.w_type = w_type

    def unit_values(self, X):
        """(bedrooms, square feet, type) of each row after filling in missing values; this is what the model sees."""
        beds = pd.to_numeric(X[BEDS_COL], errors="coerce")
        sqft = pd.to_numeric(X[SQFT_COL], errors="coerce")
        typical_sqft = beds.round().map(self.sqft_by_beds_)             # NaN if beds is missing or never seen
        beds = beds.fillna(self.beds_med_)
        sqft = sqft.fillna(typical_sqft).fillna(self.sqft_med_).clip(lower=1.0)
        types = X[TYPE_COL].where(X[TYPE_COL].notna(), "unknown").astype(str).str.lower()
        return beds.to_numpy(float), sqft.to_numpy(float), types

    def _features(self, X):
        lat = X[LAT_COL].to_numpy(float)
        lon = X[LON_COL].to_numpy(float)
        north = (lat - self.lat0_) * KM_PER_DEG_LAT
        east = (lon - self.lon0_) * KM_PER_DEG_LAT * np.cos(np.radians(self.lat0_))
        loc = np.column_stack([east, north]) / self.loc_scale_km * self.w_loc

        scores = np.column_stack([pd.to_numeric(X[c], errors="coerce").fillna(0).to_numpy(float) / SCORE_MAX[c]
                                  for c in SCORE_COLUMNS]) * self.w_score

        zips = np.zeros((len(X), len(self.zips_)))
        idx = X[ZIP_COL].map(self.zip_index_)                 # NaN for an unknown or missing ZIP -> all zeros
        ok = idx.notna().to_numpy()
        zips[np.flatnonzero(ok), idx[ok].astype(int).to_numpy()] = 1.0

        beds, sqft, types = self.unit_values(X)
        unit = np.column_stack([beds / BEDS_SCALE, np.log(sqft) / SQFT_LOG_SCALE]) * self.w_unit
        type_hot = np.zeros((len(X), len(self.types_)))
        tidx = types.map(self.type_index_)                    # NaN for a type never seen in training -> all zeros
        tok = tidx.notna().to_numpy()
        type_hot[np.flatnonzero(tok), tidx[tok].astype(int).to_numpy()] = 1.0
        return np.hstack([loc, scores, zips * self.w_zip, unit, type_hot * self.w_type])

    def fit(self, X, y):
        self.lat0_ = float(X[LAT_COL].mean())
        self.lon0_ = float(X[LON_COL].mean())
        self.zips_ = sorted(X[ZIP_COL].dropna().unique())
        self.zip_index_ = {z: i for i, z in enumerate(self.zips_)}

        beds = pd.to_numeric(X[BEDS_COL], errors="coerce")
        sqft = pd.to_numeric(X[SQFT_COL], errors="coerce")
        self.beds_med_ = float(beds.median()) if beds.notna().any() else 1.0
        self.sqft_med_ = float(sqft.median()) if sqft.notna().any() else 800.0
        both = pd.DataFrame({"beds": beds.round(), "sqft": sqft}).dropna()
        self.sqft_by_beds_ = both.groupby("beds")["sqft"].median().to_dict()
        types = X[TYPE_COL].where(X[TYPE_COL].notna(), "unknown").astype(str).str.lower()
        self.types_ = sorted(types.unique())
        self.type_index_ = {t: i for i, t in enumerate(self.types_)}

        self.y_ = np.asarray(y, dtype=float)
        self.nn_ = NearestNeighbors(n_neighbors=min(int(self.n_neighbors), len(self.y_)))
        self.nn_.fit(self._features(X))
        return self

    def neighbors(self, X):
        """(distances, row positions in the training data) of each row's neighbours, closest first."""
        return self.nn_.kneighbors(self._features(X))

    def predict_details(self, X):
        """(prediction, distances, neighbour positions, which neighbours were used) for each row."""
        dist, idx = self.neighbors(X)
        pred, used = aggregate_neighbors(self.y_[idx], self.aggregate, self.outlier_z)
        return pred, dist, idx, used

    def predict(self, X):
        return self.predict_details(X)[0]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def auto_rent_range(rents, k=IQR_K):
    """
    Rent range that keeps the bulk of the market, from the data's own distribution (Tukey's IQR fences).
    The fences are computed on log rent: rents are right-skewed, so on the raw dollar scale the upper fence sits
    close to the median and the lower fence goes negative, while on the log scale the spread is symmetric enough
    for the fences to flag only unusually extreme listings, in percentage terms.
    """
    logr = np.log(np.asarray(rents, float))
    q1, q3 = np.percentile(logr, [25, 75])
    iqr = q3 - q1
    lo = float(np.exp(q1 - k * iqr))
    hi = float(np.exp(q3 + k * iqr))
    return max(MIN_RENT, float(np.floor(lo / 10) * 10)), min(MAX_RENT, float(np.ceil(hi / 10) * 10))


def has_unit_features(df):
    """True if enough listings have bedrooms and square feet for the unit-size features to mean something."""
    return bool(df[BEDS_COL].notna().mean() >= MIN_UNIT_COVERAGE and df[SQFT_COL].notna().mean() >= MIN_UNIT_COVERAGE)


def apply_unit_basis(df, basis):
    """Add beds / sqft / property_type columns from the *_min / *_max columns (any that are missing become empty)."""
    if basis not in UNIT_BASES:
        raise ValueError(f"unit basis must be one of {UNIT_BASES}")
    for c in ("beds_min", "beds_max", "sqft_min", "sqft_max"):
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    if basis == "min":
        df[BEDS_COL], df[SQFT_COL] = df["beds_min"], df["sqft_min"]
    elif basis == "max":
        df[BEDS_COL], df[SQFT_COL] = df["beds_max"], df["sqft_max"]
    else:
        df[BEDS_COL] = df[["beds_min", "beds_max"]].mean(axis=1)
        df[SQFT_COL] = df[["sqft_min", "sqft_max"]].mean(axis=1)
    df[TYPE_COL] = df[TYPE_COL].astype(str).str.lower().where(df[TYPE_COL].notna()) if TYPE_COL in df.columns else np.nan
    return df


def load_training_data(min_rent="auto", max_rent="auto", quiet=False, iqr_k=IQR_K, unit_basis="min"):
    """
    Properties with a SiteLine score, coordinates and a plausible rent, from data/atlanta_rentals_final.csv.
    min_rent / max_rent: a number in dollars, or "auto" to take that end from the rent distribution (IQR fences).
    unit_basis: which end of a beds / sqft range to use: "min" (the cheapest unit, default), "max" or "mid".
    The range actually used is stored in the result's `.attrs["rent_range"]`, the basis in `.attrs["unit_basis"]`.
    """
    if not FINAL_CSV.exists():
        raise FileNotFoundError(f"{FINAL_CSV} not found. Run redfin_scraper.py, siteline_scorer.py and income.py first.")
    df = pd.read_csv(FINAL_CSV, dtype={ZIP_COL: str, "tract_geoid": str})
    needed = [TARGET, LAT_COL, LON_COL, ZIP_COL, "url", "Address", "in_scope", "tract_geoid", *SCORE_COLUMNS]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{FINAL_CSV.name} is missing columns {missing}. Rerun siteline_scorer.py and income.py.")

    n_all = len(df)
    df[ZIP_COL] = df[ZIP_COL].astype(str).str.extract(r"(\d{5})")[0]
    for c in [TARGET, LAT_COL, LON_COL, *SCORE_COLUMNS]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = apply_unit_basis(df, unit_basis)
    scored = df["in_scope"].astype(str).str.lower().eq("true")
    located = df[LAT_COL].notna() & df[LON_COL].notna() & df[TARGET].notna()
    sane = df[TARGET].between(MIN_RENT, MAX_RENT)
    base = scored & located & sane
    if str(min_rent).lower() == "auto" or str(max_rent).lower() == "auto":
        auto_lo, auto_hi = auto_rent_range(df.loc[base, TARGET], iqr_k)
    min_rent = auto_lo if str(min_rent).lower() == "auto" else float(min_rent)
    max_rent = auto_hi if str(max_rent).lower() == "auto" else float(max_rent)
    plausible = df[TARGET].between(min_rent, max_rent)
    out = df[scored & located & plausible].drop_duplicates(subset="url").reset_index(drop=True)
    out.attrs["rent_range"] = (min_rent, max_rent)
    out.attrs["unit_basis"] = unit_basis

    if not quiet:
        print(f"Loaded {n_all:,} rows from {FINAL_CSV.name}; using {len(out):,} for the model.")
        print(f"  dropped: {int((~scored).sum()):,} without a SiteLine score (outside the 11 counties / no coordinates), "
              f"{int((scored & located & ~plausible).sum()):,} with rent outside ${min_rent:,.0f}-${max_rent:,.0f} "
              f"(the rent range kept; change it with --min-rent / --max-rent).")
        if has_unit_features(out):
            print(f"  unit features ({unit_basis} end of ranges): bedrooms known for {out[BEDS_COL].notna().mean():.0%}, "
                  f"square feet for {out[SQFT_COL].notna().mean():.0%}, property type for {out[TYPE_COL].notna().mean():.0%} "
                  f"of listings.")
        else:
            print("  unit features: not enough listings have bedrooms and square feet (the dataset was probably scraped "
                  "before the scraper collected them). Rescrape with redfin_scraper.py --fresh to use them; "
                  "the unit-size models are skipped for now.")
    if len(out) < 50:
        raise ValueError(f"Only {len(out)} usable properties, which is too few to model.")
    return out


def make_groups(df, mode):
    """Group label per property; properties with the same label always land in the same fold."""
    if mode == "building":
        return (df[LAT_COL].round(4).astype(str) + "_" + df[LON_COL].round(4).astype(str)).to_numpy()
    if mode == "tract":
        return df["tract_geoid"].fillna("unknown").astype(str).to_numpy()
    return np.arange(len(df)).astype(str)                 # every property its own group


def group_folds(groups, n_splits, seed):
    """(train_idx, test_idx) pairs where no group is split across folds; groups are spread to keep folds similar in size."""
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    n_splits = min(n_splits, len(uniq))
    if n_splits < 2:
        raise ValueError("Need at least 2 distinct groups to split.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    order = order[np.argsort(-counts[order], kind="stable")]      # biggest groups first, ties in random order
    sizes = np.zeros(n_splits, dtype=int)
    fold_of = {}
    for i in order:
        f = int(np.argmin(sizes))
        fold_of[uniq[i]] = f
        sizes[f] += counts[i]
    fold_id = np.array([fold_of[g] for g in groups])
    all_idx = np.arange(len(groups))
    return [(all_idx[fold_id != f], all_idx[fold_id == f]) for f in range(n_splits)]


# ---------------------------------------------------------------------------
# Tuning and evaluation
# ---------------------------------------------------------------------------
def tune_params(X, y, folds, w_score_grid, w_zip_grid, options, w_unit_grid=(0.0,), w_type_grid=(0.0,)):
    """
    Choose K, the weights and the averaging method that give the lowest mean absolute error on the validation folds.

    For each combination of weights the neighbours are searched once with the largest K; every smaller K and every
    averaging option is then evaluated from that same sorted neighbour list, which makes the search fast.
    Returns (best settings dict, its mean validation MAE).
    """
    k_max = max(K_GRID)
    results = {}
    for w_score in w_score_grid:
        for w_zip in w_zip_grid:
            for w_unit in w_unit_grid:
                for w_type in w_type_grid:
                    for train_idx, val_idx in folds:
                        model = RentKNN(n_neighbors=min(k_max, len(train_idx)), w_score=w_score, w_zip=w_zip,
                                        w_unit=w_unit, w_type=w_type)
                        model.fit(X.iloc[train_idx], y[train_idx])
                        _, idx = model.neighbors(X.iloc[val_idx])
                        Y = model.y_[idx]                                  # neighbour rents, closest first
                        for k in K_GRID:
                            if k > Y.shape[1]:
                                continue
                            for aggregate, z in options:
                                pred, _ = aggregate_neighbors(Y[:, :k], aggregate, z if z is not None else 2.0)
                                results.setdefault((k, w_score, w_zip, w_unit, w_type, aggregate, z), []).append(
                                    float(np.mean(np.abs(pred - y[val_idx]))))
    best_key = min(results, key=lambda key: float(np.mean(results[key])))
    k, w_score, w_zip, w_unit, w_type, aggregate, z = best_key
    best = {"n_neighbors": int(k), "w_score": float(w_score), "w_zip": float(w_zip), "w_unit": float(w_unit),
            "w_type": float(w_type), "aggregate": aggregate}
    if z is not None:
        best["outlier_z"] = float(z)
    return best, float(np.mean(results[best_key]))


def regression_metrics(y_true, y_pred):
    err = y_pred - y_true
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "medae": float(np.median(np.abs(err))),
        "median_ape": float(np.median(np.abs(err) / y_true)),
        "r2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
    }


def spec_grids(spec):
    return dict(w_score_grid=spec["w_score"], w_zip_grid=spec["w_zip"], options=spec["options"],
                w_unit_grid=spec.get("w_unit", [0.0]), w_type_grid=spec.get("w_type", [0.0]))


def cross_validate_models(df, group_mode, outer_folds, seed):
    """Nested grouped CV. Returns (per-fold metrics, out-of-fold predictions)."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET].to_numpy(float)
    groups = make_groups(df, group_mode)
    outer = group_folds(groups, outer_folds, seed)
    units_ok = has_unit_features(df)
    specs = {n: sp for n, sp in MODEL_SPECS.items() if units_ok or n not in UNIT_MODELS}
    print(f"\nCross-validating {len(specs) + 1} models: {len(outer)} outer folds (testing sets), {INNER_FOLDS} inner folds "
          f"(validation) to tune each model; properties grouped by {group_mode}.")
    if not units_ok:
        print("  (unit-size models skipped: the data has no bedrooms / square feet yet)")

    rows, pred_frames = [], []
    for fold, (train_idx, test_idx) in enumerate(outer, 1):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        inner = group_folds(groups[train_idx], INNER_FOLDS, seed + fold)
        preds = {"mean_baseline": np.full(len(test_idx), y_train.mean())}
        chosen = {"mean_baseline": {}}
        for name, spec in specs.items():
            best, _ = tune_params(X_train, y_train, inner, **spec_grids(spec))
            preds[name] = RentKNN(**best).fit(X_train, y_train).predict(X.iloc[test_idx])
            chosen[name] = best
        for name, p in preds.items():
            rows.append({"fold": fold, "model": name, "n_train": len(train_idx), "n_test": len(test_idx),
                         **regression_metrics(y[test_idx], p), "best_params": json.dumps(chosen[name])})
        pred_frames.append(pd.DataFrame({
            "fold": fold, "url": df["url"].iloc[test_idx].to_numpy(), "Address": df["Address"].iloc[test_idx].to_numpy(),
            ZIP_COL: df[ZIP_COL].iloc[test_idx].to_numpy(), "actual": y[test_idx],
            **{f"pred_{n}": p for n, p in preds.items()}}))
        mae = {r["model"]: r["mae"] for r in rows if r["fold"] == fold}
        shown = ", ".join(f"{m} ${mae[m]:,.0f}" for m in ("full_robust", "units_robust", "all_robust") if m in mae)
        print(f"  fold {fold}/{len(outer)}: tested on {len(test_idx):,} unseen properties (MAE: {shown})")
    return pd.DataFrame(rows), pd.concat(pred_frames, ignore_index=True)


def print_cv_summary(metrics):
    present = set(metrics["model"])
    print("\n--- Cross-validated test performance (unseen properties; mean over folds, +/- std) ---")
    for m in [m for m in MODEL_ORDER if m in present]:
        s_ = metrics[metrics["model"] == m]
        print(f"{m:<20} RMSE ${s_['rmse'].mean():>6,.0f} (+/-{s_['rmse'].std():>4,.0f}) | "
              f"MAE ${s_['mae'].mean():>5,.0f} (+/-{s_['mae'].std():>4,.0f}) | typical error ${s_['medae'].mean():>5,.0f} | "
              f"median % error {s_['median_ape'].mean() * 100:>4.1f}% | R2 {s_['r2'].mean():>6.3f}")

    def by_fold(model, col):
        return metrics[metrics["model"] == model].set_index("fold")[col]

    def compare(a, b, text):
        if a not in present or b not in present:
            return
        parts = []
        for col, label in (("mae", "MAE"), ("rmse", "RMSE")):
            fa, fb = by_fold(a, col), by_fold(b, col)
            diff = (fa - fb).mean()
            parts.append(f"{label} {'lower' if diff < 0 else 'higher'} by ${abs(diff):,.0f} ({abs(diff) / fb.mean() * 100:.1f}%), "
                         f"better in {int((fa < fb).sum())} of {len(fa)} folds")
        print(f"\n{text}\n  " + "; ".join(parts))

    compare("full_robust", "full", "Outlier-robust averaging vs plain mean (full model):")
    compare("location_zip_robust", "location_zip", "Outlier-robust averaging vs plain mean (location + postal code):")
    compare("units_robust", "location_zip_robust", "Unit features (bedrooms, size, type) vs none, robust averaging:")
    compare("all_robust", "units_robust", "SiteLine scores on top of unit features (all_robust vs units_robust):")
    compare("full_robust", "location_zip_robust", "SiteLine scores vs none, without unit features:")

    def describe(p):
        text = p["aggregate"] + (f"(z={p['outlier_z']:g})" if "outlier_z" in p else "")
        return f"{text}, K={p['n_neighbors']}, weights score={p['w_score']:g} zip={p['w_zip']:g} unit={p['w_unit']:g} type={p['w_type']:g}"

    for name in ("all_robust", "units_robust", "full_robust"):
        if name in present:
            chosen = metrics[metrics["model"] == name]["best_params"].map(json.loads)
            print(f"\nSettings chosen by {name} in each fold:")
            for f, p in zip(metrics[metrics["model"] == name]["fold"], chosen):
                print(f"  fold {f}: {describe({'w_unit': 0.0, 'w_type': 0.0, **p})}")
            break


def save_scatter(preds, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the actual-vs-predicted plot.")
        return
    col = "pred_all_robust" if "pred_all_robust" in preds.columns else "pred_full_robust"
    top = float(max(preds["actual"].max(), preds[col].max()))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(preds["actual"], preds[col], s=8, alpha=0.35)
    ax.plot([0, top], [0, top], "k--", linewidth=1)
    ax.set_xlabel("Actual monthly rent ($)")
    ax.set_ylabel("Predicted monthly rent ($), unseen property")
    ax.set_title(f"KNN rent model ({col.replace('pred_', '')}): cross-validated predictions")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def tune_final(df, group_mode, seed):
    """Choose K, the weights and the averaging method for the final model using grouped CV on all the data."""
    groups = make_groups(df, group_mode)
    folds = group_folds(groups, OUTER_FOLDS, seed + 999)
    units_ok = has_unit_features(df)
    return tune_params(df[FEATURE_COLUMNS], df[TARGET].to_numpy(float), folds, W_SCORE_GRID, W_ZIP_GRID, ALL_OPTIONS,
                       w_unit_grid=W_UNIT_GRID if units_ok else [0.0], w_type_grid=W_TYPE_GRID if units_ok else [0.0])


def cmd_train(args):
    df = load_training_data(args.min_rent, args.max_rent, iqr_k=args.iqr_k, unit_basis=args.unit_basis)
    lo, hi = df.attrs["rent_range"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    if not args.no_cv:
        metrics, preds = cross_validate_models(df, args.group_by, args.outer_folds, args.seed)
        print_cv_summary(metrics)
        metrics.to_csv(MODEL_DIR / "cv_metrics.csv", index=False)
        preds.to_csv(MODEL_DIR / "cv_predictions.csv", index=False)
        save_scatter(preds, MODEL_DIR / "cv_actual_vs_predicted.png")
        summary["cv_mean_by_model"] = (metrics.groupby("model")[["rmse", "mae", "medae", "median_ape", "r2"]]
                                       .mean().round(4).to_dict("index"))
        print(f"\nSaved cv_metrics.csv, cv_predictions.csv and cv_actual_vs_predicted.png to {MODEL_DIR}")

    params, cv_mae = tune_final(df, args.group_by, args.seed)
    print(f"\nFinal settings for predict (tuned on all {len(df):,} properties): {params} "
          f"(cross-validated MAE ${cv_mae:,.0f}; this tuning figure is a bit optimistic, use the table above for accuracy)")
    PARAMS_JSON.write_text(json.dumps({
        "trained_at": datetime.now().isoformat(timespec="seconds"), "n_properties": len(df),
        "group_by": args.group_by, "min_rent": lo, "max_rent": hi, "unit_basis": args.unit_basis,
        "has_unit_features": has_unit_features(df), "best_params": params, **summary}, indent=2))
    print(f"Saved {PARAMS_JSON}")
    return 0


# ---------------------------------------------------------------------------
# Predicting a new property
# ---------------------------------------------------------------------------
def load_params():
    if PARAMS_JSON.exists():
        return json.loads(PARAMS_JSON.read_text())["best_params"]
    print("No tuned settings found (run `python knn_model.py train` first); using defaults K=8, weights 0.5, plain mean.")
    return {"n_neighbors": 8, "w_score": 0.5, "w_zip": 0.5}


def find_in_dataset(df, query):
    """Listings whose address or URL name matches `query` (e.g. a property name or street address)."""
    q = query.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", q).strip("-")
    if len(q) < 4:
        return df.iloc[0:0]
    mask = (df["Address"].astype(str).str.lower().str.contains(re.escape(q), regex=True)
            | df["url"].astype(str).str.lower().str.contains(re.escape(slug), regex=True))
    return df[mask]


def geocode_address(query):
    """(lat, lon, zip, label) from the U.S. Census geocoder, falling back to OpenStreetMap Nominatim."""
    import requests
    q = query if re.search(r"\bGA\b|georgia", query, flags=re.IGNORECASE) else f"{query}, Atlanta, GA"
    headers = {"User-Agent": "atlanta-rental-affordability-research"}
    try:
        r = requests.get("https://geocoding.geo.census.gov/geocoder/locations/onelineaddress", headers=headers, timeout=30,
                         params={"address": q, "benchmark": "Public_AR_Current", "format": "json"})
        matches = r.json()["result"]["addressMatches"]
        if matches:
            m = matches[0]
            return float(m["coordinates"]["y"]), float(m["coordinates"]["x"]), m["addressComponents"].get("zip") or None, m["matchedAddress"]
    except Exception as e:
        print(f"  Census geocoder unavailable ({type(e).__name__}); trying OpenStreetMap...")
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search", headers=headers, timeout=30,
                         params={"q": q, "format": "jsonv2", "limit": 1, "addressdetails": 1})
        hits = r.json()
        if hits:
            h = hits[0]
            return float(h["lat"]), float(h["lon"]), h.get("address", {}).get("postcode"), h.get("display_name", q)
    except Exception as e:
        print(f"  OpenStreetMap geocoder unavailable ({type(e).__name__}).")
    raise ValueError(f"Could not find '{query}'. Pass its coordinates instead: --lat 33.78 --lon -84.38")


def unit_from_listing(row):
    """Unit details of a dataset listing: what the model uses (beds, sqft, type) plus the raw ranges for display."""
    def num(v):
        return None if pd.isna(v) else float(v)
    return {"beds": num(row[BEDS_COL]), "sqft": num(row[SQFT_COL]),
            "type": None if pd.isna(row[TYPE_COL]) else str(row[TYPE_COL]),
            "beds_range": (num(row.get("beds_min")), num(row.get("beds_max"))),
            "sqft_range": (num(row.get("sqft_min")), num(row.get("sqft_max")))}


def resolve_location(args, df):
    """
    Work out lat, lon, zip and unit details for the property to predict, plus any dataset listings to leave out of the
    neighbours. Unit details come from the dataset listing when the property is found there, else they are empty.
    """
    empty_unit = {"beds": None, "sqft": None, "type": None, "beds_range": (None, None), "sqft_range": (None, None)}
    if args.lat is not None and args.lon is not None:
        return {"lat": args.lat, "lon": args.lon, "zip": args.zip, "label": f"({args.lat}, {args.lon})",
                "exclude": [], "unit": empty_unit, "from_dataset": False}
    if not args.property:
        raise ValueError("Give --lat and --lon, or --property \"name or address\".")
    hits = find_in_dataset(df, args.property)
    if 0 < len(hits) <= 25:
        first = hits.iloc[0]
        print(f"Found {len(hits)} listing(s) for '{args.property}' in the dataset; using the first "
              f"({first['Address']}). These listings are left out of the neighbours.")
        return {"lat": float(first[LAT_COL]), "lon": float(first[LON_COL]), "zip": args.zip or first[ZIP_COL],
                "label": str(first["Address"]), "exclude": hits["url"].tolist(), "unit": unit_from_listing(first),
                "from_dataset": True}
    if len(hits) > 25:
        print(f"'{args.property}' matches {len(hits)} listings, which is too generic; looking it up as an address instead.")
    lat, lon, zip_, label = geocode_address(args.property)
    zip_ = args.zip or (re.search(r"\d{5}", str(zip_)).group(0) if zip_ and re.search(r"\d{5}", str(zip_)) else None)
    return {"lat": lat, "lon": lon, "zip": zip_, "label": label, "exclude": [], "unit": empty_unit, "from_dataset": False}


def score_with_siteline(lat, lon):
    sys.path.insert(0, str(ROOT))              # so siteline_scorer.py is found however this script is launched
    from siteline_scorer import import_siteline
    siteline = import_siteline()
    row = siteline.score_sites(pd.DataFrame({"lat": [lat], "lon": [lon]}), progress=False).iloc[0]
    if not bool(row.get("in_scope")):
        raise ValueError("That location is outside SiteLine's 11-county metro-Atlanta region, so it can't be scored.")
    return row


def fmt_range(rng, unit=""):
    lo, hi = rng
    if lo is None:
        return "not listed"
    return f"{lo:g}{unit}" if hi is None or hi == lo else f"{lo:g}-{hi:g}{unit}"


def cmd_predict(args):
    saved = json.loads(PARAMS_JSON.read_text()) if PARAMS_JSON.exists() else {}
    args.min_rent = args.min_rent if args.min_rent is not None else saved.get("min_rent", "auto")
    args.max_rent = args.max_rent if args.max_rent is not None else saved.get("max_rent", "auto")
    basis = args.unit_basis or saved.get("unit_basis", "min")
    df = load_training_data(args.min_rent, args.max_rent, quiet=True, unit_basis=basis)
    lo, hi = df.attrs["rent_range"]
    print(f"Using listings with rent between ${lo:,.0f} and ${hi:,.0f}.")
    target = resolve_location(args, df)
    row = score_with_siteline(target["lat"], target["lon"])

    # The unit: what the user typed wins, then the dataset listing; anything still empty is filled in by the model.
    unit = target["unit"]
    beds = args.beds if args.beds is not None else unit["beds"]
    sqft = args.sqft if args.sqft is not None else unit["sqft"]
    ptype = (args.type.strip().lower() if args.type else None) or (unit["type"] or None)

    new = pd.DataFrame([{LAT_COL: target["lat"], LON_COL: target["lon"], ZIP_COL: target["zip"],
                         **{c: pd.to_numeric(row.get(c), errors="coerce") for c in SCORE_COLUMNS},
                         BEDS_COL: beds if beds is not None else np.nan, SQFT_COL: sqft if sqft is not None else np.nan,
                         TYPE_COL: ptype if ptype else np.nan}])
    train = df[~df["url"].isin(target["exclude"])].reset_index(drop=True)
    params = dict(load_params())
    if args.aggregate:
        params["aggregate"] = args.aggregate
    model = RentKNN(**params).fit(train[FEATURE_COLUMNS], train[TARGET].to_numpy(float))

    pred, dist, idx, used = model.predict_details(new)
    pred = float(pred[0])
    near = train.iloc[idx[0]].copy()
    near["distance"] = dist[0]
    near["used"] = np.where(used[0], "yes", "no (outlier)")
    n_used = int(used[0].sum())

    how = {"mean": "average", "median": "median", "trimmed": "trimmed average", "filtered": "average, outlier neighbors removed"}[model.aggregate]
    print(f"\nProperty: {target['label']}   ZIP: {target['zip'] or 'not given (postal code ignored)'}")
    print(f"SiteLine score: {float(pd.to_numeric(row.get('total_score'), errors='coerce')):.1f} / 46   "
          + "  ".join(f"{c.replace('score_', '')}={new.at[0, c]:g}" for c in SCORE_COLUMNS if pd.notna(new.at[0, c])))

    # ---- say exactly which unit the prediction is for
    uses_units, uses_type = model.w_unit > 0, model.w_type > 0
    given = [name for name, v in (("--beds", args.beds), ("--sqft", args.sqft), ("--type", args.type)) if v is not None]
    print("\nUnit this prediction is for:")
    if not (uses_units or uses_type):
        print("  The tuned model does not use unit features (weight 0), so bedrooms, size and type do not change the result."
              + (f" Ignored: {', '.join(given)}." if given else "")
              + ("" if saved.get("has_unit_features", True) else " (The dataset has no bedrooms / square feet yet: rescrape and retrain.)"))
    else:
        eff_beds, eff_sqft, eff_type = model.unit_values(new)
        if target["from_dataset"]:
            print(f"  The listing shows {fmt_range(unit['beds_range'], ' bed')}, {fmt_range(unit['sqft_range'], ' sq ft')}, "
                  f"type {unit['type'] or 'not listed'}; using the '{basis}' end of each range unless you override it with "
                  f"--beds / --sqft / --type.")
        if uses_units:
            beds_note = "you gave" if args.beds is not None else ("from the listing" if beds is not None else "not given, so the typical value")
            sqft_note = ("you gave" if args.sqft is not None else "from the listing") if sqft is not None else (
                f"not given, so the typical size for {eff_beds[0]:g} bedroom(s)")
            print(f"  bedrooms: {eff_beds[0]:g} ({beds_note})   floor area: {eff_sqft[0]:,.0f} sq ft ({sqft_note})")
        if uses_type:
            if not ptype:
                print("  type: not given, so it is treated as 'unknown' and matches no listing's type")
            elif ptype not in model.types_:
                print(f"  type: '{ptype}' never appears in the data (known: {', '.join(model.types_)}); it matches no listing's type")
            else:
                print(f"  type: {ptype}" + (" (you gave)" if args.type else " (from the listing)"))
        if not target["from_dataset"] and not given:
            print("  Tip: you gave no unit details, so this is a prediction for a typical unit at this location. "
                  "Add --beds, --sqft and --type for a specific one.")

    used_text = f"{n_used} of {len(near)} used" if model.aggregate in ("trimmed", "filtered") else f"{len(near)} neighbors"
    print(f"\nPredicted rent: ${pred:,.0f} / month   ({how}; {used_text}; "
          f"neighbor rents range ${near[TARGET].min():,.0f} to ${near[TARGET].max():,.0f})")
    cols = [c for c in ("Address", ZIP_COL, TARGET, BEDS_COL, SQFT_COL, TYPE_COL, "total_score", "distance") if c in near.columns]
    if model.aggregate in ("trimmed", "filtered"):
        cols.append("used")
    shown = 15

    def number(places):
        return lambda v: "" if pd.isna(v) else f"{v:,.{places}f}"
    formatters = {TARGET: number(0), BEDS_COL: number(1), SQFT_COL: number(0), "total_score": number(1), "distance": number(2)}
    print(f"\nNearest listings (closest first{'' if len(near) <= shown else f'; showing {shown} of {len(near)}'}):")
    print(near[cols].head(shown).to_string(index=False, formatters={k: v for k, v in formatters.items() if k in cols},
                                           na_rep=""))

    annual = pred * 12
    print(f"\nAffordability (threshold {args.threshold:.0%} of household income):")
    print(f"  A household needs about ${annual / args.threshold:,.0f}/year to rent this at {args.threshold:.0%} of income.")
    tract = str(row.get("tract_geoid"))
    incomes = (df.dropna(subset=["median_household_income"]).groupby("tract_geoid")["median_household_income"].first()
               if "median_household_income" in df.columns else pd.Series(dtype=float))
    if tract in incomes.index:
        income = float(incomes[tract])
        share = annual / income
        verdict = "NOT affordable" if share > args.threshold else "affordable"
        print(f"  Tract {tract} median household income is ${income:,.0f}: this rent would be {share:.0%} of it -> {verdict} "
              f"for the typical household there.")
    else:
        print(f"  No median income on file for tract {tract}, so the rent share of income can't be computed.")
    return 0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="KNN rent model: cross-validate it and predict rents for new properties.")
    sub = p.add_subparsers(dest="command", required=True)

    def rent_arg(text):
        return "auto" if text.lower() == "auto" else float(text)

    def common(sp, default_min="auto", default_max="auto", note=""):
        sp.add_argument("--min-rent", type=rent_arg, default=default_min,
                        help="ignore listings below this rent in dollars, or 'auto' (IQR fence on the data)" + note)
        sp.add_argument("--max-rent", type=rent_arg, default=default_max,
                        help="ignore listings above this rent in dollars, or 'auto' (IQR fence on the data)" + note)

    t = sub.add_parser("train", help="cross-validate the model and save the tuned settings")
    common(t)
    t.add_argument("--iqr-k", type=float, default=IQR_K,
                   help="how wide the 'auto' rent range is: Q1 - k*IQR to Q3 + k*IQR on log rent (default 1.5; 3 keeps more)")
    t.add_argument("--unit-basis", choices=UNIT_BASES, default="min",
                   help="which end of a beds / sqft range to use: min = the cheapest unit (default), max, or mid")
    t.add_argument("--no-cv", action="store_true", help="skip the evaluation; only tune and save the final settings")
    t.add_argument("--group-by", choices=["building", "tract", "none"], default="building",
                   help="keep listings of the same building (default), the same tract, or nothing together in one fold")
    t.add_argument("--outer-folds", type=int, default=OUTER_FOLDS, help="number of testing folds")
    t.add_argument("--seed", type=int, default=SEED)

    q = sub.add_parser("predict", help="predict the rent of a new property")
    common(q, None, None, " (default: the range used by the last `train`)")
    q.add_argument("--lat", type=float)
    q.add_argument("--lon", type=float)
    q.add_argument("--zip", help="postal code (optional; improves the neighbour match)")
    q.add_argument("--property", help="property name or address")
    q.add_argument("--beds", type=float, help="bedrooms of the unit (0 = studio); default: the dataset listing's, else typical")
    q.add_argument("--sqft", type=float, help="floor area in square feet; default: the listing's, else typical for the bedrooms")
    q.add_argument("--type", help="property type as in the data (apartment, home, ...); default: the listing's")
    q.add_argument("--unit-basis", choices=UNIT_BASES, default=None,
                   help="which end of a listing's beds / sqft range to use (default: the one used by the last `train`)")
    q.add_argument("--aggregate", choices=AGGREGATES, help="override the saved way of combining neighbor rents")
    q.add_argument("--threshold", type=float, default=AFFORDABILITY_THRESHOLD,
                   help="share of income above which rent is not affordable (default 0.30)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return cmd_train(args) if args.command == "train" else cmd_predict(args)


if __name__ == "__main__":
    sys.exit(main())
