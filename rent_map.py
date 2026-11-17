"""
Step 5: an interactive heat map of predicted rent across Atlanta.

Writes ONE self-contained web page, data/model/rent_heatmap.html. Open it in any browser (it needs an internet
connection to load the map tiles and the Leaflet map library). In the page you pick the unit (bedrooms, floor area,
property type) and the map recolours to the rent the KNN model predicts for that unit at every point of the city.

HOW IT WORKS
    1. A regular grid of square cells (default 500 m wide) is laid over the area your listings cover. A grid, rather
       than random points, gives even coverage with no gaps or clumps, the same cells every run, and cells of equal
       area, so colours are comparable across the map.
    2. Cells farther than --max-dist-km (default 1.5 km) from every listing are dropped: the model has nothing to
       learn from there, so the map would only be guessing. The remaining cells are scored with SiteLine, exactly like the listings. A cell's
       postal code is taken from its nearest listing.
    3. The tuned model from `python knn_model.py train` (data/model/best_params.json) predicts every cell for a
       lattice of units: each bedroom count (studio to --max-beds) x floor area (--sqft-min to --sqft-max in steps of
       --sqft-step) x each common property type. These predictions are stored in the page, so the page needs no
       server: moving a control just switches to the matching, already computed, set of numbers.
    4. The City of Atlanta's limits are drawn as an outline (downloaded once from the Census Bureau, with the city's own
       GIS as a backup, and cached in data/model/atlanta_boundary.geojson). It can be switched off in the page.
       Note that the grid itself covers wherever your listings are, which can spill a little past the city limits;
       --clip-to-city trims the cells to the city.
    5. Census tract boundaries can be shown too (a checkbox in the page). They come from the Census Bureau's TIGERweb
       service, are downloaded once for the mapped area, and are cached in data/model/tract_boundaries.geojson.
       Skip them with --no-tracts.
    6. Cells far from any listing are drawn paler, to show where the prediction rests on less evidence.

THE PAGE
    Search         type a listing's name or address, an address anywhere in Atlanta, or "lat, lon". The page shows the
                   model's estimate there for the selected unit (a listing found by name also sets the unit controls to
                   that listing's own bedrooms/size/type), the income needed, and whether it passes the threshold.
                   Listing names and addresses are searched inside the page; other addresses are looked up through
                   OpenStreetMap's Nominatim service from your browser (turn that off with --no-geocoder).
    Pin            "Drop a pin on the map", then click anywhere: the pin appears and the side panel shows the estimate
                   there, the income needed, the threshold verdict and the LIHTC score. Drag the pin to move it.
    Layout         a sidebar (metric, search/pin, unit controls) next to the map; the colour legend floats on the map.
                   On a phone the sidebar becomes a bottom sheet.
    Metric         "Predicted rent", or "Rent as share of income": a year of the predicted rent divided by the tract's
                   median household income. A slider sets the share above which rent counts as unaffordable (default 30%);
                   cells above it are red, below it blue, and "Color only the unaffordable areas" leaves the rest plain.
                   Income comes from the whole census table in data/census_income/, so every tract is covered, not
                   only those with a listing. Tracts the census gives no income for are gray.
                   "LIHTC score": SiteLine's score (out of 46) for each location, in its own green scale. It belongs to
                   the place, not the unit, so the unit controls hide.
    Light / Dark   a button in the panel header. It follows your system until you press it, then remembers your choice.
    Controls       bedrooms, floor area ("typical" button), property type, show the listings as dots, rescale the
                   colours to the current view, cell opacity.
    Hover          the predicted rent, the unit, the income a household needs to pay it at 30% of income, the
                   tract's median household income and whether the rent exceeds the 30% share, and the distance to the nearest
                   listing.
    Download       the current view as a CSV (one row per cell).
    The colour scale is the same for every unit by default, so you can see a 3-bed cost more than a studio.

USAGE
    python rent_map.py                          # needs data/model/best_params.json from `python knn_model.py train`
    python rent_map.py --cell-km 0.35           # finer cells (more cells = a bigger page and a slower first run)
    python rent_map.py --max-dist-km 1.0        # show only areas closer to listings
    python rent_map.py --clip-to-city           # also drop cells outside the city limits
    python rent_map.py --boundary atlanta.geojson   # use your own outline (and clip to it)
    python rent_map.py --no-outline
    python rent_map.py --public                 # the version to publish: no listings, no API key, written to docs/index.html
    python rent_map.py --no-tracts              # do not download or draw census tract boundaries
    python rent_map.py --rescore                # ignore the cached SiteLine scores of grid cells

    MAP TILES: CARTO's clean light/dark basemaps now require a free API key (otherwise every tile is stamped
    "API KEY REQUIRED"). Get one in a minute, with no account, at https://carto.com/basemaps/apikey/ and run
        python rent_map.py --carto-key YOUR_KEY          (or set the CARTO_API_KEY environment variable)
    Without a key the page uses OpenStreetMap's tiles, which need none. If those do not load when you double-click the
    file, serve it instead:  cd data/model && python -m http.server 8000   then open http://localhost:8000/rent_heatmap.html
    The key is written into the page, so keep the page to yourself, or restrict the key to your own use on CARTO's site.

    SiteLine scores for the grid are cached in data/model/grid_scores.csv, so only the first run is slow.
    Re-run this script after retraining or after a new scrape.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import knn_model as km                                            # noqa: E402  (shares the data loading and the model)
from siteline_scorer import CORE_COLUMNS, import_siteline, score_rentals   # noqa: E402

OUT_HTML = km.MODEL_DIR / "rent_heatmap.html"
PUBLIC_HTML = Path(__file__).resolve().parent / "docs" / "index.html"       # where --public writes (GitHub Pages serves /docs)
PROJECT_NAME = "Mapping the Affordability of Rental Properties in Atlanta"
LAB_NAME = "Emory AI.DataLab"
CREATOR = "Songjie Pu"
CONTRIBUTORS = ["Joshua Velasquez", "Yanqi Li"]
GRID_CACHE = km.MODEL_DIR / "grid_scores.csv"
MAX_CELLS = 40_000
MIN_TYPE_LISTINGS = 20                  # a property type needs this many listings to get a control
MAX_TYPES = 4
DOMAIN_PERCENTILES = (2, 98)            # colour scale limits (percentiles of all predictions)


# ---------------------------------------------------------------------------
# Model and data
# ---------------------------------------------------------------------------
def load_model(args):
    """Training listings and the tuned model, the same way `knn_model.py predict` loads them."""
    saved = json.loads(km.PARAMS_JSON.read_text()) if km.PARAMS_JSON.exists() else {}
    if not saved:
        print("No tuned settings found: run `python knn_model.py train` first for a tuned model. Using defaults for now.")
    min_rent = args.min_rent if args.min_rent is not None else saved.get("min_rent", "auto")
    max_rent = args.max_rent if args.max_rent is not None else saved.get("max_rent", "auto")
    basis = args.unit_basis or saved.get("unit_basis", "min")
    df = km.load_training_data(min_rent, max_rent, quiet=True, unit_basis=basis)
    lo, hi = df.attrs["rent_range"]
    params = km.load_params()
    model = km.RentKNN(**params).fit(df[km.FEATURE_COLUMNS], df[km.TARGET].to_numpy(float))
    print(f"Model: {params}")
    print(f"Training on {len(df):,} listings with rent between ${lo:,.0f} and ${hi:,.0f} (unit basis: {basis}).")
    return df, model, params, saved, basis


def xy_km(lat, lon, lat_ref, lon_ref):
    """Local east/north kilometres around a reference point (fine at city scale)."""
    north = (np.asarray(lat, float) - lat_ref) * km.KM_PER_DEG_LAT
    east = (np.asarray(lon, float) - lon_ref) * km.KM_PER_DEG_LAT * np.cos(np.radians(lat_ref))
    return np.column_stack([east, north])


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------
def rings_from_geojson(data):
    """All polygon rings (arrays of longitude/latitude) in a GeoJSON object (Polygons, MultiPolygons, features, collections)."""
    def rings_of(geom):
        if not isinstance(geom, dict):
            return []
        kind, coords = geom.get("type"), geom.get("coordinates")
        if kind == "Polygon":
            return [np.asarray(r, float)[:, :2] for r in coords]
        if kind == "MultiPolygon":
            return [np.asarray(r, float)[:, :2] for poly in coords for r in poly]
        if kind == "GeometryCollection":
            return [r for g in geom.get("geometries", []) for r in rings_of(g)]
        return []
    if not isinstance(data, dict):
        return []
    if data.get("type") == "FeatureCollection":
        return [r for f in data.get("features", []) for r in rings_of(f.get("geometry"))]
    if data.get("type") == "Feature":
        return rings_of(data.get("geometry"))
    return rings_of(data)


def load_boundary(path):
    """Polygon rings from a GeoJSON file."""
    rings = rings_from_geojson(json.loads(Path(path).read_text()))
    if not rings:
        raise SystemExit(f"No polygons found in {path}.")
    return rings


# The City of Atlanta's limits, from the Census Bureau (incorporated place 1304000) or, failing that, the city's own GIS.
CITY_OUTLINE_CACHE = km.MODEL_DIR / "atlanta_boundary.geojson"
CITY_SOURCES = [
    ("U.S. Census Bureau TIGERweb",
     "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Places_CouSub_ConCity_SubMCD/MapServer/4/query",
     {"where": "GEOID='1304000'", "outFields": "NAME,GEOID", "outSR": "4326", "f": "geojson"}),
    ("City of Atlanta GIS",
     "https://gis.atlantaga.gov/dpcd/rest/services/AdministrativeArea/GeopoliticalArea/MapServer/0/query",
     {"where": "1=1", "outFields": "*", "outSR": "4326", "f": "geojson"}),
]
ATLANTA_BOX = (-84.7, -84.1, 33.5, 34.1)             # lon_min, lon_max, lat_min, lat_max: sanity check on a downloaded outline


def get_city_rings(refresh=False):
    """Atlanta's city-limit rings: from the cached file, else downloaded once and cached. None if neither works."""
    if CITY_OUTLINE_CACHE.exists() and not refresh:
        rings = rings_from_geojson(json.loads(CITY_OUTLINE_CACHE.read_text()))
        if rings:
            return rings, "cached city limits"
    try:
        import requests
    except ImportError:
        print("Skipping the city outline: the 'requests' package is not installed.")
        return None, None
    for name, url, params in CITY_SOURCES:
        try:
            reply = requests.get(url, params=params, timeout=60)
            reply.raise_for_status()
            data = reply.json()
            rings = rings_from_geojson(data)
            if not rings:
                raise ValueError("no polygon in the reply")
            xs = np.concatenate([r[:, 0] for r in rings]); ys = np.concatenate([r[:, 1] for r in rings])
            if not (ATLANTA_BOX[0] < xs.min() and xs.max() < ATLANTA_BOX[1] and ATLANTA_BOX[2] < ys.min() and ys.max() < ATLANTA_BOX[3]):
                raise ValueError("the polygon is not where Atlanta is")
            km.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            CITY_OUTLINE_CACHE.write_text(json.dumps(data))
            print(f"City limits downloaded from {name} (saved to {CITY_OUTLINE_CACHE.name}).")
            return rings, name
        except Exception as e:
            print(f"  Could not get the city limits from {name}: {type(e).__name__}: {e}")
    print("No city outline this time (no internet?). Pass --boundary city.geojson to supply one, or --no-outline to skip it.")
    return None, None


def outline_for_page(rings, tol_km=0.02):
    """Rings as [[lat, lon], ...] for the page, dropping points closer than tol_km to the previous one to keep it small."""
    out = []
    for ring in rings:
        lat0 = float(ring[0, 1])
        kept = [ring[0]]
        for pt in ring[1:-1]:
            dx = (pt[0] - kept[-1][0]) * km.KM_PER_DEG_LAT * np.cos(np.radians(lat0))
            dy = (pt[1] - kept[-1][1]) * km.KM_PER_DEG_LAT
            if np.hypot(dx, dy) >= tol_km:
                kept.append(pt)
        kept.append(ring[0])
        if len(kept) >= 4:
            out.append([[round(float(y), 5), round(float(x), 5)] for x, y in kept])
    return out


# Census tract boundaries (TIGERweb). The service lists its layers, so the "Census Tracts" layer is looked up by name.
TRACT_CACHE = km.MODEL_DIR / "tract_boundaries.geojson"
TRACT_SERVICE = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer"


def get_tracts(cells, refresh=False, pad_deg=0.02):
    """
    Tract polygons covering the mapped cells, as [{"g": GEOID, "r": rings for the page}], or None if they cannot be had.
    Downloaded once for the cells' bounding box and cached (reused while the cache covers the area).
    """
    w, e = float(cells["longitude"].min()) - pad_deg, float(cells["longitude"].max()) + pad_deg
    s_, n = float(cells["latitude"].min()) - pad_deg, float(cells["latitude"].max()) + pad_deg
    data = None
    if TRACT_CACHE.exists() and not refresh:
        try:
            cached = json.loads(TRACT_CACHE.read_text())
            cw, cs, ce, cn = cached["bbox"]
            if cw <= w and cs <= s_ and ce >= e and cn >= n and cached.get("features"):
                data = cached
        except Exception:
            data = None
    if data is None:
        try:
            import requests
        except ImportError:
            print("Skipping tract boundaries: the 'requests' package is not installed.")
            return None
        try:
            layers = requests.get(TRACT_SERVICE, params={"f": "json"}, timeout=60).json().get("layers", [])
            layer = next((l["id"] for l in layers if l.get("name", "").strip().lower() == "census tracts"), 0)
            feats, offset = [], 0
            while True:
                reply = requests.get(f"{TRACT_SERVICE}/{layer}/query", timeout=120, params={
                    "where": "STATE='13'", "geometry": f"{w},{s_},{e},{n}", "geometryType": "esriGeometryEnvelope",
                    "inSR": "4326", "spatialRel": "esriSpatialRelIntersects", "outFields": "GEOID", "outSR": "4326",
                    "returnGeometry": "true", "maxAllowableOffset": "0.0001", "resultOffset": offset,
                    "resultRecordCount": 500, "f": "geojson"})
                reply.raise_for_status()
                page = reply.json()
                got = page.get("features", [])
                feats += got
                if len(got) < 500 and not page.get("exceededTransferLimit"):
                    break
                offset += len(got)
                if not got or offset > 5000:
                    break
            if not feats:
                raise ValueError("no tracts in the reply")
            data = {"type": "FeatureCollection", "bbox": [w, s_, e, n], "features": feats}
            km.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            TRACT_CACHE.write_text(json.dumps(data))
            print(f"Census tract boundaries downloaded: {len(feats)} tracts (saved to {TRACT_CACHE.name}).")
        except Exception as ex:
            print(f"  Could not get census tract boundaries: {type(ex).__name__}: {ex}\n  The map will have no tract outlines this time (use --no-tracts to stop trying).")
            return None
    keep = set(cells["tract_geoid"].dropna().astype(str)) if "tract_geoid" in cells.columns else set()
    def convert(only):
        out = []
        for f in data["features"]:
            gid = str((f.get("properties") or {}).get("GEOID", ""))
            if only and gid not in only:
                continue                                    # only the tracts that contain mapped cells
            rings = outline_for_page(rings_from_geojson(f), tol_km=0.03)
            if rings:
                out.append({"g": gid, "r": rings})
        return out
    return convert(keep) or convert(None) or None           # if no cell's tract id matched, draw everything downloaded


def inside_boundary(rings, lon, lat):
    """Even-odd point-in-polygon test over all rings (so holes and separate polygons both work); vectorized over points."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    inside = np.zeros(len(lon), dtype=bool)
    for ring in rings:
        x1, y1 = ring[:-1, 0], ring[:-1, 1]
        x2, y2 = ring[1:, 0], ring[1:, 1]
        for xa, ya, xb, yb in zip(x1, y1, x2, y2):
            if ya == yb:
                continue
            crosses = (ya > lat) != (yb > lat)
            inside ^= crosses & (lon < (xb - xa) * (lat - ya) / (yb - ya) + xa)
    return inside


def build_grid(df, cell_km, max_dist_km, boundary_rings=None):
    """
    Square cells over the listings' extent, keeping those within max_dist_km of a listing (and inside the boundary,
    if given). Returns the cells (centre latitude/longitude, grid indices i/j, distance to the nearest listing, and
    that listing's postal code) and the grid geometry the page needs.
    """
    lat, lon = df[km.LAT_COL].to_numpy(float), df[km.LON_COL].to_numpy(float)
    lat_min, lat_max, lon_min, lon_max = lat.min(), lat.max(), lon.min(), lon.max()
    lat_mid = (lat_min + lat_max) / 2
    dlat = cell_km / km.KM_PER_DEG_LAT
    dlon = cell_km / (km.KM_PER_DEG_LAT * np.cos(np.radians(lat_mid)))
    ni = int(np.ceil((lat_max - lat_min) / dlat)) + 1
    nj = int(np.ceil((lon_max - lon_min) / dlon)) + 1
    if ni * nj > MAX_CELLS * 6:
        raise SystemExit(f"A {cell_km} km grid over this area would have {ni * nj:,} cells; use a larger --cell-km.")
    I, J = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    cells = pd.DataFrame({"i": I.ravel(), "j": J.ravel()})
    cells["latitude"] = lat_min + cells["i"] * dlat
    cells["longitude"] = lon_min + cells["j"] * dlon

    lon_mid = (lon_min + lon_max) / 2
    tree = cKDTree(xy_km(lat, lon, lat_mid, lon_mid))
    d, nearest = tree.query(xy_km(cells["latitude"], cells["longitude"], lat_mid, lon_mid))
    cells["d_km"] = d
    cells["zip"] = df[km.ZIP_COL].to_numpy()[nearest]
    cells = cells[cells["d_km"] <= max_dist_km]

    if boundary_rings is not None:
        cells = cells[inside_boundary(boundary_rings, cells["longitude"], cells["latitude"])]
    cells = cells.reset_index(drop=True)
    if not len(cells):
        raise SystemExit("No grid cells left. Raise --max-dist-km or check the --boundary file.")
    if len(cells) > MAX_CELLS:
        raise SystemExit(f"{len(cells):,} cells is too many for one web page; use a larger --cell-km.")
    grid = {"lat0": float(lat_min), "lon0": float(lon_min), "dlat": float(dlat), "dlon": float(dlon), "cell_km": cell_km}
    print(f"Grid: {len(cells):,} cells of {cell_km * 1000:.0f} m within {max_dist_km} km of a listing.")
    return cells, grid


def cell_key(lat, lon):
    return pd.Series([f"{a:.6f},{b:.6f}" for a, b in zip(lat, lon)])


def score_cells(cells, rescore=False):
    """Add SiteLine scores to the cells (cached in data/model/grid_scores.csv) and drop cells outside SiteLine's region."""
    cells = cells.copy()
    cells["key"] = cell_key(cells["latitude"], cells["longitude"]).to_numpy()
    cache_cols = ["key", "tract_geoid", "in_scope", *km.SCORE_COLUMNS, "total_score"]
    cache = pd.DataFrame(columns=cache_cols)
    if GRID_CACHE.exists() and not rescore:
        cache = pd.read_csv(GRID_CACHE, dtype={"key": str, "tract_geoid": str})
    todo = cells[~cells["key"].isin(cache["key"])]
    print(f"SiteLine scores: {len(cells) - len(todo):,} cells cached, {len(todo):,} to score"
          + (" (this can take a few minutes the first time)." if len(todo) else "."))
    if len(todo):
        siteline = import_siteline()
        batch = pd.DataFrame({"Address": "grid cell " + todo["key"], km.LAT_COL: todo["latitude"].to_numpy(),
                              km.LON_COL: todo["longitude"].to_numpy(), "key": todo["key"].to_numpy()})
        scored = score_rentals(batch, siteline)
        new = scored[[c for c in cache_cols if c in scored.columns]]
        cache = pd.concat([cache, new], ignore_index=True) if len(cache) else new
        cache = cache.drop_duplicates(subset="key", keep="last")
        km.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cache.to_csv(GRID_CACHE, index=False)
    out = cells.merge(cache[cache_cols], on="key", how="left")
    in_scope = out["in_scope"].astype(str).str.lower().eq("true")
    print(f"  {int((~in_scope).sum()):,} cells are outside SiteLine's 11-county region and were dropped.")
    out = out[in_scope].reset_index(drop=True)
    for c in km.SCORE_COLUMNS + ["total_score"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if not len(out):
        raise SystemExit("No scored cells left.")
    return out


# ---------------------------------------------------------------------------
# Predictions for every unit on the lattice
# ---------------------------------------------------------------------------
def snap(value, options):
    options = np.asarray(options, float)
    return int(np.argmin(np.abs(options - value)))


def make_lattice(df, model, args):
    """The units to precompute: bedrooms x floor area x property type (a single dummy value for anything unused)."""
    uses_units, uses_type = model.w_unit > 0, model.w_type > 0
    if not uses_units and not uses_type:
        print("Note: the tuned model gives bedrooms, size and type zero weight, so the map has no unit controls. "
              "(If the data has no beds/sqft yet, rescrape, then rerun knn_model.py train.)")
    beds = list(range(0, args.max_beds + 1)) if uses_units else [None]
    sqft = list(range(args.sqft_min, args.sqft_max + 1, args.sqft_step)) if uses_units else [None]
    types = [None]
    if uses_type:
        counts = df[km.TYPE_COL].dropna().value_counts()
        common = [t for t in counts.index if counts[t] >= MIN_TYPE_LISTINGS][:MAX_TYPES]
        types = common or [None]
    typical = {}
    if uses_units:
        for b in beds:
            typical[str(b)] = int(round(model.sqft_by_beds_.get(float(b), model.sqft_med_), -1))
    default_beds = 0
    if uses_units:
        mode = df[km.BEDS_COL].round().mode()
        default_beds = snap(float(mode.iloc[0]) if len(mode) else 1, beds)
    default_sqft = snap(typical[str(beds[default_beds])], sqft) if uses_units else 0
    return {"types": types, "beds": beds, "sqft": sqft, "typical": typical, "uses_units": uses_units,
            "uses_type": bool(uses_type and types != [None]),
            "default": {"type": 0, "beds": default_beds, "sqft": default_sqft}}


def predict_lattice(model, cells, lat):
    """Predicted rent (whole dollars) for every cell, for every combination in the lattice."""
    X = pd.DataFrame({km.LAT_COL: cells["latitude"].to_numpy(), km.LON_COL: cells["longitude"].to_numpy(),
                      km.ZIP_COL: cells["zip"].to_numpy()})
    for c in km.SCORE_COLUMNS:
        X[c] = cells[c].to_numpy(float)
    X[km.BEDS_COL], X[km.SQFT_COL], X[km.TYPE_COL] = np.nan, np.nan, None
    out = []
    total = len(lat["types"]) * len(lat["beds"]) * len(lat["sqft"])
    print(f"Predicting {len(cells):,} cells for {total} unit combinations...")
    for t in lat["types"]:
        X[km.TYPE_COL] = t
        for b in lat["beds"]:
            X[km.BEDS_COL] = np.nan if b is None else float(b)
            for s in lat["sqft"]:
                X[km.SQFT_COL] = np.nan if s is None else float(s)
                out.append(np.rint(model.predict(X)).astype(int).tolist())
                if len(out) % 20 == 0:
                    print(f"  {len(out)}/{total}")
    return out


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------
def cv_mae_note(saved, uses_units):
    table = saved.get("cv_mean_by_model", {})
    for name in (["all_robust", "units_robust"] if uses_units else []) + ["full_robust", "location_zip_robust"]:
        if name in table:
            return round(float(table[name]["mae"]))
    return None


def choose_basemap(args):
    """Which map tiles the page uses. CARTO needs a free API key (https://carto.com/basemaps/apikey/); OSM needs none."""
    if args.public:                                       # a public page is readable by everyone, so it never carries a key
        if args.carto_key:
            raise SystemExit("--public pages can be read by anyone, so they must not contain an API key: drop --carto-key.")
        if args.basemap == "carto":
            raise SystemExit("--public uses OpenStreetMap tiles (no key). Drop --basemap carto.")
        if os.environ.get("CARTO_API_KEY"):
            print("Note: CARTO_API_KEY is set in your environment; --public ignores it so that it cannot end up in the page.")
        return {"provider": "none" if args.basemap == "none" else "osm", "key": None}
    key = args.carto_key or os.environ.get("CARTO_API_KEY")
    provider = args.basemap
    if provider == "auto":
        provider = "carto" if key else "osm"
    if provider == "carto" and not key:
        raise SystemExit("--basemap carto needs a key: pass --carto-key YOUR_KEY or set CARTO_API_KEY "
                         "(free, no account: https://carto.com/basemaps/apikey/).")
    if provider == "osm" and args.basemap == "auto":
        print("Basemap: OpenStreetMap tiles (no key needed). For CARTO's cleaner light/dark maps, get a free key at "
              "https://carto.com/basemaps/apikey/ and run with --carto-key YOUR_KEY.")
    return {"provider": provider, "key": key if provider == "carto" else None}


def census_income(df):
    """
    Median household income by census tract (a Series indexed by 11-digit tract GEOID). Uses the whole census table in
    data/census_income/ so that cells with no listing in their tract still get an income; falls back to the tracts that
    appear in the listings.
    """
    out = pd.Series(dtype=float)
    if "median_household_income" in df.columns and "tract_geoid" in df.columns:
        out = df.dropna(subset=["median_household_income", "tract_geoid"]).groupby("tract_geoid")["median_household_income"].first().astype(float)
    try:
        from income import load_latest_income
        table, _ = load_latest_income()
        full = table.dropna(subset=["median_household_income"]).drop_duplicates("tract_geoid").set_index("tract_geoid")["median_household_income"].astype(float)
        out = pd.concat([out, full[~full.index.isin(out.index)]])
    except Exception as e:
        print(f"Note: could not read the full census income table ({type(e).__name__}: {e}). "
              f"The affordability view will only cover tracts that contain a listing.")
    return out


def listing_name(url):
    """'.../GA/Atlanta/Crystal-Townhomes/apartment/24734600' -> 'Crystal Townhomes' (used to search by property name)."""
    m = re.search(r"/([^/]+)/[A-Za-z-]+/\d+/?(?:[?#].*)?$", str(url))
    return m.group(1).replace("-", " ") if m else ""


def build_payload(df, cells, grid, lattice, preds, model, params, saved, basis, args):
    flat = np.asarray(preds).ravel()
    lo, hi = np.percentile(flat, DOMAIN_PERCENTILES)
    income = census_income(df)
    cell_income = cells["tract_geoid"].map(income)
    listing_income = df["tract_geoid"].map(income) if "tract_geoid" in df.columns else pd.Series(np.nan, index=df.index)

    def listing_value(col, digits=None):
        s = pd.to_numeric(df[col], errors="coerce") if col in (km.BEDS_COL, km.SQFT_COL) else df[col]
        return [None if pd.isna(v) else (round(float(v), digits) if digits is not None else v) for v in s]

    lo_rent, hi_rent = df.attrs["rent_range"]
    return {
        "meta": {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "n_listings": int(len(df)),
                 "unit_basis": basis, "rent_range": [lo_rent, hi_rent], "max_dist_km": args.max_dist_km,
                 "threshold": km.AFFORDABILITY_THRESHOLD, "cv_mae": cv_mae_note(saved, lattice["uses_units"]),
                 "aggregate": model.aggregate, "k": int(model.n_neighbors), "basemap": args.basemap_config,
                 "outline_source": args.outline_source, "tracts": bool(getattr(args, "tracts", None)), "geocoder": not args.no_geocoder,
                 "public": bool(args.public), "project": PROJECT_NAME, "lab": LAB_NAME, "creator": CREATOR, "contributors": CONTRIBUTORS},
        "grid": grid,
        "cells": {
            "lat": [round(v, 6) for v in cells["latitude"]], "lon": [round(v, 6) for v in cells["longitude"]],
            "d": [round(float(v), 1 if args.public else 3) for v in cells["d_km"]], "score": [None if pd.isna(v) else round(float(v), 1) for v in cells["total_score"]],
            "zip": [None if (args.public or pd.isna(v)) else str(v) for v in cells["zip"]],
            "tract": [None if pd.isna(v) else str(v) for v in cells["tract_geoid"]],
            "income": [None if pd.isna(v) else int(v) for v in cell_income],
        },
        "lattice": lattice,
        "domain": [round(float(lo), -1), round(float(hi), -1)],
        "pred": preds,
        "outline": outline_for_page(args.outline_rings) if args.outline_rings else None,
        "tracts": getattr(args, "tracts", None),
        "listings": ({k: [] for k in ("lat", "lon", "rent", "addr", "beds", "sqft", "type", "score", "name", "income")} if args.public else {
            "lat": [round(float(v), 5) for v in df[km.LAT_COL]], "lon": [round(float(v), 5) for v in df[km.LON_COL]],
            "rent": [int(round(v)) for v in df[km.TARGET]], "addr": [str(v) for v in df["Address"]],
            "beds": listing_value(km.BEDS_COL, 1), "sqft": listing_value(km.SQFT_COL, 0),
            "type": listing_value(km.TYPE_COL),
            "score": ([None if pd.isna(v) else round(float(v), 1) for v in pd.to_numeric(df["total_score"], errors="coerce")]
                      if "total_score" in df.columns else [None] * len(df)),
            "name": [listing_name(u) for u in df["url"]],
            "income": [None if pd.isna(v) else int(v) for v in listing_income],
        }),
    }


def write_html(payload, path):
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    page = HTML_TEMPLATE.replace("/*__DATA__*/", text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    print(f"\nWrote {path} ({path.stat().st_size / 1e6:.1f} MB). Open it in a browser.")


def check_public_page(path, df):
    """After a --public build: make sure no listing address or API key text made it into the page."""
    text = path.read_text(encoding="utf-8")
    leaked = [a for a in df["Address"].astype(str) if len(a) >= 8 and a in text]
    key = os.environ.get("CARTO_API_KEY")
    if leaked or (key and key in text):
        path.unlink()
        raise SystemExit(f"Refusing to keep {path}: it contains " + (f"{len(leaked)} listing addresses" if leaked else "an API key") + ".")
    print(f"Checked {path.name}: no listing addresses and no API key in the page. Safe to publish.")


HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Atlanta Predicted Rent</title>
<meta name="author" content="Songjie Pu, Joshua Velasquez, Yanqi Li">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<style>
:root {
  --bg: #f4f4f1; --panel: #ffffff; --ink: #1b1b1a; --muted: #5c5c57; --line: #dcdcd5; --accent: #2a78d6;
  --accent-ink: #ffffff; --chip: #eef0f3; --ring: #ffffff; --pin: #d9382a; --soft: #f7f7f4;
  --shadow: 0 2px 12px rgba(0, 0, 0, .16); --side: 380px;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #161614; --panel: #1f1f1c; --ink: #ecece8; --muted: #a4a49c; --line: #34342f; --accent: #5598e7;
    --accent-ink: #0d1b2e; --chip: #2a2a26; --ring: #1f1f1c; --pin: #ff6b58; --soft: #252522; --shadow: 0 2px 12px rgba(0, 0, 0, .5);
  }
}
:root[data-theme="dark"] {
  --bg: #161614; --panel: #1f1f1c; --ink: #ecece8; --muted: #a4a49c; --line: #34342f; --accent: #5598e7;
  --accent-ink: #0d1b2e; --chip: #2a2a26; --ring: #1f1f1c; --pin: #ff6b58; --soft: #252522; --shadow: 0 2px 12px rgba(0, 0, 0, .5);
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
html, body { height: 100%; margin: 0; }
body { background: var(--bg); color: var(--ink); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; overflow: hidden; }
#map { position: absolute; top: 0; bottom: 0; right: 0; left: var(--side); background: var(--bg); }

/* sidebar */
#panel { position: absolute; z-index: 800; top: 0; bottom: 0; left: 0; width: var(--side); display: flex; flex-direction: column;
  background: var(--panel); border-right: 1px solid var(--line); box-shadow: var(--shadow); }
.top { padding: 16px 18px 12px; border-bottom: 1px solid var(--line); }
.eyebrow { font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin-bottom: 2px; }
.head { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; }
h1 { font-size: 20px; line-height: 1.2; margin: 0; letter-spacing: -.01em; }
.sub { margin: 6px 0 0; color: var(--muted); font-size: 13px; }
.scroll { flex: 1; overflow-y: auto; padding: 14px 18px 18px; }
.sec { margin: 0 0 16px; }
fieldset { border: 0; margin: 0 0 16px; padding: 0; min-width: 0; }
legend, .label { font-weight: 600; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); padding: 0; margin-bottom: 7px; display: block; }

/* controls */
.seg { display: flex; flex-wrap: wrap; gap: 6px; }
.seg label { position: relative; }
.seg input { position: absolute; opacity: 0; inset: 0; width: 100%; height: 100%; margin: 0; cursor: pointer; }
.seg span { display: block; min-width: 44px; text-align: center; padding: 7px 10px; border-radius: 8px; background: var(--chip); border: 1px solid var(--line); cursor: pointer; }
.seg input:checked + span { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); font-weight: 600; }
.seg input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: 2px; }
.seg.tabs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; padding: 4px; background: var(--chip); border: 1px solid var(--line); border-radius: 11px; }
.seg.tabs span { min-width: 0; padding: 8px 4px; border: 0; background: transparent; font-size: 13px; line-height: 1.2; }
.seg.tabs input:checked + span { background: var(--panel); color: var(--ink); box-shadow: 0 1px 3px rgba(0, 0, 0, .18); font-weight: 650; }
.row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
output { font-weight: 600; font-variant-numeric: tabular-nums; }
input[type=range] { width: 100%; accent-color: var(--accent); }
select, input[type=search] { width: 100%; padding: 8px 10px; border-radius: 8px; border: 1px solid var(--line); background: var(--soft); color: var(--ink); font: inherit; }
input[type=search] { flex: 1; min-width: 0; }
button { font: inherit; color: var(--ink); background: var(--chip); border: 1px solid var(--line); border-radius: 8px; padding: 7px 11px; cursor: pointer; }
button:hover { border-color: var(--accent); }
button.primary { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); font-weight: 600; }
button.pin-btn { width: 100%; display: flex; align-items: center; justify-content: center; gap: 8px; margin-top: 8px; font-weight: 600; }
button.pin-btn svg { width: 14px; height: 18px; fill: var(--pin); }
button.pin-btn[aria-pressed="true"] { background: var(--pin); border-color: var(--pin); color: #fff; }
button.pin-btn[aria-pressed="true"] svg { fill: #fff; }
button.icon { padding: 5px 10px; white-space: nowrap; flex: none; }
button:focus-visible, select:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.small { font-size: 12px; color: var(--muted); }
.check { display: flex; gap: 8px; align-items: center; margin: 5px 0; }
hr { border: 0; border-top: 1px solid var(--line); margin: 14px 0; }
.cands { display: flex; flex-direction: column; gap: 4px; margin: 6px 0 0; }
.cands button { text-align: left; }

/* the headline number */
#summary { font-size: 14px; margin: 0 0 16px; padding: 11px 13px; background: var(--soft); border: 1px solid var(--line); border-left: 3px solid var(--accent); border-radius: 8px; }
#summary b { font-variant-numeric: tabular-nums; }
#found { margin: 10px 0 0; padding: 11px 13px; border: 1px solid var(--line); border-radius: 10px; background: var(--soft); }
#found .big { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1.2; }
#found .verdict { font-weight: 600; margin: 5px 0; }
#found .head button { padding: 2px 8px; }
details { margin: 0 0 12px; border-top: 1px solid var(--line); padding-top: 10px; }
summary { cursor: pointer; font-weight: 600; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
details > :not(summary) { margin-top: 8px; }
details p { font-size: 12px; color: var(--muted); margin-bottom: 0; }

.credit { padding: 9px 18px; border-top: 1px solid var(--line); font-size: 11px; color: var(--muted); line-height: 1.45; }
.credit button { padding: 0; border: 0; background: none; color: var(--accent); font-size: 11px; text-decoration: underline; }
.modal-backdrop { position: fixed; inset: 0; z-index: 2000; display: flex; align-items: center; justify-content: center; padding: 16px;
  background: rgba(10, 10, 9, .6); backdrop-filter: blur(4px); }
.modal { position: relative; width: min(940px, 100%); max-height: calc(100vh - 32px); display: grid; grid-template-columns: 5fr 7fr; overflow: hidden;
  background: var(--panel); color: var(--ink); border-radius: 20px; box-shadow: 0 16px 56px rgba(0, 0, 0, .45); }
.modal:focus { outline: none; }
.m-hero { padding: 34px 30px; color: #fff; display: flex; flex-direction: column; gap: 14px; overflow-y: auto;
  background: radial-gradient(120% 90% at 0% 0%, #2f7fe0 0%, rgba(47, 127, 224, 0) 60%), linear-gradient(165deg, #12376a 0%, #0c2347 100%); }
.m-eyebrow { font-size: 11px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; opacity: .8; }
.m-hero h2 { font-size: 30px; line-height: 1.12; letter-spacing: -.02em; margin: 0; }
.m-hero .m-lead { margin: 0; font-size: 14px; line-height: 1.55; opacity: .92; }
.m-pills { display: flex; flex-wrap: wrap; gap: 6px; }
.m-pills span { padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; background: rgba(255, 255, 255, .14); border: 1px solid rgba(255, 255, 255, .25); }
.m-team { margin: auto 0 0; padding-top: 14px; border-top: 1px solid rgba(255, 255, 255, .25); font-size: 13px; line-height: 1.5; }
.m-body { padding: 28px 28px 0; overflow-y: auto; max-height: calc(100vh - 32px); }
.m-foot { position: sticky; bottom: 0; margin: 4px -28px 0; padding: 12px 28px 20px; background: var(--panel); border-top: 1px solid var(--line); }
.m-body h3 { font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin: 0 0 10px; }
.m-body h3.warn { color: var(--pin); }
.m-card { padding: 9px 12px; margin: 0 0 8px; border-radius: 10px; background: var(--soft); border: 1px solid var(--line); border-left: 3px solid var(--pin); font-size: 13px; line-height: 1.45; }
.m-card b { display: block; margin-bottom: 1px; }
.m-card.site { border-left-color: var(--accent); margin: 0 40px 18px 0; }
.m-card a, .m-hero a { color: var(--accent); }
.m-card code { font-size: 12px; }
.modal-x { position: absolute; top: 12px; right: 12px; z-index: 2; width: 34px; height: 34px; padding: 0; font-size: 20px; line-height: 1; border-radius: 9px; }
.modal .go { width: 100%; padding: 12px; font-size: 15px; border-radius: 12px; }
@media (max-width: 760px) {
  .modal { grid-template-columns: 1fr; overflow-y: auto; }
  .m-hero { padding: 24px 22px; overflow: visible; }
  .m-hero h2 { font-size: 24px; margin-right: 40px; }
  .m-body { max-height: none; overflow: visible; padding: 22px 22px 0; }
  .m-foot { margin: 4px -22px 0; padding: 12px 22px 16px; }
  .m-card.site { margin-right: 0; }
}
/* legend card floating on the map */
#legend { position: absolute; z-index: 700; right: 12px; bottom: 28px; width: 290px; padding: 11px 13px; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow); }
#legend-title { font-size: 12px; font-weight: 650; margin-bottom: 7px; }
.bar { height: 12px; border-radius: 6px; border: 1px solid var(--line); }
.ticks { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; margin-top: 3px; }
#legend-note { font-size: 11px; color: var(--muted); margin-top: 6px; line-height: 1.35; }

/* hover card */
#tip { position: fixed; z-index: 900; pointer-events: none; display: none; max-width: 300px;
  background: var(--panel); color: var(--ink); border-radius: 10px; box-shadow: var(--shadow); padding: 10px 12px; font-size: 13px; border: 1px solid var(--line); }
#tip .big { font-size: 18px; font-weight: 700; font-variant-numeric: tabular-nums; }
#tip .muted { color: var(--muted); font-size: 12px; }

/* Leaflet to match the theme */
.leaflet-container { font: inherit; }
.leaflet-control-attribution { font-size: 10px; background: color-mix(in srgb, var(--panel) 80%, transparent) !important; color: var(--muted); }
.leaflet-control-attribution a { color: var(--accent); }
.leaflet-bar a, .leaflet-bar a:hover { background: var(--panel); color: var(--ink); border-bottom-color: var(--line); }
.leaflet-bar { border: 1px solid var(--line); box-shadow: var(--shadow); }
.leaflet-tooltip { background: var(--panel); color: var(--ink); border-color: var(--line); box-shadow: var(--shadow); }
.leaflet-tooltip-top:before { border-top-color: var(--panel); } .leaflet-tooltip-bottom:before { border-bottom-color: var(--panel); }
.leaflet-tooltip-left:before { border-left-color: var(--panel); } .leaflet-tooltip-right:before { border-right-color: var(--panel); }
#map.dark-osm .leaflet-tile-pane { filter: invert(1) hue-rotate(180deg) brightness(.92) contrast(.9); }
#map.pinning, #map.pinning .leaflet-interactive, #map.pinning .leaflet-grab { cursor: crosshair !important; }
.pin-icon { background: none; border: 0; }
.pin-icon svg { display: block; filter: drop-shadow(0 2px 2px rgba(0, 0, 0, .35)); cursor: grab; }
#pin-hint { position: absolute; z-index: 750; top: 14px; left: 50%; transform: translateX(-50%); padding: 8px 14px; border-radius: 999px;
  background: var(--pin); color: #fff; font-weight: 600; box-shadow: var(--shadow); pointer-events: none; }

@media (max-width: 760px) {
  :root { --side: 0px; }
  #map { bottom: 52%; }
  #panel { top: auto; width: auto; right: 0; height: 52%; border-right: 0; border-top: 1px solid var(--line); border-radius: 14px 14px 0 0; }
  #legend { top: 8px; bottom: auto; left: 8px; right: 56px; width: auto; padding: 8px 10px; }
  #legend-note { display: none; }
}
</style>
</head>
<body>
<div id="map" role="application" aria-label="Map of predicted rent in Atlanta"></div>
<div id="pin-hint" hidden>Click the map to drop the pin · Esc to cancel</div>
<div id="legend" aria-label="Legend">
  <div id="legend-title"></div>
  <div class="bar" id="bar"></div>
  <div class="ticks"><span id="t-lo"></span><span id="t-mid"></span><span id="t-hi"></span></div>
  <div id="legend-note"></div>
</div>
<aside id="panel" aria-label="Controls">
  <div class="top">
    <div class="eyebrow">Emory AI.DataLab · Atlanta rental affordability</div>
    <div class="head">
      <h1 id="title">Predicted rent in Atlanta</h1>
      <button type="button" class="icon" id="theme" aria-label="Switch to dark mode">Dark</button>
    </div>
    <p class="sub" id="subtitle">Estimated starting monthly rent for a unit like the one below, at every point on the map.</p>
  </div>
  <div class="scroll">
    <div class="sec">
      <span class="label">Map shows</span>
      <div class="seg tabs" id="metric" role="radiogroup" aria-label="Map metric">
        <label><input type="radio" name="metric" value="rent" checked><span>Rent</span></label>
        <label><input type="radio" name="metric" value="share"><span>Rent burden</span></label>
        <label><input type="radio" name="metric" value="score"><span>LIHTC score</span></label>
      </div>
    </div>

    <p id="summary" aria-live="polite"></p>

    <form id="search-form" class="sec" role="search" autocomplete="off">
      <label class="label" for="q">Check a specific place</label>
      <div class="row"><input type="search" id="q" placeholder="Address, name, or lat, lon"><button type="submit" class="primary">Search</button></div>
      <button type="button" class="pin-btn" id="pin" aria-pressed="false"><svg viewBox="0 0 24 32" aria-hidden="true"><path d="M12 1C6 1 1.5 5.5 1.5 11.3 1.5 19 12 31 12 31s10.5-12 10.5-19.7C22.5 5.5 18 1 12 1zm0 15.3a5 5 0 1 1 0-10 5 5 0 0 1 0 10z"/></svg><span id="pin-label">Drop a pin on the map</span></button>
      <div class="small" id="search-note" style="margin-top:6px"></div>
      <div id="results" class="cands" aria-live="polite"></div>
      <div id="found" hidden></div>
    </form>

    <div id="unit-controls">
      <fieldset id="beds-set">
        <legend>Bedrooms</legend>
        <div class="seg" id="beds"></div>
      </fieldset>
      <fieldset id="sqft-set">
        <legend>Floor area</legend>
        <div class="row"><output id="sqft-out" for="sqft"></output><button type="button" id="typical"></button></div>
        <input type="range" id="sqft" aria-label="Floor area in square feet">
      </fieldset>
      <fieldset id="type-set">
        <legend>Property type</legend>
        <select id="type" aria-label="Property type"></select>
      </fieldset>
    </div>
    <p class="small" id="no-units" hidden>The tuned model does not use bedrooms, size or property type, so the map shows one prediction per location.</p>

    <fieldset id="thr-set" hidden>
      <legend>Unaffordable above</legend>
      <div class="row"><output id="thr-out" for="thr"></output><span class="small">of median household income</span></div>
      <input type="range" id="thr" min="10" max="70" step="1" value="30" aria-label="Share of income above which rent counts as unaffordable">
      <label class="check"><input type="checkbox" id="only"> Color only the unaffordable areas</label>
    </fieldset>

    <details open>
      <summary>Map display</summary>
      <label class="check"><input type="checkbox" id="show-listings"> <span id="listings-label">Show listings as dots</span></label>
      <div class="small" id="listings-note" hidden></div>
      <label class="check" id="tract-row" hidden><input type="checkbox" id="show-tracts" checked> Census tract outlines</label>
      <label class="check" id="outline-row" hidden><input type="checkbox" id="show-outline" checked> City limits outline</label>
      <label class="check"><input type="checkbox" id="rescale"> Rescale colors to this view</label>
      <label class="small" for="opacity">Cell opacity</label>
      <input type="range" id="opacity" min="30" max="100" value="80">
      <button type="button" id="download">Download this view (CSV)</button>
    </details>
    <details>
      <summary>About these numbers</summary>
      <p id="about"></p>
    </details>
  </div>
  <div class="credit"><span id="credit-line"></span><br><button type="button" id="open-about">About, credits &amp; disclaimers</button></div>
</aside>
<div id="welcome" class="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="w-title" hidden>
  <div class="modal" id="w-box" tabindex="-1">
    <button type="button" class="modal-x" id="w-x" aria-label="Close">&times;</button>
    <div class="m-hero">
      <div class="m-eyebrow" id="w-lab"></div>
      <h2 id="w-title"></h2>
      <p class="m-lead" id="w-desc"></p>
      <div class="m-pills" aria-hidden="true"><span>Rent</span><span>Rent burden</span><span>LIHTC score</span></div>
      <p class="m-team" id="w-team"></p>
    </div>
    <div class="m-body">
      <div class="m-card site"><b>LIHTC scores: SiteLine</b><span id="w-siteline"></span></div>
      <h3 class="warn">Before you use the numbers</h3>
      <div id="w-disc"></div>
      <div class="m-foot"><button type="button" class="primary go" id="w-go">Got it, explore the map</button></div>
    </div>
  </div>
</div>
<div id="tip" role="tooltip"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script id="data" type="application/json">/*__DATA__*/</script>
<script>
(function () {
  'use strict';
  const D = JSON.parse(document.getElementById('data').textContent);
  const $ = (id) => document.getElementById(id);
  const C = D.cells, LI = D.listings, LAT = D.lattice, META = D.meta;
  const PUBLIC = !!META.public;                      // public builds carry no individual listings
  const N = C.lat.length;
  const NB = LAT.beds.length, NS = LAT.sqft.length;
  const fmt = (v) => '$' + Math.round(v).toLocaleString('en-US');
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : 'Unknown');
  const bedsLabel = (b) => (b === 0 ? 'Studio' : String(b));
  const bedsText = (b) => (b === 0 ? 'studio' : b + '-bed');

  // ---- state: indices into the lattice
  const st = { t: LAT.default.type, b: LAT.default.beds, s: LAT.default.sqft, listings: false, rescale: false, opacity: 0.8,
               metric: 'rent', thr: META.threshold, only: false };
  let found = null;                                                   // the property chosen in the search box
  const comboIndex = () => (st.t * NB + st.b) * NS + st.s;

  // ---- colour: one-hue blue ramp (light = lower). In dark mode the ramp flips so higher is lighter.
  const STOPS = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
  const RGB = STOPS.map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));
  const isDark = () => {
    const forced = document.documentElement.getAttribute('data-theme');
    return forced ? forced === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
  };
  function rampRGB(t, stops) {
    const R = stops || RGB;
    t = Math.min(1, Math.max(0, t));
    if (isDark()) t = 1 - t;
    const x = t * (R.length - 1), i = Math.min(R.length - 2, Math.floor(x)), f = x - i;
    return R[i].map((v, k) => Math.round(v + (R[i + 1][k] - v) * f));
  }
  // LIHTC score: its own one-hue (green) ramp, so it is not mistaken for the blue rent map
  const SC_RGB = ['#d9f0e6', '#a6dfc7', '#66c7a4', '#2fa37f', '#1d7d61', '#125c47', '#0a3d30'].map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));
  const SCORES = C.score.filter((v) => v !== null).sort((a, b) => a - b);
  const SC_MAX = Math.max(5, Math.ceil(SCORES.length ? SCORES[Math.floor(0.98 * (SCORES.length - 1))] : 46));      // top of the scale: 98th percentile
  const ramp = (t) => 'rgb(' + rampRGB(t).join(',') + ')';
  // Cells far from any listing are mixed toward the map background (paler = less evidence). Cells stay opaque and the
  // whole layer is made translucent with one opacity, so neighbouring cells never show darker seams where they touch.
  function mixBg(rgb, frac) {
    const bg = isDark() ? [26, 26, 24] : [242, 240, 236];
    return 'rgb(' + rgb.map((v, k) => Math.round(v + (bg[k] - v) * frac)).join(',') + ')';
  }
  // Affordability colours: diverging, centred on the threshold. Blue = affordable, neutral = at the threshold,
  // red = unaffordable. The scale spans +/- 20 percentage points around the threshold (less when the threshold is low).
  const hexRGB = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const DIV = { light: ['#1f5fb4', '#8fb9ee', '#ebe9e4', '#f0a99c', '#b3281a'].map(hexRGB),
                dark: ['#4d8fe6', '#2f5c96', '#34342f', '#8e4a40', '#e8644f'].map(hexRGB) };
  const NO_INCOME = { light: [196, 196, 190], dark: [84, 84, 79] };
  const halfRange = () => Math.min(0.2, st.thr);
  function shareRGB(share) {
    const S = isDark() ? DIV.dark : DIV.light;
    let u = Math.min(1, Math.max(-1, (share - st.thr) / halfRange()));
    if (st.only && u <= 0) return S[2];
    const x = (u + 1) / 2 * (S.length - 1), i = Math.min(S.length - 2, Math.floor(x)), f = x - i;
    return S[i].map((v, k) => Math.round(v + (S[i + 1][k] - v) * f));
  }
  const noIncomeRGB = () => (isDark() ? NO_INCOME.dark : NO_INCOME.light);
  const pct = (v) => Math.round(v * 100) + '%';
  const scoreRGB = (v) => (v === null || v === undefined ? noIncomeRGB() : rampRGB(v / SC_MAX, SC_RGB));

  // ---- map
  const map = L.map('map', { zoomControl: false, preferCanvas: true, minZoom: 9, maxZoom: 18 });
  L.control.zoom({ position: 'topright' }).addTo(map);
  // Basemap tiles. CARTO's free basemaps need a (free) API key; without one they stamp "API KEY REQUIRED" on every tile.
  const BM = META.basemap;
  function makeTiles(kind) {
    if (BM.provider === 'carto') {
      return L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/' + (kind === 'dark' ? 'dark_all' : 'light_all') + '/{z}/{x}/{y}{r}.png?key=' + encodeURIComponent(BM.key),
        { attribution: '&copy; OpenStreetMap contributors &copy; CARTO', subdomains: 'abcd', maxZoom: 19 });
    }
    if (BM.provider === 'osm') {
      return L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '&copy; OpenStreetMap contributors', maxZoom: 19 });
    }
    return null;
  }
  const tilesLight = makeTiles('light');
  const tiles = { light: tilesLight, dark: BM.provider === 'osm' ? tilesLight : makeTiles('dark') };   // OSM has one style
  let activeTiles = null;
  function applyBasemap() {
    $('map').classList.toggle('dark-osm', isDark() && BM.provider === 'osm');       // OSM tiles are darkened with a CSS filter
    const want = isDark() ? tiles.dark : tiles.light;
    if (!want || activeTiles === want) return;
    if (activeTiles) map.removeLayer(activeTiles);
    activeTiles = want;
    want.addTo(map);
  }

  const G = D.grid, hLat = G.dlat / 2 * 1.03, hLon = G.dlon / 2 * 1.03;
  map.createPane('cells');
  map.getPane('cells').style.zIndex = 380;
  const canvas = L.canvas({ pane: 'cells', padding: 0.3 });
  const rects = new Array(N), lookup = new Map();
  for (let k = 0; k < N; k++) {
    rects[k] = L.rectangle([[C.lat[k] - hLat, C.lon[k] - hLon], [C.lat[k] + hLat, C.lon[k] + hLon]],
      { renderer: canvas, stroke: false, fillOpacity: 1, interactive: false }).addTo(map);
    lookup.set(Math.round((C.lat[k] - G.lat0) / G.dlat) * 100000 + Math.round((C.lon[k] - G.lon0) / G.dlon), k);
  }
  const EXT = { s: Math.min(...C.lat) - hLat, w: Math.min(...C.lon) - hLon, n: Math.max(...C.lat) + hLat, e: Math.max(...C.lon) + hLon };
  map.fitBounds([[EXT.s, EXT.w], [EXT.n, EXT.e]], { padding: [20, 20] });

  // city-limits outline (drawn above the cells; its color follows the theme)
  const outlineLayer = L.layerGroup();
  const outlineLines = (D.outline || []).map((ring) => L.polyline(ring, { weight: 2.5, opacity: 0.9, fill: false, interactive: false }));
  outlineLines.forEach((l) => outlineLayer.addLayer(l));
  if (outlineLines.length) { $('outline-row').hidden = false; }
  // census tract outlines (thin, under the city outline)
  const tractLayer = L.layerGroup();
  const tractLines = [];
  (D.tracts || []).forEach((t) => t.r.forEach((ring) => { const l = L.polyline(ring, { weight: 1, opacity: 0.6, fill: false, interactive: false }); tractLines.push(l); tractLayer.addLayer(l); }));
  if (tractLines.length) { $('tract-row').hidden = false; }
  function drawOutline() {
    tractLines.forEach((l) => l.setStyle({ color: isDark() ? '#b9b9b0' : '#55554f' }));
    if ($('show-tracts').checked && tractLines.length) { tractLayer.addTo(map); } else { map.removeLayer(tractLayer); }
    outlineLines.forEach((l) => l.setStyle({ color: isDark() ? '#f2f2ee' : '#1b1b1a' }));
    map.removeLayer(outlineLayer);                                  // re-added last so the city outline stays on top
    if ($('show-outline').checked && outlineLines.length) { outlineLayer.addTo(map); }
  }

  // listings layer (built once)
  const listingLayer = L.layerGroup();
  const dots = [];
  for (let k = 0; k < LI.lat.length; k++) {
    const bits = [];
    if (LI.beds[k] !== null) bits.push(bedsText(LI.beds[k]));
    if (LI.sqft[k] !== null) bits.push(Math.round(LI.sqft[k]).toLocaleString('en-US') + ' sq ft');
    if (LI.type[k]) bits.push(LI.type[k]);
    const m = L.circleMarker([LI.lat[k], LI.lon[k]], { radius: 4, weight: 1.5, fillOpacity: 0.95, color: '#fff' });
    const iscore = LI.score[k] !== null ? '<br>LIHTC score ' + LI.score[k] + ' / 46' : '';
    const ishare = LI.income[k] ? '<br>' + pct(LI.rent[k] * 12 / LI.income[k]) + ' of tract median income (' + fmt(LI.income[k]) + ')' : '';
    m.bindTooltip('<b>' + fmt(LI.rent[k]) + '</b> (listed, starting)<br>' + esc(LI.addr[k]) + (bits.length ? '<br>' + esc(bits.join(' · ')) : '') + ishare + iscore, { sticky: true });
    dots.push(m);
    listingLayer.addLayer(m);
  }

  // ---- controls
  const bedsBox = $('beds');
  if (LAT.uses_units) {
    LAT.beds.forEach((b, i) => {
      const lab = document.createElement('label');
      lab.innerHTML = '<input type="radio" name="beds" value="' + i + '"' + (i === st.b ? ' checked' : '') + '><span>' + bedsLabel(b) + '</span>';
      bedsBox.appendChild(lab);
    });
    const slider = $('sqft');
    slider.min = 0; slider.max = NS - 1; slider.step = 1; slider.value = st.s;
    bedsBox.addEventListener('change', (e) => {
      st.b = +e.target.value;
      st.s = nearestSqft(LAT.typical[String(LAT.beds[st.b])]);     // a new bedroom count starts at its typical size
      slider.value = st.s;
      render();
    });
    slider.addEventListener('input', () => { st.s = +slider.value; render(); });
    $('typical').addEventListener('click', () => { st.s = nearestSqft(LAT.typical[String(LAT.beds[st.b])]); slider.value = st.s; render(); });
  } else {
    $('beds-set').hidden = true; $('sqft-set').hidden = true;
  }
  function nearestSqft(v) {
    let best = 0;
    LAT.sqft.forEach((s, i) => { if (Math.abs(s - v) < Math.abs(LAT.sqft[best] - v)) best = i; });
    return best;
  }
  if (LAT.uses_type) {
    LAT.types.forEach((t, i) => { const o = document.createElement('option'); o.value = i; o.textContent = cap(t); $('type').appendChild(o); });
    $('type').value = st.t;
    $('type').addEventListener('change', (e) => { st.t = +e.target.value; render(); });
  } else {
    $('type-set').hidden = true;
  }
  if (!LAT.uses_units && !LAT.uses_type) { $('no-units').hidden = false; }
  $('metric').addEventListener('change', (e) => { st.metric = e.target.value; render(); });
  $('thr').value = Math.round(st.thr * 100);
  $('thr').addEventListener('input', (e) => { st.thr = +e.target.value / 100; render(); });
  $('only').addEventListener('change', (e) => { st.only = e.target.checked; render(); });
  $('show-listings').addEventListener('change', (e) => { st.listings = e.target.checked; render(); });
  $('show-outline').addEventListener('change', drawOutline);
  $('show-tracts').addEventListener('change', drawOutline);
  $('rescale').addEventListener('change', (e) => { st.rescale = e.target.checked; render(); });
  $('opacity').addEventListener('input', (e) => { st.opacity = +e.target.value / 100; render(); });

  // ---- drawing
  let domain = D.domain;
  function quantile(sorted, q) { return sorted[Math.floor(q * (sorted.length - 1))]; }
  function unitText() {
    const parts = [];
    if (LAT.uses_units) parts.push(bedsText(LAT.beds[st.b]) + ', ' + LAT.sqft[st.s].toLocaleString('en-US') + ' sq ft');
    if (LAT.uses_type && LAT.types[st.t]) parts.push(LAT.types[st.t]);
    return parts.join(' ') || 'typical unit';
  }
  function render() {
    const p = D.pred[comboIndex()];
    const share = st.metric === 'share', score = st.metric === 'score';
    const sorted = Float64Array.from(p).sort();
    domain = st.rescale && !share ? [quantile(sorted, 0.02), quantile(sorted, 0.98)] : D.domain;
    const span = Math.max(1, domain[1] - domain[0]);
    const shares = [];
    for (let k = 0; k < N; k++) {
      const frac = Math.min(1, C.d[k] / Math.max(0.1, META.max_dist_km));
      let rgb;
      if (score) { rgb = scoreRGB(C.score[k]); }
      else if (!share) { rgb = rampRGB((p[k] - domain[0]) / span); }
      else if (C.income[k] === null) { rgb = noIncomeRGB(); }
      else { const sh = p[k] * 12 / C.income[k]; shares.push(sh); rgb = shareRGB(sh); }
      rects[k].setStyle({ fillColor: mixBg(rgb, 0.6 * frac) });
    }
    map.getPane('cells').style.opacity = st.opacity;
    drawOutline();
    listingLayer.clearLayers();
    let shown = 0;
    if (st.listings) {
      dots.forEach((m, k) => {
        if (!score && LAT.uses_units && (LI.beds[k] === null || Math.round(LI.beds[k]) !== LAT.beds[st.b])) return;
        if (!score && LAT.uses_type && LAT.types[st.t] && LI.type[k] !== LAT.types[st.t]) return;
        let rgb;
        if (score) rgb = scoreRGB(LI.score[k]);
        else if (!share) rgb = rampRGB((LI.rent[k] - domain[0]) / span);
        else rgb = LI.income[k] ? shareRGB(LI.rent[k] * 12 / LI.income[k]) : noIncomeRGB();
        m.setStyle({ fillColor: 'rgb(' + rgb.join(',') + ')', color: isDark() ? '#22221f' : '#ffffff' });
        listingLayer.addLayer(m);
        shown++;
      });
      listingLayer.addTo(map);
    } else { map.removeLayer(listingLayer); }
    $('listings-label').textContent = st.listings ? 'Showing ' + shown.toLocaleString('en-US') + (score ? ' listings as dots' : ' listings like this unit as dots') : (score ? 'Show listings as dots' : 'Show listings like this unit as dots');
    $('listings-note').hidden = !st.listings;
    $('listings-note').textContent = score
      ? 'Dots are all listings, colored by the LIHTC score of their own location.'
      : share
      ? 'Dots are listings like the selected unit, colored by their own rent as a share of their tract\'s median income, on the same scale as the cells.'
      : 'Dots are listings like the selected unit, colored by their actual rent on the same scale: where a dot\'s color differs from the cells around it, the model is off.';

    $('thr-set').hidden = !share;
    $('unit-controls').hidden = score;                       // the LIHTC score belongs to a location, not to a unit
    $('no-units').hidden = score || LAT.uses_units || LAT.uses_type;
    $('rescale').parentElement.hidden = share || score;
    $('title').textContent = score ? 'LIHTC score in Atlanta' : share ? 'Rent burden in Atlanta' : 'Predicted rent in Atlanta';
    $('subtitle').textContent = score
      ? 'SiteLine\'s score for how well each location fits Georgia\'s Low-Income Housing Tax Credit scoring, out of 46 points.'
      : share
      ? 'Predicted rent for the unit below, for a year, as a share of the median household income of the census tract.'
      : 'Estimated starting monthly rent for a unit like the one below, at every point on the map.';
    $('thr-out').textContent = pct(st.thr);
    $('legend-title').textContent = score ? 'LIHTC score (SiteLine, out of 46)' : share ? 'Annual rent as a share of tract median income' : 'Predicted starting rent, per month · ' + unitText();

    $('sqft-out').textContent = LAT.uses_units ? LAT.sqft[st.s].toLocaleString('en-US') + ' sq ft' : '';
    if (LAT.uses_units) $('typical').textContent = 'Typical for ' + bedsText(LAT.beds[st.b]) + ': ' + LAT.typical[String(LAT.beds[st.b])].toLocaleString('en-US');
    if (score) {
      const sc = SCORES.length ? SCORES : [0];
      $('summary').innerHTML = SCORES.length ? 'LIHTC scores across the map are typically <b>' + quantile(sc, 0.5).toFixed(1) + '</b> out of 46, from ' + sc[0].toFixed(1) + ' to ' + sc[sc.length - 1].toFixed(1) + '.' + (N - SCORES.length ? ' ' + (N - SCORES.length) + ' cells have no score (gray).' : '') : 'No SiteLine scores were found for the mapped area.';
      $('bar').style.background = 'linear-gradient(to right, ' + [0, 0.25, 0.5, 0.75, 1].map((t) => 'rgb(' + rampRGB(t, SC_RGB).join(',') + ')').join(',') + ')';
      $('t-lo').textContent = '0'; $('t-mid').textContent = (SC_MAX / 2).toFixed(0); $('t-hi').textContent = SC_MAX + '+ pts';
      $('legend-note').textContent = 'Darker green = higher score (a better fit for tax-credit housing under the state\'s scoring). The scale tops out at the 98th percentile of the map. Paler cells are farther from any listing; the score itself is computed at the cell\'s own location.';
    } else if (!share) {
      const med = quantile(sorted, 0.5);
      $('summary').innerHTML = 'For a <b>' + esc(unitText()) + '</b>, predicted rent is typically <b>' + fmt(med) + '</b>, from ' + fmt(sorted[0]) + ' to ' + fmt(sorted[sorted.length - 1]) + ' across the map.';
      $('bar').style.background = 'linear-gradient(to right, ' + [0, 0.25, 0.5, 0.75, 1].map((t) => ramp(t)).join(',') + ')';
      $('t-lo').textContent = fmt(domain[0]); $('t-mid').textContent = fmt((domain[0] + domain[1]) / 2); $('t-hi').textContent = fmt(domain[1]);
      $('legend-note').textContent = (st.rescale ? 'Scale fitted to this view. ' : 'Same scale for every unit. ') + 'Paler cells are farther from any listing, so less reliable.';
    } else {
      const over = shares.filter((v) => v > st.thr).length, none = N - shares.length;
      const sortedShares = Float64Array.from(shares).sort();
      $('summary').innerHTML = shares.length
        ? 'For a <b>' + esc(unitText()) + '</b>, the rent is above <b>' + pct(st.thr) + '</b> of the tract median income in <b>' + pct(over / shares.length) + '</b> of the mapped area (' +
          over.toLocaleString('en-US') + ' of ' + shares.length.toLocaleString('en-US') + ' cells). The typical share is ' + pct(quantile(sortedShares, 0.5)) + '.' +
          (none ? ' ' + none.toLocaleString('en-US') + ' cells have no income data (gray).' : '')
        : 'No census income data was found for the mapped area, so shares cannot be computed. Put the Census B19013 table in data/census_income/ and rerun.';
      const H = halfRange(), lo = st.thr - H, hi = st.thr + H;
      const stops = [-1, -0.5, 0, 0.5, 1].map((u) => 'rgb(' + shareRGB(st.thr + u * H).join(',') + ')');
      $('bar').style.background = 'linear-gradient(to right, ' + (st.only ? [stops[2], stops[2], stops[2], stops[3], stops[4]] : stops).join(',') + ')';
      $('t-lo').textContent = pct(lo); $('t-mid').textContent = pct(st.thr) + ' (threshold)'; $('t-hi').textContent = pct(hi) + '+';
      $('legend-note').textContent = (st.only ? 'Red: rent above the threshold; everything else is left plain. ' : 'Red: rent above the threshold. Blue: below it. ') +
        'Gray: no income data. Paler cells are farther from any listing, so less reliable.';
    }
    renderFound();
  }

  // ---- hover
  const tip = $('tip');
  map.on('mousemove', (e) => {
    const gi = Math.round((e.latlng.lat - G.lat0) / G.dlat), gj = Math.round((e.latlng.lng - G.lon0) / G.dlon);
    const k = lookup.get(gi * 100000 + gj);
    if (k === undefined) { tip.style.display = 'none'; return; }
    if (st.metric === 'score') {
      tip.innerHTML = '<div class="big">' + (C.score[k] !== null ? C.score[k] + ' / 46' : 'No score') + '</div><div>SiteLine LIHTC score</div>' +
        '<div class="muted">' + nearTxt(k) + (C.tract[k] ? ' · tract ' + esc(C.tract[k]) : '') + '</div>';
      placeTip(e); return;
    }
    const rent = D.pred[comboIndex()][k], need = rent * 12 / st.thr;
    let incomeLine = 'Tract median household income: not available', head = '<div class="big">' + fmt(rent) + ' / month</div>', shareLine = '';
    if (C.income[k] !== null) {
      const sh = rent * 12 / C.income[k], over = sh > st.thr;
      incomeLine = 'Tract median income ' + fmt(C.income[k]);
      shareLine = (over ? '▲ Above' : '✓ Within') + ' the ' + pct(st.thr) + ' threshold';
      if (st.metric === 'share') head = '<div class="big">' + pct(sh) + ' of income</div><div>' + fmt(rent) + ' / month · ' + (over ? '▲ above' : '✓ within') + ' the ' + pct(st.thr) + ' threshold</div>';
      else shareLine = 'This rent is ' + pct(sh) + ' of it (' + (over ? '▲ above' : '✓ within') + ' ' + pct(st.thr) + ')';
    } else if (st.metric === 'share') { head = '<div class="big">No income data</div><div>' + fmt(rent) + ' / month</div>'; }
    tip.innerHTML = head + '<div>' + esc(unitText()) + '</div>' +
      '<div class="muted">Needs about ' + fmt(need) + '/yr household income at the ' + pct(st.thr) + ' rule</div>' +
      '<div class="muted">' + esc(incomeLine) + (shareLine && st.metric !== 'share' ? '. ' + esc(shareLine) : '') + '</div>' +
      '<div class="muted">' + nearTxt(k) + (C.score[k] !== null ? ' · SiteLine score ' + C.score[k] + '/46' : '') + (C.zip[k] ? ' · ZIP ' + esc(C.zip[k]) + ' (nearest listing)' : '') + '</div>';
    placeTip(e);
  });
  function nearTxt(k) {                              // how well the data covers this cell
    if (!PUBLIC) return 'Nearest listing ' + Math.round(C.d[k] * 1000).toLocaleString('en-US') + ' m away';
    const f = C.d[k] / Math.max(0.1, META.max_dist_km);
    return 'Data coverage here: ' + (f < 0.34 ? 'good' : f < 0.67 ? 'moderate' : 'thin');
  }
  function placeTip(e) {
    tip.style.display = 'block';
    const x = e.originalEvent.clientX, y = e.originalEvent.clientY, w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.min(window.innerWidth - w - 8, x + 14) + 'px';
    tip.style.top = Math.min(window.innerHeight - h - 8, y + 14) + 'px';
  }
  map.on('mouseout', () => { tip.style.display = 'none'; });

  // ---- download the current view
  $('download').addEventListener('click', () => {
    const p = D.pred[comboIndex()];
    const rows = ['cell_latitude,cell_longitude,predicted_rent,bedrooms,sqft,property_type,' + (PUBLIC ? '' : 'nearest_listing_m,') + 'siteline_total_score,tract_geoid,tract_median_income,rent_share_of_income,unaffordable_at_threshold_' + Math.round(st.thr * 100) + 'pct'];
    for (let k = 0; k < N; k++) {
      rows.push([C.lat[k], C.lon[k], p[k], LAT.uses_units ? LAT.beds[st.b] : '', LAT.uses_units ? LAT.sqft[st.s] : '',
        LAT.uses_type ? LAT.types[st.t] : '', ...(PUBLIC ? [] : [Math.round(C.d[k] * 1000)]), C.score[k] === null ? '' : C.score[k], C.tract[k] === null ? '' : C.tract[k],
        C.income[k] === null ? '' : C.income[k],
        C.income[k] === null ? '' : (p[k] * 12 / C.income[k]).toFixed(4), C.income[k] === null ? '' : (p[k] * 12 / C.income[k] > st.thr ? 1 : 0)].join(','));
    }
    const name = 'atlanta_predicted_rent' + (LAT.uses_units ? '_' + LAT.beds[st.b] + 'bd_' + LAT.sqft[st.s] + 'sqft' : '') + (LAT.uses_type && LAT.types[st.t] ? '_' + LAT.types[st.t] : '') + '.csv';
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([rows.join('\n')], { type: 'text/csv' }));
    a.download = name; document.body.appendChild(a); a.click(); a.remove();
  });

  // ---- search: coordinates, property names/addresses from the listings, then a geocoder for other addresses
  const foundLayer = L.layerGroup().addTo(map);
  let pinMarker = null, dragging = false, pinning = false;
  const PIN_ICON = L.divIcon({ className: 'pin-icon', iconSize: [30, 38], iconAnchor: [15, 37],
    html: '<svg viewBox="0 0 24 32" width="30" height="38" aria-hidden="true"><path d="M12 1C6 1 1.5 5.5 1.5 11.3 1.5 19 12 31 12 31s10.5-12 10.5-19.7C22.5 5.5 18 1 12 1z" fill="var(--pin)" stroke="#fff" stroke-width="1.6"/><circle cx="12" cy="11.3" r="4.2" fill="#fff"/></svg>' });
  // ---- drop a pin: press the button, then click the map. The pin can be dragged afterwards.
  function setPinning(on) {
    pinning = on;
    $('pin').setAttribute('aria-pressed', on ? 'true' : 'false');
    $('pin-label').textContent = on ? 'Click the map… (Esc to cancel)' : (found && found.pin ? 'Move the pin' : 'Drop a pin on the map');
    $('pin-hint').hidden = !on;
    $('map').classList.toggle('pinning', on);
  }
  $('pin').addEventListener('click', () => setPinning(!pinning));
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && pinning) setPinning(false); });
  map.on('click', (e) => {
    if (!pinning) return;
    setPinning(false);
    choose({ lat: e.latlng.lat, lon: e.latlng.lng, label: 'Dropped pin', k: null, pin: true });
  });
  const resBox = $('results');
  $('q').placeholder = PUBLIC ? 'Address or lat, lon' : 'Address, name, or lat, lon';
  $('show-listings').parentElement.hidden = PUBLIC;
  $('search-note').textContent = PUBLIC ? (META.geocoder ? 'Addresses are looked up with OpenStreetMap, which receives what you type. You can also type "lat, lon".' : 'Address lookup is off in this page: type "lat, lon" or drop a pin.') : META.geocoder
    ? 'Names and addresses of listings are found in this page. Other addresses are looked up with OpenStreetMap, which receives what you type.'
    : 'Address lookup is off in this page: search a listing\'s name or address, or type "lat, lon".';
  function parseLatLon(text) {
    const m = text.match(/^\s*\(?\s*(-?\d{1,3}(?:\.\d+)?)\s*[,;\s]\s*(-?\d{1,3}(?:\.\d+)?)\s*\)?\s*$/);
    if (!m) return null;
    const a = +m[1], b = +m[2];
    const valid = (la, lo) => Math.abs(la) <= 90 && Math.abs(lo) <= 180;
    const near = (la, lo) => la > EXT.s - 1 && la < EXT.n + 1 && lo > EXT.w - 1 && lo < EXT.e + 1;
    // "lat, lon" is expected, but "lon, lat" (as many GIS tools write it) is accepted when only that order lands near the map
    if (valid(b, a) && near(b, a) && !(valid(a, b) && near(a, b))) return { lat: b, lon: a };
    return valid(a, b) ? { lat: a, lon: b } : (valid(b, a) ? { lat: b, lon: a } : { bad: true });
  }
  function findListings(text) {
    const tokens = text.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
    if (PUBLIC || !tokens.length || text.trim().length < 3) return [];
    const norm = (t) => ' ' + t.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean).join(' ') + ' ';
    const phrase = tokens.join(' '), all = [], exact = [];
    for (let k = 0; k < LI.lat.length; k++) {
      const hay = norm(LI.name[k] + ' ' + LI.addr[k]);
      if (hay.includes(' ' + phrase + ' ')) exact.push(k);                    // the whole phrase, on word boundaries
      else if (tokens.every((t) => hay.includes(t))) all.push(k);             // every word somewhere
    }
    return exact.length ? exact : all;
  }
  const listingItem = (k) => ({ lat: LI.lat[k], lon: LI.lon[k], label: (LI.name[k] ? LI.name[k] + ' · ' : '') + LI.addr[k], k: k });
  function say(text) { resBox.innerHTML = ''; const d = document.createElement('div'); d.className = 'small'; d.textContent = text; resBox.appendChild(d); }
  function candidates(items, note, extra) {
    resBox.innerHTML = '';
    const d = document.createElement('div'); d.className = 'small'; d.textContent = note; resBox.appendChild(d);
    items.forEach((it) => {
      const b = document.createElement('button'); b.type = 'button'; b.textContent = it.label;
      b.addEventListener('click', () => choose(it)); resBox.appendChild(b);
    });
    if (extra) { const b = document.createElement('button'); b.type = 'button'; b.textContent = extra.text; b.addEventListener('click', extra.run); resBox.appendChild(b); }
  }
  async function geocode(text) {
    say('Looking up the address…');
    const query = /\b(GA|georgia)\b/i.test(text) ? text : text + ', Atlanta, GA';
    const url = 'https://nominatim.openstreetmap.org/search?format=jsonv2&limit=5&countrycodes=us&q=' + encodeURIComponent(query) +
      '&viewbox=' + [EXT.w - 0.3, EXT.n + 0.3, EXT.e + 0.3, EXT.s - 0.3].join(',');
    let hits;
    try {
      const r = await fetch(url, { headers: { Accept: 'application/json' } });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      hits = await r.json();
    } catch (err) {
      say('The address lookup is not available right now (' + err.message + '). You can still type "lat, lon" or a listing name.');
      return;
    }
    const items = hits.map((h) => ({ lat: +h.lat, lon: +h.lon, label: h.display_name, k: null }));
    if (!items.length) { say('No match for "' + text + '". Try a fuller address, a listing name, or "lat, lon".'); return; }
    if (items.length === 1) { choose(items[0]); return; }
    candidates(items, 'Several matches. Pick one:');
  }
  $('search-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const text = $('q').value.trim();
    if (!text) return;
    const ll = parseLatLon(text);
    if (ll) { if (ll.bad) say('Those coordinates are out of range. Use latitude, then longitude (Atlanta is about 33.75, -84.39).'); else choose({ lat: ll.lat, lon: ll.lon, label: ll.lat.toFixed(5) + ', ' + ll.lon.toFixed(5), k: null }); return; }
    const hits = findListings(text);
    if (hits.length === 1) { choose(listingItem(hits[0])); return; }
    if (hits.length > 1) {
      candidates(hits.slice(0, 8).map(listingItem), hits.length + ' listings match' + (hits.length > 8 ? ' (showing 8; be more specific)' : '') + '. Pick one:',
        META.geocoder ? { text: 'None of these: search as an address', run: () => geocode(text) } : null);
      return;
    }
    if (META.geocoder) geocode(text); else say('No listing matches "' + text + '".');
  });
  function snapIndex(options, v) { let b = 0; options.forEach((o, i) => { if (Math.abs(o - v) < Math.abs(options[b] - v)) b = i; }); return b; }
  function applyListingUnit(k) {                  // a listing found by name sets the controls to that listing's own unit
    if (LAT.uses_units) {
      if (LI.beds[k] !== null) st.b = snapIndex(LAT.beds, Math.round(LI.beds[k]));
      st.s = nearestSqft(LI.sqft[k] !== null ? LI.sqft[k] : LAT.typical[String(LAT.beds[st.b])]);
      const radios = $('beds').querySelectorAll('input'); if (radios[st.b]) radios[st.b].checked = true;
      $('sqft').value = st.s;
    }
    if (LAT.uses_type && LI.type[k]) { const i = LAT.types.indexOf(LI.type[k]); if (i >= 0) { st.t = i; $('type').value = i; } }
  }
  function choose(item) {
    found = item;
    resBox.innerHTML = '';
    if (item.k !== null) applyListingUnit(item.k);
    if (!item.pin) map.setView([item.lat, item.lon], Math.max(map.getZoom(), 15));      // a dropped pin is already where you are looking
    render();
    setPinning(false);
  }
  function estimateAt(lat, lon) {
    const fi = (lat - G.lat0) / G.dlat, fj = (lon - G.lon0) / G.dlon, i0 = Math.floor(fi), j0 = Math.floor(fj);
    const kc = lookup.get(Math.round(fi) * 100000 + Math.round(fj));
    if (kc === undefined) return null;                                  // not inside a mapped cell
    const p = D.pred[comboIndex()];
    let sw = 0, sv = 0;
    for (let di = 0; di < 2; di++) for (let dj = 0; dj < 2; dj++) {      // bilinear blend of the (up to) four surrounding cell centres
      const k = lookup.get((i0 + di) * 100000 + (j0 + dj));
      if (k === undefined) continue;
      const w = (1 - Math.abs(fi - (i0 + di))) * (1 - Math.abs(fj - (j0 + dj)));
      sw += w; sv += w * p[k];
    }
    return { rent: sw > 0 ? sv / sw : p[kc], k: kc };
  }
  function nearestListingKm(lat, lon) {
    const c = Math.cos(lat * Math.PI / 180);
    let best = Infinity;
    for (let k = 0; k < LI.lat.length; k++) {
      const dy = (LI.lat[k] - lat) * 111.19, dx = (LI.lon[k] - lon) * 111.19 * c, d = dx * dx + dy * dy;
      if (d < best) best = d;
    }
    return Math.sqrt(best);
  }
  function renderFound() {
    const box = $('found');
    if (!found) { box.hidden = true; if (pinMarker) { foundLayer.removeLayer(pinMarker); pinMarker = null; } return; }
    if (!pinMarker) {
      pinMarker = L.marker([found.lat, found.lon], { icon: PIN_ICON, draggable: true, keyboard: false, autoPan: false, title: 'Drag to move the pin' });
      pinMarker.on('dragstart', () => { dragging = true; });
      pinMarker.on('drag', () => { const ll = pinMarker.getLatLng(); found.lat = ll.lat; found.lon = ll.lng; found.k = null; found.label = 'Dropped pin'; found.pin = true; renderFound(); });
      pinMarker.on('dragend', () => { dragging = false; renderFound(); });
      foundLayer.addLayer(pinMarker);
    } else if (!dragging) { pinMarker.setLatLng([found.lat, found.lon]); }
    const est = estimateAt(found.lat, found.lon);
    const dist = nearestListingKm(found.lat, found.lon);
    let html = '<div class="head"><b>' + esc(found.label) + '</b><button type="button" id="clear-found" aria-label="Clear the search result">Clear</button></div>' +
      '<div class="small">' + found.lat.toFixed(5) + ', ' + found.lon.toFixed(5) + '</div>';
    if (found.k !== null) {
      const bits = [];
      if (LI.beds[found.k] !== null) bits.push(bedsText(LI.beds[found.k]));
      if (LI.sqft[found.k] !== null) bits.push(Math.round(LI.sqft[found.k]).toLocaleString('en-US') + ' sq ft');
      if (LI.type[found.k]) bits.push(LI.type[found.k]);
      html += '<div>Listed at <b>' + fmt(LI.rent[found.k]) + '</b>/month (starting rent)' + (bits.length ? ' · ' + esc(bits.join(' · ')) : '') + '</div>';
    }
    if (!est) {
      html += '<p>Outside the mapped area: this point is more than ' + META.max_dist_km + ' km from any listing' + (D.outline ? ' or outside the clipped city limits' : '') + ', so the model has too little to go on. ' + (dist < 100 ? 'The nearest listing is ' + dist.toFixed(1) + ' km away.' : '') + '</p>';
    } else {
      const income = C.income[est.k] !== null ? C.income[est.k] : (found.k !== null ? LI.income[found.k] : null);
      const sc = found.k !== null && LI.score[found.k] !== null ? LI.score[found.k] : C.score[est.k];
      const scLine = '<div' + (st.metric === 'score' ? ' class="verdict"' : ' class="small"') + '>SiteLine LIHTC score: ' + (sc !== null ? sc + ' / 46' : 'not available') + '</div>';
      if (st.metric === 'score') html += scLine;
      html += '<div class="big">' + fmt(est.rent) + ' / month</div><div>Model estimate for a ' + esc(unitText()) + (found.k !== null ? ' (this listing is part of the training data)' : '') + '</div>' +
        '<div class="small">Needs about ' + fmt(est.rent * 12 / st.thr) + '/yr household income at the ' + pct(st.thr) + ' rule</div>';
      if (income) {
        const sh = est.rent * 12 / income, over = sh > st.thr;
        html += '<div class="verdict">' + (over ? '▲ Unaffordable: ' : '✓ Within the threshold: ') + pct(sh) + ' of the tract median income (' + fmt(income) + ') vs ' + pct(st.thr) + '</div>';
      } else { html += '<div class="small">No median income is available for this tract, so affordability cannot be judged.</div>'; }
      if (st.metric !== 'score') html += scLine;
      html += '<div class="small">' + (PUBLIC ? '' : 'Nearest listing ' + Math.round(dist * 1000).toLocaleString('en-US') + ' m away. ') + 'Blended from the surrounding ' + Math.round(G.cell_km * 1000) + ' m cells, so it is an estimate; ' +
        '<code>python knn_model.py predict</code> gives the exact model output for one address.</div>';
    }
    box.innerHTML = html; box.hidden = false;
    $('clear-found').addEventListener('click', () => { found = null; renderFound(); setPinning(false); });
  }

  // ---- light / dark: follows the system until the button is used, then remembers the choice
  function syncTheme() {
    const dark = isDark();
    $('theme').textContent = dark ? 'Light' : 'Dark';
    $('theme').setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
  }
  try { const saved = localStorage.getItem('rent-map-theme'); if (saved === 'light' || saved === 'dark') document.documentElement.setAttribute('data-theme', saved); } catch (e) { /* storage unavailable */ }
  $('theme').addEventListener('click', () => {
    const next = isDark() ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('rent-map-theme', next); } catch (e) { /* storage unavailable */ }
    syncTheme(); applyBasemap(); render();
  });

  // ---- credits, disclaimers and the welcome block
  const SITELINE_URL = 'https://kmcalist682336.github.io/sitelineLIHTC/', SITELINE_REPO = 'https://github.com/kmcalist682336/SitelineLIHTCStartUp';
  const people = (a) => a.length < 2 ? a.join('') : a.slice(0, -1).join(', ') + ' and ' + a[a.length - 1];
  $('credit-line').innerHTML = esc(META.lab) + '<br>' + esc([META.creator].concat(META.contributors).join(' · '));
  $('w-title').textContent = META.project;
  $('w-lab').textContent = META.lab;
  $('w-desc').textContent = 'This tool estimates what rent should be for a given kind of unit anywhere in Atlanta, compares it with the income of the surrounding census tract to show where rent is out of reach, and maps each location\'s LIHTC (Low-Income Housing Tax Credit) score. Use it to explore how rent, income and tax-credit siting line up across the city.';
  $('w-team').innerHTML = 'This tool was developed by <b>' + esc(META.creator) + '</b>. Additional contributions were made by ' + esc(people(META.contributors)) + '.';
  $('w-siteline').innerHTML = 'The scores shown here are computed by <a href="' + SITELINE_URL + '" target="_blank" rel="noopener noreferrer">SiteLine</a>, an ATL Metro Area LIHTC scorer and an InvestATL &amp; Emory collaboration. ' +
    'It calculates the location-based scoring criteria of Georgia\'s 2024&ndash;2025 Qualified Allocation Plan (out of 46 points). ' +
    '<a href="' + SITELINE_URL + '" target="_blank" rel="noopener noreferrer">SiteLine app and data sources</a> &middot; <a href="' + SITELINE_REPO + '" target="_blank" rel="noopener noreferrer">source code</a>.';
  const tileCredit = BM.provider === 'carto' ? 'CARTO (with &copy; OpenStreetMap contributors)' : BM.provider === 'osm' ? '&copy; OpenStreetMap contributors' : 'none';
  const NOTES = [
    ['Model estimates, for research only', 'Rents are predictions from a K-nearest-neighbors model trained on ' + META.n_listings.toLocaleString('en-US') + ' Redfin rental listings (starting asking rents at the time they were scraped; this map was built on ' + esc(META.generated.slice(0, 10)) + '). ' +
      (META.cv_mae ? 'On properties the model had not seen, its typical error was about ' + fmt(META.cv_mae) + ' per month. ' : '') + 'They are not quotes, appraisals or rental advice, and real rents vary by building and unit.'],
    ['Affordability is an area average', '&ldquo;Rent burden&rdquo; divides a year of predicted rent by the median household income of the census tract (American Community Survey estimates, U.S. Census Bureau). It describes places, not any household, and the 30% line is only a rule of thumb you can change.'],
    ['LIHTC scores are guidance, not ground truth', 'As SiteLine itself warns, its data has rough edges and its reading of the QAP may differ from the Georgia DCA\'s, so scores should be verified and checked by a qualified professional against the DCA\'s requested data sources before any decision.']];
  if (META.geocoder) NOTES.push(['Address lookup', 'An address you type is sent from your browser to OpenStreetMap\'s Nominatim service to find its coordinates. Searching by &ldquo;lat, lon&rdquo; or dropping a pin sends nothing.']);
  NOTES.push(['Map tiles and privacy', 'The basemap (' + tileCredit + ') is loaded from its provider\'s servers, which see your IP address like any website you visit. Boundaries come from the U.S. Census Bureau (TIGERweb). This page has no analytics or cookies and only remembers your light/dark choice in your browser.']);
  NOTES.push(PUBLIC ? ['No individual listings', 'This public version shows only model predictions for map cells, not the Redfin listings behind them.']
                    : ['Private build', 'This version contains individual Redfin listings. Do not publish it; build the public one with <code>python rent_map.py --public</code>.']);
  $('w-disc').innerHTML = NOTES.map((n) => '<div class="m-card"><b>' + n[0] + '</b>' + n[1] + '</div>').join('');
  function showWelcome() { $('welcome').hidden = false; $('w-box').scrollTop = 0; $('w-box').focus(); }
  function hideWelcome() {
    $('welcome').hidden = true;
    try { sessionStorage.setItem('rent-map-welcome-seen', '1'); } catch (e) { /* storage unavailable */ }
    $('open-about').focus();
  }
  $('w-go').addEventListener('click', hideWelcome);
  $('w-x').addEventListener('click', hideWelcome);
  $('open-about').addEventListener('click', showWelcome);
  document.addEventListener('keydown', (e) => {
    if ($('welcome').hidden) return;
    if (e.key === 'Escape') { hideWelcome(); return; }
    if (e.key === 'Tab') {                                           // keep keyboard focus inside the dialog
      const f = [$('w-x'), ...$('welcome').querySelectorAll('a'), $('w-go')], i = f.indexOf(document.activeElement);
      const next = e.shiftKey ? (i <= 0 ? f.length - 1 : i - 1) : (i === f.length - 1 ? 0 : i + 1);
      e.preventDefault(); f[next].focus();
    }
  });
  let seen = false;
  try { seen = sessionStorage.getItem('rent-map-welcome-seen') === '1'; } catch (e) { seen = false; }
  if (!seen) showWelcome();

  // ---- notes
  $('about').textContent = 'Predictions come from a K-nearest-neighbors model (' + META.k + ' neighbors, ' + META.aggregate + ') trained on ' +
    META.n_listings.toLocaleString('en-US') + ' Redfin rental listings, using location, SiteLine LIHTC scores, postal code' + (LAT.uses_units ? ', bedrooms, floor area' : '') + (LAT.uses_type ? ' and property type' : '') +
    '. Redfin rents are starting prices, so these are starting rents. ' +
    (META.cv_mae ? 'On properties the model had not seen, its typical error was about ' + fmt(META.cv_mae) + ' per month. ' : '') +
    (META.tracts ? 'Census tract boundaries: U.S. Census Bureau TIGERweb. ' : '') +
    (META.outline_source ? 'City limits outline: ' + META.outline_source + '. ' : '') +
    'Cells are ' + Math.round(G.cell_km * 1000) + ' m wide and only shown within ' + META.max_dist_km + ' km of a listing. Combinations far from what is listed (for example a 4-bed studio-sized unit) are extrapolations. Generated ' + META.generated + '.';

  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { syncTheme(); applyBasemap(); render(); });
  syncTheme();
  applyBasemap();
  render();
})();
</script>
</body>
</html>
'''


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def rent_arg(text):
    return "auto" if text.lower() == "auto" else float(text)


def build_parser():
    p = argparse.ArgumentParser(description="Interactive heat map of predicted rent across Atlanta.")
    p.add_argument("--cell-km", type=float, default=0.5, help="grid cell width in km (default 0.5)")
    p.add_argument("--max-dist-km", type=float, default=1.5, help="drop cells farther than this from every listing (default 1.5)")
    p.add_argument("--boundary", help="GeoJSON polygon to use instead of the downloaded city limits: it is drawn as the "
                                      "outline and cells outside it are dropped")
    p.add_argument("--no-outline", action="store_true", help="do not draw (or download) the city-limits outline")
    p.add_argument("--clip-to-city", action="store_true", help="drop grid cells outside the city limits")
    p.add_argument("--no-tracts", action="store_true", help="do not download or draw census tract boundaries")
    p.add_argument("--refresh-tracts", action="store_true", help="download the census tract boundaries again")
    p.add_argument("--refresh-outline", action="store_true", help="download the city limits again")
    p.add_argument("--max-beds", type=int, default=4, help="largest bedroom count on the map (default 4)")
    p.add_argument("--sqft-min", type=int, default=400)
    p.add_argument("--sqft-max", type=int, default=2400)
    p.add_argument("--sqft-step", type=int, default=200, help="floor-area step of the slider (default 200)")
    p.add_argument("--out", type=Path, default=None, help="output page (default data/model/rent_heatmap.html, or docs/index.html with --public)")
    p.add_argument("--public", action="store_true",
                   help="build a page that is safe to publish: no individual listings (no dots, addresses or names), no "
                        "listing search, no API key (OpenStreetMap tiles); written to docs/index.html for GitHub Pages")
    p.add_argument("--basemap", choices=["auto", "carto", "osm", "none"], default="auto",
                   help="map tiles: carto (needs a free key), osm, or none; auto = carto if a key is given, else osm")
    p.add_argument("--carto-key", default=None, help="free CARTO basemap API key (or set the CARTO_API_KEY environment variable)")
    p.add_argument("--no-geocoder", action="store_true",
                   help="turn off address lookup in the page's search box (it is sent to OpenStreetMap Nominatim); "
                        "property names and coordinates still work")
    p.add_argument("--rescore", action="store_true", help="ignore cached SiteLine scores of grid cells")
    p.add_argument("--min-rent", type=rent_arg, default=None, help="dollars or 'auto'; default: the range used by the last `knn_model.py train`")
    p.add_argument("--max-rent", type=rent_arg, default=None, help="dollars or 'auto'; default: the range used by the last `knn_model.py train`")
    p.add_argument("--unit-basis", choices=km.UNIT_BASES, default=None, help="default: the one used by the last `train`")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.basemap_config = choose_basemap(args)            # checked first, so a missing key fails before the slow steps
    args.outline_rings, args.outline_source = None, None
    if args.boundary:
        args.outline_rings, args.outline_source = load_boundary(args.boundary), Path(args.boundary).name
    elif not args.no_outline:
        args.outline_rings, args.outline_source = get_city_rings(args.refresh_outline)
    if args.clip_to_city and args.outline_rings is None:
        raise SystemExit("--clip-to-city needs the city limits: drop --no-outline, or fix the download problem reported above.")
    df, model, params, saved, basis = load_model(args)
    clip = args.outline_rings if (args.boundary or args.clip_to_city) else None
    cells, grid = build_grid(df, args.cell_km, args.max_dist_km, clip)
    cells = score_cells(cells, args.rescore)
    args.tracts = None if args.no_tracts else get_tracts(cells, args.refresh_tracts)
    lattice = make_lattice(df, model, args)
    preds = predict_lattice(model, cells, lattice)
    payload = build_payload(df, cells, grid, lattice, preds, model, params, saved, basis, args)
    out = args.out or (PUBLIC_HTML if args.public else OUT_HTML)
    write_html(payload, out)
    if args.public:
        check_public_page(out, df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
