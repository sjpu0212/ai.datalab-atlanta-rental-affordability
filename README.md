# Atlanta Rental Affordability: data pipeline

Builds the dataset for the project *Mapping the Affordability of Rental Properties in Atlanta* (Emory Center for AI
Learning): every Redfin rental listing in Atlanta, scored with a LIHTC "desirability" score and joined to
census-tract median household income.

## Team

Emory AI.DataLab. This tool was developed by Songjie Pu. Additional contributions were made by Joshua Velasquez and Yanqi Li.

## Acknowledgements and disclaimer

The LIHTC scores come from [SiteLine](https://kmcalist682336.github.io/sitelineLIHTC/), an ATL Metro Area LIHTC scorer and
an InvestATL & Emory collaboration that computes the location-based criteria of Georgia's 2024-2025 Qualified Allocation
Plan ([source](https://github.com/kmcalist682336/SitelineLIHTCStartUp)). Income data are American Community Survey estimates and boundaries come from
TIGERweb, both from the U.S. Census Bureau; map tiles are (c) OpenStreetMap contributors.

Everything here is a model estimate for research. Predicted rents are not quotes or rental advice, affordability is a
census-tract average and says nothing about any one household, and, as SiteLine itself says, the LIHTC scores are
guidance, not ground truth, and should be checked by a qualified professional.

## Repository layout

```
redfin_scraper.py                      Step 1: scrape Redfin rentals  -> data/redfin_rentals/
siteline_scorer.py                     Step 2: add SiteLine LIHTC scores -> data/redfin_rentals_lihtc_scores.csv
income.py                              Step 3: add tract income from the local census file -> data/atlanta_rentals_final.csv
knn_model.py                           Step 4: K-nearest-neighbors rent model: cross-validate it, predict new properties
rent_map.py                            Step 5: interactive heat map of predicted rent -> data/model/rent_heatmap.html
requirements.txt
data/
  redfin_rentals/                      every scrape, saved as redfin_atlanta_rentals_<YYYY-MM-DD_HHMM>.csv
    redfin_progress/                   per-page checkpoints for an in-progress scrape (cleared when it finishes)
  census_income/                       census tract income downloads (drop new ones here; the newest is used)
  redfin_rentals_lihtc_scores.csv      newest scrape + LIHTC scores (output of step 2)
  atlanta_rentals_final.csv            newest scrape + LIHTC scores + tract + income (output of step 3)
  model/                               cross-validation results and the tuned model settings (output of step 4)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install chromium
```

SiteLine's scorer is **not** included in this repository. Clone it next to this project and install it once:

```bash
git clone https://github.com/kmcalist682336/SitelineLIHTCStartUp.git ../SitelineLIHTCStartUp
cd ../SitelineLIHTCStartUp && pip install -e ".[geo]"
```

If you cloned it elsewhere, set `SITELINE_REPO` to that folder. Credit for the scorer goes to its authors; see their
repository for its documentation and terms. Scores are estimates and not official DCA scores.

## Running

```bash
python redfin_scraper.py        # about 71 pages with deliberate pauses; takes over an hour
python siteline_scorer.py       # seconds
python income.py                # uses the newest file in data/census_income/
python knn_model.py train        # cross-validate the rent model and save its tuned settings
python knn_model.py predict --lat 33.7660 --lon -84.3491 --zip 30307      # or --property "name or address"
python rent_map.py               # build the interactive map (needs the tuned settings from `knn_model.py train`)
```

Useful options for the scraper: `--max-pages 3` (test run), `--headed` (visible browser, helps if blocked),
`--fresh` (discard saved progress). If Redfin blocks the run it stops with its progress saved; run the same command
later to resume. A dataset version is only saved once all pages have been collected.

## The rent model

`knn_model.py` predicts a rental's monthly price by averaging the rents of its K most similar listings, where similar
means close in location (latitude/longitude), in SiteLine section scores, and in postal code. Median income is
deliberately not a predictor: it is already reflected in location, and it is what predicted rent is later compared
against to judge affordability (annual rent above 30% of tract median household income = not affordable).

The neighbors' rents can be combined four ways: plain **mean**, **median**, **trimmed** mean (drop the top and bottom
20%), or **filtered** mean (drop neighbors whose rent is far from the neighbors' median, measured in percent terms with
the median absolute deviation, then average the rest). Cross-validation picks the method, and how strict the filter is.
`predict` marks which neighbors were dropped as outliers.

`train` evaluates it with nested cross-validation on properties the model has not seen (testing folds, with validation
folds inside the training data to choose K, the weights and the averaging method) and compares six models on the same
folds: a mean baseline, location only, location plus postal code, the same with outlier-robust averaging, the full model
and the full model with robust averaging. The summary prints the head-to-head gaps: robust vs plain averaging, and with
vs without LIHTC scores. Listings of one building are kept in the same fold so that twin units cannot inflate the
scores. Results are written to `data/model/`.

**Unit features.** Besides location, SiteLine scores and postal code, the model matches on bedrooms, floor area and
property type (from the scraper's `beds_*`, `sqft_*` and `property_type` columns). Buildings with several unit types have
ranges, and the rent is the starting price, so by default the smallest unit is used (`--unit-basis min`; `max` and `mid`
are available to compare). In `predict`, a property found in the dataset brings its own unit details; for coordinates or
a new address give `--beds`, `--sqft` and `--type` (anything omitted is filled with a typical value, and `predict` prints
what it assumed). `train` adds two models, `units_robust` and `all_robust`, so you can see what unit features add and
whether the LIHTC scores still matter once they are included. They need data scraped with the current scraper.

**Rent range.** Before modeling, listings with extreme rents are removed. By default the range comes from the data
itself: Tukey's IQR fences (Q1 - 1.5 x IQR to Q3 + 1.5 x IQR) computed on log rent, because rents are right-skewed
(dollar-scale fences would flag far too many ordinary listings). Listings outside $300-$10,000 are always dropped as
data errors. Override with `--max-rent 4500`, `--min-rent 600`, or `--iqr-k 3` (a wider, more permissive range). The range is
saved with the tuned settings and `predict` reuses it. Excluded listings are not scored, so reported errors describe the
kept rent range, not the whole market.

`predict` scores a new location with SiteLine, finds its nearest listings, and reports the predicted rent and what
household income is needed to afford it. It accepts coordinates, or a property name or address (looked up in the
dataset first, then geocoded, which needs internet).

## The rent map

`rent_map.py` writes one web page, `data/model/rent_heatmap.html`; open it in a browser (it loads the map tiles and the
Leaflet library from the internet). CARTO's clean basemaps now need a free API key (without one, tiles are stamped
"API KEY REQUIRED"): get one at <https://carto.com/basemaps/apikey/> and run `python rent_map.py --carto-key YOUR_KEY`. Without
a key the script falls back to OpenStreetMap tiles. It lays a grid of 500 m cells over the area your listings cover, drops cells more
than 1.5 km from any listing (the model has nothing to learn from there), scores each cell with SiteLine, and uses the
tuned model to predict the rent of every cell for every combination of bedrooms, floor area and property type. The page
stores those predictions, so no server is needed: change bedrooms, floor area or type and the map recolors. Hover a cell
for the predicted rent, the household income needed at the 30% rule, the tract's median income, and how far the nearest
listing is. Cells far from listings are drawn paler. The City of Atlanta's limits are drawn as an outline (downloaded once from the Census Bureau, with the city's own GIS as a
backup, and cached in `data/model/atlanta_boundary.geojson`; toggle it in the page). Options: `--cell-km`, `--max-dist-km`,
`--clip-to-city` (drop cells outside the city), `--boundary city.geojson` (use your own outline and clip to it),
`--no-outline`, `--sqft-step`. SiteLine scores for the grid are cached in `data/model/grid_scores.csv`, so only the
first run is slow. Rerun the script after retraining or after a new scrape. The page contains Redfin-derived data, so
treat it like the data files (it is git-ignored with the rest of `data/model/`).

**Search, pin, affordability and theme.** The panel has a *Check a specific place* box that accepts a listing's name or address, an
address anywhere in Atlanta, or `lat, lon`, plus a *Drop a pin on the map* button (press it, click the map, then drag the pin to move it). It marks the spot and shows the model's estimate for the unit currently
selected (a listing found by name also sets bedrooms, size and type to that listing's own), the income needed, and
whether the rent passes the threshold. The estimate is blended from the surrounding grid cells, so for the exact model
output of one address use `python knn_model.py predict`. Listing names are searched inside the page; other addresses are
sent from your browser to OpenStreetMap's Nominatim service (disable with `--no-geocoder`). *Show: Rent as share of
income* recolors the map by a year of predicted rent divided by the tract's median household income, with a slider for
the share above which rent is unaffordable (default 30%): red above it, blue below it, gray where the census gives no
income. Income for every tract comes from the whole table in `data/census_income/`. A *Light/Dark* button in the panel
header switches the theme (it follows your system until you use it, then remembers your choice).
Census tract outlines (a checkbox under *Map display*) come from the Census Bureau's TIGERweb service, are downloaded once for the mapped area, and are cached in `data/model/tract_boundaries.geojson` (git-ignored); `--no-tracts` skips them.

*Show: LIHTC score* colors each cell by SiteLine's score for that location (out of 46), the same score the model uses as a predictor; it depends on the place only, so the bedroom, size and type controls hide in that view.

## Publishing the map on GitHub Pages

The scraped Redfin listings must not be published, so the page you publish is a separate build:

```
python rent_map.py --public
```

`--public` leaves out every individual listing (no dots, addresses or names, and no listing search), refuses any CARTO API
key (it uses OpenStreetMap tiles, which need none, and ignores a `CARTO_API_KEY` environment variable), writes to
`docs/index.html`, and then checks the file for listing addresses and key text. Commit `docs/index.html` and push, then in
the repository go to Settings -> Pages, choose "Deploy from a branch", branch `main`, folder `/docs`. The site appears at
`https://<username>.github.io/<repository>/`. The normal build (`python rent_map.py`, written to `data/model/`) contains the
listings and stays on your computer.

## Notes


- Redfin's Terms of Use restrict automated access. The scraper is deliberately slow; use the data responsibly.
- Redfin's price for an apartment building is a starting price, so the model predicts the listed (starting) rent.
- `siteline_scorer.py` only scores locations inside SiteLine's 11 metro-Atlanta counties. Listings elsewhere keep
  empty scores and have no tract/income.
- Income comes from the Census Bureau ACS 5-year estimates (table B19013), matched on the 11-digit tract GEOID. It is
  read from a file in `data/census_income/` (a copy is committed to this repository; see the README in that folder for
  how to download a newer one). When Census publishes a newer release, add the new file to the folder and rerun `income.py`.
