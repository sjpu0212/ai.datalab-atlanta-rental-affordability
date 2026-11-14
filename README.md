# Atlanta Rental Affordability: data pipeline

Builds the dataset for the project *Mapping the Affordability of Rental Properties in Atlanta* (Emory Center for AI
Learning, Group 17): every Redfin rental listing in Atlanta, scored with a LIHTC "desirability" score and joined to
census-tract median household income.

## Repository layout

```
redfin_scraper.py                      Step 1: scrape Redfin rentals  -> data/redfin_rentals/
siteline_scorer.py                     Step 2: add SiteLine LIHTC scores -> data/redfin_rentals_lihtc_scores.csv
income.py                              Step 3: add tract income from the local census file -> data/atlanta_rentals_final.csv
knn_model.py                           Step 4: K-nearest-neighbors rent model: cross-validate it, predict new properties
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

## Notes


- Redfin's Terms of Use restrict automated access. The scraper is deliberately slow; use the data responsibly.
- Redfin's price for an apartment building is a starting price, so the model predicts the listed (starting) rent.
- `siteline_scorer.py` only scores locations inside SiteLine's 11 metro-Atlanta counties. Listings elsewhere keep
  empty scores and have no tract/income.
- Income comes from the Census Bureau ACS 5-year estimates (table B19013), matched on the 11-digit tract GEOID. It is
  read from a file in `data/census_income/` (a copy is committed to this repository; see the README in that folder for
  how to download a newer one). When Census publishes a newer release, add the new file to the folder and rerun `income.py`.
