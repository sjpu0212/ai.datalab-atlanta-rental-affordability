# Census tract income data

`income.py` reads its income table from this folder. **To update the income data, drop the new download in here.** The
script uses the most recent file automatically (newest ACS year in the file name, then newest file date), so older
files can stay as history.

A copy of the download is committed in this folder, so the project runs without it. You only need the steps below to
update the data (or if the copy is missing).

## What to download

**U.S. Census Bureau, American Community Survey (ACS) 5-Year Estimates, Detailed Tables, table B19013 "Median
Household Income in the Past 12 Months", for all census tracts in Georgia.** The 2024 release (2020–2024 estimates,
published January 29, 2026) is the newest as of this writing.

1. Go to <https://data.census.gov> and search for `B19013`.
2. Open the table *Median Household Income in the Past 12 Months (in 2024 Inflation-Adjusted Dollars)*, from the
   ACS 5-Year Estimates Detailed Tables. If it opens another year, change the vintage.
3. Under **Geos**, choose **Tract**, then **Georgia**, then **All Census Tracts within Georgia**, and apply.
4. Click **Download table**, pick the ACS 5-Year Estimates vintage and the CSV format, and download.
5. Put the downloaded `.zip` in this folder. You don't need to unzip it. An extracted `...B19013-Data.csv` works too.
   The file name should keep the Census's own name (for example `ACSDT5Y2024.B19013-Data.csv`), because the script
   reads the ACS year from it.

The buttons on data.census.gov change from time to time, so the labels above may differ slightly. The file only needs
a `GEO_ID` column with the tract ID (like `1400000US13121001600`) and the `B19013_001E` estimate column.

## Notes

- Use the 2020 ACS 5-year release or later. Earlier releases use 2010 tract boundaries, so many tracts won't match.
- Some tracts have no estimate (shown as `-`), and very high incomes are reported only as `250,000+`. The final
  dataset leaves the first empty and marks the second in an `income_top_coded` column.
- The metadata and notes files inside the download (`...Column-Metadata.csv`, `...Table-Notes.txt`) are ignored.
