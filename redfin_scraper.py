"""
Step 1: scrape Atlanta rental listings from Redfin.

Reads the JSON-LD metadata on every page of https://www.redfin.com/city/30756/GA/Atlanta/rentals and saves one
timestamped CSV per complete scrape to data/redfin_rentals/.

Columns: Address, City, State, Postal Code, latitude, longitude, url, Rental Price,
         rent_min, rent_max, beds_min, beds_max, sqft_min, sqft_max, property_type, price_raw, beds_raw, sqft_raw
(JSON-LD price, numberOfRooms and floorSize each hold one number or a range such as "1204-1500", "1-2" or
"1,150-1,350". Redfin's numberOfRooms is the number of BEDROOMS (0 = studio), so it is stored as beds_*.
"Rental Price" is the low end of the price range, i.e. the starting price of the cheapest unit, and equals rent_min.)

Usage (run from anywhere):
    python redfin_scraper.py                  # scrape every page
    python redfin_scraper.py --max-pages 3    # quick test (not saved as a version unless --save-partial)
    python redfin_scraper.py --fresh          # ignore and clear any saved progress first
    python redfin_scraper.py --headed         # show the browser window (can help if you get blocked)

Each page is checkpointed to data/redfin_rentals/redfin_progress/ as it is fetched, so if Redfin blocks the
run you can wait a while and run the same command again to resume. After a complete scrape the checkpoint
files are cleared, so the next run starts fresh.

Note: Redfin's Terms of Use restrict automated access. Keep the pacing slow and use the data responsibly.
"""

import argparse
import asyncio
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

# ---------------------------------------------------------------------------
# Paths (relative to this file, so the script works from any working directory)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RENTALS_DIR = ROOT / "data" / "redfin_rentals"
PROGRESS_DIR = RENTALS_DIR / "redfin_progress"
VERSION_PREFIX = "redfin_atlanta_rentals"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = "https://www.redfin.com/city/30756/GA/Atlanta/rentals"
HEADLESS = True                # Use --headed on the command line if you get blocked

# Pacing / anti-block
DELAY_RANGE = (10.0, 20.0)       # Seconds between page requests
BLOCK_RETRIES = 3                # Retries of the SAME page after a block
BLOCK_COOLDOWN = (240.0, 360.0)  # Seconds to wait before each retry (about 4-6 minutes)
BATCH_SIZE = 5                   # Pages fetched per batch before a long rest
BATCH_REST = (180.0, 300.0)      # Seconds to rest between batches (about 3-5 minutes)
BLOCKED_RESOURCE_TYPES = {"image", "media", "font"}   # Not needed: we only read the ld+json script

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--no-sandbox",
    "--disable-dev-shm-usage",
]
BLOCK_TITLE_KEYWORDS = ["robot", "access denied", "verify", "captcha", "error", "just a moment"]
BLOCK_BODY_MARKERS = ["the request could not be satisfied", "verify you are a human", "captcha"]

COLUMNS = ["Address", "City", "State", "Postal Code", "latitude", "longitude", "url", "Rental Price",
           "rent_min", "rent_max", "beds_min", "beds_max", "sqft_min", "sqft_max", "property_type",
           "price_raw", "beds_raw", "sqft_raw"]

PRICE_LIMITS = (100, 50_000)           # monthly rent; outside this the price is treated as bad data (-> empty)
BEDS_LIMITS = (0, 10)                  # values outside these limits are treated as bad data (-> empty)
SQFT_LIMITS = (100, 10_000)
# schema.org unitCode (UN/CEFACT) -> multiplier to square feet. FTK = square foot, MTK = square metre.
SQFT_UNIT_FACTORS = {"FTK": 1.0, "MTK": 10.7639104}


# ---------------------------------------------------------------------------
# JSON-LD parsing
# ---------------------------------------------------------------------------
_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")        # the comma in '1,150', but not in '1, 2'


def parse_range(value, limits):
    """
    Turn a JSON-LD size/rooms value into (low, high). Handles:
      '2' -> (2, 2)            '1-2' -> (1, 2)             '0-2' -> (0, 2)  (0 = studio)
      '1,080' -> (1080, 1080)  '1,150-1,350' -> (1150, 1350)       '346 - 817', '346–817', '346 to 817'
      'Studio' / 'Studio-2' -> (0, 0) / (0, 2)             1.5, 2 (plain numbers)
      {'minValue': 1, 'maxValue': 2} or {'value': '1-2'} (schema.org QuantitativeValue); lists are merged
      None, '', '-', 'N/A', or text with no number -> (None, None)
    Numbers outside `limits` make the whole value (None, None), since they are almost certainly bad data.
    """
    if value is None:
        return None, None
    if isinstance(value, dict):
        lo, hi = value.get("minValue"), value.get("maxValue")
        if lo is not None or hi is not None:
            value = f"{lo if lo is not None else hi}-{hi if hi is not None else lo}"
        else:
            value = value.get("value")
    if isinstance(value, (list, tuple)):
        parts = [parse_range(v, limits) for v in value]
        parts = [p for p in parts if p[0] is not None]
        return (min(p[0] for p in parts), max(p[1] for p in parts)) if parts else (None, None)
    text = _THOUSANDS_COMMA.sub("", str(value).strip().lower())
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    if "studio" in text:
        numbers.append(0.0)
    if not numbers:
        return None, None
    lo, hi = min(numbers), max(numbers)
    if lo < limits[0] or hi > limits[1]:
        return None, None
    return lo, hi


def sqft_factor(floor_size):
    """Multiplier that converts a floorSize to square feet, from its unitCode (missing -> assume square feet; unknown -> None)."""
    unit = floor_size.get("unitCode") if isinstance(floor_size, dict) else None
    if not unit:
        return 1.0
    return SQFT_UNIT_FACTORS.get(str(unit).strip().upper())


def property_type_from_url(url):
    """'.../Crystal-Townhomes/apartment/24734600' -> 'apartment'; '.../home/177414114' -> 'home'."""
    match = re.search(r"/([A-Za-z-]+)/\d+/?(?:[?#].*)?$", str(url))
    return match.group(1).lower() if match else None


def iter_ld_objects(soup):
    """Yield every dict found in every <script type="application/ld+json"> block."""
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text()
        if not raw or not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for obj in (data if isinstance(data, list) else [data]):
            if isinstance(obj, dict):
                yield obj


def extract_offer_price(offers):
    """
    Price of a listing as (low, high, raw text). A single price gives low == high; a range ('1204-1500', or an
    AggregateOffer with lowPrice/highPrice) gives both ends. offers may be a dict or a list of dicts.
    """
    lows, highs, raws = [], [], []
    for offer in (offers if isinstance(offers, list) else [offers]):
        if not isinstance(offer, dict):
            continue
        for key in ("price", "lowPrice", "highPrice"):
            if offer.get(key) is None:
                continue
            lo, hi = parse_range(offer[key], PRICE_LIMITS)
            if lo is not None:
                lows.append(lo)
                highs.append(hi)
                raws.append(str(offer[key]))
    if not lows:
        return None, None, None
    return min(lows), max(highs), "|".join(dict.fromkeys(raws))


def parse_listings(html):
    """
    Each listing appears as two JSON objects sharing the same 'url':
      - an Accommodation (address, geo, size)
      - a Product (offers -> price)
    Merge them on url. Site-level JSON-LD blocks (no street address, e.g. the Redfin homepage) are dropped.
    """
    soup = BeautifulSoup(html, "html.parser")
    merged = {}

    for obj in iter_ld_objects(soup):
        url = obj.get("url")
        if not url:
            continue
        rec = merged.setdefault(url, {"url": url})

        addr = obj.get("address")
        if isinstance(addr, dict):
            rec["Address"] = addr.get("streetAddress")
            rec["City"] = addr.get("addressLocality")
            rec["State"] = addr.get("addressRegion")
            rec["Postal Code"] = addr.get("postalCode")

        geo = obj.get("geo")
        if isinstance(geo, dict):
            # float() directly so negative longitudes keep their sign
            try:
                rec["latitude"] = float(geo.get("latitude"))
                rec["longitude"] = float(geo.get("longitude"))
            except (TypeError, ValueError):
                pass

        if "numberOfRooms" in obj:                  # Redfin puts the number of bedrooms here (0 = studio)
            lo, hi = parse_range(obj["numberOfRooms"], BEDS_LIMITS)
            rec["beds_min"], rec["beds_max"] = lo, hi
            rec["beds_raw"] = json.dumps(obj["numberOfRooms"]) if not isinstance(obj["numberOfRooms"], str) else obj["numberOfRooms"]

        if "floorSize" in obj:
            size = obj["floorSize"]
            factor = sqft_factor(size)
            if factor is None:                      # an unit we don't know how to convert: leave the size empty, don't guess
                print(f"  Warning: unknown floorSize unitCode {size.get('unitCode')!r} for {url}; sqft left empty.")
            else:
                lo, hi = parse_range(size, (SQFT_LIMITS[0] / factor, SQFT_LIMITS[1] / factor))
                if lo is not None:
                    lo, hi = round(lo * factor), round(hi * factor)
                rec["sqft_min"], rec["sqft_max"] = lo, hi
            rec["sqft_raw"] = size.get("value") if isinstance(size, dict) else size

        rec["property_type"] = property_type_from_url(url)

        if "offers" in obj:
            low, high, raw = extract_offer_price(obj["offers"])
            if low is not None:
                rec["Rental Price"] = low              # the starting price, as before (single prices: the price itself)
                rec["rent_min"], rec["rent_max"], rec["price_raw"] = low, high, raw

    return [rec for rec in merged.values() if rec.get("Address")]


def get_total_pages(html):
    """Read N from <span class="pageText" ...>Viewing page 1 of N</span>."""
    soup = BeautifulSoup(html, "html.parser")
    span = soup.select_one('span.pageText[data-rf-test-name="download-and-save-page-number-text"]')
    if span is None:
        span = soup.select_one("span.pageText")
    if span is None:
        return None
    match = re.search(r"of\s+(\d+)", span.get_text(" ", strip=True))
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Progress (checkpoint) helpers
# ---------------------------------------------------------------------------
def page_path(page_num):
    return PROGRESS_DIR / f"page_{page_num:03d}.csv"


def meta_path():
    return PROGRESS_DIR / "meta.json"


def save_page(page_num, df_page):
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    df_page.to_csv(page_path(page_num), index=False)


def load_page(page_num):
    """Return the saved DataFrame for a page, or None if it isn't on disk."""
    path = page_path(page_num)
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"Postal Code": str})


def save_total_pages(total_pages):
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path().write_text(json.dumps({"total_pages": total_pages}))


def load_total_pages():
    try:
        return int(json.loads(meta_path().read_text())["total_pages"])
    except Exception:
        return None


def clear_progress():
    """Delete checkpoint files (keeps the folder itself)."""
    if not PROGRESS_DIR.exists():
        return
    for f in list(PROGRESS_DIR.glob("page_*.csv")) + [meta_path()]:
        if f.exists():
            f.unlink()


# ---------------------------------------------------------------------------
# Versioned dataset helpers
# ---------------------------------------------------------------------------
def save_rentals_version(df):
    """Save df as data/redfin_rentals/redfin_atlanta_rentals_<YYYY-MM-DD_HHMM>.csv and return the path."""
    RENTALS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    path = RENTALS_DIR / f"{VERSION_PREFIX}_{timestamp}.csv"
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
    return path


def latest_rentals_path():
    """Path of the newest saved version, or None if there isn't one."""
    files = sorted(RENTALS_DIR.glob(f"{VERSION_PREFIX}_*.csv"))
    return files[-1] if files else None


def load_latest_rentals():
    """Load the newest saved version (None if there isn't one)."""
    path = latest_rentals_path()
    if path is None:
        print(f"No saved versions found in {RENTALS_DIR}. Run the scraper first.")
        return None
    df = pd.read_csv(path, dtype={"Postal Code": str})
    print(f"Loaded {path.name} ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
async def is_blocked(page, html):
    title = (await page.title()).lower()
    if any(k in title for k in BLOCK_TITLE_KEYWORDS):
        return True
    lowered = html.lower()
    return any(m in lowered for m in BLOCK_BODY_MARKERS)


async def block_heavy_resources(route):
    """Skip images / media / fonts to cut the number of requests per page."""
    if route.request.resource_type in BLOCKED_RESOURCE_TYPES:
        await route.abort()
    else:
        await route.continue_()


async def scrape_single_page(url, need_total=False, headless=HEADLESS):
    """
    Fetch ONE catalog page: fresh Playwright session -> browser -> page -> parse -> close.
    Returns (df_page, total_pages, status, title) where status is "OK", "BLOCKED" or "EMPTY".
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless, args=LAUNCH_ARGS)
        context = await browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1440, "height": 900},
            locale="en-US",
            timezone_id="America/New_York",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        await Stealth().apply_stealth_async(context)
        await context.route("**/*", block_heavy_resources)
        page = await context.new_page()

        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        try:
            await page.wait_for_selector(
                'script[type="application/ld+json"]', state="attached", timeout=15000
            )
        except Exception:
            print("   Warning: no ld+json script appeared within timeout.")

        if need_total:
            try:
                await page.wait_for_selector("span.pageText", state="attached", timeout=10000)
            except Exception:
                pass

        # Light scroll in case more cards/scripts are injected lazily
        for _ in range(3):
            await page.mouse.wheel(0, 1500)
            await asyncio.sleep(0.6)

        html = await page.content()
        title = await page.title()
        blocked = await is_blocked(page, html)

        await browser.close()

    if blocked:
        return pd.DataFrame(columns=COLUMNS), None, "BLOCKED", title

    df_page = pd.DataFrame(parse_listings(html))
    for col in COLUMNS:
        if col not in df_page.columns:
            df_page[col] = pd.NA
    df_page = df_page[COLUMNS]

    status = "OK" if len(df_page) else "EMPTY"
    return df_page, get_total_pages(html), status, title


async def scrape_redfin_rentals(max_pages=None, start_page=1, resume=True, headless=HEADLESS):
    """
    Loop over every catalog page and stack the per-page DataFrames.
    Returns (df, complete) where `complete` is True only if every page 1..N was collected.
    """
    page_frames = []                                          # one DataFrame per page, in page order
    collected = set()                                         # page numbers we have data for
    total_pages = load_total_pages() if resume else None      # None until read from a fetched page
    fetched_this_run = 0

    for page_num in range(1, 10_000):
        if total_pages is not None and page_num > total_pages:
            break
        if max_pages is not None and page_num > max_pages:
            print(f"   Reached --max-pages cap ({max_pages}). Stopping.")
            break

        # Reuse pages already saved on disk (resume), or skip pages before start_page
        saved = load_page(page_num) if (resume or page_num < start_page) else None
        if saved is not None and len(saved):
            print(f"--- Page {page_num}: loaded {len(saved)} listings from disk ---")
            page_frames.append(saved)
            collected.add(page_num)
            continue
        if page_num < start_page:
            continue

        # Page 1 uses BASE_URL; page n uses BASE_URL/page-n
        url = BASE_URL if page_num == 1 else f"{BASE_URL}/page-{page_num}"
        print(f"--- Fetching page {page_num} of {total_pages or '?'}: {url} ---")

        for attempt in range(BLOCK_RETRIES + 1):
            df_page, tp, status, title = await scrape_single_page(
                url, need_total=(total_pages is None), headless=headless
            )
            if status != "BLOCKED":
                break
            if attempt < BLOCK_RETRIES:
                wait = random.uniform(*BLOCK_COOLDOWN)
                print(f"   BLOCKED (title: '{title}'). Cooling down {wait / 60:.1f} min, "
                      f"retry {attempt + 1}/{BLOCK_RETRIES}...")
                await asyncio.sleep(wait)

        if status == "BLOCKED":
            print(f"   BLOCKED after {BLOCK_RETRIES} retries (title: '{title}'). Stopping.")
            print(f"   Progress is saved in '{PROGRESS_DIR}'. Run again later to resume from page {page_num}.")
            break

        if total_pages is None:
            if tp is None:
                print("   Warning: could not find the total page count. Scraping this page only.")
                total_pages = page_num
            else:
                total_pages = tp
                save_total_pages(total_pages)
                print(f"   Total pages detected: {total_pages}")

        print(f"   Parsed {len(df_page)} listings.")
        if status == "EMPTY":
            print(f"   FAILED to find listings (title: '{title}').")
            break

        save_page(page_num, df_page)      # write this page to disk right away
        page_frames.append(df_page)       # and add it to the collection
        collected.add(page_num)
        fetched_this_run += 1

        last_page = total_pages if max_pages is None else min(total_pages, max_pages)
        if page_num < last_page:
            if fetched_this_run % BATCH_SIZE == 0:
                rest = random.uniform(*BATCH_REST)
                print(f"   Batch of {BATCH_SIZE} done. Resting {rest / 60:.1f} min...")
                await asyncio.sleep(rest)
            else:
                await asyncio.sleep(random.uniform(*DELAY_RANGE))

    if not page_frames:
        return pd.DataFrame(columns=COLUMNS), False

    # Final DataFrame = every page stacked together
    df = pd.concat(page_frames, ignore_index=True)
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    complete = total_pages is not None and collected >= set(range(1, total_pages + 1))
    return df, complete


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Scrape Atlanta rental listings from Redfin.")
    p.add_argument("--max-pages", type=int, default=None, help="stop after this many pages (for testing)")
    p.add_argument("--start-page", type=int, default=1, help="first page to fetch from the network")
    p.add_argument("--fresh", action="store_true", help="clear saved progress and start over")
    p.add_argument("--headed", action="store_true", help="show the browser window instead of running headless")
    p.add_argument("--save-partial", action="store_true",
                   help="save a version even if the scrape did not reach the last page")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.fresh:
        clear_progress()

    df, complete = asyncio.run(scrape_redfin_rentals(
        max_pages=args.max_pages,
        start_page=args.start_page,
        resume=True,
        headless=not args.headed and HEADLESS,
    ))

    if df.empty:
        print("No listings collected.")
        return 1

    print(f"\nRows: {len(df)} | missing Rental Price: {df['Rental Price'].isna().sum()}")
    if not complete and not args.save_partial:
        print("Scrape is incomplete, so no dataset version was saved. Checkpointed pages are in "
              f"'{PROGRESS_DIR}'; run the same command again to resume (or use --save-partial).")
        return 1

    save_rentals_version(df)
    if complete:
        clear_progress()
    return 0


if __name__ == "__main__":
    sys.exit(main())
