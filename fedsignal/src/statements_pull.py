import io
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

US_BASE_URL  = 'https://www.federalreserve.gov'
RBA_BASE_URL = 'https://www.rba.gov.au'
SBP_BASE_URL = 'https://www.sbp.org.pk'
BOE_BASE_URL = 'https://www.bankofengland.co.uk'

# Data directory: project/src/Back_End/data/
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
DATA_DIR.mkdir(exist_ok=True)

# Canonical country keys used for filenames and the labels index
COUNTRY_KEYS = {
    'united states':   'united_states',
    'us':              'united_states',
    'usa':             'united_states',
    'fed':             'united_states',
    'federal reserve': 'united_states',
    'fomc':            'united_states',
    'australia':       'australia',
    'au':              'australia',
    'rba':             'australia',
    'pakistan':        'pakistan',
    'pk':              'pakistan',
    'sbp':             'pakistan',
    'united kingdom':  'united_kingdom',
    'uk':              'united_kingdom',
    'britain':         'united_kingdom',
    'great britain':   'united_kingdom',
    'england':         'united_kingdom',
    'boe':             'united_kingdom',
    'bank of england': 'united_kingdom',
}
BOE_BASE_URL = 'https://www.bankofengland.co.uk'

# FOMC meeting dates 2000–2010 — hardcoded because pre-2011 Fed historical
# pages use a different link format that the scraper can't pick up automatically
_US_DATES_2000_2010 = [
    # 2000
    '20000201', '20000321', '20000516', '20000628', '20000822', '20001003', '20001115',
    # 2001
    '20010131', '20010320', '20010418', '20010515', '20010627', '20010821',
    '20010917', '20011002', '20011106', '20011211',
    # 2002
    '20020130', '20020319', '20020507', '20020626', '20020813', '20020924', '20021106', '20021210',
    # 2003
    '20030129', '20030318', '20030506', '20030625', '20030812', '20030916', '20031028', '20031209',
    # 2004
    '20040128', '20040316', '20040504', '20040630', '20040810', '20040921', '20041110', '20041214',
    # 2005
    '20050202', '20050322', '20050503', '20050630', '20050809', '20050920', '20051101', '20051213',
    # 2006
    '20060131', '20060328', '20060510', '20060629', '20060808', '20060920', '20061025', '20061212',
    # 2007
    '20070131', '20070321', '20070509', '20070618', '20070807', '20070917', '20071024', '20071211',
    # 2008 (includes Jan 22 emergency cut)
    '20080122', '20080130', '20080318', '20080430', '20080625', '20080805',
    '20080916', '20081008', '20081029', '20081216',
    # 2009
    '20090128', '20090318', '20090429', '20090624', '20090812', '20090923', '20091104', '20091216',
    # 2010
    '20100127', '20100316', '20100428', '20100623', '20100810', '20100921', '20101103', '20101214',
]

# Hardcoded recent FOMC meeting dates (2021-2026) — historical pages lag
_US_RECENT_DATES = [
    '20260128', '20251210', '20251029', '20250917',
    '20250730', '20250618', '20250507', '20250319',
    '20250129', '20241218', '20241107', '20240918',
    '20240731', '20240612', '20240501', '20240320',
    '20240131', '20231213', '20231101', '20230920',
    '20230726', '20230614', '20230503', '20230322',
    '20230201', '20221214', '20221102', '20220921',
    '20220727', '20220615', '20220504', '20220316',
    '20220126', '20211215', '20211103', '20210922',
    '20210728', '20210616', '20210428', '20210317',
    '20210127',
]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get(url, timeout=15):
    """GET url; return Response on HTTP 200, None on any error."""
    try:
        resp = requests.get(url, timeout=timeout, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; research-scraper/1.0)'
        })
        if resp.status_code == 200:
            return resp
    except requests.RequestException as e:
        print(f'Request error [{url}]: {e}')
    return None


def _sleep():
    time.sleep(1)


def extract_text_html(soup, extra_selectors=None):
    """Try multiple div patterns used across different eras of central bank websites."""
    selectors = extra_selectors or []
    selectors += [
        ('div', 'col-xs-12 col-sm-8 col-md-9'),
        ('div', 'col-xs-12 col-sm-8 col-md-8'),
        ('div', 'col-xs-12 col-sm-12 col-md-9'),
        ('div', 'col-xs-12'),
    ]
    for tag, cls in selectors:
        el = soup.find(tag, class_=cls)
        if el:
            return el.get_text(strip=True)
    for fallback in [
        soup.find('article'),
        soup.find('div', id='content'),
        soup.find('div', id='leftText'),
        soup.find('div', class_='page-content'),
    ]:
        if fallback:
            return fallback.get_text(strip=True)
    return None


def extract_text_pdf(pdf_bytes):
    """Extract text from raw PDF bytes using pdfplumber."""
    if pdfplumber is None:
        return None
    try:
        pages = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and len(text.strip()) > 50:
                    pages.append(text.strip())
        return '\n'.join(pages) if pages else None
    except Exception as e:
        print(f'PDF extraction error: {e}')
        return None


# ---------------------------------------------------------------------------
# United States (FOMC)
# ---------------------------------------------------------------------------

def _discover_us_dates():
    """Return sorted list of FOMC meeting dates as YYYYMMDD strings."""
    meeting_dates = list(_US_RECENT_DATES) + list(_US_DATES_2000_2010)

    # Scrape historical meeting dates 2000–2020
    for year in range(2020, 1999, -1):
        url = f'{US_BASE_URL}/monetarypolicy/fomchistorical{year}.htm'
        resp = _get(url)
        if resp:
            soup = BeautifulSoup(resp.text, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'pressreleases/monetary' in href and href.endswith('a.htm'):
                    date = href.split('monetary')[1].replace('a.htm', '')
                    if len(date) == 8 and date.isdigit() and date not in meeting_dates:
                        meeting_dates.append(date)
                        print(f'Found historical date: {date}')
        _sleep()

    return sorted(set(meeting_dates))


def scrape_united_states():
    """
    Scrape FOMC statements, minutes, and press conferences.
    Returns a dict keyed by YYYY-MM-DD with sub-keys 'statement', 'minutes', 'press_conference'.
    Also returns a flat statements-only dict for consistency with other country scrapers.
    """
    meeting_dates = _discover_us_dates()
    print(f'\nTotal meeting dates: {len(meeting_dates)}')

    statements       = {}
    minutes          = {}
    press_conferences = {}

    for date in meeting_dates:
        date_iso = f'{date[:4]}-{date[4:6]}-{date[6:]}'

        # Statement
        resp = _get(f'{US_BASE_URL}/newsevents/pressreleases/monetary{date}a.htm')
        if resp:
            text = extract_text_html(BeautifulSoup(resp.text, 'html.parser'))
            if text:
                statements[date_iso] = text
                print(f'Statement:        {date_iso}')

        # Minutes
        resp = _get(f'{US_BASE_URL}/monetarypolicy/fomcminutes{date}.htm')
        if resp:
            text = extract_text_html(BeautifulSoup(resp.text, 'html.parser'))
            if text:
                minutes[date_iso] = text
                print(f'Minutes:          {date_iso}')

        # Press conference (available from April 2011 onward)
        if date >= '20110427':
            resp = _get(f'{US_BASE_URL}/monetarypolicy/fomcpresconf{date}.htm')
            if resp:
                text = extract_text_html(BeautifulSoup(resp.text, 'html.parser'))
                if text:
                    press_conferences[date_iso] = text
                    print(f'Press conference: {date_iso}')

        _sleep()

    print(f'\n--- US Summary ---')
    print(f'Statements:        {len(statements)}')
    print(f'Minutes:           {len(minutes)}')
    print(f'Press conferences: {len(press_conferences)}')
    if statements:
        print(f'Date range:        {min(statements)} to {max(statements)}')

    return statements


# ---------------------------------------------------------------------------
# Australia (Reserve Bank of Australia)
# ---------------------------------------------------------------------------

def _discover_rba_links():
    """
    Scrape RBA rate decision pages for all years (1990-present).
    Each year has its own page at /monetary-policy/int-rate-decisions/YYYY/.
    The link text IS the date (e.g. "10 December 2024").
    Returns list of (date_iso, url) tuples.
    """
    current_year = datetime.now().year
    links = []
    seen = set()

    for year in range(current_year, 1999, -1):
        url = f'{RBA_BASE_URL}/monetary-policy/int-rate-decisions/{year}/'
        resp = _get(url)
        if not resp:
            _sleep()
            continue

        soup = BeautifulSoup(resp.text, 'html.parser')
        found_any = False

        for a in soup.find_all('a', href=True):
            href = a['href']
            if '/media-releases/' not in href:
                continue
            if not re.search(r'/mr-\d{2}-\d+', href):
                continue

            abs_url = href if href.startswith('http') else RBA_BASE_URL + href
            if abs_url in seen:
                continue

            date_text = a.get_text(strip=True)
            date_iso = None
            for fmt in ('%d %B %Y', '%d %b %Y'):
                try:
                    dt = datetime.strptime(date_text.strip(), fmt)
                    date_iso = dt.strftime('%Y-%m-%d')
                    break
                except ValueError:
                    continue

            if date_iso:
                seen.add(abs_url)
                links.append((date_iso, abs_url))
                found_any = True
                print(f'Found RBA date: {date_iso}')

        _sleep()

    if not links:
        print('Warning: No RBA statement links found — the index page structure may have changed.')
    return links


def scrape_australia():
    """
    Scrape Reserve Bank of Australia monetary policy decision statements.
    Returns {YYYY-MM-DD: text}.
    """
    links = _discover_rba_links()
    statements = {}

    for date_iso, url in links:
        resp = _get(url)
        if resp:
            soup = BeautifulSoup(resp.text, 'html.parser')
            text = extract_text_html(soup)
            if text:
                statements[date_iso] = text
                print(f'RBA statement: {date_iso}')
        _sleep()

    print(f'\n--- Australia Summary ---')
    print(f'Statements: {len(statements)}')
    if statements:
        print(f'Date range: {min(statements)} to {max(statements)}')

    return statements


# ---------------------------------------------------------------------------
# Pakistan (State Bank of Pakistan)
# ---------------------------------------------------------------------------

def _discover_sbp_links():
    """
    Scrape the SBP MPC page for Monetary Policy Statement PDF links.
    Returns list of (date_iso, url) tuples.
    Date precision is month only (day defaults to 01).
    """
    index_url = f'{SBP_BASE_URL}/m_policy/mon.asp'
    resp = _get(index_url)
    if not resp:
        print('Warning: Could not fetch SBP index page — returning empty list.')
        return []

    soup = BeautifulSoup(resp.text, 'html.parser')
    links = []
    seen = set()

    for a in soup.find_all('a', href=True):
        href = a['href']
        if not href.lower().endswith('.pdf'):
            continue
        if 'MPS' not in href and 'mps' not in href.lower():
            continue

        # Only fetch English statements — skip Urdu and Compendium variants
        filename = href.split('/')[-1]
        if 'Urdu' in filename or 'Compendium' in filename:
            continue

        # Force HTTPS — older links use http:// which times out
        if href.startswith('http://'):
            href = 'https://' + href[7:]
        url = href if href.startswith('https') else 'https://' + SBP_BASE_URL.lstrip('https://') + '/' + href.lstrip('/')

        # Extract month and year from filename e.g. MPS-Feb-2024-Eng.pdf
        m = re.search(r'MPS[_-](\w+)[_-](\d{4})', filename, re.IGNORECASE)
        if not m:
            continue

        month_str, year_str = m.group(1), m.group(2)
        date_iso = None
        for fmt in ('%b-%Y', '%B-%Y'):
            try:
                dt = datetime.strptime(f'{month_str}-{year_str}', fmt)
                date_iso = dt.strftime('%Y-%m-01')
                break
            except ValueError:
                continue

        if date_iso and url not in seen:
            seen.add(url)
            links.append((date_iso, url))

    if not links:
        print('Warning: No SBP PDF links found — the index page structure may have changed.')
    return links


def scrape_pakistan():
    """
    Scrape State Bank of Pakistan Monetary Policy Statements (PDFs).
    Returns {YYYY-MM-01: text}.
    Requires pdfplumber: pip install pdfplumber
    """
    if pdfplumber is None:
        raise RuntimeError(
            'pdfplumber is required for Pakistan scraping. '
            'Install it with: pip install pdfplumber'
        )

    links = _discover_sbp_links()
    statements = {}

    for date_iso, url in links:
        resp = _get(url)
        if resp:
            text = extract_text_pdf(resp.content)
            if text:
                statements[date_iso] = text
                print(f'SBP statement: {date_iso}')
        _sleep()

    print(f'\n--- Pakistan Summary ---')
    print(f'Statements: {len(statements)}')
    if statements:
        print(f'Date range: {min(statements)} to {max(statements)}')

    return statements


# ---------------------------------------------------------------------------
# United Kingdom (Bank of England)
# ---------------------------------------------------------------------------

def _extract_text_boe(soup):
    """
    BoE pages have no wrapper div — collect h1/h2/h3/p elements sequentially.
    """
    h1 = soup.find('h1')
    if not h1:
        return extract_text_html(soup)
    parts = [h1.get_text(strip=True)]
    for el in h1.find_next_siblings(['h2', 'h3', 'p']):
        text = el.get_text(strip=True)
        if text:
            parts.append(text)
    return '\n'.join(parts) if parts else None


def scrape_united_kingdom():
    """
    Scrape Bank of England Monetary Policy Summary and Minutes.
    The BoE index is JS-rendered so we probe each month directly using the
    known URL pattern. We reuse the response for text extraction (one request
    per month, not two). 0.25s delay between requests.
    Returns {YYYY-MM-01: text}.
    """
    month_names = [
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
    ]
    current_year  = datetime.now().year
    current_month = datetime.now().month
    statements = {}

    for year in range(current_year, 1996, -1):
        for month_idx, month_name in enumerate(month_names, start=1):
            if year == current_year and month_idx > current_month:
                continue
            slug = f'{month_name}-{year}'
            url  = f'{BOE_BASE_URL}/monetary-policy-summary-and-minutes/{year}/{slug}'
            resp = _get(url)
            if resp:
                date_iso = f'{year}-{month_idx:02d}-01'
                soup = BeautifulSoup(resp.text, 'html.parser')
                text = _extract_text_boe(soup)
                if text:
                    statements[date_iso] = text
                    print(f'BoE statement: {date_iso}')
            time.sleep(0.25)

    print(f'\n--- UK Summary ---')
    print(f'Statements: {len(statements)}')
    if statements:
        print(f'Date range: {min(statements)} to {max(statements)}')

    return statements


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

COUNTRY_ALIASES = {
    'united states':   scrape_united_states,
    'us':              scrape_united_states,
    'usa':             scrape_united_states,
    'fed':             scrape_united_states,
    'federal reserve': scrape_united_states,
    'fomc':            scrape_united_states,
    'australia':       scrape_australia,
    'au':              scrape_australia,
    'rba':             scrape_australia,
    'pakistan':        scrape_pakistan,
    'pk':              scrape_pakistan,
    'sbp':             scrape_pakistan,
    'united kingdom':  scrape_united_kingdom,
    'uk':              scrape_united_kingdom,
    'britain':         scrape_united_kingdom,
    'great britain':   scrape_united_kingdom,
    'england':         scrape_united_kingdom,
    'boe':             scrape_united_kingdom,
    'bank of england': scrape_united_kingdom,
}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _country_file(country_key: str) -> Path:
    return DATA_DIR / f'{country_key}_statements.json'


def save_statements(country_key: str, statements: dict):
    """Save statements and their labels to data/{country_key}_statements.json."""
    labels = sorted(statements.keys())
    payload = {'labels': labels, 'statements': statements}
    path = _country_file(country_key)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'Saved {len(statements)} statements -> {path}')
    _update_labels_index()


def load_statements(country: str) -> dict:
    """
    Load saved statements for a country from disk.
    Returns {date: text} dict, or empty dict if not yet scraped.
    """
    key = COUNTRY_KEYS.get(country.strip().lower())
    if key is None:
        raise ValueError(f"Unknown country '{country}'.")
    path = _country_file(key)
    if not path.exists():
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)['statements']


def get_labels(country: str) -> list:
    """
    Return the sorted list of date labels for a country's saved statements.
    """
    key = COUNTRY_KEYS.get(country.strip().lower())
    if key is None:
        raise ValueError(f"Unknown country '{country}'.")
    path = _country_file(key)
    if not path.exists():
        return []
    with open(path, encoding='utf-8') as f:
        return json.load(f)['labels']


def _update_labels_index():
    """Rebuild data/all_labels.json from all saved country files."""
    index = {}
    for country_key in set(COUNTRY_KEYS.values()):
        path = _country_file(country_key)
        if path.exists():
            with open(path, encoding='utf-8') as f:
                index[country_key] = json.load(f)['labels']
    index_path = DATA_DIR / 'all_labels.json'
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2)
    print(f'Labels index updated -> {index_path}')


def get_all_labels() -> dict:
    """Return {country_key: [labels]} for all saved countries."""
    path = DATA_DIR / 'all_labels.json'
    if not path.exists():
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def scrape_statements(country: str) -> dict:
    """
    Scrape central bank policy statements for the given country, save to disk,
    and return the {date: text} dict.

    Args:
        country: Country name or alias (case-insensitive).
                 Supported: 'United States', 'Australia', 'Pakistan', 'United Kingdom'
                 and common abbreviations/aliases.

    Returns:
        dict mapping date strings (YYYY-MM-DD) to statement text.
    """
    key = COUNTRY_KEYS.get(country.strip().lower())
    if key is None:
        supported = sorted({k for k in COUNTRY_KEYS if len(k) > 3})
        raise ValueError(
            f"Unknown country '{country}'. "
            f"Supported values: {supported}"
        )
    fn = COUNTRY_ALIASES.get(country.strip().lower())
    statements = fn()
    save_statements(key, statements)
    return statements


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    country = sys.argv[1] if len(sys.argv) > 1 else 'United States'
    results = scrape_statements(country)
    print(f'\nFetched {len(results)} statements for "{country}"')
