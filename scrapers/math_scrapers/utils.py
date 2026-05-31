# utils.py
# Shared helpers used by every scraper in this project.
# Import from here instead of duplicating code across scraper files.

import json
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    USER_AGENT,
    REQUEST_TIMEOUT,
    RETRY_TOTAL,
    RETRY_BACKOFF_FACTOR,
    RETRY_STATUS_FORCELIST,
    RATE_LIMIT_BACKOFF_BASE,
)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def load_scraped_urls(path: Path) -> set[str]:
    """
    Read a JSONL output file and return the set of URLs already scraped.

    This allows every scraper to skip previously downloaded items and resume
    from where it left off without re-fetching data.
    """
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["url"])
            except (json.JSONDecodeError, KeyError):
                # Skip malformed lines silently; they won't be re-added.
                pass
    return done


def append_jsonl(path: Path, record: dict) -> None:
    """Append a single record as a JSON line to *path*, creating the file if needed."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(checkpoint_path: Path, data: dict) -> None:
    """
    Persist checkpoint *data* as JSON so a scraper can resume after a crash.

    Example data: {"category": "liceum", "last_id": 312}
    """
    with checkpoint_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_checkpoint(checkpoint_path: Path) -> dict:
    """
    Load checkpoint data written by *save_checkpoint*.

    Returns an empty dict if the file does not exist or is corrupt.
    """
    if not checkpoint_path.exists():
        return {}
    try:
        with checkpoint_path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# HTTP session factory
# ---------------------------------------------------------------------------

def get_session(extra_headers: dict | None = None) -> requests.Session:
    """
    Create a requests.Session pre-configured with:
      - a standard User-Agent header
      - automatic retry on common server errors (5xx)
      - optional extra headers

    HTTP 429 (Too Many Requests) is intentionally excluded from the automatic
    retry list because it requires exponential back-off; handle it with
    *request_with_backoff* instead.
    """
    session = requests.Session()
    headers = {"User-Agent": USER_AGENT}
    if extra_headers:
        headers.update(extra_headers)
    session.headers.update(headers)

    retry_policy = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=RETRY_STATUS_FORCELIST,
        # Do not raise on redirect; allow the session to follow them.
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry_policy))
    session.mount("http://", HTTPAdapter(max_retries=retry_policy))
    return session


# ---------------------------------------------------------------------------
# Rate-limit-aware GET
# ---------------------------------------------------------------------------

def request_with_backoff(
    session: requests.Session,
    url: str,
    timeout: int = REQUEST_TIMEOUT,
    max_attempts: int = 5,
    quiet: bool = False,
) -> requests.Response | None:
    """
    Perform a GET request and handle HTTP 429 with exponential back-off.

    On a 429 response the function sleeps for RATE_LIMIT_BACKOFF_BASE * 2^attempt
    seconds before retrying.  Other HTTP errors are raised immediately.
    Returns None if all attempts are exhausted or a non-retryable error occurs.
    """
    backoff = RATE_LIMIT_BACKOFF_BASE
    for attempt in range(max_attempts):
        try:
            response = session.get(url, timeout=timeout)
        except requests.RequestException as exc:
            vprint(f"[network error] {url}: {exc}", quiet=quiet)
            return None

        if response.status_code == 429:
            vprint(
                f"[429 rate-limited] {url} – sleeping {backoff}s (attempt {attempt + 1}/{max_attempts})",
                quiet=quiet,
            )
            time.sleep(backoff)
            backoff *= 2  # exponential back-off
            continue

        return response

    vprint(f"[gave up] {url} after {max_attempts} attempts due to rate limiting.", quiet=quiet)
    return None


# ---------------------------------------------------------------------------
# Verbosity helper
# ---------------------------------------------------------------------------

def vprint(message: str, quiet: bool = False) -> None:
    """Print *message* unless *quiet* is True."""
    if not quiet:
        print(message)
