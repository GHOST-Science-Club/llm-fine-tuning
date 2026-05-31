# math_edu_forum.py
# Scraper for the math.edu.pl discussion forum.
#
# Strategy:
#   Iterate over task IDs within each category (sp, liceum, studia, zadania).
#   For each ID construct the thread URL, scrape all pages, and write a JSONL
#   record. A checkpoint file tracks the last (category, task_id) pair so the
#   scraper can resume after an interruption.
#
# Usage:
#   python math_edu_forum.py [--categories CAT ...] [--output OUTPUT]
#                            [--start-id N] [--quiet]

import argparse
import json
import re
import time
import random
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from config import (
    MATH_EDU_BASE,
    MATH_EDU_SOURCE,
    MATH_EDU_CATEGORIES,
    MATH_EDU_CATEGORY_MAX_ID,
    MATH_EDU_SLEEP,
    MATH_EDU_RETRIES,
    REQUEST_TIMEOUT,
)
from utils import load_scraped_urls, append_jsonl, save_checkpoint, load_checkpoint, get_session, vprint

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MathEduScraper/1.0)",
    "Accept-Language": "pl,en;q=0.9",
}


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def get_soup(url: str, session: requests.Session, retries: int = MATH_EDU_RETRIES, quiet: bool = False) -> BeautifulSoup | None:
    """
    Fetch *url* and return a parsed BeautifulSoup, or None on failure.

    Retries up to *retries* times with linear back-off on network errors.
    Returns None for HTTP 404 (thread does not exist) without retrying.
    Handles HTTP 429 with exponential back-off before retrying.
    """
    backoff = 5
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            if attempt < retries:
                time.sleep(3 * attempt)
                continue
            vprint(f"      [HTTP error] {url}: {exc}", quiet=quiet)
            return None

        if resp.status_code == 404:
            return None  # Thread does not exist; not an error

        if resp.status_code == 429:
            vprint(f"      [429] {url} - sleeping {backoff}s", quiet=quiet)
            time.sleep(backoff)
            backoff *= 2
            continue

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            vprint(f"      [HTTP error] {url}: {exc}", quiet=quiet)
            return None

        resp.encoding = "utf-8"
        return BeautifulSoup(resp.text, "html.parser")

    return None


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def parse_date(raw: str) -> str:
    """
    Convert a raw date string from the forum into an ISO-8601 timestamp.
    Falls back to returning the raw string unchanged if no known format matches.
    """
    raw = raw.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            pass
    return raw


# ---------------------------------------------------------------------------
# Thread validity check
# ---------------------------------------------------------------------------

def thread_exists(soup: BeautifulSoup) -> bool:
    """
    Return True if *soup* contains a posts table with the expected structure.
    The forum renders a table with an "Autor" header row for valid threads.
    """
    if soup is None:
        return False
    for t in soup.find_all("table"):
        first_row = t.find("tr")
        if first_row and "Autor" in first_row.get_text():
            for row in t.find_all("tr"):
                if len(row.find_all("td")) == 2:
                    return True
    return False


# ---------------------------------------------------------------------------
# Post extraction
# ---------------------------------------------------------------------------

def parse_posts_from_page(soup: BeautifulSoup, index_offset: int = 0) -> list[dict]:
    """
    Extract all posts from a single page of a thread.

    Each post is stored in a two-column table row: author (left) and content
    (right). The first line of the content cell is treated as the post date
    when it matches the expected date format.
    """
    posts = []

    # Find the posts table by looking for the "Autor" header column
    table = None
    for t in soup.find_all("table"):
        first_row = t.find("tr")
        if first_row and "Autor" in first_row.get_text():
            table = t
            break

    if table is None:
        return posts

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) != 2:
            continue  # Skip header or separator rows

        author_cell, content_cell = cells

        if not author_cell.get_text(strip=True) and not content_cell.get_text(strip=True):
            continue  # Skip completely empty rows

        # Extract author; prefer the linked username, fall back to plain text
        a_tag = author_cell.find("a", href=re.compile(r'/forum/uzytkownik'))
        if a_tag:
            author = a_tag.get_text(strip=True)
        else:
            raw = author_cell.get_text(separator=" ", strip=True)
            author = raw.split()[0] if raw else "unknown"

        # Split content into lines and check if the first line is a date
        raw_text = content_cell.get_text(separator="\n").strip()
        lines = raw_text.split("\n")

        date_iso = ""
        content_start = 0
        if lines:
            first_line = lines[0].strip()
            if re.match(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', first_line):
                date_iso = parse_date(first_line)
                content_start = 1  # Skip the date line from the content body

        content_lines = [l for l in lines[content_start:]]
        content = "\n".join(content_lines).strip()
        content = re.sub(r'\n{3,}', '\n\n', content)

        contains_images = bool(re.search(r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)', content, re.IGNORECASE))

        if not content and not contains_images:
            continue  # Ignore structurally empty posts

        posts.append({
            "index":           index_offset + len(posts),
            "author":          author,
            "date":            date_iso,
            "content":         content,
            "contains_images": contains_images,
        })

    return posts


# ---------------------------------------------------------------------------
# Thread scraper
# ---------------------------------------------------------------------------

def scrape_thread(cat_key: str, task_id: int, session: requests.Session, quiet: bool = False) -> dict | None:
    """
    Scrape one thread identified by *cat_key* and *task_id*.

    Pages are numbered from 0 and collected until no higher page number is
    found among the pagination links, or a page returns no new posts.
    Returns None if the thread does not exist or has no posts.
    """
    base_url = f"{MATH_EDU_BASE}/forum/temat,{cat_key},{task_id}"
    all_posts = []
    title = ""
    page = 0

    while True:
        url = f"{base_url},{page}"
        soup = get_soup(url, session, quiet=quiet)

        # Validate thread existence on the first page only
        if page == 0:
            if not thread_exists(soup):
                return None
            h2 = soup.find("h2")
            title = h2.get_text(strip=True) if h2 else f"Zadanie nr {task_id}"

        if soup is None:
            break

        page_posts = parse_posts_from_page(soup, index_offset=len(all_posts))
        all_posts.extend(page_posts)

        # Check whether any pagination link points to a higher page number
        next_exists = any(
            int(m.group(1)) > page
            for a in soup.find_all("a", href=re.compile(rf'/forum/temat,{cat_key},{task_id},\d+'))
            if (m := re.search(r',(\d+)$', a.get("href", "")))
        )

        if not next_exists or not page_posts:
            break  # No more pages or current page was empty

        page += 1
        time.sleep(random.uniform(*MATH_EDU_SLEEP))

    if not all_posts:
        return None

    return {
        "source":     MATH_EDU_SOURCE,
        "url":        f"{base_url},0",
        "title":      title,
        "scraped_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "posts":      all_posts,
    
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scraper for math.edu.pl forum -> JSONL")
    parser.add_argument("--categories", "-c", nargs="+", choices=list(MATH_EDU_CATEGORIES.keys()),
                        default=list(MATH_EDU_CATEGORIES.keys()), metavar="CATEGORY",
                        help="Categories to scrape (default: all)")
    parser.add_argument("--output", "-o", default="math_edu_forum.jsonl")
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    output_path     = Path(args.output)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    scraped_urls = load_scraped_urls(output_path)
    session      = get_session(extra_headers=HEADERS)

    # Load checkpoint to resume from last (category, task_id) if available
    checkpoint      = load_checkpoint(checkpoint_path)
    resume_category = checkpoint.get("category", "")
    resume_id       = checkpoint.get("last_id", args.start_id)

    already = len(scraped_urls)
    if already:
        vprint(f"  Already downloaded {already} threads.\n", quiet=args.quiet)

    total_saved = 0

    for cat_key in args.categories:
        cat_label = MATH_EDU_CATEGORIES[cat_key]
        max_id    = MATH_EDU_CATEGORY_MAX_ID[cat_key]

        # Determine starting ID: resume from checkpoint for the matching category,
        # skip categories that appear before the checkpointed one entirely
        if resume_category and cat_key == resume_category:
            start_id = resume_id
        elif resume_category and list(MATH_EDU_CATEGORIES.keys()).index(cat_key) < \
                list(MATH_EDU_CATEGORIES.keys()).index(resume_category):
            continue
        else:
            start_id = args.start_id

        total_ids = max_id - start_id + 1

        vprint(f"\n{'='*60}", quiet=args.quiet)
        vprint(f"  Category: {cat_label} [{cat_key}]", quiet=args.quiet)
        vprint(f"  Problems #{start_id} - #{max_id}  ({total_ids} iterations)", quiet=args.quiet)
        vprint(f"{'='*60}", quiet=args.quiet)

        for task_id in range(start_id, max_id + 1):
            url   = f"{MATH_EDU_BASE}/forum/temat,{cat_key},{task_id},0"
            width = len(str(max_id))

            if url in scraped_urls:
                vprint(f"  [#{task_id:{width}}/{max_id}] skipped (already downloaded)", quiet=args.quiet)
                continue

            if not args.quiet:
                print(f"  [#{task_id:{width}}/{max_id}] scraping: {url}", end="", flush=True)

            try:
                record = scrape_thread(cat_key, task_id, session, quiet=args.quiet)
            except requests.RequestException as exc:
                vprint(f"  [error] {url}: {exc}", quiet=args.quiet)
                save_checkpoint(checkpoint_path, {"category": cat_key, "last_id": task_id})
                continue

            if record:
                append_jsonl(output_path, record)
                scraped_urls.add(url)
                total_saved += 1
                vprint(f"  ✓  {record['title']}  ({len(record['posts'])} posts)", quiet=args.quiet)
            else:
                vprint("  -  no posts", quiet=args.quiet)

            # Save checkpoint after each task so we can resume precisely
            save_checkpoint(checkpoint_path, {"category": cat_key, "last_id": task_id + 1})
            time.sleep(random.uniform(*MATH_EDU_SLEEP))

    vprint(f"\n{'='*60}", quiet=args.quiet)
    vprint(f"  Done - saved {total_saved} new threads -> {output_path}", quiet=args.quiet)
    vprint(f"{'='*60}", quiet=args.quiet)


if __name__ == "__main__":
    main()
