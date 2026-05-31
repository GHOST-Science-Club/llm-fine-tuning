# zadania_info.py
# Scraper for the forum.zadania.info discussion forum.
#
# Strategy:
#   Iterate over thread IDs from --start-id to --max-id.  For each ID, build
#   the thread URL, scrape all pages (handling pagination), and append a JSONL
#   record.  A checkpoint file records the last successfully processed ID so
#   the scraper can resume after an interruption.
#
# Usage:
#   python zadania_info.py [--output OUTPUT] [--start-id N] [--max-id N]
#                          [--quiet]

import argparse
import time
import random
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from config import (
    ZADANIA_INFO_BASE_URL,
    ZADANIA_INFO_SOURCE,
    ZADANIA_INFO_DEFAULT_START_ID,
    ZADANIA_INFO_DEFAULT_MAX_ID,
    ZADANIA_INFO_SLEEP,
    REQUEST_TIMEOUT,
)
from utils import (
    load_scraped_urls,
    append_jsonl,
    save_checkpoint,
    load_checkpoint,
    get_session,
    request_with_backoff,
    vprint,
)


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def extract_post_content(content_div) -> str:
    """
    Return the text content of a post, converting inline LaTeX script tags
    to standard dollar-sign notation ($ … $ or $$ … $$).

    forum.zadania.info injects LaTeX via <script type="math/tex"> tags, which
    BeautifulSoup can read but browsers would execute as JavaScript.  We
    replace them with plain-text equivalents before extracting the text.
    """
    # Work on a fresh parse so we don't mutate the outer tree
    from bs4 import BeautifulSoup as _BS
    div = _BS(str(content_div), "html.parser")

    for script in div.find_all("script", type=True):
        script_type = script.get("type", "")
        latex = script.string or ""
        if "mode=display" in script_type:
            # Display-mode math: rendered on its own line
            script.replace_with(f"\n$$\n{latex}\n$$\n")
        elif "math/tex" in script_type:
            # Inline math
            script.replace_with(f"${latex}$")

    return div.get_text(separator="\n", strip=True)


# ---------------------------------------------------------------------------
# Thread scraper
# ---------------------------------------------------------------------------

def scrape_thread(
    url: str,
    session: requests.Session,
    quiet: bool = False,
) -> dict | None:
    """
    Scrape a single thread and return a record dict, or None if invalid/empty.

    Handles pagination by following "next" links until none remain.
    """
    response = request_with_backoff(session, url, quiet=quiet)
    if response is None:
        return None

    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # A page with no .post divs is not a valid thread
    if not soup.find("div", class_="post"):
        return None

    # Some threads use h1, some use h2 for the title
    title_tag = (
        soup.find("h2", class_="topic-title")
        or soup.find("h1", class_="topic-title")
    )
    title = title_tag.get_text(strip=True) if title_tag else ""
    scraped_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    posts: list[dict] = []
    page_url: str | None = url

    while page_url:
        # Fetch subsequent pages (first page already in `soup`)
        if page_url != url:
            response = request_with_backoff(session, page_url, quiet=quiet)
            if response is None:
                vprint(f"  [error] Could not fetch page: {page_url}", quiet=quiet)
                break
            soup = BeautifulSoup(response.text, "html.parser")

        for post in soup.find_all("div", class_="post"):
            # Try two common author selectors
            author_tag = (
                post.select_one(".avatar-container a.username")
                or post.select_one("a.username")
            )
            author = author_tag.get_text(strip=True) if author_tag else "unknown"

            # Prefer the machine-readable datetime attribute when available
            time_tag = post.find("time", datetime=True)
            iso_date = time_tag["datetime"] if time_tag else None

            content_div = post.find("div", class_="content")
            if content_div:
                content = extract_post_content(content_div)
                contains_images = bool(content_div.find("img", class_="postimage"))
            else:
                content = ""
                contains_images = False

            posts.append({
                "index":           len(posts),
                "author":          author,
                "date":            iso_date,
                "content":         content,
                "contains_images": contains_images,
            })

        # Follow pagination
        next_link = soup.select_one("a[rel='next']")
        page_url = urljoin(url, next_link["href"]) if next_link else None

    if not posts:
        return None

    return {
        "source":     ZADANIA_INFO_SOURCE,
        "url":        url,
        "title":      title,
        "scraped_at": scraped_at,
        "posts":      posts,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scraper for forum.zadania.info → JSONL")
    parser.add_argument("--output", "-o", default="output_zadania_info.jsonl",
                        help="Path to the JSONL output file")
    parser.add_argument("--start-id", type=int, default=ZADANIA_INFO_DEFAULT_START_ID,
                        help=f"First thread ID to attempt (default: {ZADANIA_INFO_DEFAULT_START_ID})")
    parser.add_argument("--max-id", type=int, default=ZADANIA_INFO_DEFAULT_MAX_ID,
                        help=f"Last thread ID to attempt (default: {ZADANIA_INFO_DEFAULT_MAX_ID})")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress progress output")
    args = parser.parse_args()

    output_path = Path(args.output)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    scraped_urls = load_scraped_urls(output_path)
    session = get_session()

    # Resume from checkpoint if available (overrides --start-id)
    checkpoint = load_checkpoint(checkpoint_path)
    start_id = checkpoint.get("last_id", args.start_id)
    if start_id != args.start_id:
        vprint(f"Resuming from checkpoint: thread ID {start_id}", quiet=args.quiet)

    total_threads = 0
    total_posts = 0

    vprint(
        f"Starting scrape. Already downloaded: {len(scraped_urls)} threads.",
        quiet=args.quiet,
    )

    for thread_id in range(start_id, args.max_id + 1):
        url = ZADANIA_INFO_BASE_URL.format(thread_id)

        if url in scraped_urls:
            vprint(f"[{thread_id}] Skipped (already downloaded)", quiet=args.quiet)
            continue

        time.sleep(random.uniform(*ZADANIA_INFO_SLEEP))

        try:
            thread = scrape_thread(url, session, quiet=args.quiet)
        except Exception as exc:
            vprint(f"[{thread_id}] Error: {exc} — skipping", quiet=args.quiet)
            save_checkpoint(checkpoint_path, {"last_id": thread_id})
            continue

        if thread is None:
            vprint(f"[{thread_id}] No thread found", quiet=args.quiet)
            save_checkpoint(checkpoint_path, {"last_id": thread_id + 1})
            continue

        append_jsonl(output_path, thread)
        scraped_urls.add(url)
        total_threads += 1
        total_posts += len(thread["posts"])

        vprint(
            f"[{thread_id}] '{thread['title']}' — {len(thread['posts'])} posts "
            f"(total: {total_posts})",
            quiet=args.quiet,
        )
        # Save checkpoint after each successful write
        save_checkpoint(checkpoint_path, {"last_id": thread_id + 1})

    vprint(f"\nDone! New threads saved to '{args.output}'.", quiet=args.quiet)


if __name__ == "__main__":
    main()
