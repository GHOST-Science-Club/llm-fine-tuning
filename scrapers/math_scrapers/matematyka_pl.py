# matematyka_pl.py
# Scraper for the matematyka.pl discussion forum.
#
# Strategy:
#   1. Fetch the XML sitemap to discover all category (forum) URLs.
#   2. Paginate through each category page to collect individual thread URLs.
#   3. Scrape every thread (with pagination) and write each as a JSONL record.
#
# Usage:
#   python matematyka_pl.py [--output OUTPUT] [--quiet]

import argparse
import json
import re
import time
import random
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from config import (
    MATEMATYKA_PL_BASE,
    MATEMATYKA_PL_SITEMAP,
    MATEMATYKA_PL_CATEGORY_SLEEP,
    MATEMATYKA_PL_THREAD_SLEEP,
    MATEMATYKA_PL_PAGE_SLEEP,
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

# Regex that identifies a thread URL on matematyka.pl (contains "-tNNN.html")
THREAD_URL_RE = re.compile(r'-t\d+\.html')


# ---------------------------------------------------------------------------
# Sitemap / category discovery
# ---------------------------------------------------------------------------

def get_category_urls(session: requests.Session, quiet: bool = False) -> list[str]:
    """
    Parse the XML sitemap and return all category (viewforum) URLs.

    The sitemap lists every section of the forum; we filter for pages whose
    URL contains 'viewforum.php', which are the category index pages.
    """
    vprint("Fetching sitemap …", quiet=quiet)
    response = request_with_backoff(session, MATEMATYKA_PL_SITEMAP, quiet=quiet)
    if response is None:
        vprint("[error] Could not fetch sitemap.", quiet=quiet)
        return []

    # lxml is required for XML parsing; listed in requirements.txt
    soup = BeautifulSoup(response.content, "xml")
    category_urls = [
        loc.text.strip()
        for loc in soup.find_all("loc")
        if "viewforum.php" in loc.text.strip()
    ]
    vprint(f"Found {len(category_urls)} categories in sitemap.", quiet=quiet)
    return category_urls


# ---------------------------------------------------------------------------
# Thread URL collection
# ---------------------------------------------------------------------------

def get_thread_urls_from_category(
    category_url: str,
    session: requests.Session,
    quiet: bool = False,
) -> set[str]:
    """
    Collect all thread URLs from a single category, following pagination links.

    Each category page lists topics as anchor tags; we extract those whose
    href matches the thread URL pattern and strip query/fragment noise.
    """
    thread_urls: set[str] = set()
    page_url: str | None = category_url

    while page_url:
        response = request_with_backoff(session, page_url, quiet=quiet)
        if response is None:
            vprint(f"  [error] Could not fetch category page: {page_url}", quiet=quiet)
            break

        soup = BeautifulSoup(response.text, "html.parser")

        for anchor in soup.find_all("a", href=True):
            full_url = urljoin(MATEMATYKA_PL_BASE, anchor["href"])
            if THREAD_URL_RE.search(full_url) and "matematyka.pl" in full_url:
                # Remove query string and fragment to get a canonical URL
                thread_urls.add(full_url.split("?")[0].split("#")[0])

        # Follow the "next page" link if present
        next_link = soup.select_one("a[rel='next']")
        if next_link and next_link.get("href"):
            page_url = urljoin(MATEMATYKA_PL_BASE, next_link["href"].split("?")[0])
        else:
            page_url = None

        time.sleep(random.uniform(*MATEMATYKA_PL_CATEGORY_SLEEP))

    return thread_urls


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def extract_post_content(content_div) -> str:
    """
    Return the plain-text content of a forum post.

    Smiley images add no textual value and are removed before extraction.
    """
    # Remove smiley/emoji images that clutter the text
    for tag in content_div.find_all(class_="smilies"):
        tag.decompose()
    return content_div.get_text(separator="\n", strip=True)


# ---------------------------------------------------------------------------
# Thread scraper
# ---------------------------------------------------------------------------

def scrape_thread(
    url: str,
    session: requests.Session,
    quiet: bool = False,
) -> dict | None:
    """
    Scrape a single thread and return a record dict, or None if empty/invalid.

    Each thread may span multiple pages; we follow pagination until there is
    no "next" link.  Posts are appended in chronological order.
    """
    response = request_with_backoff(session, url, quiet=quiet)
    if response is None:
        raise requests.RequestException(f"Failed to fetch thread: {url}")

    soup = BeautifulSoup(response.text, "html.parser")

    # A page without any .post divs is not a valid thread
    if not soup.find("div", class_="post"):
        return None

    title_tag = soup.find("h2", class_="topic-title")
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
            # Author is inside a nested selector; fall back to "unknown"
            author = "unknown"
            author_span = post.select_one("p.author span.responsive-hide strong a")
            if author_span:
                author = author_span.get_text(strip=True)

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
                "index": len(posts),
                "author": author,
                "date": iso_date,
                "content": content,
                "contains_images": contains_images,
            })

        # Move to the next page of the thread, if any
        next_link = soup.select_one("a[rel='next']")
        if next_link and next_link.get("href"):
            page_url = urljoin(url, next_link["href"].split("?")[0])
            time.sleep(random.uniform(*MATEMATYKA_PL_PAGE_SLEEP))
        else:
            page_url = None

    if not posts:
        return None

    return {
        "source": "matematyka.pl",
        "url": url,
        "title": title,
        "scraped_at": scraped_at,
        "posts": posts,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scraper for matematyka.pl forum")
    parser.add_argument("--output", "-o", default="output_matematyka_pl.jsonl",
                        help="Path to the JSONL output file")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress progress output")
    args = parser.parse_args()

    output_path = Path(args.output)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    scraped_urls = load_scraped_urls(output_path)
    session = get_session()

    total_threads = 0
    total_posts = 0

    vprint(f"Already downloaded: {len(scraped_urls)} threads.", quiet=args.quiet)

    category_urls = get_category_urls(session, quiet=args.quiet)

    # Load checkpoint to know which category was last being processed
    checkpoint = load_checkpoint(checkpoint_path)
    last_category = checkpoint.get("last_category", "")
    skip_until_checkpoint = bool(last_category)

    with output_path.open("a", encoding="utf-8") as out_file:
        for cat_url in category_urls:
            # Resume from the checkpointed category
            if skip_until_checkpoint:
                if cat_url == last_category:
                    skip_until_checkpoint = False
                else:
                    continue

            vprint(f"\nCategory: {cat_url}", quiet=args.quiet)
            thread_urls = get_thread_urls_from_category(cat_url, session, quiet=args.quiet)
            vprint(f"  Found {len(thread_urls)} threads.", quiet=args.quiet)

            for thread_url in thread_urls:
                if thread_url in scraped_urls:
                    continue

                time.sleep(random.uniform(*MATEMATYKA_PL_THREAD_SLEEP))

                try:
                    thread = scrape_thread(thread_url, session, quiet=args.quiet)
                except requests.RequestException as exc:
                    vprint(f"  [error] {thread_url}: {exc}", quiet=args.quiet)
                    continue

                if thread is None:
                    vprint(f"  [skip] {thread_url}", quiet=args.quiet)
                    continue

                # Write immediately and flush so progress survives a crash
                out_file.write(json.dumps(thread, ensure_ascii=False) + "\n")
                out_file.flush()
                scraped_urls.add(thread_url)

                total_threads += 1
                total_posts += len(thread["posts"])
                vprint(
                    f"  [ok] '{thread['title']}' — {len(thread['posts'])} posts "
                    f"(total: {total_posts})",
                    quiet=args.quiet,
                )

            # Save checkpoint after finishing a category
            save_checkpoint(checkpoint_path, {"last_category": cat_url})

    vprint(f"\nDone — output saved to '{args.output}'.", quiet=args.quiet)


if __name__ == "__main__":
    main()