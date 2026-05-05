import argparse
import json
import re
import time
import random
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

BASE = "https://matematyka.pl"
SITEMAP_URL = "https://matematyka.pl/sitemap-1.xml"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; forum-scraper/1.0)"}
THREAD_RE = re.compile(r'-t\d+\.html')


def load_scraped_urls(path: Path) -> set[str]:
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["url"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def get_category_urls(session: requests.Session):
    print("getting the sitemap")
    r = session.get(SITEMAP_URL, timeout=15)
    soup = BeautifulSoup(r.content, "xml")

    category_urls = [loc.text.strip() for loc in soup.find_all("loc") if "viewforum.php" in loc.text.strip()]
    print(f"found {len(category_urls)} in sitemap")
    return category_urls


def get_thread_urls_from_category(category_url: str, session: requests.Session):
    thread_urls = set()
    page_url = category_url

    while page_url:
        try:
            r = session.get(page_url, timeout=15)
        except Exception as e:
            print(f"  error getting {page_url}: {e}")
            break

        soup = BeautifulSoup(r.text, "html.parser")

        for a in soup.find_all("a", href=True):
            full = urljoin(BASE, a["href"])
            if THREAD_RE.search(full) and "matematyka.pl" in full:
                thread_urls.add(full.split("?")[0].split("#")[0]) 

        next_link = soup.select_one("a[rel='next']")
        page_url = urljoin(BASE, next_link["href"].split("?")[0]) if next_link and next_link.get("href") else None
        time.sleep(random.uniform(0.2, 0.6))

    return thread_urls


def extract_content(content_div) -> str:
    return content_div.get_text(separator="\n", strip=True)


def scrape_thread(url: str, session: requests.Session) -> dict | None:
    try:
        r = session.get(url, timeout=15)
    except Exception:
        raise

    soup = BeautifulSoup(r.text, "html.parser")
    if not soup.find("div", class_="post"):
        return None

    title_tag = soup.find("h2", class_="topic-title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    scraped_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    posts = []
    page_url = url

    while page_url:
        if page_url != url:
            try:
                r = session.get(page_url, timeout=15)
                soup = BeautifulSoup(r.text, "html.parser")
            except Exception as e:
                print(f"  error downloading the threas {page_url}: {e}")
                break

        for post in soup.find_all("div", class_="post"):
            author = "unknown"
            author_span = post.select_one("p.author span.responsive-hide strong a")
            if author_span:
                author = author_span.get_text(strip=True)

            time_tag = post.find("time", datetime=True)
            iso_date = time_tag["datetime"] if time_tag else None

            content_div = post.find("div", class_="content")
            if content_div:
                content = extract_content(content_div)
                contains_images = bool(content_div.find("img"))
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

        next_link = soup.select_one("a[rel='next']")
        page_url = urljoin(url, next_link["href"].split("?")[0]) if next_link else None
        if page_url:
            time.sleep(random.uniform(0.2, 0.6))

    if not posts:
        return None

    return {
        "source": "matematyka.pl",
        "url": url,
        "title": title,
        "scraped_at": scraped_at,
        "posts": posts,
    }


def main():
    parser = argparse.ArgumentParser(description="Scraper matematyka.pl")
    parser.add_argument("--output", "-o", default="output_matematykapl1.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output)
    scraped_urls = load_scraped_urls(output_path)
    session = get_session()

    total_threads = 0
    total_posts = 0

    print(f" {len(scraped_urls)} threads downloaded")

    category_urls = get_category_urls(session)

    for cat_url in category_urls:
        print(f"\nKategoria: {cat_url}")
        thread_urls = get_thread_urls_from_category(cat_url, session)
        print(f"  found {len(thread_urls)} threads")

        for thread_url in thread_urls:
            if thread_url in scraped_urls:
                continue

            time.sleep(random.uniform(0.2, 0.6))
            try:
                thread = scrape_thread(thread_url, session)

                if thread is None:
                    print(f"  [skip] {thread_url}")
                    continue

                append_jsonl(output_path, thread)
                scraped_urls.add(thread_url)

                total_threads += 1
                total_posts += len(thread["posts"])
                print(f"  [ok] '{thread['title']}' — {len(thread['posts'])} posts, sum {total_posts})")

            except Exception as e:
                print(f"  [error] {thread_url}: {e}")

    print(f"\nend - downloaded '{args.output}'.")


if __name__ == "__main__":
    main()