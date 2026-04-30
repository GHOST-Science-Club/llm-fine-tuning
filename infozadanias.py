import argparse
import json
import random
import time
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://forum.zadania.info/viewtopic.php?t={}"
SOURCE = "forum.zadania.info"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; forum-scraper/1.0)"}


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


def extract_content(content_div) -> str:
    div = BeautifulSoup(str(content_div), "html.parser")
    for script in div.find_all("script", type=True):
        script_type = script.get("type", "")
        latex = script.string or ""
        if "mode=display" in script_type:
            script.replace_with(f"\n$$\n{latex}\n$$\n")
        elif "math/tex" in script_type:
            script.replace_with(f"${latex}$")
    return div.get_text(separator="\n", strip=True)


def scrape_thread(url: str, session: requests.Session) -> dict | None:
    try:
        response = session.get(url, timeout=15)
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    if not soup.find("div", class_="post"):
        return None

    title_tag = soup.find("h2", class_="topic-title") or soup.find("h1", class_="topic-title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    scraped_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    posts = []
    page_url = url

    while page_url:
        if page_url != url:
            try:
                response = session.get(page_url, timeout=15)
                soup = BeautifulSoup(response.text, "html.parser")
            except requests.RequestException:
                break

        for post in soup.find_all("div", class_="post"):
            author_tag = post.select_one(".avatar-container a.username") or post.select_one("a.username")
            author = author_tag.get_text(strip=True) if author_tag else "unknown"

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
        page_url = urljoin(url, next_link["href"]) if next_link else None

    if not posts:
        return None

    return {
        "source": SOURCE,
        "url": url,
        "title": title,
        "scraped_at": scraped_at,
        "posts": posts,
    }


def main():
    parser = argparse.ArgumentParser(description="Scraper forum.zadania.info")
    parser.add_argument("--output", "-o", default="output_forum.zadania.info.jsonl", help="Plik wyjściowy")
    parser.add_argument("--start-id", type=int, default=1, help="Początkowe ID wątku")
    parser.add_argument("--max-id", type=int, default=103900, help="Końcowe ID wątku")
    args = parser.parse_args()

    output_path = Path(args.output)
    scraped_urls = load_scraped_urls(output_path)
    
    session = requests.Session()
    session.headers.update(HEADERS)

    total_threads = 0
    total_posts = 0

    print(f"Rozpoczynam pobieranie. Już pobrano: {len(scraped_urls)} wątków.")

    for thread_id in range(args.start_id, args.max_id + 1):
        url = BASE_URL.format(thread_id)
        
        if url in scraped_urls:
            print(f"[{thread_id}] Pominięto (już pobrano)")
            continue

        time.sleep(random.uniform(0.1, 0.9))
        try:
            thread = scrape_thread(url, session)

            if thread is None:
                print(f"[{thread_id}] Brak wątku")
                continue

            append_jsonl(output_path, thread)
            scraped_urls.add(url)
            total_threads += 1
            total_posts += len(thread["posts"])
            print(f"[{thread_id}] '{thread['title']}' — {len(thread['posts'])} postów (suma: {total_posts})")

        except Exception as e:
            print(f"[{thread_id}] Error: {e} — pomijam")

    print(f"\nGotowe! Zapisano nowe wątki w '{args.output}'.")


if __name__ == "__main__":
    main()