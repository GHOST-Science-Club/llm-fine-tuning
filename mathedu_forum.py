import argparse
import json
import logging
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.math.edu.pl"
SOURCE = "math.edu.pl"

CATEGORIES = {
    "sp":      "Szkoła podstawowa",
    "liceum":  "Szkoła ponadpodstawowa",
    "studia":  "Uczelnie wyższe",
    "zadania": "Zadania różne",
}

CATEGORY_MAX_ID = {
    "sp":      1046,
    "liceum":  6437,
    "studia":  6630,
    "zadania": 346,
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MathEduScraper/1.0)",
    "Accept-Language": "pl,en;q=0.9",
}

logging.basicConfig(level=logging.WARNING) 

session = requests.Session()
session.headers.update(HEADERS)

def get_soup(url: str, retries: int = 3) -> BeautifulSoup | None:
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=20)
     
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as exc:
            if attempt < retries:
                time.sleep(3 * attempt)
            else:
                print(f"      [BŁĄD HTTP] {url}: {exc}")
    return None


def parse_date(raw: str) -> str:
    raw = raw.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            pass
    return raw


def thread_exists(soup: BeautifulSoup) -> bool:
  
    if soup is None:
        return False
    for t in soup.find_all("table"):
        first_row = t.find("tr")
        if first_row and "Autor" in first_row.get_text():
            for row in t.find_all("tr"):
                if len(row.find_all("td")) == 2:
                    return True
    return False

def parse_posts_from_page(soup: BeautifulSoup) -> list[dict]:
  
    posts = []

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
            continue

        author_cell, content_cell = cells

        if not author_cell.get_text(strip=True) and not content_cell.get_text(strip=True):
            continue

        
        a_tag = author_cell.find("a", href=re.compile(r'/forum/uzytkownik'))
        if a_tag:
            author = a_tag.get_text(strip=True)
        else:
            raw = author_cell.get_text(separator=" ", strip=True)
            author = raw.split()[0] if raw else "unknown"

        raw_text = content_cell.get_text(separator="\n").strip()
        lines = raw_text.split("\n")

        date_iso = ""
        content_start = 0
        if lines:
            first_line = lines[0].strip()
            if re.match(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', first_line):
                date_iso = parse_date(first_line)
                content_start = 1

        content_lines = [l for l in lines[content_start:]]
        content = "\n".join(content_lines).strip()
        content = re.sub(r'\n{3,}', '\n\n', content)

        contains_images = False
        for img in content_cell.find_all("img"):
            src = img.get("src", "").lower()
            if not any(x in src for x in ["smil", "icon", "emoji"]):
                contains_images = True
                break

        if not content and not contains_images:
            continue

        posts.append({
            "author": author,
            "date": date_iso,
            "content": content,
            "contains_images": contains_images,
        })

    return posts

def scrape_thread(cat_key: str, task_id: int) -> dict | None:
    
    base_url = f"{BASE_URL}/forum/temat,{cat_key},{task_id}"
    all_posts = []
    title = ""
    page = 0

    while True:
        url = f"{base_url},{page}"
        soup = get_soup(url)

        if page == 0:
            if not thread_exists(soup):
                return None
            h2 = soup.find("h2")
            title = h2.get_text(strip=True) if h2 else f"Zadanie nr {task_id}"

        if soup is None:
            break

        page_posts = parse_posts_from_page(soup)
        all_posts.extend(page_posts)


        next_exists = any(
            int(m.group(1)) > page
            for a in soup.find_all("a", href=re.compile(rf'/forum/temat,{cat_key},{task_id},\d+'))
            if (m := re.search(r',(\d+)$', a.get("href", "")))
        )

        if not next_exists or not page_posts:
            break

        page += 1
        time.sleep(random.uniform(0, 0.8))

    if not all_posts:
        return None

    for i, post in enumerate(all_posts):
        post["index"] = i

    return {
        "source": SOURCE,
        "url": f"{base_url},0",
        "title": title,
        "scraped_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "posts": all_posts,
    }

def load_scraped_urls(path: Path) -> set[str]:
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["url"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Scraper forum math.edu.pl → JSONL")
    parser.add_argument(
        "--categories", "-c",
        nargs="+",
        choices=list(CATEGORIES.keys()),
        default=list(CATEGORIES.keys()),
        metavar="KAT",
        
    )
    parser.add_argument(
        "--output", "-o",
        default="math_edu_forum.jsonl",
        
    )
    parser.add_argument(
        "--start-id",
        type=int, default=1,
        
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    scraped_urls = load_scraped_urls(output_path)
    already = len(scraped_urls)
    if already:
        print(f"  allready downloaded {already} threads.\n")

    total_saved = 0

    for cat_key in args.categories:
        cat_label = CATEGORIES[cat_key]
        max_id = CATEGORY_MAX_ID[cat_key]
        start_id = args.start_id
        total_ids = max_id - start_id + 1

        print(f"\n{'='*60}")
        print(f"  Category: {cat_label} [{cat_key}]")
        print(f"  Problems #{start_id} – #{max_id}  ({total_ids} iterations)")
        print(f"{'='*60}")

        for task_id in range(start_id, max_id + 1):
            url = f"{BASE_URL}/forum/temat,{cat_key},{task_id},0"
            width = len(str(max_id))

            if url in scraped_urls:
                print(f"  [#{task_id:{width}}/{max_id}] skipped (allready there)")
                continue

            print(f"  [#{task_id:{width}}/{max_id}] scraped: {url}", end="", flush=True)
            record = scrape_thread(cat_key, task_id)

            if record:
                append_jsonl(output_path, record)
                scraped_urls.add(url)
                total_saved += 1
                print(f"  ✓  {record['title']}  ({len(record['posts'])} posts)")
            else:
                print(f"  –  no post")

            time.sleep(random.uniform(0, 0.8))

    print(f"\n{'='*60}")
    print(f"  end, total save {total_saved} new threads  → {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()