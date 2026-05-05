import argparse
import time
import json
import re
import random
from pathlib import Path
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

BASE_URL     = "https://www.math.edu.pl"
SOLUTION_URL = BASE_URL + "/rozwiazanie.php"

SECTIONS = [
    ("zadania dla szóstoklasisty",       "sp"),
    ("średnia arytmetyczna",             "srednia-arytmetyczna"),
    ("procenty",                         "procenty"),
    ("prędkość, droga, czas",            "predkosc-droga-czas"),
    ("zbiory",                           "zbiory"),
    ("permutacje",                       "permutacje"),
    ("wariacje",                         "wariacje"),
    ("kombinacje",                       "kombinacje"),
    ("równania z wartością bezwzględną", "rownania-wartosc-bezwzgledna"),
    ("ciąg arytmetyczny",                "ciag-arytmetyczny"),
    ("przelewanie",                      "przelewanie"),
    ("zadania ciekawe",                  "ciekawe"),
    ("zadania różne",                    "rozne"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

NAV_NOISE = [
    "logowanie", "math.edu.pl", "arytmetyka", "algebra", "geometria",
    "analiza", "zadania matematyczne", "ciekawostki", "wzory matematyczne",
    "narzędzia", "szukaj", "matematyka »", "zbiór zadań", "powrót do",
    "wersja do druku", "następne zadanie", "poprzednie zadanie", "© 20", "kontakt",
]


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


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch(url: str, session: requests.Session) -> BeautifulSoup:
    resp = session.get(url, timeout=15)
    resp.encoding = "iso-8859-2"
    return BeautifulSoup(resp.text, "html.parser")


def clean(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    lines = [l for l in lines if not any(n in l.lower() for n in NAV_NOISE)]
    return "\n".join(lines).strip()


def parse(soup: BeautifulSoup, url: str) -> dict:
    body = soup.find("body")

    title = next(
        (b.get_text(strip=True) for b in body.find_all("b")
         if re.match(r"Zadanie\s+\d+", b.get_text(strip=True))),
        ""
    )

    segments, current = [], []
    for el in body.children:
        if getattr(el, "name", None) == "hr":
            segments.append("\n".join(current).strip())
            current = []
        else:
            text = el.get_text(separator="\n") if el.name else str(el)
            current += [l.strip() for l in text.splitlines() if l.strip()]
    segments.append("\n".join(current).strip())

    raw_segment = clean(segments[0]).removeprefix(title).strip() if len(segments) > 0 else ""

    rozw_split = re.split(r"(?m)^Rozwi[ąa]zanie\s*$", raw_segment, maxsplit=1, flags=re.IGNORECASE)
    if len(rozw_split) == 2:
        problem  = rozw_split[0].strip()
        solution = rozw_split[1].strip()
    else:
        problem  = raw_segment
        solution = re.sub(r"^Rozwi[ąa]zanie\s*", "", clean(segments[1]), flags=re.IGNORECASE).strip() if len(segments) > 1 else ""

    odp_match = re.search(r"\bOdp[p]?[\.\:]", problem)
    if odp_match:
        answer_part = problem[odp_match.start():].strip()
        problem     = problem[:odp_match.start()].strip()
        solution    = (answer_part + "\n" + solution).strip() if solution else answer_part

    contains_images = bool(body.find("img"))

    next_url = next(
        (BASE_URL + a["href"] for a in soup.find_all("a")
         if "następne zadanie" in a.get_text().lower()),
        None
    )

    return {
        "title":           title,
        "problem":         problem,
        "solution":        solution,
        "contains_images": contains_images,
        "url":             url,
        "next_url":        next_url,
    }


def build_thread(data: dict, section: str) -> dict:
    scraped_at = now_iso()
    return {
        "source":     "math.edu.pl",
        "url":        data["url"],
        "title":      f"{section} – {data['title']}",
        "scraped_at": scraped_at,
        "posts": [
            {
                "index":           0,
                "author":          "math.edu.pl",
                "date":            scraped_at,
                "content":         data["problem"],
                "contains_images": data["contains_images"],
            },
            {
                "index":           1,
                "author":          "math.edu.pl",
                "date":            scraped_at,
                "content":         data["solution"],
                "contains_images": False,
            },
        ],
    }


def scrape_section(name: str, section: str, session: requests.Session, output_path: Path, scraped_urls: set) -> int:
    print(f"\n  {name} ({section})")
    problem_id = 1
    consecutive_misses = 0
    MAX_MISSES = 5
    saved_in_section = 0

    while True:
        url = f"{SOLUTION_URL}?dzial={section}&id={problem_id}"
        
        if url in scraped_urls:
            print(f"    [{problem_id:>3}] skipped")
            problem_id += 1
            consecutive_misses = 0
            continue

        print(f"    [{problem_id:>3}] {url}")
        data = parse(fetch(url, session), url)

        if not data["title"] and not data["problem"]:
            consecutive_misses += 1
            if consecutive_misses >= MAX_MISSES:
                break
            problem_id += 1
            continue

        consecutive_misses = 0
        thread = build_thread(data, name)
        
        append_jsonl(output_path, thread)
        scraped_urls.add(url)
        saved_in_section += 1

        if data["next_url"] is None:
            break

        problem_id += 1
        time.sleep(random.uniform(0, 0.8))

    print(f"    -> saved {saved_in_section} problems from this category")
    return saved_in_section


def main():
    parser = argparse.ArgumentParser(description="math_edu_scraper")
    parser.add_argument("--output", "-o", default="math_edu_not_forum.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output)
    scraped_urls = load_scraped_urls(output_path)
    
    session = requests.Session()
    session.headers.update(HEADERS)

    print(f" current {len(scraped_urls)}- tasks saved")
    total_saved = 0

    for name, section in SECTIONS:
        saved = scrape_section(name, section, session, output_path, scraped_urls)
        total_saved += saved

    print(f"\n the end {total_saved} problems saved")


if __name__ == "__main__":
    main()