import argparse
import time
import json
import re
import random
from pathlib import Path
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

# --- Configuration & Constants ---
BASE_URL     = "https://www.math.edu.pl"
SOLUTION_URL = BASE_URL + "/rozwiazanie.php"

# Map of human-readable category names to their URL query parameters
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

# Keywords used to filter out standard website navigation and footer text
NAV_NOISE = [
    "logowanie", "math.edu.pl", "arytmetyka", "algebra", "geometria",
    "analiza", "zadania matematyczne", "ciekawostki", "wzory matematyczne",
    "narzędzia", "szukaj", "matematyka »", "zbiór zadań", "powrót do",
    "wersja do druku", "następne zadanie", "poprzednie zadanie", "© 20", "kontakt",
]

def mathml_to_latex(element) -> str:
    """Recursive conversion of MathML DOM elements to LaTeX strings."""
    
    # Handle base text nodes
    if not element.name:
        return element.string.strip() if element.string else ""
    
    # Recursively process children for wrapper nodes
    if element.name in ["math", "mrow"]:
        return "".join([mathml_to_latex(child) for child in element.children])
    
    # Handle fractions (mfrac -> \frac{num}{den})
    if element.name == "mfrac":
        children = [c for c in element.children if c.name or (isinstance(c, str) and c.strip())]
        num = mathml_to_latex(children[0]) if len(children) > 0 else ""
        den = mathml_to_latex(children[1]) if len(children) > 1 else ""
        return rf"\frac{{{num}}}{{{den}}}"
    
    # Handle superscripts / exponents (msup -> base^{exp})
    if element.name == "msup":
        children = [c for c in element.children if c.name or (isinstance(c, str) and c.strip())]
        base = mathml_to_latex(children[0]) if len(children) > 0 else ""
        exp = mathml_to_latex(children[1]) if len(children) > 1 else ""
        return rf"{base}^{{{exp}}}"

    # Fix for multiplication symbols inside operator (mo) tags
    text = element.get_text().strip()
    if element.name == "mo":
        if text in ["·", "⋅", "•", "*"]:
            return r" \cdot "
        elif text == "×":
            return r" \times "

    # Fallback for simple elements (mn, mi, mo)
    return text

def load_scraped_urls(path: Path) -> set[str]:
    """Reads the JSONL output file to build a set of already processed URLs."""
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
    """Appends a single dictionary record as a JSON line to the output file."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def now_iso() -> str:
    """Returns the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def fetch(url: str, session: requests.Session) -> BeautifulSoup:
    """Fetches a URL and returns a BeautifulSoup object with the correct Polish encoding."""
    resp = session.get(url, timeout=15)
    resp.encoding = "iso-8859-2"
    return BeautifulSoup(resp.text, "html.parser")

def clean(text: str) -> str:
    """Removes empty lines and lines containing known website navigation noise."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    lines = [l for l in lines if not any(n in l.lower() for n in NAV_NOISE)]
    return "\n".join(lines).strip()

def parse(soup: BeautifulSoup, url: str) -> dict:
    """Extracts the title, problem, solution, and next URL from the page HTML."""
    body = soup.find("body")
    if not body: 
        return {"title": "", "problem": "", "solution": "", "contains_images": False, "url": url, "next_url": None}

    # Convert all <math> tags into inline LaTeX
    for math_tag in body.find_all("math"):
        try:
            latex_repr = mathml_to_latex(math_tag)
            math_tag.replace_with(f" ${latex_repr}$ ")
        except Exception:
            continue 

    # Look for bold text matching "Zadanie <number>" to use as the title
    title = next(
        (b.get_text(strip=True) for b in body.find_all("b")
         if re.match(r"Zadanie\s+\d+", b.get_text(strip=True))),
        ""
    )

    # Split the body's content based on horizontal rules (<hr>)
    segments, current = [], []
    for el in body.children:
        if getattr(el, "name", None) == "hr":
            segments.append("\n".join(current).strip())
            current = []
        else:
            text = el.get_text(separator="\n") if el.name else str(el)
            current += [l.strip() for l in text.splitlines() if l.strip()]
    segments.append("\n".join(current).strip())

    # The first segment usually contains the problem description
    raw_segment = clean(segments[0]).removeprefix(title).strip() if len(segments) > 0 else ""

    # Split the text where the word "Rozwiązanie" (Solution) appears
    rozw_split = re.split(r"(?m)^Rozwi[ąa]zanie\s*$", raw_segment, maxsplit=1, flags=re.IGNORECASE)
    if len(rozw_split) == 2:
        problem  = rozw_split[0].strip()
        solution = rozw_split[1].strip()
    else:
        # If not found in the first segment, check the second segment
        problem  = raw_segment
        solution = re.sub(r"^Rozwi[ąa]zanie\s*", "", clean(segments[1]), flags=re.IGNORECASE).strip() if len(segments) > 1 else ""

    # Look for the answer prefix ("Odp.") to separate it from the main problem text
    odp_match = re.search(r"\bOdp[p]?[\.\:]", problem)
    if odp_match:
        answer_part = problem[odp_match.start():].strip()
        problem     = problem[:odp_match.start()].strip()
        solution    = (answer_part + "\n" + solution).strip() if solution else answer_part

    contains_images = bool(body.find("img"))

    # Find the link to the next task to enable crawling
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
    """Formats the parsed data into the standard JSON thread structure."""
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
    """Iterates through all task IDs in a given section and saves them."""
    print(f"\n  {name} ({section})")
    problem_id = 1
    consecutive_misses = 0
    MAX_MISSES = 5  # Stop trying if we hit 5 invalid pages in a row
    saved_in_section = 0

    while True:
        url = f"{SOLUTION_URL}?dzial={section}&id={problem_id}"
        
        # Skip if already downloaded in a previous run
        if url in scraped_urls:
            print(f"    [{problem_id:>3}] skipped")
            problem_id += 1
            consecutive_misses = 0
            continue

        print(f"    [{problem_id:>3}] {url}")
        data = parse(fetch(url, session), url)

        # Check if the page is empty/invalid
        if not data["title"] and not data["problem"]:
            consecutive_misses += 1
            if consecutive_misses >= MAX_MISSES:
                break
            problem_id += 1
            continue

        # Reset misses, build the JSON record, and append it to the file
        consecutive_misses = 0
        thread = build_thread(data, name)
        
        append_jsonl(output_path, thread)
        scraped_urls.add(url)
        saved_in_section += 1

        # Stop if there is no "następne zadanie" (next task) link
        if data["next_url"] is None:
            break

        problem_id += 1
        time.sleep(random.uniform(0, 0.8)) # Rate limiting

    print(f"    -> saved {saved_in_section} problems from this category")
    return saved_in_section

def main():
    """Main execution entry point."""
    parser = argparse.ArgumentParser(description="math_edu_scraper")
    parser.add_argument("--output", "-o", default="math_edu_not_forum.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output)
    scraped_urls = load_scraped_urls(output_path)
    
    session = requests.Session()
    session.headers.update(HEADERS)

    print(f" current {len(scraped_urls)}- tasks saved")
    total_saved = 0

    # Loop through predefined categories and scrape them one by one
    for name, section in SECTIONS:
        saved = scrape_section(name, section, session, output_path, scraped_urls)
        total_saved += saved

    print(f"\n the end {total_saved} problems saved")

if __name__ == "__main__":
    main()