# Polish Math Forums & Exercises Scrapers

This repository contains a collection of Python scrapers designed to extract mathematical discussions, tasks, and solutions from popular Polish educational websites and forums. The scraped data is uniformly saved in JSONL format, making it easy to process for NLP tasks or dataset generation.

## Project Structure

The project relies on a shared architecture to handle rate-limiting, error recovery, and data formatting:
* **`config.py`**: Centralized configuration for request timeouts, retry logic, rate-limit backoffs, and scraper-specific constants (like maximum category IDs and sleep intervals).
* **`utils.py`**: Shared utilities for all scrapers. It handles robust HTTP sessions (automatic retries for 5xx errors, exponential backoff for HTTP 429 Too Many Requests), loading/saving JSONL records, and managing checkpoint files to resume scraping after interruptions.

## Scraper Methodologies

This project includes four distinct scrapers. Below is a breakdown of how each one operates to safely and effectively extract data.

### 1. `matematyka_pl.py` (matematyka.pl Forum)
* **Methodology:** This scraper uses a top-down discovery approach. 
    1. It first fetches the site's XML sitemap to locate all forum category URLs (`viewforum.php`). 
    2. It paginates through each category to collect canonical thread URLs (`-tNNN.html`).
    3. It then visits each thread, follows the internal pagination, and extracts individual posts chronologically, stripping out noisy elements like smiley images.
* **Note:** This scraper relies on parsing an XML sitemap, which requires the `lxml` library (see the Troubleshooting section).

### 2. `zadania_info.py` (forum.zadania.info)
* **Methodology:** This scraper uses an ID-iteration approach.
    1. It loops through thread IDs sequentially (e.g., from 1 to 103900) and constructs the URLs directly.
    2. It fetches the thread, handles any pagination via "next" links, and extracts the posts.
    3. **Key Feature:** The forum dynamically injects LaTeX via `<script type="math/tex">` tags. The scraper intercepts these tags and safely converts them into standard inline (`$...$`) and display (`$$...$$`) plain-text LaTeX markers before saving.

### 3. `math_edu.py` vs. `math_edu_forum.py`
A critical distinction in this project is how we handle `math.edu.pl`, as it hosts both a repository of math tasks and a traditional discussion forum. 

* **`math_edu.py` (Website Tasks & Solutions):**
    * **Target:** The main educational website consisting of curated math problems and answers.
    * **Methodology:** Iterates over specific mathematical topics (e.g., fractions, permutations) and task IDs by hitting the `rozwiazanie.php` endpoint. It parses the page structure based on HTML `<hr>` tags and text keywords to cleanly separate the "Problem" from the "Solution". 
    * **Key Feature:** Converts MathML DOM elements recursively into standard LaTeX strings.

* **`math_edu_forum.py` (Discussion Forum):**
    * **Target:** The user discussion forum attached to the site.
    * **Methodology:** Iterates over thread IDs bounded by predefined maximums for specific categories (primary school, high school, university, etc.). It constructs the forum URL structure (`/forum/temat,CAT,ID,PAGE`) and scrapes all paginated content.
    * **Key Feature:** Relies on parsing custom two-column HTML tables to pair post authors with their content, using regex to properly isolate datestamps from the text bodies.

## Setup & Installation

1. Ensure you have Python 3.10+ installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Each script can be run directly from the command line. They will automatically generate `.checkpoint.json` files alongside their output `.jsonl` files, allowing you to stop and resume them seamlessly.
