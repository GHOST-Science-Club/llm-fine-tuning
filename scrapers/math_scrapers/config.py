# config.py
# Central configuration for all scrapers.
# Adjust these values without touching scraper logic.

# ---------------------------------------------------------------------------
# Common HTTP settings
# ---------------------------------------------------------------------------

# Default request timeout in seconds
REQUEST_TIMEOUT = 15

# Number of automatic retries on server errors (5xx)
RETRY_TOTAL = 3

# Multiplier for urllib3 exponential back-off between retries (seconds)
RETRY_BACKOFF_FACTOR = 1

# HTTP status codes that trigger an automatic retry
RETRY_STATUS_FORCELIST = [500, 502, 503, 504]

# Seconds to sleep before re-attempting after an HTTP 429 (Too Many Requests).
# Each consecutive 429 doubles this value (exponential back-off).
RATE_LIMIT_BACKOFF_BASE = 5  # seconds

# Shared User-Agent header used by every scraper
USER_AGENT = "Mozilla/5.0 (compatible; forum-scraper/1.0)"

# ---------------------------------------------------------------------------
# matematyka.pl scraper
# ---------------------------------------------------------------------------

MATEMATYKA_PL_BASE = "https://matematyka.pl"
MATEMATYKA_PL_SITEMAP = "https://matematyka.pl/sitemap-1.xml"


# ---------------------------------------------------------------------------
# math.edu.pl forum scraper
# ---------------------------------------------------------------------------

MATH_EDU_BASE = "https://www.math.edu.pl"
MATH_EDU_SOURCE = "math.edu.pl"

# Category keys mapped to human-readable labels
MATH_EDU_CATEGORIES = {
    "sp":      "Primary school",
    "liceum":  "Secondary school",
    "studia":  "University",
    "zadania": "Miscellaneous problems",
}

# Highest known task ID per category (upper bound for iteration)
MATH_EDU_CATEGORY_MAX_ID = {
    "sp":      1046,
    "liceum":  6437,
    "studia":  6630,
    "zadania": 346,
}



# Number of HTTP retries for get_soup()
MATH_EDU_RETRIES = 3

# ---------------------------------------------------------------------------
# forum.zadania.info scraper
# ---------------------------------------------------------------------------

ZADANIA_INFO_BASE_URL = "https://forum.zadania.info/viewtopic.php?t={}"
ZADANIA_INFO_SOURCE = "forum.zadania.info"

# Default thread-ID range to scrape
ZADANIA_INFO_DEFAULT_START_ID = 1
ZADANIA_INFO_DEFAULT_MAX_ID = 103900

# ---------------------------------------------------------------------------
# math.edu.pl exercise scraper (math_edu.py)
# ---------------------------------------------------------------------------

MATH_EDU_EX_BASE_URL = "https://www.math.edu.pl"
MATH_EDU_EX_SOLUTION_URL = MATH_EDU_EX_BASE_URL + "/rozwiazanie.php"
# ---------------------------------------------------------------------------
# sleepers
# ---------------------------------------------------------------------------


# Random sleep range (seconds) between thread requests
ZADANIA_INFO_SLEEP = (0.1, 0.9)

# Random sleep range (seconds) between category page requests
MATEMATYKA_PL_CATEGORY_SLEEP = (0.1, 0.9)

# Random sleep range (seconds) between thread requests
MATEMATYKA_PL_THREAD_SLEEP = (0.1, 0.9)

# Random sleep range (seconds) between pagination requests inside a thread
MATEMATYKA_PL_PAGE_SLEEP = (0.1, 0.9)
# Random sleep range (seconds) between task requests
MATH_EDU_SLEEP = (0.1, 0.9)

