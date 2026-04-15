import requests
from config import  API_KEY, MODEL, BASE_URL, DEBUG

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Call the LLM API (OpenAI-compatible).
    Swap BASE_URL / auth headers here when moving to PCSS.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    response = requests.post(
        BASE_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    if DEBUG and response.status_code != 200:
        print(f"\n[API ERROR] Payload: {payload}")
        print(f"[API ERROR] Server response: {response.text}")

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

def debug(stage: str, content: str) -> None:
    """Print a labelled debug block when DEBUG=true."""
    if not DEBUG:
        return
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  [DEBUG] {stage}")
    print(sep)
    print(content)
    print(sep)