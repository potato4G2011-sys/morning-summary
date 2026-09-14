#!/usr/bin/env python3
"""
Morning health summary.
Pulls yesterday's sleep, HRV, and resting heart rate from Google Health
using the ghealth CLI, sends it to Gemini for a coaching style summary,
then posts the result to Telegram.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

import requests

# Secrets come from environment variables, never hardcode these
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

GHEALTH_BIN = os.environ.get("GHEALTH_BIN", "./ghealth")
GEMINI_MODEL = "gemini-flash-latest"

COACHING_PROMPT = """You are a fitness coach and recovery specialist for a junior tennis
player. Look at the training load, HRV, resting heart rate, and sleep data below.
Identify the trend between them and explain what is driving any negative trend.
Then give a plan in two sections: what to start or keep doing, and what to stop
or avoid. Be specific to this data, not generic. Flag anything that looks like
overtraining or insufficient recovery for his age. Keep it direct and practical,
no fluff, no filler sentences.

Data:
{data}
"""


def run_ghealth(args):
    """Run a ghealth command and return parsed JSON."""
    result = subprocess.run(
        [GHEALTH_BIN] + args,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def get_yesterday_range():
    yesterday = datetime.now() - timedelta(days=1)
    start = yesterday.strftime("%Y-%m-%d")
    return start


def collect_health_data():
    start = get_yesterday_range()

    sleep = run_ghealth(["data", "sleep", "list", "--from", start])
    hrv = run_ghealth(["data", "heart-rate-variability", "list", "--from", start])
    rhr = run_ghealth(["data", "heart-rate", "list", "--from", start, "--limit", "50"])

    return {
        "sleep": sleep.get("dataPoints", []),
        "hrv": hrv.get("dataPoints", []),
        "heart_rate": rhr.get("dataPoints", []),
    }


def call_gemini(data, max_retries=4):
    prompt = COACHING_PROMPT.format(data=json.dumps(data, indent=2))

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

    for attempt in range(max_retries):
        response = requests.post(
            url,
            headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 4096},
            },
            timeout=60,
        )

        if response.status_code == 503 and attempt < max_retries - 1:
            wait = (2 ** attempt) + 1  # 2, 3, 5, 9 seconds
            print(f"Gemini overloaded (503), retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
            continue

        if not response.ok:
            print(f"Gemini error {response.status_code}: {response.text}", file=sys.stderr)

        response.raise_for_status()
        result = response.json()
        candidate = result["candidates"][0]

        if candidate.get("finishReason") == "MAX_TOKENS":
            print("Warning: response hit the token limit and may be cut off", file=sys.stderr)

        return candidate["content"]["parts"][0]["text"]

    raise RuntimeError("Gemini stayed unavailable after retries")


def format_for_telegram(text):
    """Convert standard Markdown (### headers, **bold**) into what
    Telegram's legacy Markdown parse mode actually understands."""
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip("#").strip()
        if line.startswith("#") and stripped:
            lines.append(f"*{stripped}*")
        else:
            lines.append(line)
    result = "\n".join(lines)
    return result.replace("**", "*")


def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    formatted = format_for_telegram(text)
    # Telegram caps a single message at 4096 characters, split if needed
    chunks = [formatted[i:i + 4000] for i in range(0, len(formatted), 4000)] or [formatted]
    for chunk in chunks:
        response = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "Markdown"},
            timeout=30,
        )
        if not response.ok:
            # Formatting broke on a mismatched symbol, resend as plain text
            # rather than losing the message entirely
            print(f"Telegram Markdown parse failed, retrying as plain text: {response.text}", file=sys.stderr)
            response = requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk},
                timeout=30,
            )
        response.raise_for_status()


def main():
    try:
        data = collect_health_data()
    except subprocess.CalledProcessError as e:
        print(f"ghealth command failed: {e.stderr}", file=sys.stderr)
        sys.exit(1)

    if not any(data.values()):
        send_telegram("No health data found for yesterday. Check the Fitbit Air sync.")
        return

    summary = call_gemini(data)
    send_telegram(f"Morning recovery summary\n\n{summary}")


if __name__ == "__main__":
    main()
