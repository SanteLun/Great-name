"""Collect dated CBR release headlines; retain provenance, never invent articles."""
from __future__ import annotations
import hashlib
import re
from pathlib import Path
import requests
from html import unescape
import pandas as pd


def main():
    dates = ["2023-02-10", "2023-03-17", "2023-04-28", "2023-06-09", "2023-07-21",
             "2023-08-15", "2023-09-15", "2023-10-27", "2023-12-15", "2024-02-16", "2024-03-22",
             "2024-04-26", "2024-06-07", "2024-07-26", "2024-09-13", "2024-10-25", "2024-12-20"]
    rows = []
    failures = []
    for day in dates:
        stamp = pd.Timestamp(day).strftime("%d%m%Y")
        url = f"https://www.cbr.ru/press/pr/?file={stamp}_133000key.htm"
        if day == "2023-06-09":
            url = "https://www.cbr.ru/press/pr/?file=09062023_133024key.htm"
        elif day == "2023-08-15":
            url = "https://www.cbr.ru/press/pr/?file=15082023_103000Key.htm"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        response.encoding = "utf-8"
        heading = re.search(r"<h1\b[^>]*>(.*?)</h1>", response.text, re.S | re.I)
        title = unescape(re.sub(r"<[^>]+>", " ", heading.group(1))).strip() if heading else ""
        title = " ".join(title.split())
        if not title or not re.search("ключев", title, re.I):
            failures.append({"date": day, "url": url, "reason": "unexpected headline"})
            continue
        page_text = unescape(re.sub(r"<[^>]+>", " ", response.text))
        if pd.Timestamp(day).strftime("%d.%m.%Y") not in page_text:
            failures.append({"date": day, "url": url, "reason": "unverified publication timestamp"})
            continue
        rows.append({"date": day, "title": title, "text": title, "source": "Пресс-служба Банка России",
                     "url": url, "source_sha256": hashlib.sha256(response.content).hexdigest()})
        print(day, title, flush=True)
    output = Path("data/news/cbr_releases.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    pd.DataFrame(failures, columns=["date", "url", "reason"]).to_csv(output.with_name("collection_failures.csv"), index=False)


if __name__ == "__main__":
    main()
