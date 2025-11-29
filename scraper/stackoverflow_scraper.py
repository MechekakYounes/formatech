import requests
import pandas as pd
import time

API_URL = "https://api.stackexchange.com/2.3/questions"
tags_input = input("Enter tags separated by commas (e.g., 'python,pandas,keras'): ")
tags = [tag.strip() for tag in tags_input.split(",")]
file_name = tags_input.replace(",", "_") + "_questions.csv"
questions = []

for tagged in tags:  # distribute 2500 questions evenly among tags
    print(f"[+] Fetching questions for tag: {tagged}")
    page = 1
    has_more = True

    while has_more and page <= 25:  # limit to first 2500 questions (25 pages * 100 questions/page)
        print(f"Fetching page {page}...")
        params = {
            "order": "desc",
            "sort": "creation",
            "tagged": tagged,
            "site": "stackoverflow",
            "pagesize": 100,
            "page": page
        }

        res = requests.get(API_URL, params=params)
        if res.status_code != 200:
            print(f"Error: {res.status_code}")
            break

        data = res.json()

        for q in data.get("items", []):
            questions.append({
                "tag": tagged,
                "question_id": q["question_id"],
                "display_name": q["owner"].get("display_name", "unknown"),
                "title": q["title"],
                "answer_count": q["answer_count"],
                "score": q["score"],  # upvote count determines which question is more useful
                "is_answered": q["is_answered"],
            })

        has_more = data.get("has_more", False)  # API indicates if more pages exist
        page += 1
        time.sleep(1)  # polite delay to avoid throttling

df = pd.DataFrame(questions)
print(f"Total questions fetched: {len(df)}")
df.to_csv(file_name, index=False)
