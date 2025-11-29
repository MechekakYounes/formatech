import requests
import pandas as pd
import time

API_URL = "https://api.stackexchange.com/2.3/questions"
tagged = input("Enter the tag to filter questions by (e.g., 'python'): ") #subject to scrape 
file_name = tagged + "_questions.csv"

params = {
    "order": "desc",
    "sort": "creation",
    "tagged": tagged,
    "site": "stackoverflow",
    "pagesize": 100
}

res = requests.get(API_URL, params=params).json()
#print(res)

questions = []
page = 1
has_more = True
while has_more:
    print(f"Fetching page {page}...")
    params["page"] = page
    res = requests.get(API_URL, params=params)
    data = res.json()

    if res.status_code != 200:
        print(f"Error: {res.status_code}")
        break

    for q in data["items"]:
     questions.append({
        "question_id": q["question_id"],
        "display_name": q["owner"]["display_name"],
        "title": q["title"],
        "answer_count": q["answer_count"],
        "score": q["score"], #upvote count determine which question is more usefull
        "is_answered": q["is_answered"],

    })
    has_more = data.get("has_more", False)  # API indicates if more pages exist
    page += 1
    time.sleep(1) 

df = pd.DataFrame(questions)
print(f"Total questions fetched: {len(df)}")
df.to_csv(file_name, index=False)


