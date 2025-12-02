import requests
import pandas as pd
import time

"""
scrapes stackOverflow questions based on user-specified tags and saves them to a CSV file
used the StackExchange API to fetch questions with a delay of 1 second between requests to avoid throttling
tried to fetch up to 2500 questions per tag, distributed evenly across multiple tags if provided
2500 is the maximum allowed by the API for a single query

"""
def is_valid_tag(tag: str) -> bool:

    url = "https://api.stackexchange.com/2.3/tags"
    params = {
        "inname": tag,
        "site": "stackoverflow",
        "pagesize": 1
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()

        # "items" contains tags similar or matching the name
        for item in data.get("items", []):
            if item.get("name", "").lower() == tag.lower():
                return True
        
        return False

    except Exception as e:
        print(f"[Error] Could not validate tag '{tag}': {e}")
        return False

API_URL = "https://api.stackexchange.com/2.3/questions"
tags_input = input("Enter tags separated by commas (e.g., 'python,pandas,keras'): ")
tags = [tag.strip() for tag in tags_input.split(",")]
file_name =  'C:\\Users\\Administrator\\Desktop\\Bi\\facebook_sentiment_intelligence\\scraper\\raw_datasets\\' + tags_input.replace(",", "_") + "_questions.csv"
questions = []
generate_file = True



for tagged in tags:  # distribute 2500 questions evenly among tags
    if (not is_valid_tag(tagged)):
        generate_file = False
        print(f"[!] Tag '{tagged}' is not valid. Skipping...")
        break
    page = 1
    print(f"[+] Fetching questions for tag: {tagged}")
    has_more = True

    while has_more and page <= 25 :  # limit to first 2500 questions (25 pages * 100 questions/page)
        print(f"Fetching page {page} of tag {tagged}...")
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
                "body": q.get("body", ""),
                "answer_count": q["answer_count"],
                "score": q["score"],  # upvote count determines which question is more useful
                "is_answered": q["is_answered"],
            })

        has_more = data.get("has_more", False)  # API indicates if more pages exist
        page += 1
        time.sleep(1)  # delay to avoid throttling
if generate_file:
   df = pd.DataFrame(questions)
   print(f"Total questions fetched: {len(df)}")
   df.to_csv(file_name, index=False)
