import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


raw_dir_path = 'C:\\Users\\Administrator\\Desktop\\Bi\\facebook_sentiment_intelligence\\scraper\\cleaned_datasets\\'
file_name = os.path.join(raw_dir_path, "python_pandas_keras_questions_cleaned.csv") 
print(f"[+] Loading cleaned data from {file_name}...")

df = pd.read_csv(file_name)  
titles = df["title"].astype(str).tolist()

print("[+] Loaded", len(titles), "questions")


print("[+] Loading Sentence-BERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")  

print("[+] Computing embeddings...")
embeddings = model.encode(titles, show_progress_bar=True)

print("[+] Running BERTopic topic modeling...")
topic_model = BERTopic(
    language="english",
    embedding_model=model,
    min_topic_size=10,       # adjust based on dataset size
    n_gram_range=(1, 3),     # better topic quality
    calculate_probabilities=True
)

topics, probabilities = topic_model.fit_transform(titles, embeddings)

df["topic"] = topics

df.to_csv("questions_with_topics.csv", index=False)
topic_model.save("stackoverflow_topic_model")

print("[+] Done! Topics saved to questions_with_topics.csv")


info = topic_model.get_topic_info()
info.to_csv("topic_summary.csv", index=False)
print(info)