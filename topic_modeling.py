import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


raw_dir_path = 'C:\\Users\\Administrator\\Desktop\\Bi\\facebook_sentiment_intelligence\\scraper\\cleaned_datasets\\'
file_name = os.path.join(raw_dir_path, "javascript_questions_cleaned.csv") 

df = pd.read_csv(file_name)  
titles = df["title"].astype(str).tolist()


model = SentenceTransformer("all-MiniLM-L6-v2")  
embeddings = model.encode(titles, show_progress_bar=True)

topic_model = BERTopic(
    language="english",
    embedding_model=model,
    min_topic_size=10,       
    n_gram_range=(1, 3),     
    calculate_probabilities=True
)

topics, probabilities = topic_model.fit_transform(titles, embeddings)

df["topic"] = topics

df.to_csv("questions_with_topics.csv", index=False)
#topic_model.save("topics_model")

print("[+] Done! Topics saved to questions_with_topics.csv")


info = topic_model.get_topic_info()
info.to_csv("topic_summary.csv", index=False)
