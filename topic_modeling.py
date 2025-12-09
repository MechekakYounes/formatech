import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

def compute_difficulty(df: pd.DataFrame) -> pd.DataFrame:

    if "view_count" not in df.columns:
        df["view_count"] = 0
    if "score" not in df.columns:
        df["score"] = 0
    if "answer_count" not in df.columns:
        df["answer_count"] = 0
    if "is_answered" not in df.columns:
        df["is_answered"] = False

    # Prevent division by zero
    df["score"] = df["score"].fillna(0)
    df["answer_count"] = df["answer_count"].fillna(0)
    df["is_answered"] = df["is_answered"].fillna(False)
    df["view_count"] = df.get("view_count", 0).fillna(0)

    # Normalize views so they remain comparable
    df["view_norm"] = (df["view_count"] - df["view_count"].min()) / (
        df["view_count"].max() - df["view_count"].min() + 1
    )

    df["difficulty"] = (
        1 / (1 + df["score"]) +
        1 / (1 + df["answer_count"]) +
        (0.5 * (~df["is_answered"])) + 
        (1 / (1 + df["view_norm"]))
    )

    return df

cleaned_dir_path = 'C:\\Users\\Administrator\\Desktop\\Bi\\facebook_sentiment_intelligence\\scraper\\cleaned_datasets\\'
file_name= "python_pandas_keras_questions_cleaned.csv"
file_path = os.path.join(cleaned_dir_path, file_name) 
out_file_name = file_name.replace("questions_cleaned", "topics_modeling")
df = pd.read_csv(file_path)  
titles = df["title"].astype(str).tolist()


model = SentenceTransformer("all-MiniLM-L6-v2")  
embeddings = model.encode(titles, show_progress_bar=True)

#model init 
topic_model = BERTopic(
    language="english",
    embedding_model=model,
    min_topic_size=15,       
    n_gram_range=(1, 3),     
    calculate_probabilities=True
)

topics, probabilities = topic_model.fit_transform(titles, embeddings)

df["topic"] = topics
df.to_csv("questions_with_topics.csv", index=False)
df = compute_difficulty(df)
df.to_csv("questions_with_difficulty.csv", index=False)



info = topic_model.get_topic_info()
info.to_csv(out_file_name, index=False)
topic_model.save("topics_model")
