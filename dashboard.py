import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import requests
import plotly.express as px
from typing import Optional
from bs4 import BeautifulSoup
import re 
# Try importing BERTopic
try:
    from bertopic import BERTopic
    BER_AVAILABLE = True
except Exception:
    BER_AVAILABLE = False

st.set_page_config(page_title="Educational Insights", layout="wide")


# Utilities: scraper & helpers
CACHE_CSV = "scraped_cache.csv"  # cache file for scraped results
MODEL_FOLDER_DEFAULT = "topics_model"  # default model folder


def is_valid_tag(tag: str) -> bool:
    """Check if a StackOverflow tag exists via StackExchange API"""
    url = "https://api.stackexchange.com/2.3/tags"
    params = {"inname": tag, "site": "stackoverflow", "pagesize": 1}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        for item in data.get("items", []):
            if item.get("name", "").lower() == tag.lower():
                return True
        return False
    except Exception:
        return False

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "question_id" in df.columns:
        df.drop_duplicates(subset=["question_id"], inplace=True)

    if "body" in df.columns:
        df["body"] = df["body"].astype(str).apply(lambda x: BeautifulSoup(x, "html.parser").get_text())

    code_pattern = r"```.*?```|`.*?`"
    if "body" in df.columns:
        df["body"] = df["body"].apply(lambda x: re.sub(code_pattern, " ", x, flags=re.DOTALL))

    text_cols = ["title", "body"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: " ".join(x.split()))

    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE
    )
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: emoji_pattern.sub("", x))

    df["text"] = df["title"].astype(str) + " " + df["body"].astype(str)
    return df


def scrape_stackoverflow(tags_input: str, max_pages_per_tag: int = 25) -> pd.DataFrame:
    """
    Scrape up to pages_per_tag * 100 questions per tag (2500 by default).
    Returns a DataFrame with title, body, score, view_count, answer_count, is_answered, tag.
    """
    tags = [t.strip() for t in tags_input.split(",") if t.strip()]
    if not tags:
        raise ValueError("No tags provided.")

    # Validate tags
    for tag in tags:
        if not is_valid_tag(tag):
            raise ValueError(f"Invalid tag: {tag}")

    API_URL = "https://api.stackexchange.com/2.3/questions"
    rows = []

    for tag in tags:
        page = 1
        has_more = True
        while has_more and page <= max_pages_per_tag:
            params = {
                "order": "desc",
                "sort": "creation",
                "tagged": tag,
                "site": "stackoverflow",
                "pagesize": 100,
                "page": page,
                "filter": "withbody",  # try to get body if available
            }
            r = requests.get(API_URL, params=params, timeout=15)
            if r.status_code != 200:
                # If rate limited or error, stop this tag
                break
            data = r.json()
            items = data.get("items", [])
            for q in items:
                rows.append({
                    "tag": tag,
                    "title": q.get("title", ""),
                    "body": q.get("body", "") or "",
                    "score": q.get("score", 0),
                    "view_count": q.get("view_count", 0),
                    "answer_count": q.get("answer_count", 0),
                    "is_answered": q.get("is_answered", False),
                })
            has_more = data.get("has_more", False)
            page += 1
            time.sleep(1) # to avoid hitting rate limits must wait for 1 second
    df = pd.DataFrame(rows)
    return df


def compute_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a stable difficulty column on the DataFrame."""
    # Ensure columns exist and numeric
    df["view_count"] = pd.to_numeric(df.get("view_count", 0), errors="coerce").fillna(0)
    df["score"] = pd.to_numeric(df.get("score", 0), errors="coerce").fillna(0)
    df["answer_count"] = pd.to_numeric(df.get("answer_count", 0), errors="coerce").fillna(0)
    df["is_answered"] = df.get("is_answered", False)
    df["is_answered"] = df["is_answered"].astype(bool).fillna(False)

    # view normalization (safe)
    min_v = df["view_count"].min() if not df["view_count"].empty else 0
    max_v = df["view_count"].max() if not df["view_count"].empty else 0
    denom = (max_v - min_v) + 1
    df["view_norm"] = (df["view_count"] - min_v) / denom

    # difficulty formula
    df["difficulty"] = (
         (1.5 * df["score"]) +
        (-0.25 * df["answer_count"]) +
        (0.5 * (~df["is_answered"])) +
        ( df["view_norm"])
    )

    return df


# Model loader (cached)
@st.cache_resource
def load_bertopic_model(folder: str) -> Optional[BERTopic]:
    if not BER_AVAILABLE:
        return None
    if not os.path.exists(folder):
        return None
    try:
        model = BERTopic.load(folder)
        return model
    except Exception:
        return None


def topic_keywords_from_model(model: BERTopic, top_n: int = 5):
    mapping = {}
    try:
        info = model.get_topic_info()
        for tid in info["Topic"]:
            if tid == -1:
                continue
            kws = model.get_topic(tid)
            mapping[int(tid)] = ", ".join([w for w, _ in kws[:top_n]])
    except Exception:
        pass
    return mapping


# UI: Sidebar controls
st.sidebar.title("Settings")

mode = st.sidebar.radio("Data source", ["Load CSV", "Scrape StackOverflow"])

if mode == "Load CSV":
    data_path = st.sidebar.text_input("CSV Path", "questions_with_difficulty.csv")
    load_csv_btn = st.sidebar.button("Load CSV")
else:
    tags_input = st.sidebar.text_input("Tags (comma separated)", "python,pandas")
    scrape_now_btn = st.sidebar.button("Scrape & Analyze")

model_folder = st.sidebar.text_input("BERTopic model folder", MODEL_FOLDER_DEFAULT)
use_cache = st.sidebar.checkbox("Use scrape cache (scraped_cache.csv)", value=True)
clear_cache_btn = st.sidebar.button("Clear cached scrape")

# load model (if available)
topic_model = load_bertopic_model(model_folder) if BER_AVAILABLE else None
topic_keyword_map = topic_keywords_from_model(topic_model) if topic_model else {}


# Manage cache clearing

if clear_cache_btn:
    if os.path.exists(CACHE_CSV):
        os.remove(CACHE_CSV)
        st.sidebar.success("Cache file removed.")
    else:
        st.sidebar.info("No cache file found.")


# Data loading & session_state
# ensure session keys exist
if "scraped_df" not in st.session_state:
    st.session_state.scraped_df = None
if "last_tags" not in st.session_state:
    st.session_state.last_tags = None

# LOAD CSV mode
if mode == "Load CSV" and load_csv_btn:
    try:
        df_loaded = pd.read_csv(data_path)
        df_loaded = compute_difficulty(df_loaded)
        st.session_state.scraped_df = df_loaded
        st.success(f"CSV loaded ({len(df_loaded)} rows)")
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")

# SCRAPE mode
if mode == "Scrape StackOverflow" and scrape_now_btn:
    try:
        # if using cache and cache exists and tags same as last, load cache
        use_saved = False
        if use_cache and os.path.exists(CACHE_CSV) and st.session_state.last_tags == tags_input:
            try:
                cached = pd.read_csv(CACHE_CSV)
                st.session_state.scraped_df = cached
                st.success(f"Loaded cached scrape ({len(cached)} rows)")
                use_saved = True
            except Exception:
                use_saved = False

        if not use_saved:
            with st.spinner("Scraping StackOverflow (this can take several minutes)..."):
                df_scraped = scrape_stackoverflow(tags_input)
                if df_scraped.empty:
                    st.warning("Scraper returned no rows.")
                    st.session_state.scraped_df = df_scraped
                else:
                    df_scraped = compute_difficulty(df_scraped)
                    df_scraped = clean_data(df_scraped)
                    # Run BERTopic transform if model present, else topic column stays -1
                    if topic_model:
                        with st.spinner("Running BERTopic for topics..."):
                            docs = (df_scraped["title"].astype(str) + " " + df_scraped["body"].astype(str)).tolist()
                            topics, probs = topic_model.transform(docs)
                            df_scraped["topic"] = topics
                            # reduce probs matrix into single confidence value per doc
                            df_scraped["probability"] = np.max(probs, axis=1)
                    else:
                        df_scraped["topic"] = -1
                    st.session_state.scraped_df = df_scraped
                    st.session_state.last_tags = tags_input
                    # save cache
                    if use_cache:
                        try:
                            df_scraped.to_csv(CACHE_CSV, index=False)
                        except Exception:
                            pass
                    st.success(f"Scraped & processed ({len(df_scraped)} rows).")
    except Exception as e:
        st.error(f"Error during scraping: {e}")


# After this point, use df = st.session_state.scraped_df
df = st.session_state.get("scraped_df", None)


# Pages
page = st.sidebar.radio("Select Page", ["Overview", "Topics", "Difficulty", "Explorer", "Trending"])

# Guard: if no data loaded show instructions
if df is None:
    st.markdown("# StackOverflow — Education Insights")
    st.info("No data loaded. Choose 'Load CSV' and click Load CSV or choose 'Scrape StackOverflow' and click Scrape & Analyze.")
    st.stop()

# basic safety: ensure dtype and required columns exist
df["title"] = df.get("title", "").astype(str)
df["body"] = df.get("body", "").astype(str)
df["score"] = pd.to_numeric(df.get("score", 0), errors="coerce").fillna(0)
df["view_count"] = pd.to_numeric(df.get("view_count", 0), errors="coerce").fillna(0)
df["answer_count"] = pd.to_numeric(df.get("answer_count", 0), errors="coerce").fillna(0)
if "difficulty" not in df.columns:
    df = compute_difficulty(df)

# ------------------ Overview ------------------
if page == "Overview":
    st.title("StackOverflow — Education Insights (Overview)")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Questions", f"{len(df):,}")
    col2.metric("Distinct Topics", int(df["topic"].nunique()) if "topic" in df else "N/A")
    col3.metric("Avg Difficulty", round(df["difficulty"].mean(), 3) if "difficulty" in df else "N/A")
    col4.metric("Avg Score", round(df["score"].mean(), 3))

    st.markdown("## Most Discussed Topics in{tags_input}")

    if topic_model and "topic" in df.columns:
        freq = df[df["topic"] != -1]["topic"].value_counts(normalize=True).head(4) * 100
        top_topics = freq.reset_index()
        top_topics.columns = ["topic", "percentage"]
        top_topics["title"] = top_topics["topic"].map(lambda t: topic_keyword_map.get(t, ", ".join([w[0] for w in topic_model.get_topic(t)[:3]] if t != -1 else ["Other"])))

        # donut chart
        fig = px.pie(top_topics, values="percentage", names="title", hole=0.55, title="Top 4 Topics (by share)")
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Topic Breakdown (Top 4)")
        st.table(top_topics[["topic", "title", "percentage"]].round(2))

    else:
        st.info("No BERTopic model loaded — topics are numeric IDs.")

    # topic frequency bar
    if "topic" in df.columns:
        freq_all = df["topic"].value_counts().reset_index()
        freq_all.columns = ["topic", "count"]
        fig2 = px.bar(freq_all.head(40), x="topic", y="count", title="Topic Frequency (top 40)")
        st.plotly_chart(fig2, use_container_width=True)

    # tags if exist
    if "tag" in df.columns:
        tag_counts = df["tag"].str.split(",").explode().value_counts().reset_index()
        tag_counts.columns = ["tag", "count"]
        fig3 = px.bar(tag_counts.head(40), x="tag", y="count", title="Tag Distribution (top 40)")
        st.plotly_chart(fig3, use_container_width=True)


# ------------------ Topics ------------------
elif page == "Topics":
    st.title("Topics (inspect and label)")
    if "topic" not in df.columns:
        st.warning("No topic column available.")
        st.stop()
    unique_topics = sorted(df["topic"].unique())
    sel = st.selectbox("Choose topic", options=unique_topics, index=0)

    # show keywords if model exists
    if topic_model and sel != -1:
        kws = topic_model.get_topic(sel)
        st.markdown(f"**Top keywords:** {', '.join([w for w, _ in kws[:10]])}")
    else:
        st.info("No model keywords available (BERTopic not loaded).")

    top_n = st.slider("Top N questions", 1, 50, 10)
    cols_to_show = [c for c in ["score", "title", "answer_count", "view_count", "difficulty"] if c in df.columns]
    st.table(df[df["topic"] == sel][cols_to_show].head(top_n))


# ------------------ Difficulty ------------------
elif page == "Difficulty":
    st.title("Difficulty Analysis")
    if "difficulty" not in df.columns:
        st.info("No difficulty column.")
        st.stop()
    fig = px.histogram(df, x="difficulty", nbins=40, title="Difficulty Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # average difficulty by topic
    if "topic" in df.columns:
        topic_diff = df.groupby("topic")["difficulty"].mean().reset_index().sort_values("difficulty", ascending=False)
        st.markdown("### Average Difficulty per Topic")
        st.dataframe(topic_diff.head(40))


# ------------------ Explorer ------------------
elif page == "Explorer":
    st.title("Question Explorer")
    q = st.text_input("Search in title", "")
    topic_filter = st.selectbox("Filter by topic", ["All"] + df["topic"].astype(str).unique().tolist())
    df_ex = df.copy()
    if q:
        df_ex = df_ex[df_ex["title"].str.contains(q, case=False, na=False)]
    if topic_filter != "All":
        df_ex = df_ex[df_ex["topic"] == int(topic_filter)]
    st.dataframe(df_ex[["topic", "score", "title", "difficulty", "view_count", "answer_count"]].head(500))
    if not df_ex.empty:
        csv = df_ex.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", csv, "so_filtered.csv", "text/csv")


# ------------------ Trending ------------------
elif page == "Trending":
    st.title("Trending / Priority Topics")
    df["priority"] = df["score"] * (df["view_count"] + 1)
    top_by_priority = df.sort_values("priority", ascending=False).head(200)
    st.dataframe(top_by_priority[["topic", "priority", "score", "title", "view_count", "answer_count"]].head(200))


# Footer
st.markdown("---")
st.markdown("Built for curriculum insights — powered by BERTopic & StackOverflow.")
