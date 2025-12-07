import streamlit as st
import pandas as pd
import os
import plotly.express as px
from typing import Optional


try:
    from bertopic import BERTopic
    BER_AVAILABLE = True
except Exception:
    BER_AVAILABLE = False

st.set_page_config(page_title="SO Edu Insights", layout="wide")

# ---------- Helper functions ----------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # safe columns
    if "topic" not in df.columns:
        df["topic"] = -1
    if "difficulty" not in df.columns:
        df["difficulty"] = None
    if "score" not in df.columns:
        df["score"] = 0
    if "view_count" not in df.columns:
        df["view_count"] = 0
    if "answer_count" not in df.columns:
        df["answer_count"] = 0
    return df

@st.cache_data
def load_topic_model(folder: str) -> Optional[BERTopic]:
    if BER_AVAILABLE and os.path.exists(folder):
        try:
            return BERTopic.load(folder)
        except Exception:
            return None
    return None

def topic_keywords_from_model(model, top_n=5):
    info = model.get_topic_info()
    mapping = {}
    for tid in info["Topic"]:
        if tid == -1:
            continue
        kws = model.get_topic(tid)
        mapping[int(tid)] = ", ".join([w for w, _ in kws[:top_n]])
    return mapping

# ------- Sidebar / inputs -------
st.sidebar.title("Settings")
data_path = st.sidebar.text_input("Path to CSV", "questions_with_difficulty.csv")
model_folder = st.sidebar.text_input("BERTopic model folder (optional)", "topics_model")
refresh = st.sidebar.button("Reload data")

# ---------- Load data & model ----------
df = load_data(data_path)
topic_model = None
topic_keyword_map = {}
if BER_AVAILABLE and os.path.exists(model_folder):
    topic_model = load_topic_model(model_folder)
    if topic_model:
        topic_keyword_map = topic_keywords_from_model(topic_model, top_n=5)

# ---------- Layout / Pages ----------
page = st.sidebar.radio("Select Page", ["Overview", "Topics", "Difficulty", "Explorer", "Trending"])


if page == "Overview":
    st.title("StackOverflow — Education Insights (Overview)")

    # ---- Metrics ----
    col1, col2, col3, col4 = st.columns(4)
    total_q = len(df)
    total_topics = df["topic"].nunique()
    avg_difficulty = df["difficulty"].dropna().mean() if df["difficulty"].notna().any() else None
    avg_score = df["score"].mean()

    col1.metric("Total Questions", f"{total_q:,}")
    col2.metric("Distinct Topics", f"{total_topics}")
    col3.metric("Avg Difficulty", f"{avg_difficulty:.2f}" if avg_difficulty is not None else "N/A")
    col4.metric("Avg Score", f"{avg_score:.2f}")


    st.markdown("Most Discussed Topics")

    if topic_model:
        # 1. Build topic titles
        def get_topic_titles(topic_model):
            topic_info = topic_model.get_topic_info()
            titles = {}
            for _, row in topic_info.iterrows():
                tid = row["Topic"]
                if tid == -1:
                    continue
                words = topic_model.get_topic(tid)
                if words:
                    titles[tid] = ", ".join([w[0] for w in words[:3]])
                else:
                    titles[tid] = f"Topic {tid}"
            return titles

        topic_titles = get_topic_titles(topic_model)

        # 2. Top 4 topics by frequency
        top_freq = (
            df[df["topic"] != -1]["topic"]
            .value_counts(normalize=True)
            .head(4) * 100
        ).reset_index()
        top_freq.columns = ["topic", "percentage"]
        top_freq["title"] = top_freq["topic"].map(topic_titles)

        # 3. Donut chart
        fig_donut = px.pie(
            top_freq,
            values="percentage",
            names="title",
            hole=0.55,
            title="Top 4 Most Discussed Topics"
        )
        fig_donut.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_donut, use_container_width=True)

        # 4. Table
        st.markdown("### Topic Breakdown")
        st.table(
            top_freq[["topic", "title", "percentage"]].round(2)
        )
    else:
        st.warning("BERTopic model not found — cannot generate topic titles or donut chart.")

    # ============================
    #  OLD FREQUENCY BAR CHART
    # ============================
    st.markdown("### Topic Frequency (Raw)")

    freq = df["topic"].value_counts().reset_index()
    freq.columns = ["topic", "count"]
    fig = px.bar(freq.head(40), x="topic", y="count", labels={"topic":"Topic","count":"Questions"})
    st.plotly_chart(fig, use_container_width=True)

    # ============================
    #   TAG DISTRIBUTION
    # ============================
    st.markdown("###  Tag distribution (if available)")
    if "tag" in df.columns:
        tag_counts = df['tag'].str.split(",").explode().value_counts().reset_index()
        tag_counts.columns = ["tag", "count"]
        fig2 = px.bar(tag_counts.head(40), x="tag", y="count")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No `tag` column found in CSV.")

# ---------- Topics ----------
elif page == "Topics":
    st.title(" Topics (inspect and label)")
    st.markdown("Select a topic to see keywords and top questions.")

    unique_topics = sorted(df["topic"].unique())
    sel = st.selectbox("Choose topic", options=unique_topics, index=0)
    st.markdown(f"#### Topic: {sel}")

    # show keywords from model if available, else show top words via df
    if sel in topic_keyword_map:
        st.markdown(f"**Top keywords (from BERTopic):** {topic_keyword_map[sel]}")
    else:
        # fallback: compute top words from titles in topic
        sample = df[df["topic"] == sel]["title"].astype(str)
        if not sample.empty:
            words = " ".join(sample).lower().split()
            from collections import Counter
            common = Counter(words).most_common(20)
            st.markdown("**Top words (fallback):** " + ", ".join([w for w, _ in common[:10]]))
        else:
            st.info("No documents for this topic")

    top_n = st.slider("Top N questions", 1, 30, 10)
    top_questions = df[df["topic"] == sel].sort_values("score", ascending=False).head(top_n)
    if top_questions.empty:
        st.warning("No questions in this topic.")
    else:
        st.table(top_questions[["score","title","answer_count","view_count"]].reset_index(drop=True))

# ---------- Difficulty ----------
elif page == "Difficulty":
    st.title("⚖️ Difficulty Analysis")
    st.markdown("Distribution of difficulty and average difficulty per topic.")

    # distribution
    if df["difficulty"].notna().any():
        fig = px.histogram(df, x="difficulty", nbins=40, title="Difficulty Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No `difficulty` column found. Run compute_difficulty first.")

    # avg difficulty by topic
    if "difficulty" in df.columns:
        topic_diff = df.groupby("topic")["difficulty"].mean().reset_index().sort_values("difficulty", ascending=False)
        fig2 = px.bar(topic_diff.head(30), x="topic", y="difficulty", title="Average Difficulty by Topic (top 30)")
        st.plotly_chart(fig2, use_container_width=True)

# ---------- Explorer ----------
elif page == "Explorer":
    st.title("🔎 Question Explorer")
    q = st.text_input("Search in titles (keywords)", "")
    topic_filter = st.selectbox("Filter by topic", options=["All"] + [str(t) for t in sorted(df["topic"].unique())])
    min_score = st.slider("Min score", int(df["score"].min()), int(df["score"].max()), int(df["score"].min()))
    min_views = st.number_input("Min views", min_value=0, value=0)

    df_explore = df.copy()
    if q:
        df_explore = df_explore[df_explore["title"].str.contains(q, case=False, na=False)]
    if topic_filter != "All":
        df_explore = df_explore[df_explore["topic"] == int(topic_filter)]
    df_explore = df_explore[df_explore["score"] >= min_score]
    df_explore = df_explore[df_explore["view_count"] >= min_views]

    st.write(f"Results: {len(df_explore)}")
    st.dataframe(df_explore[["topic","score","title","difficulty","view_count","answer_count"]].reset_index(drop=True).head(200))

    if not df_explore.empty:
        csv = df_explore.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", csv, file_name="so_filtered.csv", mime="text/csv")

# ---------- Trending ----------
elif page == "Trending":
    st.title("🔥 Trending / Priority Topics")
    # priority metric: trending_score if available else score*views
    if "trending_score" in df.columns:
        df["priority"] = df["trending_score"]
    else:
        df["priority"] = df["score"] * (df["view_count"] + 1)

    topic_priority = df.groupby("topic")["priority"].sum().reset_index().sort_values("priority", ascending=False)
    st.markdown("### Top priority topics")
    st.dataframe(topic_priority.head(30).reset_index(drop=True))

    st.markdown("### Top priority questions")
    topq = df.sort_values("priority", ascending=False).head(50)
    st.dataframe(topq[["topic","priority","score","title","link","view_count","answer_count"]].reset_index(drop=True))

# ---------- Footer ----------
st.markdown("---")
st.markdown("Built for curriculum insights — use the Explorer to filter and export questions for teaching materials.")
