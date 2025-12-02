import streamlit as st
import pandas as pd
import spacy
import re
from collections import Counter, defaultdict
import plotly.express as px

# ============================================================
# LOAD SPACY ONCE
# ============================================================
nlp = spacy.load("en_core_web_sm")

# ============================================================
# HELPERS (from your first script)
# ============================================================
def clean_text(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenize_spacy(text: str):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or token.like_num or len(token) <= 1:
            continue
        if token.pos_ in {"DET", "CCONJ", "SCONJ", "ADP", "PRON"}:
            continue
        tokens.append(token.lemma_)
    return tokens

def ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# ============================================================
# CONCEPT EXTRACTION LOGIC (rewrapped from your script)
# ============================================================
def extract_concepts(df):

    # Config
    MIN_UNIGRAM_FREQ = 2
    MIN_BIGRAM_FREQ = 2
    MIN_TRIGRAM_FREQ = 2
    MIN_COHESION_BIGRAM = 0.25
    MIN_COHESION_TRIGRAM = 0.20
    MAX_CONCEPTS = 200

    keywords = df["Keyword Term"].astype(str).tolist()

    # Build maps
    unigram_counts = Counter()
    bigram_counts = Counter()
    trigram_counts = Counter()
    phrase_map = defaultdict(set)

    for kw in keywords:
        cleaned = clean_text(kw)
        tokens = tokenize_spacy(cleaned)
        if not tokens:
            continue

        for w in tokens:
            unigram_counts[w] += 1
            phrase_map[w].add(kw)

        for bg in ngrams(tokens, 2):
            bigram_counts[bg] += 1
            phrase_map[bg].add(kw)

        for tg in ngrams(tokens, 3):
            trigram_counts[tg] += 1
            phrase_map[tg].add(kw)

    # Filter strong n-grams
    strong_unigrams = {w: c for w, c in unigram_counts.items() if c >= MIN_UNIGRAM_FREQ}

    strong_bigrams = {}
    for bg, cnt in bigram_counts.items():
        if cnt < MIN_BIGRAM_FREQ:
            continue
        w1, w2 = bg.split()
        coh = cnt / max(unigram_counts.get(w1, 1), unigram_counts.get(w2, 1))
        if coh >= MIN_COHESION_BIGRAM:
            strong_bigrams[bg] = cnt

    strong_trigrams = {}
    for tg, cnt in trigram_counts.items():
        if cnt < MIN_TRIGRAM_FREQ:
            continue
        w1, w2, w3 = tg.split()
        coh = cnt / max(unigram_counts.get(w1, 1), unigram_counts.get(w2, 1), unigram_counts.get(w3, 1))
        if coh >= MIN_COHESION_TRIGRAM:
            strong_trigrams[tg] = cnt

    # Build concepts
    concepts = {}
    for tg, cnt in strong_trigrams.items():
        concepts[tg] = {"phrase": tg, "count": cnt, "keywords": phrase_map[tg], "arity": 3}
    for bg, cnt in strong_bigrams.items():
        concepts[bg] = {"phrase": bg, "count": cnt, "keywords": phrase_map[bg], "arity": 2}
    for w, cnt in strong_unigrams.items():
        concepts[w] = {"phrase": w, "count": cnt, "keywords": phrase_map[w], "arity": 1}

    # Dominance filter
    items = sorted(concepts.values(), key=lambda x: (-x["arity"], -x["count"]))
    to_remove = set()
    for P in items:
        for Q in items:
            if Q["arity"] <= P["arity"]:
                continue
            if P["keywords"].issubset(Q["keywords"]):
                to_remove.add(P["phrase"])
                break

    filtered = [c for c in concepts.values() if c["phrase"] not in to_remove]
    filtered = sorted(filtered, key=lambda x: (-x["count"], -x["arity"], x["phrase"]))
    filtered = filtered[:MAX_CONCEPTS]

    # Build final dataframe
    total_impressions = df["Keyword Impressions"].sum()

    rows = []
    for c in filtered:
        kws = list(c["keywords"])
        df_c = df[df["Keyword Term"].isin(kws)]

        imp_sum = df_c["Keyword Impressions"].sum()
        clicks_sum = df_c["Keyword Clicks"].sum()
        rev_sum = df_c["Revenue"].sum()

        rows.append({
            "concept": c["phrase"],
            "concept_count": c["count"],
            "matched_keywords_count": len(kws),
            "exposure_pct": round((imp_sum / total_impressions) * 100, 2),
            "rpc": round(rev_sum / clicks_sum, 2) if clicks_sum else 0,
            "ctr": round((clicks_sum / imp_sum) * 100, 2) if imp_sum else 0,
            "matched_keywords": "; ".join(kws)
        })

    return pd.DataFrame(rows)

# ============================================================
# TREEMAP GENERATOR (rewrapped from your second script)
# ============================================================
def create_beautiful_treemap(df):
    df["ctr_norm"] = (df["ctr"] - df["ctr"].min()) / (df["ctr"].max() - df["ctr"].min())

    fig = px.treemap(
        df,
        path=["concept"],
        values="exposure_pct",
        color="ctr_norm",
        color_continuous_scale=["#FF6B6B", "#FFA94D", "#FFD43B", "#82C91E", "#2F9E44"],
        maxdepth=1
    )

    fig.update_traces(
        texttemplate="<b>%{label}</b>",
        hovertemplate="<b>%{label}</b><br>Exposure: %{value}%<br>CTR: %{customdata[0]}%",
        customdata=df[["ctr"]],
        tiling=dict(pad=4)
    )

    fig.update_layout(
        title="Concept Exposure Treemap",
        width=1200, height=800,
        margin=dict(t=60, l=10, r=10, b=10)
    )

    return fig

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Keyword Concept Engine", layout="wide")

st.title("Concept Frequency Manager")

uploaded_file = st.file_uploader("Upload csv file with columns - [Keyword Term,Keyword Impressions,Keyword Clicks,Revenue]", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Name normalization
    if "Keyword Term" not in df.columns:
        if "keyword term" in df.columns:
            df.rename(columns={"keyword term": "Keyword Term"}, inplace=True)
        else:
            st.error("CSV must contain 'Keyword Term' column.")
            st.stop()

    st.success("File uploaded successfully!")

    st.subheader("Step 1: Extracting Concepts...")
    concepts_df = extract_concepts(df)

    st.write("### Preview of Concepts")
    st.dataframe(concepts_df.head(20))

    st.download_button(
        label="📥 Download concept_output.csv",
        data=concepts_df.to_csv(index=False),
        file_name="concept_output.csv",
        mime="text/csv"
    )

    st.subheader("Step 2: Generating Treemap...")
    fig = create_beautiful_treemap(concepts_df)
    st.plotly_chart(fig, use_container_width=True)

    treemap_html = fig.to_html(full_html=True)
    st.download_button(
        label="📥 Download Treemap HTML",
        data=treemap_html,
        file_name="concept_treemap.html",
        mime="text/html"
    )
