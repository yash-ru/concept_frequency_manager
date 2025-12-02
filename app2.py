import streamlit as st
import pandas as pd
import plotly.express as px
import spacy
import re
from urllib.parse import urlparse
from collections import Counter, defaultdict

import subprocess
import sys

st.set_page_config(page_title="Concept Frequency", layout="wide")
st.title("Concept Frequency Dashboard")


# ================================
# LOAD SPACY ON STREAMLIT CLOUD
# ================================
@st.cache_resource
def load_spacy_model():
    try:
        # Try loading normally
        nlp = spacy.load("en_core_web_sm")
    except:
        # First-time Streamlit Cloud install
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    return nlp

nlp = load_spacy_model()

# ------------------------------------------------------
# Helper Functions (shared)
# ------------------------------------------------------
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


# ------------------------------------------------------
# ========== TAB 1: KEYWORD DENSITY + TREEMAP ==========
# ------------------------------------------------------
def keyword_clean_text(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def extract_keyword_concepts(df):

    MIN_UNIGRAM_FREQ = 2
    MIN_BIGRAM_FREQ = 2
    MIN_TRIGRAM_FREQ = 2
    MIN_COHESION_BIGRAM = 0.25
    MIN_COHESION_TRIGRAM = 0.20
    MAX_CONCEPTS = 200

    keywords = df["Keyword Term"].astype(str).tolist()

    unigram_counts = Counter()
    bigram_counts = Counter()
    trigram_counts = Counter()
    phrase_map = defaultdict(set)

    for kw in keywords:
        cleaned = keyword_clean_text(kw)
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

    # filter strong
    strong_unigrams = {w: c for w, c in unigram_counts.items() if c >= MIN_UNIGRAM_FREQ}

    strong_bigrams = {}
    for bg, cnt in bigram_counts.items():
        if cnt >= MIN_BIGRAM_FREQ:
            w1, w2 = bg.split()
            coh = cnt / max(unigram_counts.get(w1, 1), unigram_counts.get(w2, 1))
            if coh >= MIN_COHESION_BIGRAM:
                strong_bigrams[bg] = cnt

    strong_trigrams = {}
    for tg, cnt in trigram_counts.items():
        if cnt >= MIN_TRIGRAM_FREQ:
            w1, w2, w3 = tg.split()
            coh = cnt / max(unigram_counts.get(w1, 1),
                            unigram_counts.get(w2, 1),
                            unigram_counts.get(w3, 1))
            if coh >= MIN_COHESION_TRIGRAM:
                strong_trigrams[tg] = cnt

    # concept list
    concepts = {}
    for tg, cnt in strong_trigrams.items():
        concepts[tg] = {"phrase": tg, "count": cnt, "keywords": phrase_map[tg], "arity": 3}
    for bg, cnt in strong_bigrams.items():
        concepts[bg] = {"phrase": bg, "count": cnt, "keywords": phrase_map[bg], "arity": 2}
    for w, cnt in strong_unigrams.items():
        concepts[w] = {"phrase": w, "count": cnt, "keywords": phrase_map[w], "arity": 1}

    # dominance
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
    filtered = sorted(filtered, key=lambda x: (-x["count"], -x["arity"]))

    final = filtered[:MAX_CONCEPTS]

    # build df
    total_impr = df["Keyword Impressions"].sum()
    rows = []
    for c in final:
        kws = list(c["keywords"])
        df_c = df[df["Keyword Term"].isin(kws)]

        imp_sum = df_c["Keyword Impressions"].sum()
        clicks_sum = df_c["Keyword Clicks"].sum()
        rev_sum = df_c["Revenue"].sum()

        rows.append({
            "concept": c["phrase"],
            "concept_count": c["count"],
            "matched_keywords_count": len(kws),
            "exposure_pct": round((imp_sum / total_impr) * 100, 2) if total_impr else 0,
            "rpc": round(rev_sum / clicks_sum, 2) if clicks_sum else 0,
            "ctr": round((clicks_sum / imp_sum) * 100, 2) if imp_sum else 0,
            "matched_keywords": "; ".join(kws)
        })

    return pd.DataFrame(rows)

def create_keyword_treemap(df):
    df["ctr_norm"] = (df["ctr"] - df["ctr"].min()) / (df["ctr"].max() - df["ctr"].min())

    fig = px.treemap(
        df,
        path=["concept"],
        values="exposure_pct",
        color="ctr_norm",
        color_continuous_scale=["#FF6B6B", "#FFA94D", "#FFD43B", "#82C91E", "#2F9E44"],
        maxdepth=1
    )
    return fig


# ------------------------------------------------------
# ========== TAB 2: URL DENSITY + TREEMAP ==========
# ------------------------------------------------------
def extract_domain(url):
    u = urlparse(url.lower())
    net = u.netloc if u.netloc else u.path.split('/')[0]
    return re.sub(r"^www\.", "", net)

def is_homepage(url):
    u = urlparse(url.lower())
    return u.path.strip("/") == ""

def slug_from_url(url):
    u = urlparse(url.lower())
    path = u.path
    path = re.sub(r"\.(html|php|jsp)$", "", path)
    path = re.sub(r"[/\-_+]", " ", path)
    path = re.sub(r"\b\d+\b", " ", path)
    return re.sub(r"\s+", " ", path).strip()

def clean_url_to_keywords(url):
    domain = extract_domain(url)
    slug = slug_from_url(url)
    if is_homepage(url):
        return domain.replace(".", " ")
    return (domain.replace(".", " ") + " " + slug).strip()

def extract_url_concepts(df):
    """Direct port of your URL density script"""
    # cleaning
    df["cleaned"] = df["URL"].apply(clean_url_to_keywords)

    records = list(zip(df["cleaned"], df["URL"]))

    # COUNTS
    unigram = Counter()
    bigram = Counter()
    trigram = Counter()
    phrase_map = defaultdict(set)

    for cleaned, orig in records:
        tokens = tokenize_spacy(cleaned)
        if not tokens:
            continue

        for w in tokens:
            unigram[w] += 1
            phrase_map[w].add(orig)

        for bg in ngrams(tokens, 2):
            bigram[bg] += 1
            phrase_map[bg].add(orig)

        for tg in ngrams(tokens, 3):
            trigram[tg] += 1
            phrase_map[tg].add(orig)

    # apply your thresholds
    MIN_UNI, MIN_BI, MIN_TRI = 2, 2, 2
    COH_BI, COH_TRI = 0.25, 0.20

    strong_uni = {w: c for w, c in unigram.items() if c >= MIN_UNI}

    strong_bi = {}
    for bg, cnt in bigram.items():
        if cnt >= MIN_BI:
            w1, w2 = bg.split()
            coh = cnt / max(unigram.get(w1, 1), unigram.get(w2, 1))
            if coh >= COH_BI:
                strong_bi[bg] = cnt

    strong_tri = {}
    for tg, cnt in trigram.items():
        if cnt >= MIN_TRI:
            w1, w2, w3 = tg.split()
            coh = cnt / max(unigram.get(w1, 1), unigram.get(w2, 1), unigram.get(w3, 1))
            if coh >= COH_TRI:
                strong_tri[tg] = cnt

    # build concept objects
    concepts = {}
    for k, v in strong_tri.items():
        concepts[k] = {"phrase": k, "count": v, "urls": phrase_map[k], "arity": 3}
    for k, v in strong_bi.items():
        concepts[k] = {"phrase": k, "count": v, "urls": phrase_map[k], "arity": 2}
    for k, v in strong_uni.items():
        concepts[k] = {"phrase": k, "count": v, "urls": phrase_map[k], "arity": 1}

    # dominance filter
    items = sorted(concepts.values(), key=lambda x: (-x["arity"], -x["count"]))
    to_remove = set()
    for P in items:
        for Q in items:
            if Q["arity"] <= P["arity"]:
                continue
            if P["urls"].issubset(Q["urls"]):
                to_remove.add(P["phrase"])
                break

    final = [c for c in concepts.values() if c["phrase"] not in to_remove]

    # final df
    total_impr = df["Page Impression"].sum()

    rows = []
    for c in final:
        urls = list(c["urls"])
        df_c = df[df["URL"].isin(urls)]

        imp_sum = df_c["Page Impression"].sum()
        clicks_sum = df_c["Keyword Clicks"].sum()
        rev_sum = df_c["Revenue"].sum()

        rows.append({
            "concept": c["phrase"],
            "concept_count": c["count"],
            "matched_url_count": len(urls),
            "exposure_pct": round((imp_sum / total_impr) * 100, 2) if total_impr else 0,
            "rpc": round(rev_sum / clicks_sum, 2) if clicks_sum else 0,
            "ctr": round((clicks_sum / imp_sum) * 100, 2) if imp_sum else 0,
            "matched_urls": "; ".join(urls)
        })

    return pd.DataFrame(rows)

def create_url_treemap(df):
    df["ctr_norm"] = (df["ctr"] - df["ctr"].min()) / (df["ctr"].max() - df["ctr"].min())

    fig = px.treemap(
        df,
        path=["concept"],
        values="exposure_pct",
        color="ctr_norm",
        color_continuous_scale=["#FF6B6B", "#FFA94D", "#FFD43B", "#82C91E", "#2F9E44"],
        maxdepth=1
    )
    return fig


# ------------------------------------------------------
#                   STREAMLIT TABS
# ------------------------------------------------------
tab1, tab2 = st.tabs(["Keyword Concept", "URL Concept"])


# ------------------------------------------------------
# TAB 1 — KEYWORD ENGINE
# ------------------------------------------------------
with tab1:

    uploaded = st.file_uploader(
        "Upload Keyword CSV (Keyword Term, Keyword Impressions, Keyword Clicks, Revenue)",
        type=["csv"],
        key="kw_upload"
    )

    if uploaded:
        df = pd.read_csv(uploaded)

        st.subheader("Extracting Keyword Concepts...")
        kw_concepts = extract_keyword_concepts(df)

        st.write("### Preview")
        st.dataframe(kw_concepts.head())

        st.download_button(
            "📥 Download keyword_concepts.csv",
            kw_concepts.to_csv(index=False),
            "keyword_concepts.csv",
            mime="text/csv"
        )

        fig = create_keyword_treemap(kw_concepts)
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "📥 Download Keyword Treemap HTML",
            fig.to_html(full_html=True),
            file_name="keyword_treemap.html",
            mime="text/html"
        )


# ------------------------------------------------------
# TAB 2 — URL ENGINE
# ------------------------------------------------------
with tab2:

    uploaded2 = st.file_uploader(
        "Upload URL CSV (URL, Page Impression, Keyword Clicks, Revenue)",
        type=["csv"],
        key="url_upload"
    )

    if uploaded2:
        df2 = pd.read_csv(uploaded2)

        st.subheader("Extracting URL Concepts...")
        url_concepts = extract_url_concepts(df2)

        st.write("### Preview")
        st.dataframe(url_concepts.head())

        st.download_button(
            "📥 Download url_concepts.csv",
            url_concepts.to_csv(index=False),
            "url_concepts.csv",
            mime="text/csv"
        )

        fig2 = create_url_treemap(url_concepts)
        st.plotly_chart(fig2, use_container_width=True)

        st.download_button(
            "📥 Download URL Treemap HTML",
            fig2.to_html(full_html=True),
            file_name="url_treemap.html",
            mime="text/html"
        )

