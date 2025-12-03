import streamlit as st
import pandas as pd
import plotly.express as px
import re
from urllib.parse import urlparse
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#from auth import ensure_user

st.set_page_config(page_title="Concept Frequency", layout="wide")

# Get user email
#user_email = ensure_user()

# Continue with your real app
#st.title("My Internal App")
#st.write(f"Welcome, {user_email}")

# ------------------------------------------------------
# Download NLTK data (only once, cached)
# ------------------------------------------------------
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    
    return stopwords.words('english'), WordNetLemmatizer()

STOP_WORDS, lemmatizer = download_nltk_data()

st.title("Concept Frequency Dashboard")

# ------------------------------------------------------
# NLTK-based tokenizer (automatic stopwords + lemmatization)
# ------------------------------------------------------
def tokenize_nltk(text: str):
    """Professional tokenizer using NLTK - handles lakhs of keywords"""
    text = str(text).lower()
    # Remove special characters but keep apostrophes for contractions
    text = re.sub(r'[^a-z0-9\s\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into words
    tokens = text.split()
    
    # Filter and lemmatize
    processed = []
    for token in tokens:
        # Skip if too short or is a number
        if len(token) <= 2 or token.isdigit():
            continue
        # Skip stopwords
        if token in STOP_WORDS:
            continue
        # Lemmatize (convert to base form: running -> run)
        lemma = lemmatizer.lemmatize(token)
        processed.append(lemma)
    
    return processed

def ngrams(tokens, n):
    """Generate n-grams from token list"""
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


# ------------------------------------------------------
# ========== KEYWORD CONCEPT EXTRACTION ==========
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

    # Process with progress bar for large datasets
    progress_bar = st.progress(0)
    total = len(keywords)
    
    for idx, kw in enumerate(keywords):
        cleaned = keyword_clean_text(kw)
        tokens = tokenize_nltk(cleaned)
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
        
        # Update progress every 100 items
        if idx % 100 == 0:
            progress_bar.progress((idx + 1) / total)
    
    progress_bar.progress(1.0)
    progress_bar.empty()

    # Filter strong unigrams
    strong_unigrams = {w: c for w, c in unigram_counts.items() if c >= MIN_UNIGRAM_FREQ}

    # Filter strong bigrams with cohesion
    strong_bigrams = {}
    for bg, cnt in bigram_counts.items():
        if cnt >= MIN_BIGRAM_FREQ:
            parts = bg.split()
            if len(parts) == 2:
                w1, w2 = parts
                coh = cnt / max(unigram_counts.get(w1, 1), unigram_counts.get(w2, 1))
                if coh >= MIN_COHESION_BIGRAM:
                    strong_bigrams[bg] = cnt

    # Filter strong trigrams with cohesion
    strong_trigrams = {}
    for tg, cnt in trigram_counts.items():
        if cnt >= MIN_TRIGRAM_FREQ:
            parts = tg.split()
            if len(parts) == 3:
                w1, w2, w3 = parts
                coh = cnt / max(unigram_counts.get(w1, 1),
                                unigram_counts.get(w2, 1),
                                unigram_counts.get(w3, 1))
                if coh >= MIN_COHESION_TRIGRAM:
                    strong_trigrams[tg] = cnt

    # Build concept list
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
    filtered = sorted(filtered, key=lambda x: (-x["count"], -x["arity"]))

    final = filtered[:MAX_CONCEPTS]

    # Build dataframe
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
            "frequency_pct": round((imp_sum / total_impr) * 100, 2) if total_impr else 0,
            "rpc": round(rev_sum / clicks_sum, 2) if clicks_sum else 0,
            "ctr": round((clicks_sum / imp_sum) * 100, 2) if imp_sum else 0,
            "matched_keywords": "; ".join(kws)
        })

    return pd.DataFrame(rows)

def create_keyword_treemap(df):
    if df.empty or df["ctr"].max() == df["ctr"].min():
        df["ctr_norm"] = 0.5
    else:
        df["ctr_norm"] = (df["ctr"] - df["ctr"].min()) / (df["ctr"].max() - df["ctr"].min())

    fig = px.treemap(
        df,
        path=["concept"],
        values="frequency_pct",
        color="ctr_norm",
        color_continuous_scale=["#FF6B6B", "#FFA94D", "#FFD43B", "#82C91E", "#2F9E44"],
        maxdepth=1,
        title="Keyword Concept Distribution",
        width=1400, 
        height=600  
    )
    fig.update_traces(textinfo="label+value+percent parent")
    return fig


# ------------------------------------------------------
# ========== URL CONCEPT EXTRACTION ==========
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
    path = re.sub(r"\.(html|php|jsp|aspx)$", "", path)
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
    df["cleaned"] = df["URL"].apply(clean_url_to_keywords)
    records = list(zip(df["cleaned"], df["URL"]))

    unigram = Counter()
    bigram = Counter()
    trigram = Counter()
    phrase_map = defaultdict(set)

    # Process with progress bar
    progress_bar = st.progress(0)
    total = len(records)

    for idx, (cleaned, orig) in enumerate(records):
        tokens = tokenize_nltk(cleaned)
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
        
        if idx % 100 == 0:
            progress_bar.progress((idx + 1) / total)
    
    progress_bar.progress(1.0)
    progress_bar.empty()

    # Apply thresholds
    MIN_UNI, MIN_BI, MIN_TRI = 2, 2, 2
    COH_BI, COH_TRI = 0.25, 0.20

    strong_uni = {w: c for w, c in unigram.items() if c >= MIN_UNI}

    strong_bi = {}
    for bg, cnt in bigram.items():
        if cnt >= MIN_BI:
            parts = bg.split()
            if len(parts) == 2:
                w1, w2 = parts
                coh = cnt / max(unigram.get(w1, 1), unigram.get(w2, 1))
                if coh >= COH_BI:
                    strong_bi[bg] = cnt

    strong_tri = {}
    for tg, cnt in trigram.items():
        if cnt >= MIN_TRI:
            parts = tg.split()
            if len(parts) == 3:
                w1, w2, w3 = parts
                coh = cnt / max(unigram.get(w1, 1), unigram.get(w2, 1), unigram.get(w3, 1))
                if coh >= COH_TRI:
                    strong_tri[tg] = cnt

    # Build concept objects
    concepts = {}
    for k, v in strong_tri.items():
        concepts[k] = {"phrase": k, "count": v, "urls": phrase_map[k], "arity": 3}
    for k, v in strong_bi.items():
        concepts[k] = {"phrase": k, "count": v, "urls": phrase_map[k], "arity": 2}
    for k, v in strong_uni.items():
        concepts[k] = {"phrase": k, "count": v, "urls": phrase_map[k], "arity": 1}

    # Dominance filter
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

    # Build dataframe
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
            "frequency_pct": round((imp_sum / total_impr) * 100, 2) if total_impr else 0,
            "rpc": round(rev_sum / clicks_sum, 2) if clicks_sum else 0,
            "ctr": round((clicks_sum / imp_sum) * 100, 2) if imp_sum else 0,
            "matched_urls": "; ".join(urls)
        })

    return pd.DataFrame(rows)

def create_url_treemap(df):
    if df.empty or df["ctr"].max() == df["ctr"].min():
        df["ctr_norm"] = 0.5
    else:
        df["ctr_norm"] = (df["ctr"] - df["ctr"].min()) / (df["ctr"].max() - df["ctr"].min())

    fig = px.treemap(
        df,
        path=["concept"],
        values="frequency_pct",
        color="ctr_norm",
        color_continuous_scale=["#FF6B6B", "#FFA94D", "#FFD43B", "#82C91E", "#2F9E44"],
        maxdepth=1,
        title="URL Concept Distribution",
        width=1400, 
        height=600  
    )
    fig.update_traces(textinfo="label+value+percent parent")
    return fig


# ------------------------------------------------------
#                   STREAMLIT TABS
# ------------------------------------------------------
tab1, tab2 = st.tabs(["Keyword Concept", "URL Concept"])


# ------------------------------------------------------
# TAB 1 — KEYWORD ENGINE
# ------------------------------------------------------
with tab1:

    st.markdown("🔗 [Sample Analytics link](https://cm.analytics.mn/reports/analyse?hash=b32ba06d7f2a4ba67a2f5b1db847519c)")
    
    uploaded = st.file_uploader(
        "Upload Keyword CSV (Required columns: Keyword Term, Keyword Impressions, Keyword Clicks, Revenue)",
        type=["csv"],
        key="kw_upload",
        help="CSV should contain: Keyword Term, Keyword Impressions, Keyword Clicks, Revenue"
    )

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            
            # Validate columns
            required_cols = ["Keyword Term", "Keyword Impressions", "Keyword Clicks", "Revenue"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info(f"Your CSV has: {', '.join(df.columns.tolist())}")
            else:
                st.success(f"✅ Loaded {len(df):,} keywords")

                with st.spinner("Extracting keyword concepts... This may take a minute for large datasets"):
                    kw_concepts = extract_keyword_concepts(df)

                
                col1, col2 = st.columns([3, 1])
                
                #with col1:
                st.write("### 📊 Concept Analysis Results")
                st.dataframe(
                    kw_concepts.head(50),
                    use_container_width=True,
                    height=400
                )
                
                #with col2:                   
                st.download_button(
                    "📥 Download Full Results CSV",
                    kw_concepts.to_csv(index=False),
                    "keyword_concepts.csv",
                    mime="text/csv",
                    use_container_width=True
                    )

                fig = create_keyword_treemap(kw_concepts)
                st.plotly_chart(fig, use_container_width=True)

                st.download_button(
                    "📥 Download Treemap HTML",
                    fig.to_html(full_html=True),
                    file_name="keyword_treemap.html",
                    mime="text/html"
                )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please check your CSV format and try again.")


# ------------------------------------------------------
# TAB 2 — URL ENGINE
# ------------------------------------------------------
with tab2:

    st.markdown("🔗 [Sample Analytics link](https://cm.analytics.mn/reports/analyse?hash=749ac6e6705bdc0ff2d4e4c64ea83bdd)")
    
    uploaded2 = st.file_uploader(
        "Upload URL CSV (Required columns: URL, Page Impression, Keyword Clicks, Revenue)",
        type=["csv"],
        key="url_upload",
        help="CSV should contain: URL, Page Impression, Keyword Clicks, Revenue"
    )

    if uploaded2:
        try:
            df2 = pd.read_csv(uploaded2)
            
            # Validate columns
            required_cols = ["URL", "Page Impression", "Keyword Clicks", "Revenue"]
            missing_cols = [col for col in required_cols if col not in df2.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info(f"Your CSV has: {', '.join(df2.columns.tolist())}")
            else:
                st.success(f"✅ Loaded {len(df2):,} URLs")

                with st.spinner("🔄 Extracting URL concepts... This may take a minute for large datasets"):
                    url_concepts = extract_url_concepts(df2)
                
                col1, col2 = st.columns([3, 1])
                
                #with col1:
                st.write("### 📊 Concept Analysis Results")
                st.dataframe(
                    url_concepts.head(50),
                    use_container_width=True,
                    height=400
                )
                
                #with col2:
                    
                st.download_button(
                    "📥 Download Full Results CSV",
                    url_concepts.to_csv(index=False),
                    "url_concepts.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                fig2 = create_url_treemap(url_concepts)
                st.plotly_chart(fig2, use_container_width=True)

                st.download_button(
                    "📥 Download Treemap HTML",
                    fig2.to_html(full_html=True),
                    file_name="url_treemap.html",
                    mime="text/html"
                )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please check your CSV format and try again.")
