import pandas as pd
import spacy
import re
from urllib.parse import urlparse
from collections import Counter, defaultdict

# ===============================
# CONFIG
# ===============================
MIN_UNIGRAM_FREQ = 2
MIN_BIGRAM_FREQ = 2
MIN_TRIGRAM_FREQ = 2

MIN_COHESION_BIGRAM = 0.25
MIN_COHESION_TRIGRAM = 0.20

MAX_CONCEPTS = 200  # safety cap

nlp = spacy.load("en_core_web_sm")


# ===============================
# URL CLEANING
# ===============================
def extract_domain(url: str) -> str:
    """Return domain without www."""
    u = urlparse(url.lower())
    net = u.netloc if u.netloc else u.path.split('/')[0]
    return re.sub(r"^www\.", "", net)


def is_homepage(url: str) -> bool:
    """Detect whether URL points to homepage."""
    u = urlparse(url.lower())
    path = u.path.strip("/")
    qs = u.query.strip()
    fr = u.fragment.strip()

    if path == "":
        return True

    return False


def slug_from_url(url: str) -> str:
    """Extract meaningful text from URL path."""
    u = urlparse(url.lower())
    path = u.path

    # Remove file extensions
    path = re.sub(r"\.(html|htm|php|aspx|jsp)$", "", path)

    # Replace separators with spaces
    path = re.sub(r"[/\-_+]", " ", path)

    # remove numbers
    path = re.sub(r"\b\d+\b", " ", path)

    # remove duplicates
    path = re.sub(r"\s+", " ", path).strip()
    return path


def clean_url_to_keywords(url: str) -> str:
    """
    Convert URL into keyword-like text.
    Homepage → domain name
    Deep URL → domain + cleaned slug
    """
    domain = extract_domain(url)
    slug = slug_from_url(url)

    if is_homepage(url):
        return domain.replace(".", " ")

    # keep domain token in path concepts (per your instruction)
    combined = domain.replace(".", " ") + " " + slug
    combined = re.sub(r"\s+", " ", combined).strip()
    return combined


# ===============================
# NLP HELPERS
# ===============================
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


# ===============================
# CORE EXTRACTION
# ===============================
def build_ngram_maps(records):
    unigram_counts = Counter()
    bigram_counts = Counter()
    trigram_counts = Counter()
    phrase_map = defaultdict(set)

    for url, original_url in records:
        tokens = tokenize_spacy(url)
        if not tokens:
            continue

        for w in tokens:
            unigram_counts[w] += 1
            phrase_map[w].add(original_url)

        for bg in ngrams(tokens, 2):
            bigram_counts[bg] += 1
            phrase_map[bg].add(original_url)

        for tg in ngrams(tokens, 3):
            trigram_counts[tg] += 1
            phrase_map[tg].add(original_url)

    return unigram_counts, bigram_counts, trigram_counts, phrase_map


def filter_strong_ngrams(unigram_counts, bigram_counts, trigram_counts):
    strong_unigrams = {w: c for w, c in unigram_counts.items() if c >= MIN_UNIGRAM_FREQ}

    strong_bigrams = {}
    for bg, cnt in bigram_counts.items():
        if cnt < MIN_BIGRAM_FREQ:
            continue
        w1, w2 = bg.split()
        f1 = unigram_counts.get(w1, 1)
        f2 = unigram_counts.get(w2, 1)
        coh = cnt / max(f1, f2)
        if coh >= MIN_COHESION_BIGRAM:
            strong_bigrams[bg] = cnt

    strong_trigrams = {}
    for tg, cnt in trigram_counts.items():
        if cnt < MIN_TRIGRAM_FREQ:
            continue
        w1, w2, w3 = tg.split()
        f1 = unigram_counts.get(w1, 1)
        f2 = unigram_counts.get(w2, 1)
        f3 = unigram_counts.get(w3, 1)
        coh = cnt / max(f1, f2, f3)
        if coh >= MIN_COHESION_TRIGRAM:
            strong_trigrams[tg] = cnt

    return strong_unigrams, strong_bigrams, strong_trigrams


def build_initial_concepts(strong_unigrams, strong_bigrams, strong_trigrams, phrase_map):
    concepts = {}
    for tg, cnt in strong_trigrams.items():
        concepts[tg] = {"phrase": tg, "count": cnt, "keywords": set(phrase_map[tg]), "arity": 3}
    for bg, cnt in strong_bigrams.items():
        concepts[bg] = {"phrase": bg, "count": cnt, "keywords": set(phrase_map[bg]), "arity": 2}
    for w, cnt in strong_unigrams.items():
        concepts[w] = {"phrase": w, "count": cnt, "keywords": set(phrase_map[w]), "arity": 1}
    return concepts


# ===============================
# DOMINANCE FILTERING
# ===============================
def dominance_filter(concepts):
    items = sorted(concepts.values(), key=lambda x: (-x['arity'], -x['count']))
    to_remove = set()

    for i, P in enumerate(items):
        if P['phrase'] in to_remove:
            continue
        for Q in items:
            if Q['arity'] <= P['arity']:
                continue
            if P['keywords'].issubset(Q['keywords']):
                to_remove.add(P['phrase'])
                break

    filtered = {ph: v for ph, v in concepts.items() if ph not in to_remove}
    return filtered, to_remove


# ===============================
# FINAL CLEANUP
# ===============================
def finalize_concepts(filtered_concepts):
    final_list = sorted(filtered_concepts.values(), key=lambda x: (-x['count'], -x['arity'], x['phrase']))
    if len(final_list) > MAX_CONCEPTS:
        final_list = final_list[:MAX_CONCEPTS]
    return final_list


# ===============================
# EXECUTION
# ===============================
if __name__ == "__main__":
    df = pd.read_csv("URL_data.csv")

    # Clean URLs → text concepts
    df["cleaned"] = df["URL"].apply(clean_url_to_keywords)

    records = list(zip(df["cleaned"], df["URL"]))

    # N-gram building
    unigram_counts, bigram_counts, trigram_counts, phrase_map = build_ngram_maps(records)
    strong_unigrams, strong_bigrams, strong_trigrams = filter_strong_ngrams(unigram_counts, bigram_counts, trigram_counts)
    concepts = build_initial_concepts(strong_unigrams, strong_bigrams, strong_trigrams, phrase_map)
    filtered_concepts, removed = dominance_filter(concepts)
    final_concepts = finalize_concepts(filtered_concepts)

    # Total impressions for exposure %
    total_impressions = df["Page Impression"].sum()

    rows = []
    for c in final_concepts:
        urls_for_c = list(c["keywords"])
        df_c = df[df["URL"].isin(urls_for_c)]

        imp_sum = df_c["Page Impression"].sum()
        clicks_sum = df_c["Keyword Clicks"].sum()
        rev_sum = df_c["Revenue"].sum()

        exposure_pct = (imp_sum / total_impressions) * 100 if total_impressions else 0
        rpc = rev_sum / clicks_sum if clicks_sum else 0
        ctr = clicks_sum / imp_sum if imp_sum else 0

        rows.append({
            "concept": c["phrase"],
            "concept_count": c["count"],
            "matched_url_count": len(c["keywords"]),
            "exposure_pct": round(exposure_pct, 2),
            "rpc": round(rpc, 2),
            "ctr": round(ctr * 100, 2),
            "matched_urls": "; ".join(sorted(c["keywords"]))
        })

    out = pd.DataFrame(rows)
    out.to_csv("URL_concept_output.csv", index=False)

    print(f"Input URLs: {len(df)}")
    print(f"Unigrams (>=min): {len(strong_unigrams)}")
    print(f"Bigrams (strong): {len(strong_bigrams)}")
    print(f"Trigrams (strong): {len(strong_trigrams)}")
    print(f"Initial concepts total: {len(concepts)}")
    print(f"Removed by dominance: {len(removed)}")
    print(f"Final concepts saved: {len(final_concepts)} → URL_concept_output.csv")
