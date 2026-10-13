"""
Travel Destination Recommender - NLP Mini Project
==================================================

EXPERIMENT 1 - PROBLEM STATEMENT
--------------------------------
Real-world NLP application: a Personal-Assistant style *recommendation system*
combined with *text categorization*, *spelling correction* and *question
answering style* query understanding.

Problem: Travellers describe what they want in free text, e.g.
    "peaceful hill station in winter under Rs 8000 with trekking"
The system must understand that text and recommend the best destinations
from a dataset (destinations.csv).

NLP pipeline implemented (experiment number -> function):
    Exp 2  Tokenization, Filtration & Script Validation -> validate_script, tokenize, filter_tokens
    Exp 3  Stop-word removal, Stemming, Lemmatization   -> remove_stopwords, stem_tokens, lemmatize_tokens
    Exp 4  Morphological analysis & word generation     -> morphological_analysis, generate_word_forms
    Exp 5  N-gram model                                 -> NGramModel
    Exp 6  POS taggers (study + tagging)                -> build_taggers, compare_taggers
    Exp 7  Chunking                                     -> chunk_text, extract_noun_phrases
    Exp 8  Named Entity Recognition                     -> recognize_entities
    Exp 9  Text similarity                              -> similarity_report (+ TF-IDF cosine)

Install:  pip install nltk pandas scikit-learn
"""

import math
import re
import unicodedata
from collections import Counter

import nltk
import pandas as pd
from nltk import RegexpParser, RegexpTagger, UnigramTagger, ne_chunk, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.metrics import edit_distance
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 0. NLTK RESOURCES
# --------------------------------------------------

NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    "maxent_ne_chunker": "chunkers/maxent_ne_chunker",
    "maxent_ne_chunker_tab": "chunkers/maxent_ne_chunker_tab",
    "words": "corpora/words",
    "treebank": "corpora/treebank",
}


def ensure_nltk_resources():
    """Download any missing NLTK data (runs once, then cached)."""
    for package, path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_resources()

STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS |= {"rs", "inr", "rupee", "rupees"}  # currency words carry no topic meaning
porter = PorterStemmer()
snowball = SnowballStemmer("english")
lancaster = LancasterStemmer()
lemmatizer = WordNetLemmatizer()


# ==================================================
# EXPERIMENT 2: TOKENIZATION, FILTRATION, SCRIPT VALIDATION
# ==================================================

def validate_script(text):
    """
    Script validation: make sure every letter belongs to the Latin script
    (English). Returns (is_valid, list_of_offending_characters).
    """
    bad = sorted({
        ch for ch in text
        if ch.isalpha() and "LATIN" not in unicodedata.name(ch, "")
    })
    return len(bad) == 0, bad


def tokenize(text):
    """Split text into word tokens (NLTK word_tokenize, regex fallback)."""
    text = text.lower().replace("-", " ")
    try:
        return word_tokenize(text)
    except LookupError:
        return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]", text)


def filter_tokens(tokens):
    """
    Filtration: drop punctuation, numbers, symbols and 1-letter tokens,
    keeping only alphabetic words.
    """
    return [t for t in tokens if t.isalpha() and len(t) > 1]


def strip_noise(text):
    """Remove URLs, e-mail addresses and HTML tags before tokenizing."""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


# ==================================================
# EXPERIMENT 3: STOP WORDS, STEMMING, LEMMATIZATION
# ==================================================

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOP_WORDS]


def stem_tokens(tokens, stemmer=porter):
    return [stemmer.stem(t) for t in tokens]


def wordnet_pos(tag):
    """Convert a Penn Treebank tag to a WordNet POS."""
    return {
        "J": wordnet.ADJ,
        "V": wordnet.VERB,
        "N": wordnet.NOUN,
        "R": wordnet.ADV,
    }.get(tag[0], wordnet.NOUN)


def lemmatize_tokens(tokens):
    """POS-aware lemmatization ('hiking' -> 'hike', 'forts' -> 'fort')."""
    return [
        lemmatizer.lemmatize(word, wordnet_pos(tag))
        for word, tag in pos_tag(tokens)
    ]


def preprocess(text):
    """
    Full preprocessing pipeline. Returns every intermediate stage so it
    can be displayed for the lab report.
    """
    cleaned = strip_noise(text)
    tokens = tokenize(cleaned)
    filtered = filter_tokens(tokens)

    # Lemmatize BEFORE removing stop words so the POS tagger sees full sentences.
    lemmas_all = lemmatize_tokens(filtered)
    keep = [i for i, t in enumerate(filtered) if t not in STOP_WORDS]

    no_stop = [filtered[i] for i in keep]
    lemmas = [lemmas_all[i] for i in keep]
    stems = stem_tokens(no_stop)

    return {
        "tokens": tokens,
        "filtered": filtered,
        "no_stopwords": no_stop,
        "stems": stems,
        "lemmas": lemmas,
    }


def stem_list(text):
    """Stems of all filtered tokens (stop words kept, for phrase matching)."""
    return stem_tokens(filter_tokens(tokenize(strip_noise(text))))


def contains_keyword(stems, keyword):
    """
    Stem-based keyword/phrase match: 'mountains' matches 'mountain',
    'trekking' matches 'trek', and 'sea' never matches inside 'season'.
    """
    key = stem_list(keyword)
    n = len(key)
    return n > 0 and any(stems[i:i + n] == key for i in range(len(stems) - n + 1))


def nlp_analyzer(doc):
    """Custom TF-IDF analyzer: lemma unigrams + lemma bigrams."""
    lemmas = preprocess(doc)["lemmas"]
    return lemmas + [f"{a} {b}" for a, b in zip(lemmas, lemmas[1:])]


# ==================================================
# EXPERIMENT 4: MORPHOLOGICAL ANALYSIS & WORD GENERATION
# ==================================================

PREFIXES = sorted(
    ["un", "re", "dis", "in", "im", "non", "over", "under", "pre", "sub", "inter", "mis"],
    key=len, reverse=True,
)
SUFFIXES = sorted(
    ["ness", "ment", "ation", "tion", "ing", "ed", "ly", "ful", "less",
     "able", "ous", "ive", "al", "er", "est", "es", "s"],
    key=len, reverse=True,
)


def morphological_analysis(word):
    """
    Break a word into prefix + root + suffix and report its stem, lemma
    and part of speech.
    """
    w = word.lower()
    prefix = next((p for p in PREFIXES if w.startswith(p) and len(w) - len(p) >= 3), "")
    rest = w[len(prefix):]
    suffix = next((s for s in SUFFIXES if rest.endswith(s) and len(rest) - len(s) >= 3), "")
    root = rest[:len(rest) - len(suffix)] if suffix else rest

    tag = pos_tag([w])[0][1]
    return {
        "word": w,
        "prefix": prefix or "-",
        "root": root,
        "suffix": suffix or "-",
        "stem": porter.stem(w),
        "lemma": lemmatizer.lemmatize(w, wordnet_pos(tag)),
        "pos": tag,
    }


def generate_word_forms(word):
    """
    Word generation: inflected forms by spelling rules plus derivationally
    related words from WordNet.
    """
    w = word.lower()
    base = min(
        {lemmatizer.lemmatize(w, "n"), lemmatizer.lemmatize(w, "v")}, key=len
    )

    inflected = {base}

    # Plural / 3rd-person singular
    if base.endswith(("s", "x", "z", "ch", "sh")):
        inflected.add(base + "es")
    elif base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        inflected.add(base[:-1] + "ies")
    else:
        inflected.add(base + "s")

    # Past tense and -ing form
    if base.endswith("e"):
        inflected.update({base + "d", base[:-1] + "ing"})
    elif len(base) <= 4 and re.search(r"[^aeiou][aeiou][^aeiouwxy]$", base):
        inflected.update({base + base[-1] + "ed", base + base[-1] + "ing"})
    else:
        inflected.update({base + "ed", base + "ing"})

    derived = set()
    for synset in wordnet.synsets(base):
        for lemma in synset.lemmas():
            for related in lemma.derivationally_related_forms():
                derived.add(related.name().lower().replace("_", " "))

    return {
        "base": base,
        "inflected": sorted(inflected),
        "derived": sorted(derived - inflected)[:10],
    }


# ==================================================
# EXPERIMENT 5: N-GRAM MODEL
# ==================================================

class NGramModel:
    """
    A simple n-gram language model with add-one (Laplace) smoothing.
    Used to suggest the next word for a travel query and to score text.
    """

    def __init__(self, n=2):
        self.n = n
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = set()

    def train(self, token_lists):
        for tokens in token_lists:
            padded = ["<s>"] * (self.n - 1) + list(tokens) + ["</s>"]
            self.vocab.update(padded)
            for gram in ngrams(padded, self.n):
                self.ngram_counts[gram] += 1
                self.context_counts[gram[:-1]] += 1

    def _context(self, words):
        return tuple((["<s>"] * (self.n - 1) + list(words))[-(self.n - 1):])

    def probability(self, word, context_words):
        ctx = self._context(context_words)
        return (self.ngram_counts[ctx + (word,)] + 1) / (
            self.context_counts[ctx] + len(self.vocab)
        )

    def predict_next(self, context_words, k=3):
        ctx = self._context(context_words)
        candidates = [
            (gram[-1], count)
            for gram, count in self.ngram_counts.items()
            if gram[:-1] == ctx and gram[-1] != "</s>"
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        return candidates[:k]

    def perplexity(self, tokens):
        padded = ["<s>"] * (self.n - 1) + list(tokens) + ["</s>"]
        log_prob = 0.0
        count = 0
        for i in range(self.n - 1, len(padded)):
            log_prob += math.log(self.probability(padded[i], padded[:i]))
            count += 1
        return math.exp(-log_prob / count) if count else float("inf")


def top_ngrams(token_lists, n=2, k=5):
    counter = Counter()
    for tokens in token_lists:
        counter.update(ngrams(tokens, n))
    return [(" ".join(g), c) for g, c in counter.most_common(k)]


# ==================================================
# EXPERIMENT 6: POS TAGGERS
# ==================================================

def build_taggers():
    """
    Three different POS taggers to compare:
      1. NLTK averaged perceptron (statistical, best)
      2. Regex tagger (rule-based, uses word endings)
      3. Unigram tagger trained on the Penn Treebank (most frequent tag per word)
    """
    regex_tagger = RegexpTagger([
        (r"^-?\d+(\.\d+)?$", "CD"),
        (r".*ing$", "VBG"),
        (r".*ed$", "VBD"),
        (r".*ly$", "RB"),
        (r".*(ful|ous|al|ive|able|less)$", "JJ"),
        (r".*s$", "NNS"),
        (r".*", "NN"),
    ])

    taggers = {
        "Averaged Perceptron": pos_tag,
        "Regex (rule-based)": regex_tagger.tag,
    }

    try:
        from nltk.corpus import treebank
        unigram = UnigramTagger(treebank.tagged_sents()[:3000], backoff=regex_tagger)
        taggers["Unigram (Treebank)"] = unigram.tag
    except Exception:
        pass

    return taggers


def compare_taggers(tokens):
    return {name: tag(tokens) for name, tag in TAGGERS.items()}


# ==================================================
# EXPERIMENT 7: CHUNKING
# ==================================================

CHUNK_GRAMMAR = r"""
NP: {<DT>?<JJ.*>*<NN.*>+}
PP: {<IN><NP>}
VP: {<VB.*><NP|PP>*}
"""
chunk_parser = RegexpParser(CHUNK_GRAMMAR)


def chunk_text(text):
    """Return the chunk tree (NP / PP / VP) for a piece of text."""
    tokens = filter_tokens(tokenize(strip_noise(text)))
    return chunk_parser.parse(pos_tag(tokens))


def extract_noun_phrases(text):
    tree = chunk_text(text)
    return [
        " ".join(word for word, _ in subtree.leaves())
        for subtree in tree.subtrees(filter=lambda t: t.label() == "NP")
    ]


GENERIC_NOUNS = {
    "place", "trip", "holiday", "vacation", "destination", "something",
    "budget", "rupee", "rs", "inr", "season", "weather", "idea", "option",
}


def key_phrases(text):
    """Head nouns of noun phrases, used as extra matching signals."""
    heads = []
    for phrase in extract_noun_phrases(text):
        head = lemmatizer.lemmatize(phrase.split()[-1], "n")
        if head not in GENERIC_NOUNS and head not in heads:
            heads.append(head)
    return heads


# ==================================================
# EXPERIMENT 9 (helpers): TEXT SIMILARITY MEASURES
# ==================================================

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def dice(a, b):
    a, b = set(a), set(b)
    return 2 * len(a & b) / (len(a) + len(b)) if (a or b) else 0.0


def edit_similarity(a, b):
    longest = max(len(a), len(b))
    return 1 - edit_distance(a, b) / longest if longest else 1.0


def similarity_report(text_a, text_b):
    """Compare two texts with four different similarity measures."""
    lemmas_a = preprocess(text_a)["lemmas"]
    lemmas_b = preprocess(text_b)["lemmas"]
    vec_a = vectorizer.transform([text_a])
    vec_b = vectorizer.transform([text_b])
    return {
        "TF-IDF cosine": float(cosine_similarity(vec_a, vec_b)[0][0]),
        "Jaccard": jaccard(lemmas_a, lemmas_b),
        "Dice": dice(lemmas_a, lemmas_b),
        "Edit-distance similarity": edit_similarity(text_a.lower(), text_b.lower()),
    }


# --------------------------------------------------
# 1. LOAD AND PREPARE DATASET
# --------------------------------------------------

df = pd.read_csv("destinations.csv")

required_columns = {"name", "state", "description", "type", "best_season", "budget"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing columns in destinations.csv: {missing_columns}")

for column in ["name", "state", "description", "type", "best_season"]:
    df[column] = df[column].fillna("").astype(str)

df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
df = df.dropna(subset=["budget"]).reset_index(drop=True)

if df.empty:
    raise ValueError("No valid destinations found in destinations.csv.")

df["features"] = df["name"] + " " + df["state"] + " " + df["description"] + " " + df["type"]

# NLP-preprocessed columns used for matching.
df["lemmas"] = df["features"].apply(lambda t: preprocess(t)["lemmas"])
df["lemma_set"] = df["lemmas"].apply(set)
df["stems"] = df["features"].apply(stem_list)

DATASET_LEMMAS = set().union(*df["lemma_set"])

# TF-IDF over the lemmatized text (Exp 3 + Exp 5 + Exp 9).
vectorizer = TfidfVectorizer(analyzer=nlp_analyzer, sublinear_tf=True)
tfidf_matrix = vectorizer.fit_transform(df["features"])

# N-gram language models trained on destination descriptions (Exp 5).
bigram_model = NGramModel(n=2)
bigram_model.train(df["lemmas"])
trigram_model = NGramModel(n=3)
trigram_model.train(df["lemmas"])

# POS taggers (Exp 6).
TAGGERS = build_taggers()


# --------------------------------------------------
# SPELLING CORRECTION (edit distance)
# --------------------------------------------------

CATEGORY_KEYWORDS = {
    "Mountains": ["mountain", "mountains", "mountainous", "hill", "hills",
                  "hill station", "snow", "snowfall"],
    "Beach": ["beach", "beaches", "sea", "seaside", "coastal", "island", "islands"],
    "Historical": ["historical", "history", "historic", "fort", "forts", "palace",
                   "palaces", "heritage", "culture"],
    "Nature": ["nature", "greenery", "forest", "forests", "waterfall", "waterfalls",
               "tea plantations", "wildlife", "scenery"],
    "Adventure": ["adventure", "rafting", "camping", "trekking", "trek", "hiking",
                  "hike", "scuba diving", "skiing", "mountain biking"],
}

SEASONS = ["Summer", "Winter", "Monsoon", "Spring"]

PEACEFUL_KEYWORDS = ["peaceful", "quiet", "calm", "relaxing", "tranquil", "less crowded"]
COLD_KEYWORDS = ["cold", "cold weather", "snow", "snowfall", "chilly", "cool climate"]
TREKKING_KEYWORDS = ["trekking", "trek", "treks", "hiking", "hike", "hikes"]


def build_vocabulary():
    vocab = set()
    for text in df["features"]:
        vocab.update(filter_tokens(tokenize(strip_noise(text))))
    for words in CATEGORY_KEYWORDS.values():
        for phrase in words:
            vocab.update(phrase.split())
    for group in (SEASONS, PEACEFUL_KEYWORDS, COLD_KEYWORDS, TREKKING_KEYWORDS):
        for phrase in group:
            vocab.update(phrase.lower().split())
    return vocab


VOCABULARY = build_vocabulary()


def correct_spelling(query):
    """
    Fix misspelled words using minimum edit distance against the dataset
    vocabulary. Real English words (found in WordNet) are left untouched.
    """
    corrections = {}
    sorted_vocab = sorted(VOCABULARY)

    def fix(match):
        word = match.group(0)
        lower = word.lower()
        if (len(lower) < 4 or lower in VOCABULARY
                or lower in STOP_WORDS or wordnet.synsets(lower)):
            return word
        best = min(sorted_vocab, key=lambda v: edit_distance(lower, v))
        limit = 1 if len(lower) <= 6 else 2
        if edit_distance(lower, best) <= limit:
            corrections[word] = best
            return best
        return word

    return re.sub(r"[A-Za-z]+", fix, query), corrections


# ==================================================
# EXPERIMENT 8: NAMED ENTITY RECOGNITION
# ==================================================

BUDGET_PATTERN = re.compile(
    r"(?:₹|rs\.?\s*)\s*(\d[\d,]*)|(\d[\d,]*)\s*(?:rupees|inr)\b",
    flags=re.IGNORECASE,
)
DURATION_PATTERN = re.compile(r"(\d+)\s*(days?|nights?|weeks?)", flags=re.IGNORECASE)


def build_gazetteer():
    """Lookup list of known destination and state names."""
    gazetteer = {}
    for _, row in df.iterrows():
        for value, kind in ((row["name"], "DESTINATION"), (row["state"], "STATE")):
            value = value.strip().lower()
            if value:
                gazetteer[value] = kind
    return gazetteer


GAZETTEER = build_gazetteer()


def recognize_entities(query):
    """
    Hybrid NER:
      - gazetteer lookup for destinations / states (works on lowercase text)
      - NLTK statistical NER (GPE, PERSON, ORGANIZATION) for capitalised text
      - rule/regex entities: MONEY, SEASON, DURATION
    """
    entities = []
    lowered = query.lower()

    for phrase, kind in GAZETTEER.items():
        if re.search(r"\b" + re.escape(phrase) + r"\b", lowered):
            entities.append({"text": phrase, "label": kind, "source": "gazetteer"})

    try:
        for subtree in ne_chunk(pos_tag(word_tokenize(query))):
            if hasattr(subtree, "label"):
                text = " ".join(word for word, _ in subtree.leaves())
                if not any(e["text"] == text.lower() for e in entities):
                    entities.append({"text": text, "label": subtree.label(),
                                     "source": "nltk ne_chunk"})
    except LookupError:
        pass

    match = BUDGET_PATTERN.search(query)
    if match:
        entities.append({"text": match.group(0).strip(), "label": "MONEY", "source": "regex"})

    for match in DURATION_PATTERN.finditer(query):
        entities.append({"text": match.group(0), "label": "DURATION", "source": "regex"})

    stems = stem_list(query)
    for season in SEASONS:
        if contains_keyword(stems, season):
            entities.append({"text": season.lower(), "label": "SEASON", "source": "rule"})

    return entities


# --------------------------------------------------
# 2. EXTRACT USER PREFERENCES
# --------------------------------------------------

def extract_preferences(query):
    """Identify categories, season, budget and other preferences."""
    stems = stem_list(query)

    preferences = {
        "category": None,
        "categories": [],
        "season": None,
        "max_budget": None,
        "peaceful": False,
        "cold_weather": False,
        "trekking": False,
    }

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(contains_keyword(stems, k) for k in keywords):
            preferences["categories"].append(category)

    if preferences["categories"]:
        preferences["category"] = preferences["categories"][0]

    for season in SEASONS:
        if contains_keyword(stems, season):
            preferences["season"] = season
            break

    budget_match = BUDGET_PATTERN.search(query)
    if budget_match:
        value = next(g for g in budget_match.groups() if g is not None)
        preferences["max_budget"] = int(value.replace(",", ""))

    preferences["peaceful"] = any(contains_keyword(stems, k) for k in PEACEFUL_KEYWORDS)
    preferences["cold_weather"] = any(contains_keyword(stems, k) for k in COLD_KEYWORDS)
    preferences["trekking"] = any(contains_keyword(stems, k) for k in TREKKING_KEYWORDS)

    return preferences


# --------------------------------------------------
# 3. RECOMMENDATION ENGINE
# --------------------------------------------------

def recommend_destinations(query, top_n=3):
    """
    Rank destinations using:
    1. TF-IDF cosine similarity on lemmatized text (+ Jaccard overlap)
    2. Category / season / peaceful / cold / trekking matching (stem based)
    3. Named entities (destination or state mentioned by the user)
    4. Noun-phrase head words found by chunking
    5. Budget matching
    """
    if not query or not query.strip():
        return []

    is_valid, _ = validate_script(query)
    if not is_valid:
        return []

    query, _ = correct_spelling(query)

    preferences = extract_preferences(query)
    entities = recognize_entities(query)
    mentioned = {
        e["text"].lower() for e in entities
        if e["label"] in ("DESTINATION", "STATE", "GPE")
    }
    query_lemmas = preprocess(query)["lemmas"]
    phrase_heads = [h for h in key_phrases(query) if h in DATASET_LEMMAS]

    query_vector = vectorizer.transform([query])
    text_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    results = []

    for index, row in df.iterrows():
        score = float(text_scores[index]) + 0.10 * jaccard(query_lemmas, row["lemma_set"])
        reasons = []
        row_stems = row["stems"]
        destination_type = row["type"].strip().lower()

        if preferences["categories"]:
            if any(destination_type == c.lower() for c in preferences["categories"]):
                score += 0.20
                reasons.append("Matches your preferred travel category")

        if preferences["season"]:
            if row["best_season"].strip().lower() == preferences["season"].lower():
                score += 0.15
                reasons.append("Matches your preferred travel season")

        if preferences["peaceful"]:
            terms = ["peaceful", "quiet", "relaxing", "tranquil", "nature",
                     "scenic", "misty", "less crowded"]
            if any(contains_keyword(row_stems, t) for t in terms):
                score += 0.10
                reasons.append("Matches peaceful or nature-related preferences")

        if preferences["cold_weather"]:
            terms = ["cold", "snow", "snowfall", "mountain", "mountains",
                     "misty", "cool", "winter", "chilly"]
            if any(contains_keyword(row_stems, t) for t in terms):
                score += 0.10
                reasons.append("Matches your cold-weather preference")

        if preferences["trekking"]:
            terms = ["trekking", "trek", "treks", "hiking", "hike", "adventure",
                     "mountain biking", "rafting", "camping"]
            if any(contains_keyword(row_stems, t) for t in terms):
                score += 0.10
                reasons.append("Matches your trekking or hiking preference")

        # Named-entity boost (Exp 8).
        if row["name"].strip().lower() in mentioned:
            score += 0.30
            reasons.append("Destination you mentioned by name")
        elif row["state"].strip().lower() in mentioned:
            score += 0.20
            reasons.append("Located in the state you mentioned")

        # Chunking boost (Exp 7): key noun phrases found in the destination.
        if phrase_heads:
            matched = [h for h in phrase_heads if h in row["lemma_set"]]
            if matched:
                score += 0.10 * len(matched) / len(phrase_heads)
                reasons.append("Matches key phrases: " + ", ".join(matched))

        within_budget = True
        if preferences["max_budget"] is not None:
            within_budget = row["budget"] <= preferences["max_budget"]
            if within_budget:
                score += 0.15
                reasons.append("Within your stated budget")
            else:
                score -= 0.20
                reasons.append("Exceeds your stated budget")

        results.append({
            "name": row["name"],
            "state": row["state"],
            "description": row["description"],
            "type": row["type"],
            "best_season": row["best_season"],
            "budget": int(row["budget"]),
            "similarity": round(score, 3),
            "within_budget": within_budget,
            "reasons": reasons,
            "preferences": preferences,
        })

    if preferences["max_budget"] is not None:
        results.sort(key=lambda p: (p["within_budget"], p["similarity"]), reverse=True)
    else:
        results.sort(key=lambda p: p["similarity"], reverse=True)

    return results[:max(0, top_n)]


# --------------------------------------------------
# 4. PIPELINE DEMO (prints the output of every experiment)
# --------------------------------------------------

def heading(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def show_pipeline(query, top_result=None):
    heading("EXPERIMENT 2: Tokenization, Filtration & Script Validation")
    valid, bad = validate_script(query)
    print("Script valid (Latin/English):", valid, "" if valid else f"-> bad chars: {bad}")
    corrected, fixes = correct_spelling(query)
    print("Spelling corrections        :", fixes or "none")
    print("Corrected query             :", corrected)
    tokens = tokenize(strip_noise(corrected))
    print("Tokens                      :", tokens)
    print("After filtration            :", filter_tokens(tokens))

    heading("EXPERIMENT 3: Stop Words, Stemming, Lemmatization")
    steps = preprocess(corrected)
    print("Stop words removed :", steps["no_stopwords"])
    print("Porter stems       :", steps["stems"])
    print("Snowball stems     :", stem_tokens(steps["no_stopwords"], snowball))
    print("Lancaster stems    :", stem_tokens(steps["no_stopwords"], lancaster))
    print("Lemmas (POS-aware) :", steps["lemmas"])

    heading("EXPERIMENT 4: Morphological Analysis & Word Generation")
    for word in steps["no_stopwords"][:5]:
        info = morphological_analysis(word)
        print(f"{info['word']:<12} prefix={info['prefix']:<6} root={info['root']:<10} "
              f"suffix={info['suffix']:<5} stem={info['stem']:<10} "
              f"lemma={info['lemma']:<10} pos={info['pos']}")
    print()
    for word in steps["no_stopwords"][:3]:
        forms = generate_word_forms(word)
        print(f"Word generation for '{word}': inflected={forms['inflected']} "
              f"derived={forms['derived']}")

    heading("EXPERIMENT 5: N-Gram Model")
    lemmas = steps["lemmas"]
    print("Query bigrams  :", [" ".join(g) for g in ngrams(lemmas, 2)])
    print("Query trigrams :", [" ".join(g) for g in ngrams(lemmas, 3)])
    print("Top corpus bigrams :", top_ngrams(df["lemmas"], 2))
    print("Top corpus trigrams:", top_ngrams(df["lemmas"], 3))
    if lemmas:
        print(f"Next word after '{lemmas[-1]}' (bigram) :", bigram_model.predict_next([lemmas[-1]]))
        print(f"Next word after '{' '.join(lemmas[-2:])}' (trigram):",
              trigram_model.predict_next(lemmas[-2:]))
        print("Bigram perplexity of query:", round(bigram_model.perplexity(lemmas), 2))

    heading("EXPERIMENT 6: POS Tagging (3 taggers compared)")
    plain_tokens = filter_tokens(tokenize(strip_noise(corrected)))
    for name, tagged in compare_taggers(plain_tokens).items():
        print(f"{name}:\n   {tagged}")

    heading("EXPERIMENT 7: Chunking")
    tree = chunk_text(corrected)
    print(tree)
    print("\nNoun phrases:", extract_noun_phrases(corrected))
    print("Key head nouns:", key_phrases(corrected))

    heading("EXPERIMENT 8: Named Entity Recognition")
    for entity in recognize_entities(corrected):
        print(f"{entity['label']:<12} {entity['text']:<20} (via {entity['source']})")

    heading("EXPERIMENT 9: Text Similarity")
    if top_result:
        doc = f"{top_result['name']} {top_result['state']} {top_result['description']} {top_result['type']}"
        print(f"Query vs. top result ({top_result['name']}):")
        for measure, value in similarity_report(corrected, doc).items():
            print(f"   {measure:<26}{value:.3f}")


# --------------------------------------------------
# 5. COMMAND-LINE TEST
# --------------------------------------------------

if __name__ == "__main__":

    query = input("Where would you like to travel? ")

    valid, bad = validate_script(query)
    if not valid:
        print(f"Please type in English (Latin script). Unsupported characters: {bad}")
        raise SystemExit

    results = recommend_destinations(query)

    show_pipeline(query, top_result=results[0] if results else None)

    heading("RECOMMENDED DESTINATIONS")
    corrected_query, _ = correct_spelling(query)
    print("Detected preferences:")
    for key, value in extract_preferences(corrected_query).items():
        print(f"   {key}: {value}")
    print()

    if not results:
        print("No destinations available.")
    else:
        for index, place in enumerate(results, start=1):
            print(f"{index}. {place['name']}, {place['state']}")
            print(f"   Category: {place['type']}")
            print(f"   Best season: {place['best_season']}")
            print(f"   Estimated budget: ₹{place['budget']:,}")
            print(f"   Ranking score: {place['similarity']:.3f}")
            print("   Budget status:", "Within budget" if place["within_budget"] else "Over budget")
            if place["reasons"]:
                print("   Reasons: " + "; ".join(place["reasons"]))
            print()
