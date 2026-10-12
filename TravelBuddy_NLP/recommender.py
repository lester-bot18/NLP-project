import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. LOAD AND PREPARE DATASET
# --------------------------------------------------

df = pd.read_csv("destinations.csv")

required_columns = {
    "name",
    "state",
    "description",
    "type",
    "best_season",
    "budget"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns in destinations.csv: {missing_columns}"
    )

# Handle missing text values.
for column in [
    "name",
    "state",
    "description",
    "type",
    "best_season"
]:
    df[column] = df[column].fillna("").astype(str)

# Ensure budgets contain numeric values.
df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
df = df.dropna(subset=["budget"]).reset_index(drop=True)

if df.empty:
    raise ValueError(
        "No valid destinations found in destinations.csv."
    )

# Combine destination information for text matching.
df["features"] = (
    df["name"] + " " +
    df["state"] + " " +
    df["description"] + " " +
    df["type"]
)

# Convert destination descriptions into TF-IDF vectors.
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

tfidf_matrix = vectorizer.fit_transform(df["features"])


# --------------------------------------------------
# 2. EXTRACT USER PREFERENCES
# --------------------------------------------------

CATEGORY_KEYWORDS = {
    "Mountains": [
        "mountain",
        "mountains",
        "mountainous",
        "hill",
        "hills",
        "hill station",
        "snow",
        "snowfall"
    ],

    "Beach": [
        "beach",
        "beaches",
        "sea",
        "seaside",
        "coastal",
        "island",
        "islands"
    ],

    "Historical": [
        "historical",
        "history",
        "historic",
        "fort",
        "forts",
        "palace",
        "palaces",
        "heritage",
        "culture"
    ],

    "Nature": [
        "nature",
        "greenery",
        "forest",
        "forests",
        "waterfall",
        "waterfalls",
        "tea plantations",
        "wildlife",
        "scenery"
    ],

    "Adventure": [
        "adventure",
        "rafting",
        "camping",
        "trekking",
        "trek",
        "hiking",
        "hike",
        "scuba diving",
        "skiing",
        "mountain biking"
    ]
}

SEASONS = [
    "Summer",
    "Winter",
    "Monsoon",
    "Spring"
]


def contains_keyword(text, keyword):
    """
    Match a keyword or phrase without matching it
    accidentally inside another word.
    """
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def extract_preferences(query):
    """
    Identify categories, season, budget, and other
    travel preferences from the user's query.
    """

    text = query.lower()

    preferences = {
        "category": None,
        "categories": [],
        "season": None,
        "max_budget": None,
        "peaceful": False,
        "cold_weather": False,
        "trekking": False
    }

    # Detect all matching travel categories.
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(
            contains_keyword(text, keyword)
            for keyword in keywords
        ):
            preferences["categories"].append(category)

    # Keep the first category for backward compatibility.
    if preferences["categories"]:
        preferences["category"] = preferences["categories"][0]

    # Detect the preferred season.
    for season in SEASONS:
        if contains_keyword(text, season.lower()):
            preferences["season"] = season
            break

    # Detect budget formats such as:
    # ₹8000, Rs. 8000, Rs 8,000, 8000 rupees, 8000 INR.
    budget_match = re.search(
        r"(?:₹|rs\.?\s*)\s*(\d[\d,]*)"
        r"|(\d[\d,]*)\s*(?:rupees|inr)\b",
        text,
        flags=re.IGNORECASE
    )

    if budget_match:
        value = next(
            group
            for group in budget_match.groups()
            if group is not None
        )

        preferences["max_budget"] = int(
            value.replace(",", "")
        )

    # Detect peaceful or relaxing preferences.
    peaceful_keywords = [
        "peaceful",
        "quiet",
        "calm",
        "relaxing",
        "tranquil",
        "less crowded"
    ]

    preferences["peaceful"] = any(
        contains_keyword(text, keyword)
        for keyword in peaceful_keywords
    )

    # Detect cold-weather preferences.
    cold_keywords = [
        "cold",
        "cold weather",
        "snow",
        "snowfall",
        "chilly",
        "cool climate"
    ]

    preferences["cold_weather"] = any(
        contains_keyword(text, keyword)
        for keyword in cold_keywords
    )

    # Detect trekking and hiking preferences.
    trekking_keywords = [
        "trekking",
        "trek",
        "treks",
        "hiking",
        "hike",
        "hikes"
    ]

    preferences["trekking"] = any(
        contains_keyword(text, keyword)
        for keyword in trekking_keywords
    )

    return preferences


# --------------------------------------------------
# 3. RECOMMENDATION ENGINE
# --------------------------------------------------

def recommend_destinations(query, top_n=3):
    """
    Rank destinations using:
    1. TF-IDF and cosine similarity
    2. Category matching
    3. Season matching
    4. Peaceful destination preference
    5. Cold-weather preference
    6. Trekking preference
    7. Budget matching
    """

    if not query or not query.strip():
        return []

    preferences = extract_preferences(query)

    # Convert the user's query into a TF-IDF vector.
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity against all destinations.
    text_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    results = []

    for index, row in df.iterrows():

        # Start with the textual similarity score.
        score = float(text_scores[index])
        reasons = []

        destination_type = row["type"].strip().lower()

        # ------------------------------------------
        # Category matching
        # ------------------------------------------

        if preferences["categories"]:

            if any(
                destination_type == category.lower()
                for category in preferences["categories"]
            ):
                score += 0.20
                reasons.append(
                    "Matches your preferred travel category"
                )

        # ------------------------------------------
        # Season matching
        # ------------------------------------------

        if preferences["season"]:

            if (
                row["best_season"].strip().lower()
                == preferences["season"].lower()
            ):
                score += 0.15
                reasons.append(
                    "Matches your preferred travel season"
                )

        # ------------------------------------------
        # Peaceful destination matching
        # ------------------------------------------

        if preferences["peaceful"]:

            peaceful_text = (
                row["description"] + " " +
                row["name"] + " " +
                row["type"]
            ).lower()

            peaceful_terms = [
                "peaceful",
                "quiet",
                "relaxing",
                "tranquil",
                "nature",
                "scenic",
                "misty",
                "less crowded"
            ]

            if any(
                contains_keyword(peaceful_text, term)
                for term in peaceful_terms
            ):
                score += 0.10
                reasons.append(
                    "Matches peaceful or nature-related preferences"
                )

        # ------------------------------------------
        # Cold-weather matching
        # ------------------------------------------

        if preferences["cold_weather"]:

            cold_text = (
                row["description"] + " " +
                row["name"] + " " +
                row["type"]
            ).lower()

            cold_terms = [
                "cold",
                "snow",
                "snowfall",
                "mountain",
                "mountains",
                "misty",
                "cool",
                "winter",
                "chilly"
            ]

            if any(
                contains_keyword(cold_text, term)
                for term in cold_terms
            ):
                score += 0.10
                reasons.append(
                    "Matches your cold-weather preference"
                )

        # ------------------------------------------
        # Trekking and hiking matching
        # ------------------------------------------

        if preferences["trekking"]:

            trekking_text = (
                row["description"] + " " +
                row["name"] + " " +
                row["type"]
            ).lower()

            trekking_terms = [
                "trekking",
                "trek",
                "treks",
                "hiking",
                "hike",
                "adventure",
                "mountain biking",
                "rafting",
                "camping"
            ]

            if any(
                contains_keyword(trekking_text, term)
                for term in trekking_terms
            ):
                score += 0.10
                reasons.append(
                    "Matches your trekking or hiking preference"
                )

        # ------------------------------------------
        # Budget matching
        # ------------------------------------------

        within_budget = True

        if preferences["max_budget"] is not None:

            within_budget = (
                row["budget"] <= preferences["max_budget"]
            )

            if within_budget:
                score += 0.15
                reasons.append(
                    "Within your stated budget"
                )
            else:
                score -= 0.20
                reasons.append(
                    "Exceeds your stated budget"
                )

        # Add destination to the result list.
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
            "preferences": preferences
        })

    # Prioritize destinations within budget first.
    # Within each group, sort by the combined ranking score.
    if preferences["max_budget"] is not None:

        results.sort(
            key=lambda place: (
                place["within_budget"],
                place["similarity"]
            ),
            reverse=True
        )

    else:

        results.sort(
            key=lambda place: place["similarity"],
            reverse=True
        )

    return results[:max(0, top_n)]


# --------------------------------------------------
# 4. COMMAND-LINE TEST
# --------------------------------------------------

if __name__ == "__main__":

    query = input(
        "Where would you like to travel? "
    )

    preferences = extract_preferences(query)

    print("\nDetected preferences:")

    for key, value in preferences.items():
        print(f"{key}: {value}")

    results = recommend_destinations(query)

    print("\nRecommended destinations:\n")

    if not results:

        print("No destinations available.")

    else:

        for index, place in enumerate(results, start=1):

            print(
                f"{index}. {place['name']}, {place['state']}"
            )

            print(
                f"   Category: {place['type']}"
            )

            print(
                f"   Best season: {place['best_season']}"
            )

            print(
                f"   Estimated budget: ₹{place['budget']:,}"
            )

            print(
                f"   Ranking score: {place['similarity']:.3f}"
            )

            if place["within_budget"]:
                print("   Budget status: Within budget")
            else:
                print("   Budget status: Over budget")

            if place["reasons"]:
                print(
                    "   Reasons: " +
                    "; ".join(place["reasons"])
                )

            print()
