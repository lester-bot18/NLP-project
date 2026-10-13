import re
from html import escape

import streamlit as st

from recommender import (
    recommend_destinations,
    extract_preferences
)


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="TravelBuddy - AI Travel Assistant",
    layout="wide"
)


# --------------------------------------------------
# 2. CUSTOM CSS (single light theme)
# --------------------------------------------------

CSS = """
:root {
    --bg: #F4EDE4;
    --surface: #FFFFFF;
    --text: #1B2A41;
    --muted: #6B7280;
    --primary: #B07A50;
    --border: #E3D5C5;
    --chip-bg: rgba(176, 122, 80, 0.16);
}

.stApp { background: var(--bg); }
[data-testid="stHeader"] { background: transparent; }

/* Full-width layout so the page fills the desktop screen */
.block-container { max-width: 100%; padding: 1.5rem 3rem 6rem 3rem; }

.stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stMarkdownContainer"] { color: var(--text); }
.stApp [data-testid="stCaptionContainer"] p { color: var(--muted); text-align: center; }
.stApp hr { border-color: var(--border); }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0B1D3A 0%, #16294A 100%); }
.stApp [data-testid="stSidebar"] p, .stApp [data-testid="stSidebar"] li,
.stApp [data-testid="stSidebar"] label, .stApp [data-testid="stSidebar"] h1,
.stApp [data-testid="stSidebar"] h2, .stApp [data-testid="stSidebar"] h3,
.stApp [data-testid="stSidebar"] h4, .stApp [data-testid="stSidebar"] span { color: #F3EFE8; }
.stApp [data-testid="stSidebar"] hr { border-color: rgba(255, 255, 255, 0.15); }
.stApp .side-name { font-size: 1.35rem; font-weight: 800; letter-spacing: 2px; color: #F2B705;
    margin-bottom: 4px; }
.stApp .side-chip { background: rgba(255, 255, 255, 0.08); border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 12px; padding: 8px 12px; margin-bottom: 8px; font-size: 0.88rem; color: #F3EFE8; }

/* ---------- Hero ---------- */
.stApp .hero { background: linear-gradient(180deg, #9CC3E6 0%, #EADBC4 62%, #D9B48A 100%);
    border-radius: 28px; padding: 48px 56px; margin-bottom: 22px; width: 100%;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.12); }
.stApp .hero-badge { display: inline-block; background: #2F3A4F; border-radius: 18px;
    padding: 12px 22px; margin-bottom: 24px; }
.stApp .hero-brand { color: #F2B705; font-weight: 800; letter-spacing: 3px; font-size: 1.15rem; }
.stApp .hero-title { color: #14213D; font-size: 3rem; font-weight: 800; line-height: 1.15;
    margin: 0 0 12px 0; }
.stApp .hero-sub { color: #14213D; font-size: 1.15rem; font-weight: 600; margin: 0 0 8px 0; }
.stApp .hero-desc { color: #14213D; opacity: 0.85; font-size: 1.02rem; margin: 0 0 22px 0; }
.stApp .pill-row { display: flex; flex-wrap: wrap; gap: 10px; }
.stApp .pill { background: rgba(255, 255, 255, 0.55); color: #14213D; border-radius: 14px;
    padding: 9px 18px; font-weight: 700; font-size: 0.92rem; }

/* ---------- Chat ---------- */
.stApp [data-testid="stChatMessage"] { background: var(--surface); border: 1px solid var(--border);
    border-radius: 20px; padding: 14px 22px; }
.stApp [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--chip-bg); }
/* No avatar icons in the chat */
.stApp [data-testid="stChatMessageAvatarUser"],
.stApp [data-testid="stChatMessageAvatarAssistant"] { display: none; }
[data-testid="stBottom"] > div { background: var(--bg); }
.stApp [data-testid="stChatInput"] { border-radius: 999px; border: 1.5px solid var(--primary);
    background: var(--surface); }
.stApp [data-testid="stChatInput"] > div { background: var(--surface); border-radius: 999px; }
.stApp [data-testid="stChatInput"]:focus-within,
.stApp [data-testid="stChatInput"] > div:focus-within { border-color: var(--primary);
    box-shadow: 0 0 0 1px var(--primary); }
.stApp [data-testid="stChatInput"] textarea { color: var(--text); }

/* ---------- Section titles and chips ---------- */
.stApp .section-title { color: var(--text); font-size: 1.35rem; font-weight: 800; margin: 6px 0 10px 0; }
.stApp .chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; }
.stApp .chip { background: var(--chip-bg); color: var(--text); border-radius: 999px;
    padding: 6px 14px; font-size: 0.85rem; font-weight: 600; }

/* ---------- Destination cards ---------- */
.stApp .dest-card { background: var(--surface); border: 1px solid var(--border); border-radius: 24px;
    overflow: hidden; margin: 14px 0 20px 0; width: 100%; box-shadow: 0 8px 22px rgba(0, 0, 0, 0.10); }
.stApp .dest-head { position: relative; padding: 26px 32px 22px 32px; color: #FFFFFF; }
.stApp .dest-rank { position: absolute; top: 18px; right: 24px; background: rgba(255, 255, 255, 0.22);
    border-radius: 999px; padding: 4px 14px; font-weight: 800; color: #FFFFFF; }
.stApp .dest-title { color: #FFFFFF; font-size: 1.9rem; font-weight: 800; }
.stApp .dest-state { color: rgba(255, 255, 255, 0.92); font-weight: 600; margin-top: 2px; }
.stApp .dest-body { padding: 20px 32px 24px 32px; }
.stApp .dest-desc { color: var(--text); font-size: 1rem; line-height: 1.55; margin: 0 0 16px 0; }
.stApp .info-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;
    margin-bottom: 16px; }
.stApp .info-pill { background: var(--bg); border: 1.5px solid var(--border); border-radius: 14px;
    padding: 12px 16px; }
.stApp .info-pill span { display: block; color: var(--muted); font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px; }
.stApp .info-pill b { color: var(--text); font-size: 1.1rem; }
.stApp .badge { display: inline-block; border-radius: 12px; padding: 8px 14px; font-weight: 700;
    font-size: 0.9rem; margin-bottom: 14px; }
.stApp .badge.ok { background: rgba(46, 160, 67, 0.16); color: #2EA043; }
.stApp .badge.warn { background: rgba(242, 183, 5, 0.20); color: #C98A00; }
.stApp .why { color: var(--text); font-weight: 800; margin-bottom: 8px; }
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# Header colour for each destination category.
CATEGORY_GRADIENT = {
    "Mountains": "linear-gradient(135deg, #3E5C80, #7FA3C7)",
    "Beach": "linear-gradient(135deg, #1E7F9C, #5FC1D6)",
    "Historical": "linear-gradient(135deg, #8E5A34, #C48E5E)",
    "Nature": "linear-gradient(135deg, #2F6B3F, #6FAE72)",
    "Adventure": "linear-gradient(135deg, #B8501B, #E8962A)",
}
DEFAULT_GRADIENT = "linear-gradient(135deg, #0B1D3A, #3C4658)"


def render_chips(items):
    chips = "".join(f'<span class="chip">{escape(str(item))}</span>' for item in items)
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)


def destination_card(index, place, max_budget):
    gradient = CATEGORY_GRADIENT.get(place["type"].strip().title(), DEFAULT_GRADIENT)

    parts = [
        '<div class="dest-card">',
        f'<div class="dest-head" style="background: {gradient};">',
        f'<span class="dest-rank">#{index}</span>',
        f'<div class="dest-title">{escape(place["name"])}</div>',
        f'<div class="dest-state">📍 {escape(place["state"])}</div>',
        '</div>',
        '<div class="dest-body">',
        f'<p class="dest-desc">{escape(place["description"])}</p>',
        '<div class="info-row">',
        f'<div class="info-pill"><span>Category</span><b>{escape(place["type"])}</b></div>',
        f'<div class="info-pill"><span>Best season</span><b>{escape(place["best_season"])}</b></div>',
        f'<div class="info-pill"><span>Estimated budget</span><b>₹{place["budget"]:,}</b></div>',
        f'<div class="info-pill"><span>Ranking score</span><b>{place["similarity"]:.3f}</b></div>',
        '</div>',
    ]

    # Display budget status if specified.
    if max_budget is not None:
        if place["within_budget"]:
            parts.append('<div class="badge ok">Within your stated budget</div>')
        else:
            parts.append('<div class="badge warn">Exceeds your stated budget</div>')

    # Explain the recommendation.
    if place["reasons"]:
        reason_chips = "".join(
            f'<span class="chip">{escape(reason)}</span>' for reason in place["reasons"]
        )
        parts.append('<div class="why">Why we recommend this destination:</div>')
        parts.append(f'<div class="chip-row">{reason_chips}</div>')

    parts.append('</div></div>')
    return "".join(parts)


# --------------------------------------------------
# 3. HEADER (hero banner)
# --------------------------------------------------

st.markdown(
    '<div class="hero">'
    '<div class="hero-badge"><span class="hero-brand">TRAVELBUDDY</span></div>'
    '<h1 class="hero-title">Explore the best destinations across India!</h1>'
    '<p class="hero-sub">Your AI-Powered Travel Recommendation Assistant</p>'
    '<p class="hero-desc">Tell TravelBuddy what kind of trip you want. '
    'Our NLP recommendation engine will suggest destinations '
    'based on your interests, preferences, and budget.</p>'
    '<div class="pill-row">'
    '<span class="pill">NLP Powered</span>'
    '<span class="pill">Season Aware</span>'
    '<span class="pill">₹ Budget Aware</span>'
    '</div></div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# 4. SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        '<div class="side-name">TRAVELBUDDY</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Discover destinations tailored to your preferences."
    )

    st.divider()

    st.header("About the Project")

    st.write(
        "TravelBuddy is an NLP-based travel recommendation "
        "chatbot that analyzes user queries and recommends "
        "suitable tourist destinations."
    )

    st.subheader("Technologies Used")

    st.markdown(
        """
        - **Python** — Core programming language
        - **Pandas** — Dataset processing
        - **NLTK** — Tokenization, lemmatization, POS tagging, chunking, NER
        - **TF-IDF** — Text feature extraction
        - **Cosine Similarity** — Text similarity calculation
        - **Rule-Based NLP** — Preference and intent detection
        - **Streamlit** — Interactive web interface
        """
    )

    st.divider()

    st.subheader("Try asking")

    example_queries = [
        "I want cold weather and mountains",
        "Suggest beaches with water sports",
        "I love historical forts and palaces",
        "Recommend peaceful nature destinations",
        "I want mountains and trekking",
        "Suggest places under ₹8000",
        "I want to travel in summer"
    ]

    for example in example_queries:
        st.markdown(
            f'<div class="side-chip">{escape(example)}</div>',
            unsafe_allow_html=True
        )


# --------------------------------------------------
# 5. MESSAGE INTENT DETECTION
# --------------------------------------------------

def detect_intent(query):

    text = query.lower().strip()

    # Normalize punctuation and repeated spaces.
    text = re.sub(r"[^\w\s₹]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    greetings = [
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "good morning",
        "good afternoon",
        "good evening",
        "howdy"
    ]

    thanks = [
        "thanks",
        "thank you",
        "thankyou",
        "thanks a lot"
    ]

    farewells = [
        "bye",
        "goodbye",
        "see you",
        "see you later",
        "good night"
    ]

    if text in greetings:
        return "greeting"

    if text in thanks:
        return "thanks"

    if text in farewells:
        return "farewell"

    # Treat common conversational questions separately.
    small_talk = [
        "how are you",
        "how are you doing",
        "whats up",
        "what is up",
        "who are you",
        "what can you do"
    ]

    if text in small_talk:
        return "small_talk"

    # Check whether the message contains travel-related words.
    travel_keywords = [
        "travel",
        "trip",
        "visit",
        "destination",
        "destinations",
        "holiday",
        "holidays",
        "vacation",
        "mountain",
        "mountains",
        "hill",
        "hills",
        "beach",
        "beaches",
        "sea",
        "island",
        "nature",
        "forest",
        "waterfall",
        "wildlife",
        "historical",
        "history",
        "fort",
        "palace",
        "heritage",
        "culture",
        "adventure",
        "trekking",
        "trek",
        "hiking",
        "rafting",
        "camping",
        "skiing",
        "summer",
        "winter",
        "monsoon",
        "spring",
        "peaceful",
        "quiet",
        "relaxing",
        "cold",
        "snow",
        "snowfall",
        "budget",
        "rupees",
        "inr",
        "beaches",
        "₹",
        "rs"
    ]

    # Also recognize budget expressions such as 8000 or 8,000.
    has_budget = re.search(
        r"\b\d[\d,]*\b",
        text
    ) is not None

    if any(
        keyword in text
        for keyword in travel_keywords
    ) or has_budget:
        return "travel"

    return "unknown"


# --------------------------------------------------
# 6. CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! Welcome to TravelBuddy!\n\n"
                "I'm here to help you discover your next "
                "travel destination.\n\n"
                "Tell me what you enjoy, which season you "
                "prefer, or your approximate budget."
            )
        }
    ]


# Display previous messages.
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# 7. USER INPUT
# --------------------------------------------------

query = st.chat_input(
    "Where would you like to travel?"
)


# --------------------------------------------------
# 8. PROCESS USER QUERY
# --------------------------------------------------

if query:

    # Display and save the user's message.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    # Detect the user's intent.
    intent = detect_intent(query)

    # Generate the appropriate response.
    with st.chat_message("assistant"):

        # ------------------------------------------
        # GREETING
        # ------------------------------------------

        if intent == "greeting":

            response = (
                "Hello! Welcome to TravelBuddy!\n\n"
                "I'm your travel recommendation assistant. "
                "Tell me what kind of trip you're planning!\n\n"
                "**You could ask me:**\n\n"
                "- I want cold weather and mountains\n"
                "- Suggest peaceful places under ₹8000\n"
                "- I want beaches and water sports\n"
                "- Recommend historical places in winter"
            )

            st.markdown(response)

        # ------------------------------------------
        # THANKS
        # ------------------------------------------

        elif intent == "thanks":

            response = (
                "You're welcome!\n\n"
                "Whenever you're ready, tell me what kind "
                "of destination you'd like to explore!"
            )

            st.markdown(response)

        # ------------------------------------------
        # FAREWELL
        # ------------------------------------------

        elif intent == "farewell":

            response = (
                "Goodbye! Thanks for using TravelBuddy. "
                "Have a wonderful day, and happy travels!"
            )

            st.markdown(response)

        # ------------------------------------------
        # SMALL TALK
        # ------------------------------------------

        elif intent == "small_talk":

            response = (
                "I'm doing great and ready to help you "
                "plan a trip!\n\n"
                "Tell me what kind of destination you're "
                "looking for, and I'll recommend some places."
            )

            st.markdown(response)

        # ------------------------------------------
        # UNKNOWN OR UNSUPPORTED MESSAGE
        # ------------------------------------------

        elif intent == "unknown":

            response = (
                "I'm TravelBuddy, your travel recommendation "
                "assistant!\n\n"
                "I can help you discover destinations based "
                "on your interests, preferred season, weather, "
                "and budget.\n\n"
                "Try asking something like:\n\n"
                "- Suggest mountains for a winter trip\n"
                "- Recommend peaceful nature destinations\n"
                "- Find historical places under ₹8000"
            )

            st.markdown(response)

        # ------------------------------------------
        # TRAVEL QUERY
        # ------------------------------------------

        else:

            preferences = extract_preferences(query)

            results = recommend_destinations(
                query,
                top_n=3
            )

            if results:

                st.markdown(
                    '<div class="section-title">Recommended Destinations</div>',
                    unsafe_allow_html=True
                )

                # ----------------------------------
                # DETECTED PREFERENCES
                # ----------------------------------

                st.markdown(
                    '<div class="section-title">Preferences Detected</div>',
                    unsafe_allow_html=True
                )

                detected = []

                if preferences["categories"]:

                    detected.extend(
                        preferences["categories"]
                    )

                if preferences["season"]:

                    detected.append(
                        preferences["season"]
                    )

                if preferences["max_budget"] is not None:

                    detected.append(
                        f"Budget: ₹{preferences['max_budget']:,}"
                    )

                if preferences["peaceful"]:
                    detected.append("Peaceful places")

                if preferences["cold_weather"]:
                    detected.append("Cold weather")

                if preferences["trekking"]:
                    detected.append("Trekking and hiking")

                if detected:

                    render_chips(detected)

                else:

                    st.info(
                        "No specific travel preferences were "
                        "detected. Recommendations are based "
                        "mainly on textual similarity."
                    )

                st.divider()

                # ----------------------------------
                # RECOMMENDATION CARDS
                # ----------------------------------

                for index, place in enumerate(
                    results,
                    start=1
                ):

                    st.markdown(
                        destination_card(
                            index,
                            place,
                            preferences["max_budget"]
                        ),
                        unsafe_allow_html=True
                    )

                # ----------------------------------
                # CHAT RESPONSE
                # ----------------------------------

                recommended_names = [
                    place["name"]
                    for place in results
                ]

                response = (
                    "Based on your query, I recommend "
                    + ", ".join(recommended_names)
                    + ". Check the recommendations above "
                    "to see why these destinations may "
                    "suit your trip!"
                )

            else:

                response = (
                    "I couldn't find any destinations in "
                    "the dataset. Try describing your "
                    "preferred travel category, season, "
                    "or budget."
                )

                st.info(response)

    # Save the assistant's response exactly once.
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )


# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "TravelBuddy | NLP-Based Travel Recommendation System"
)

st.caption(
    "Recommendations depend on the available dataset. "
    "Budget values are illustrative estimates."
)