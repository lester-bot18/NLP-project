import re
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
    page_icon="✈️",
    layout="wide"
)


# --------------------------------------------------
# 2. HEADER
# --------------------------------------------------

st.title("✈️ TravelBuddy")

st.subheader(
    "Your AI-Powered Travel Recommendation Assistant"
)

st.write(
    "Tell TravelBuddy what kind of trip you want. "
    "Our NLP recommendation engine will suggest destinations "
    "based on your interests, preferences, and budget."
)

st.divider()


# --------------------------------------------------
# 3. SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🌍 TravelBuddy")

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
        st.markdown(f"- {example}")


# --------------------------------------------------
# 4. MESSAGE INTENT DETECTION
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
# 5. CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 Welcome to TravelBuddy! ✈️\n\n"
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
# 6. USER INPUT
# --------------------------------------------------

query = st.chat_input(
    "Where would you like to travel?"
)


# --------------------------------------------------
# 7. PROCESS USER QUERY
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
                "Hello! 👋 Welcome to TravelBuddy! ✈️\n\n"
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
                "You're welcome! 😊\n\n"
                "Whenever you're ready, tell me what kind "
                "of destination you'd like to explore!"
            )

            st.markdown(response)

        # ------------------------------------------
        # FAREWELL
        # ------------------------------------------

        elif intent == "farewell":

            response = (
                "Goodbye! 👋 Thanks for using TravelBuddy. "
                "Have a wonderful day, and happy travels! ✈️"
            )

            st.markdown(response)

        # ------------------------------------------
        # SMALL TALK
        # ------------------------------------------

        elif intent == "small_talk":

            response = (
                "I'm doing great and ready to help you "
                "plan a trip! 😊✈️\n\n"
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
                "assistant! 😊\n\n"
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
                    "## 🌍 Recommended Destinations"
                )

                # ----------------------------------
                # DETECTED PREFERENCES
                # ----------------------------------

                st.markdown(
                    "### 🔎 Preferences Detected"
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

                    for preference in detected:
                        st.markdown(f"- {preference}")

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

                    with st.container():

                        st.markdown(
                            f"### {index}. {place['name']}, "
                            f"{place['state']}"
                        )

                        st.write(
                            place["description"]
                        )

                        col1, col2, col3 = st.columns(3)

                        with col1:

                            st.metric(
                                "Category",
                                place["type"]
                            )

                        with col2:

                            st.metric(
                                "Estimated Budget",
                                f"₹{place['budget']:,}"
                            )

                        with col3:

                            st.metric(
                                "Ranking Score",
                                f"{place['similarity']:.3f}"
                            )

                        st.caption(
                            f"Best season: {place['best_season']}"
                        )

                        # Display budget status if specified.
                        if preferences["max_budget"] is not None:

                            if place["within_budget"]:

                                st.success(
                                    "Within your stated budget"
                                )

                            else:

                                st.warning(
                                    "Exceeds your stated budget"
                                )

                        # Explain the recommendation.
                        if place["reasons"]:

                            st.markdown(
                                "**Why we recommend this destination:**"
                            )

                            for reason in place["reasons"]:

                                st.markdown(
                                    f"- {reason}"
                                )

                        st.divider()

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
# 8. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "TravelBuddy | NLP-Based Travel Recommendation System"
)

st.caption(
    "Recommendations depend on the available dataset. "
    "Budget values are illustrative estimates."
)
