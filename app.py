import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Indonesian Food Recommendation",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS — STYLING & RESPONSIVENESS
# =========================================================

st.markdown(
    """
    <style>

    /* -------- Global -------- */
    .stApp {
        background: linear-gradient(180deg, #fff8f1 0%, #fffdfb 100%);
    }

    .block-container {
        max-width: 900px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* -------- Header -------- */
    .app-header {
        background: linear-gradient(135deg, #ff7e5f 0%, #ff5e62 45%, #ffb347 100%);
        border-radius: 20px;
        padding: 1.8rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(255, 94, 98, 0.25);
        text-align: center;
    }

    .app-header h1 {
        color: white;
        font-size: clamp(1.5rem, 4vw, 2.2rem);
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    .app-header p {
        color: rgba(255, 255, 255, 0.92);
        font-size: clamp(0.85rem, 2.2vw, 1rem);
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    /* -------- Sidebar -------- */
    section[data-testid="stSidebar"] {
        background: #fff3e9;
        border-right: 1px solid #ffe0c7;
    }

    /* Force readable text colors inside sidebar, regardless of
       light/dark theme, since the sidebar background is always light */
    section[data-testid="stSidebar"] * {
        color: #3a2a1e !important;
    }

    section[data-testid="stSidebar"] h3 {
        color: #d9480f !important;
    }

    /* Metric widgets (Total Resep / Total Kolom) */
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #d9480f !important;
        font-weight: 800;
    }

    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #7a5c46 !important;
    }

    /* Code blocks used for example prompts */
    section[data-testid="stSidebar"] code {
        color: #d9480f !important;
        background: #fff !important;
    }

    section[data-testid="stSidebar"] pre {
        background: #fff !important;
        border: 1px solid #ffd8ae !important;
    }

    /* Clear-history button */
    section[data-testid="stSidebar"] button {
        color: #3a2a1e !important;
        background: #fff !important;
        border: 1px solid #ffd8ae !important;
    }

    section[data-testid="stSidebar"] button:hover {
        border-color: #ff7e5f !important;
        color: #d9480f !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #ffe0c7 !important;
    }

    /* -------- Chat bubbles -------- */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 0.4rem 0.2rem;
        margin-bottom: 0.4rem;
    }

    /* -------- Recipe card -------- */
    .recipe-card {
        background: white;
        border: 1px solid #ffe3cf;
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .recipe-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(255, 94, 98, 0.12);
    }

    .recipe-title {
        font-size: clamp(1rem, 2.5vw, 1.25rem);
        font-weight: 700;
        color: #d9480f;
        margin-bottom: 0.3rem;
    }

    .recipe-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0.5rem 0;
    }

    .badge {
        display: inline-block;
        padding: 0.22rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        white-space: nowrap;
    }

    .badge-category {
        background: #fff0e0;
        color: #d9480f;
        border: 1px solid #ffd8ae;
    }

    .badge-score {
        background: #e6f7ee;
        color: #1a7f4e;
        border: 1px solid #b9ecd1;
    }

    .badge-sim {
        background: #eef2ff;
        color: #4338ca;
        border: 1px solid #d6dcfb;
    }

    .recipe-ingredients {
        font-size: 0.9rem;
        color: #444;
        line-height: 1.5;
        background: #fafafa;
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        margin-top: 0.4rem;
    }

    .tag-chip {
        display: inline-block;
        background: #ffe8d6;
        color: #b3541e;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        margin: 0.15rem 0.3rem 0.15rem 0;
        font-weight: 600;
    }

    /* -------- Random CTA (Bingung Mau Makan Apa?) -------- */

    .random-cta-wrap {
        margin-bottom: 1.4rem;
    }

    .random-cta-label {
        text-align: center;
        font-size: 0.85rem;
        color: #b3541e;
        font-weight: 600;
        margin-bottom: 0.5rem;
        letter-spacing: 0.02em;
    }

    div[data-testid="stMarkdown"]:has(#random-cta-anchor) + div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #ffb347 0%, #ff7e5f 55%, #ff5e62 100%);
        color: white !important;
        border: none;
        border-radius: 16px;
        padding: 0.9rem 1.2rem;
        font-size: clamp(1rem, 2.6vw, 1.15rem);
        font-weight: 800;
        box-shadow: 0 8px 22px rgba(255, 94, 98, 0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    div[data-testid="stMarkdown"]:has(#random-cta-anchor) + div.stButton > button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 12px 28px rgba(255, 94, 98, 0.45);
        color: white !important;
        border: none;
    }

    div[data-testid="stMarkdown"]:has(#random-cta-anchor) + div.stButton > button:active {
        transform: translateY(0px) scale(0.99);
    }

    /* -------- Responsive tweaks -------- */
    @media (max-width: 640px) {
        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
        .app-header {
            padding: 1.2rem 1rem;
            border-radius: 14px;
        }
        .recipe-card {
            padding: 0.9rem 1rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("Indonesian_Food_Recipes.csv")

    # Reset index agar index dataset sesuai dengan
    # posisi baris pada TF-IDF matrix
    df = df.reset_index(drop=True)

    return df


rekomendasi_makanan = load_data()


# =========================================================
# SIDEBAR — DATASET INFO
# =========================================================

with st.sidebar:

    st.markdown("### 📊 Dataset Information")

    col_a, col_b = st.columns(2)

    with col_a:
        st.metric("Total Resep", f"{len(rekomendasi_makanan):,}")

    with col_b:
        st.metric("Total Kolom", f"{len(rekomendasi_makanan.columns)}")

    st.divider()

    st.markdown("### 💡 Tips Pencarian")
    st.caption(
        "Sebutkan **bahan** (mis. *ayam, tahu, udang*) dan "
        "**preferensi rasa** (mis. *pedas, manis, gurih*) "
        "supaya rekomendasi lebih akurat."
    )

    st.markdown("**Contoh:**")
    st.code("Saya ingin makanan ayam yang pedas", language=None)
    st.code("Cari resep tahu tempe yang gurih", language=None)

    st.divider()

    if st.button("🗑️ Bersihkan Riwayat Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# MEMBUAT TEXT UNTUK TF-IDF
# =========================================================

def create_recipe_text(df):

    """
    Membuat satu kolom teks gabungan untuk TF-IDF
    berdasarkan kolom yang tersedia pada dataset.
    """

    text_columns = []

    # Kolom nama makanan
    if "Title" in df.columns:
        text_columns.append(
            df["Title"].fillna("").astype(str)
        )

    # Kolom kategori
    if "Category" in df.columns:
        text_columns.append(
            df["Category"].fillna("").astype(str)
        )

    # Kolom bahan
    if "Ingredients Cleaned" in df.columns:
        text_columns.append(
            df["Ingredients Cleaned"]
            .fillna("")
            .astype(str)
        )

    # Jika terdapat kolom ingredients dengan nama lain
    elif "Ingredients" in df.columns:
        text_columns.append(
            df["Ingredients"]
            .fillna("")
            .astype(str)
        )

    # Beberapa kemungkinan nama kolom instruksi
    instruction_columns = [
        "Instructions",
        "Instructions Cleaned",
        "Steps",
        "Directions",
        "Recipe"
    ]

    for column in instruction_columns:

        if column in df.columns:

            text_columns.append(
                df[column]
                .fillna("")
                .astype(str)
            )

            break

    # Pastikan ada minimal satu kolom
    if not text_columns:

        raise ValueError(
            "Tidak ditemukan kolom yang dapat digunakan "
            "untuk membuat Recipe Text."
        )

    # Gabungkan seluruh teks
    recipe_text = text_columns[0]

    for text in text_columns[1:]:

        recipe_text = (
            recipe_text + " " + text
        )

    return recipe_text


# =========================================================
# TF-IDF
# =========================================================

@st.cache_resource
def create_tfidf(df):

    # Membuat teks resep
    recipe_text = create_recipe_text(df)

    # TF-IDF
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    tfidf_matrix = tfidf.fit_transform(
        recipe_text
    )

    return tfidf, tfidf_matrix


tfidf, tfidf_matrix = create_tfidf(
    rekomendasi_makanan
)


# =========================================================
# INGREDIENT DICTIONARY
# =========================================================

common_ingredients = [

    "ayam",
    "daging",
    "sapi",
    "kambing",
    "ikan",
    "udang",
    "cumi",
    "kerang",
    "telur",
    "tahu",
    "tempe",

    "kentang",
    "wortel",
    "kol",
    "kangkung",
    "bayam",
    "brokoli",
    "tomat",
    "terong",
    "jagung",
    "jamur",

    "nasi",
    "mie",
    "tepung",
    "keju",
    "santan",

    "cabai",
    "cabe",
    "bawang",
    "bawang putih",
    "bawang merah",

    "jeruk",
    "lemon"
]


# =========================================================
# PREFERENCE DICTIONARY
# =========================================================

preference_keywords = {

    "pedas": [
        "pedas",
        "cabe",
        "cabai",
        "sambal",
        "lada",
        "merica"
    ],

    "manis": [
        "manis",
        "gula",
        "madu",
        "kecap manis"
    ],

    "asam": [
        "asam",
        "jeruk",
        "cuka",
        "asam jawa"
    ],

    "gurih": [
        "gurih",
        "kaldu",
        "santan",
        "keju"
    ]
}


# =========================================================
# DETECT INGREDIENT
# =========================================================

def detect_ingredients(user_input):

    user_input = user_input.lower()

    detected = []

    for ingredient in common_ingredients:

        pattern = (
            r"\b"
            + re.escape(ingredient)
            + r"\b"
        )

        if re.search(
            pattern,
            user_input
        ):

            detected.append(
                ingredient
            )

    return detected


# =========================================================
# DETECT PREFERENCE
# =========================================================

def detect_preferences(user_input):

    user_input = user_input.lower()

    detected = []

    for preference, keywords in preference_keywords.items():

        for keyword in keywords:

            pattern = (
                r"\b"
                + re.escape(keyword)
                + r"\b"
            )

            if re.search(
                pattern,
                user_input
            ):

                detected.append(
                    preference
                )

                break

    return detected


# =========================================================
# FILTER RECIPE
# =========================================================

def filter_by_ingredients(
    df,
    ingredients
):

    candidate_df = df.copy()

    # Jika kolom Ingredients Cleaned tersedia
    if "Ingredients Cleaned" in candidate_df.columns:

        ingredient_column = (
            candidate_df["Ingredients Cleaned"]
            .fillna("")
            .astype(str)
        )

    elif "Ingredients" in candidate_df.columns:

        ingredient_column = (
            candidate_df["Ingredients"]
            .fillna("")
            .astype(str)
        )

    else:

        # Jika tidak ada kolom ingredients,
        # jangan melakukan filtering
        return candidate_df


    for ingredient in ingredients:

        pattern = (
            r"\b"
            + re.escape(ingredient)
            + r"\b"
        )

        candidate_df = candidate_df[
            ingredient_column.loc[
                candidate_df.index
            ].str.contains(
                pattern,
                case=False,
                na=False,
                regex=True
            )
        ]

        # Update ingredient column
        ingredient_column = ingredient_column.loc[
            candidate_df.index
        ]

    return candidate_df


# =========================================================
# PREFERENCE SCORE
# =========================================================

def calculate_preference_score(
    ingredients_text,
    preferences
):

    ingredients_text = str(
        ingredients_text
    ).lower()

    score = 0

    for preference in preferences:

        keywords = preference_keywords.get(
            preference,
            []
        )

        for keyword in keywords:

            pattern = (
                r"\b"
                + re.escape(keyword)
                + r"\b"
            )

            if re.search(
                pattern,
                ingredients_text
            ):

                score += 1

    return score


# =========================================================
# RECOMMENDATION ENGINE
# =========================================================

def rekomendasi(
    user_input,
    top_n=5
):

    # -----------------------------------------
    # Deteksi bahan
    # -----------------------------------------

    ingredients = detect_ingredients(
        user_input
    )

    # -----------------------------------------
    # Deteksi preferensi
    # -----------------------------------------

    preferences = detect_preferences(
        user_input
    )

    # -----------------------------------------
    # Filter berdasarkan bahan
    # -----------------------------------------

    if ingredients:

        candidate_df = filter_by_ingredients(
            rekomendasi_makanan,
            ingredients
        ).copy()

    else:

        candidate_df = (
            rekomendasi_makanan.copy()
        )


    # -----------------------------------------
    # Jika tidak ada kandidat
    # -----------------------------------------

    if len(candidate_df) == 0:

        return (
            None,
            ingredients,
            preferences
        )


    # -----------------------------------------
    # TF-IDF SIMILARITY
    # -----------------------------------------

    candidate_indices = (
        candidate_df.index
    )

    candidate_matrix = (
        tfidf_matrix[
            candidate_indices
        ]
    )

    # Transform input user
    user_vector = tfidf.transform([
        user_input
    ])

    # Cosine Similarity
    similarity_scores = (
        cosine_similarity(
            user_vector,
            candidate_matrix
        )
        .flatten()
    )

    candidate_df[
        "Similarity"
    ] = similarity_scores


    # -----------------------------------------
    # PREFERENCE SCORE
    # -----------------------------------------

    if "Ingredients Cleaned" in candidate_df.columns:

        ingredient_text = (
            candidate_df[
                "Ingredients Cleaned"
            ]
            .fillna("")
            .astype(str)
        )

    elif "Ingredients" in candidate_df.columns:

        ingredient_text = (
            candidate_df[
                "Ingredients"
            ]
            .fillna("")
            .astype(str)
        )

    else:

        ingredient_text = pd.Series(
            "",
            index=candidate_df.index
        )


    candidate_df[
        "Preference Score"
    ] = ingredient_text.apply(
        lambda x:
        calculate_preference_score(
            x,
            preferences
        )
    )


    # -----------------------------------------
    # NORMALIZATION
    # -----------------------------------------

    max_preference = (
        candidate_df[
            "Preference Score"
        ].max()
    )

    if max_preference > 0:

        candidate_df[
            "Preference Normalized"
        ] = (
            candidate_df[
                "Preference Score"
            ]
            / max_preference
        )

    else:

        candidate_df[
            "Preference Normalized"
        ] = 0


    # -----------------------------------------
    # FINAL SCORE
    # -----------------------------------------

    candidate_df[
        "Final Score"
    ] = (
        candidate_df[
            "Similarity"
        ] * 0.7
        +
        candidate_df[
            "Preference Normalized"
        ] * 0.3
    )


    # -----------------------------------------
    # RANKING
    # -----------------------------------------

    hasil = (
        candidate_df
        .sort_values(
            "Final Score",
            ascending=False
        )
        .head(top_n)
    )


    return (
        hasil,
        ingredients,
        preferences
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="app-header">
        <h1>🍽️ Indonesian Food Recommendation</h1>
        <p>Temukan rekomendasi makanan berdasarkan bahan dan preferensi rasa Anda</p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

if "trigger_random" not in st.session_state:

    st.session_state.trigger_random = False


# =========================================================
# RANDOM CTA — "Bingung Mau Makan Apa?"
# =========================================================

st.markdown(
    """
    <div class="random-cta-wrap">
        <div class="random-cta-label">🤔 Nggak tahu mau makan apa hari ini?</div>
    </div>
    <div id="random-cta-anchor"></div>
    """,
    unsafe_allow_html=True
)

if st.button(
    "🎲 Bingung Mau Makan Apa? Klik Aja!",
    use_container_width=True,
    key="random_cta_button"
):
    st.session_state.trigger_random = True
    st.rerun()


# =========================================================
# HELPER — BUILD RECIPE CARD HTML
# =========================================================

def build_recipe_card(rank, row):

    title = row["Title"] if "Title" in row.index else "Resep"
    category = row["Category"] if "Category" in row.index else None

    if "Ingredients Cleaned" in row.index:
        ingredients_text = row["Ingredients Cleaned"]
    elif "Ingredients" in row.index:
        ingredients_text = row["Ingredients"]
    else:
        ingredients_text = None

    meta_badges = f'<span class="badge badge-score">🏆 Skor {row["Final Score"]:.2f}</span>'
    meta_badges += f'<span class="badge badge-sim">🎯 Kemiripan {row["Similarity"]:.2f}</span>'

    if category:
        meta_badges += f'<span class="badge badge-category">📂 {category}</span>'

    ingredients_html = ""
    if ingredients_text:
        ingredients_html = (
            f'<div class="recipe-ingredients">🧂 <b>Bahan:</b> {ingredients_text}</div>'
        )

    card = f"""
    <div class="recipe-card">
        <div class="recipe-title">{rank}. {title}</div>
        <div class="recipe-meta">{meta_badges}</div>
        {ingredients_html}
    </div>
    """

    return card


# =========================================================
# HELPER — BUILD RANDOM RECIPE CARD HTML
# =========================================================

def build_random_card(rank, row):

    title = row["Title"] if "Title" in row.index else "Resep"
    category = row["Category"] if "Category" in row.index else None

    if "Ingredients Cleaned" in row.index:
        ingredients_text = row["Ingredients Cleaned"]
    elif "Ingredients" in row.index:
        ingredients_text = row["Ingredients"]
    else:
        ingredients_text = None

    meta_badges = '<span class="badge badge-score">🎲 Rekomendasi Acak</span>'

    if category:
        meta_badges += f'<span class="badge badge-category">📂 {category}</span>'

    ingredients_html = ""
    if ingredients_text:
        ingredients_html = (
            f'<div class="recipe-ingredients">🧂 <b>Bahan:</b> {ingredients_text}</div>'
        )

    card = f"""
    <div class="recipe-card">
        <div class="recipe-title">{rank}. {title}</div>
        <div class="recipe-meta">{meta_badges}</div>
        {ingredients_html}
    </div>
    """

    return card


# =========================================================
# EMPTY STATE
# =========================================================

if not st.session_state.messages:

    st.info(
        "👋 Halo! Ceritakan bahan atau selera makanan yang kamu "
        "inginkan, misalnya *\"saya ingin makanan ayam yang pedas\"*, "
        "atau klik tombol **🎲 Bingung Mau Makan Apa?** di sidebar "
        "kalau kamu mau kejutan, dan aku akan carikan resep yang cocok."
    )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    avatar = "🧑" if message["role"] == "user" else "🍽️"

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):

        if message.get("type") == "recipes":

            st.write(message["intro"])

            if message.get("tags_html"):
                st.markdown(message["tags_html"], unsafe_allow_html=True)

            for card_html in message["cards"]:
                st.markdown(card_html, unsafe_allow_html=True)

        else:

            st.write(message["content"])


# =========================================================
# RANDOM RECOMMENDATION (Bingung Mau Makan Apa?)
# =========================================================

if st.session_state.trigger_random:

    st.session_state.trigger_random = False

    with st.chat_message("assistant", avatar="🍽️"):

        with st.spinner("🎲 Memilihkan resep acak untukmu..."):

            n_sample = min(5, len(rekomendasi_makanan))
            random_df = rekomendasi_makanan.sample(n_sample).copy()

        intro = "🎲 Bingung mau makan apa? Coba resep pilihan acak ini:"
        st.write(intro)

        cards = []

        for i, (_, row) in enumerate(random_df.iterrows(), start=1):
            card_html = build_random_card(i, row)
            st.markdown(card_html, unsafe_allow_html=True)
            cards.append(card_html)

        st.session_state.messages.append({
            "role": "assistant",
            "type": "recipes",
            "intro": intro,
            "tags_html": "",
            "cards": cards
        })


# =========================================================
# USER INPUT
# =========================================================

user_input = st.chat_input(
    "Contoh: Saya ingin makanan ayam yang pedas"
)


# =========================================================
# PROCESS INPUT
# =========================================================

if user_input:

    # -----------------------------------------
    # Simpan & tampilkan pesan user
    # -----------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user", avatar="🧑"):
        st.write(user_input)


    # -----------------------------------------
    # Recommendation
    # -----------------------------------------

    with st.spinner("🔍 Mencari resep terbaik untukmu..."):

        hasil, ingredients, preferences = rekomendasi(
            user_input,
            top_n=5
        )


    # -----------------------------------------
    # Assistant response
    # -----------------------------------------

    with st.chat_message("assistant", avatar="🍽️"):

        if hasil is None:

            response = (
                "Maaf, aku belum menemukan resep yang sesuai dengan "
                "bahan yang kamu sebutkan. Coba gunakan bahan lain, ya! 🙏"
            )

            st.write(response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

        else:

            intro = "🍽️ Berikut rekomendasi makanan yang cocok untuk kamu:"
            st.write(intro)

            tags_html = ""

            if ingredients:
                tags_html += "".join(
                    f'<span class="tag-chip">🥩 {ing}</span>' for ing in ingredients
                )

            if preferences:
                tags_html += "".join(
                    f'<span class="tag-chip">❤️ {pref}</span>' for pref in preferences
                )

            if tags_html:
                st.markdown(tags_html, unsafe_allow_html=True)

            cards = []

            for i, (_, row) in enumerate(hasil.iterrows(), start=1):
                card_html = build_recipe_card(i, row)
                st.markdown(card_html, unsafe_allow_html=True)
                cards.append(card_html)

            st.session_state.messages.append({
                "role": "assistant",
                "type": "recipes",
                "intro": intro,
                "tags_html": tags_html,
                "cards": cards
            })