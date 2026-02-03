import streamlit as st
import pickle
from pathlib import Path
import pandas as pd
from PIL import Image
from typing import List, Dict, Any

# *** CONFIGURATION ***
st.set_page_config(page_title="StellarMatch - NADA Recommender", layout="wide")
IMAGES_PATH = Path("apod_data/APOC")

# *** DATA PERSISTENCE ***
@st.cache_resource
def load_assets(file_path: str = "stellarmatch_model.pkl") -> Dict[str, Any]:
    """
    Load the serialized model components from the pickle file
    :param file_path:
    :return:
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Initialize assets
try:
    assets = load_assets()
    df = assets["dataframe"]
    cosine_sim = assets["cosine_sim"]
    indices = assets["indices"]
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'stellarmatch_model.pkl' exits.")
    st.stop()

# *** NLP ANALYTICS ***
def get_top_words(text_series: pd.Series, n: int = 10) -> pd.Series:
    """
    Analyzes frequency of words in a text series to identify dominant themes
    :param text_series:
    :param n:
    :return:
    """
    words = ' '.join(text_series.dropna().astype(str)).lower().split()

    filtered_words = [w for w in words if len(w) > 3]
    return pd.Series(filtered_words).value_counts().head(n)

# *** CORE LOGIC ***
def get_recommendations(title: str, top_n: int = 5) -> List[int]:
    """
    Retrieve the indices of the most similar pictures based on cosine similarity
    :param title: exact title of the astronomical picture object.
    :param top_n: number of recommendations to return
    :return: list of integer indices representing the recommended rows.
    """
    idx = indices[title]

    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    return [i[0] for i in sim_scores[1 : top_n + 1]]

# *** UI COMPONENTS ***
def render_sidebar(recommendation_df: pd.DataFrame = None) -> None:
    """
    Renders the sidebar containing project metadata and NLP visualization
    :param recommendation_df:
    :return:
    """
    with st.sidebar:
        st.image(image="https://www.nasa.gov/wp-content/themes/nasa/assets/images/nasa-logo.svg", width=80)
        st.title("Project Insights")

        # Global Dataset Metrics
        st.metric(label="Library size", value=f'{len(df)} images')

        st.divider()

        # Visualization 1: Global Keyword Frequency
        st.subheader("Global Catalog Themes")
        global_words = get_top_words(df['clean_metadata'])
        st.bar_chart(global_words)
        st.caption("Most frequent terms across the entire NASA dataset.")

        # Visualization 2: Dynamic Recommendation Keywords
        if recommendation_df is not None:
            st.divider()
            st.subheader("Current Match Keywords")
            rec_words = get_top_words(recommendation_df['clean_metadata'])
            st.bar_chart(rec_words, color="#FF4B4B")
            st.caption("Dominant terms within your 5 recommendations (Semantic Context)")


def display_image_card(row: pd.Series, is_original: bool = False) -> None:
    """
    Render an individual image card with the metadata
    :param row:
    :param is_original:
    """
    img_path = IMAGES_PATH / (str(row["Filename"]) + ".jpg")

    if img_path.exists():
        img = Image.open(img_path)
        st.image(img, use_container_width=True)
        if not is_original:
            with st.expander('View Details'):
                # st.write(f"**Date:** {row.get(key='Filename', default='N/A')}")
                st.write(f"**Date:** 20{str(row['Filename'])[0:2]}-{str(row['Filename'])[2:4]}-{str(row['Filename'])[4:6]}")
                st.caption(row["explanation"])
    else:
        st.warning(f"Media asset missing: {row['Filename']}")

# *** MAIN INTERFACE ***
def main():
    st.title("🌌 StellarMatch")
    st.markdown("### Content-Based Discovery for NASA's APOD Dataset")

    selected_title = st.selectbox(
        label= "Search for an astronomical object:",
        options=df['Title'].unique()
    )

    if st.button(label="Generate Recommendations", type="primary"):
        rec_indices = get_recommendations(selected_title)
        recs_df = df.iloc[rec_indices]

        render_sidebar(recs_df)

        st.divider()

        col_original, _ = st.columns([1, 2])
        with col_original:
            st.subheader("Your selection")
            original_row = df[df['Title'] == selected_title].iloc[0]
            display_image_card(original_row, is_original=True)

        st.divider()
        st.subheader("Similar wonders you might like")

        # Recommendations grid
        cols = st.columns(5)
        for i, idx in enumerate(rec_indices):
            with cols[i]:
                row = df.iloc[idx]
                st.write(f"**{row['Title']}**")
                display_image_card(row, is_original=False)

if __name__ == "__main__":
    main()