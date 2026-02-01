import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("books_cleaned.csv")  # <-- change to your file name

df = load_data()

# -----------------------------
# Combine Features
# -----------------------------
df['combined'] = (
    df['title'].fillna('') + " " +
    df['authors'].fillna('') + " " +
    df['content'].fillna('')
)

# -----------------------------
# Vectorize
# -----------------------------
@st.cache_resource
def build_model(text_data):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000
    )
    tfidf_matrix = vectorizer.fit_transform(text_data)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

similarity_matrix = build_model(df['combined'])

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_books(book_title, top_n=5):
    matches = df[df['title'] == book_title]

    if matches.empty:
        return None

    idx = matches.index[0]

    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_books = sim_scores[1:top_n+1]
    recs = df.iloc[[i[0] for i in top_books]][
        ['title', 'authors', 'average_rating', 'isbn']
    ].copy()

    recs['Buy Link'] = recs['isbn'].apply(
        lambda x: f"https://www.amazon.in/dp/{str(x)}"
    )

    recs['Reviews'] = recs['isbn'].apply(
        lambda x: f"https://www.goodreads.com/search?q={str(x)}"
    )

    return recs


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Book Recommender", layout="centered")

st.title("📚 Smart Book Recommendation System")
st.write("Type a book title and get similar book recommendations")

book_input = st.selectbox(
    "Search and select a book:",
    options=sorted(df['title'].dropna().unique())
)


if st.button("Recommend"):
    results = recommend_books(book_input)
    
    if results is None:
        st.warning("No matching book found. Try another title!")
    else:
        st.success("Here are your recommendations:")
        for _, row in results.iterrows():
            st.markdown(f"### 📘 {row['title']}")
            st.write(f"**Author:** {row['authors']}")
            st.write(f"⭐ Rating: {row['average_rating']}")
            st.markdown(f"🛒 [Buy on Amazon]({row['Buy Link']})")
            st.markdown(f"📝 [Read Reviews]({row['Reviews']})")
            st.divider()

