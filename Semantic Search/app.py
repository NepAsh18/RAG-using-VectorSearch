import streamlit as st
from embedding_utils import semantic_search

# Streamlit UI setup
st.set_page_config(page_title="🎬 Movie Semantic Search", layout="wide")

st.title("🎬 Semantic Movie Search (Free Hugging Face + MongoDB Atlas)")
st.markdown(
    "Search movies by *meaning*, not just keywords! "
    "Powered by Hugging Face embeddings + MongoDB vector search."
)

query = st.text_input("Enter your search query:", placeholder="e.g. imaginary characters from outer space at war")

if st.button("🔍 Search"):
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Searching movies..."):
            try:
                results = semantic_search(query)
                if not results:
                    st.info("No matching movies found.")
                else:
                    for doc in results:
                        st.subheader(doc.get("title", "Unknown Title"))
                        st.write(doc.get("plot", "No plot available."))
                        st.divider()
            except Exception as e:
                st.error(f"Error: {e}")
