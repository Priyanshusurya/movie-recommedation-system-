import requests
import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse

# Load movies dataset
movies = pd.read_csv("new_df.csv")

# Create feature vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Compute similarity
similarity = cosine_similarity(vectors)

# Function to fetch poster, rating, IMDb ID using OMDb API
def fetch_movie_details(movie_title):
    api_key = "f3d4e762"  # your OMDb API key
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()

    # Poster
    poster = data.get("Poster") if data.get("Poster") and data["Poster"] != "N/A" else "https://via.placeholder.com/300x450.png?text=No+Image"

    # Rating
    rating = data.get("imdbRating") if data.get("imdbRating") and data["imdbRating"] != "N/A" else "No Rating"

    # IMDb link
    imdb_id = data.get("imdbID")
    imdb_link = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else "https://www.imdb.com/"

    # YouTube trailer link
    query = urllib.parse.quote(movie_title + " trailer")
    youtube_link = f"https://www.youtube.com/results?search_query={query}"

    return poster, rating, imdb_link, youtube_link


# Save and load similarity
with open("movies.pkl", "wb") as f:
    pickle.dump(similarity, f)
similarity = pickle.load(open('movies.pkl', 'rb'))


# Recommendation function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distance = similarity[movie_index]
    movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []
    recommended_ratings = []
    recommended_imdb_links = []
    recommended_youtube_links = []

    for i in movie_list:
        title = movies.iloc[i[0]].title
        poster, rating, imdb_link, youtube_link = fetch_movie_details(title)
        recommended_movies.append(title)
        recommended_posters.append(poster)
        recommended_ratings.append(rating)
        recommended_imdb_links.append(imdb_link)
        recommended_youtube_links.append(youtube_link)

    return (
        recommended_movies,
        recommended_posters,
        recommended_ratings,
        recommended_imdb_links,
        recommended_youtube_links
    )


# Streamlit UI
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movies['title'].values
)

if st.button('Show Recommendation'):
    names, posters, ratings, imdb_links, youtube_links = recommend(selected_movie)

    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(posters[i])
            st.markdown(f"**{names[i]}**")
            st.markdown(f"⭐ IMDb Rating: {ratings[i]}")
            st.markdown(f"[🎞 Watch Trailer]({youtube_links[i]})")
            st.markdown(f"[🎬 IMDb Page]({imdb_links[i]})")

