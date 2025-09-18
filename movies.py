import requests
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Function to fetch poster and rating using OMDb API
# -------------------------------
def fetch_poster_and_rating(movie_title):
    api_key = "f3d4e762"  # your OMDb API key
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()

    # Poster
    poster = data.get("Poster") if data.get("Poster") and data["Poster"] != "N/A" else "https://via.placeholder.com/300x450.png?text=No+Image"

    # Rating
    rating = data.get("imdbRating") if data.get("imdbRating") and data["imdbRating"] != "N/A" else "No Rating"

    return poster, rating


# -------------------------------
# Load dataset
# -------------------------------
movies = pd.read_csv("tmdb_5000_movies.csv")

# -------------------------------
# Build similarity matrix (instead of loading pickle)
# -------------------------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['overview'].fillna(''))
similarity = cosine_similarity(vectors)


# -------------------------------
# Recommendation function
# -------------------------------
def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
    except IndexError:
        return [], [], []

    distance = similarity[movie_index]
    movie_list = sorted(
        list(enumerate(distance)), reverse=True, key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_movies_posters = []
    recommended_movies_ratings = []

    for i in movie_list:
        title = movies.iloc[i[0]].title
        recommended_movies.append(title)
        poster, rating = fetch_poster_and_rating(title)
        recommended_movies_posters.append(poster)
        recommended_movies_ratings.append(rating)

    return recommended_movies, recommended_movies_posters, recommended_movies_ratings


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movies['title'].values
)

if st.button('Show Recommendation'):
    names, posters, ratings = recommend(selected_movie)

    if not names:
        st.error("Movie not found in dataset.")
    else:
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                st.text(f"{names[i]} \n⭐ {ratings[i]}")
                st.image(posters[i])
