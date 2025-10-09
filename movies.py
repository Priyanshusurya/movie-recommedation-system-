import requests
import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from youtubesearchpython import VideosSearch  # ✅ For trailer links


# Load movies dataset
movies = pd.read_csv("new_df.csv")

# Create feature vectors
cv = CountVectorizer(max_features=5000, stop_words='english')

# Load movies dataset

vectors = cv.fit_transform(movies['tags']).toarray()


# Compute similarity
similarity = cosine_similarity(vectors)

# Function to fetch poster and rating using OMDb API
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

# ✅ New function: fetch YouTube trailer link
def fetch_trailer_link(movie_title):
    try:
        search_query = f"{movie_title} official trailer"
        videos_search = VideosSearch(search_query, limit=1)
        result = videos_search.result()
        if result["result"]:
            return result["result"][0]["link"]
        else:
            return "https://www.youtube.com"  # fallback
    except Exception as e:
        return "https://www.youtube.com"

# Save and reload similarity matrix
with open("movies.pkl", "wb") as f:
    pickle.dump(similarity, f)
similarity = pickle.load(open('movies.pkl', 'rb'))

# Recommendation function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distance = similarity[movie_index]
    movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    recommended_movies_posters = []
    recommended_movies_ratings = []
    recommended_movies_trailers = []
    
    for i in movie_list:
        title = movies.iloc[i[0]].title
        recommended_movies.append(title)
        poster, rating = fetch_poster_and_rating(title)
        trailer = fetch_trailer_link(title)
        recommended_movies_posters.append(poster)
        recommended_movies_ratings.append(rating)
        recommended_movies_trailers.append(trailer)
    
    return recommended_movies, recommended_movies_posters, recommended_movies_ratings, recommended_movies_trailers

# Streamlit UI
st.title("🎬 Movie Recommendation System with Trailers")

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movies['title'].values
)

if st.button('Show Recommendation'):
    names, posters, ratings, trailers = recommend(selected_movie)
    
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.text(f"{names[i]} \n⭐ {ratings[i]}")
            st.image(posters[i])
            st.markdown(f"[🎥 Watch Trailer]({trailers[i]})", unsafe_allow_html=True)

vectors = cv.fit_transform(movies['tags']).toarray()

# Compute similarity
similarity = cosine_similarity(vectors)

# Function to fetch poster and rating using OMDb API
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

# Load dataset
movies = pd.read_csv("new_df.csv")
with open("movies.pkl", "wb") as f:
    pickle.dump(similarity, f)

# Load precomputed similarity matrix
similarity = pickle.load(open('movies.pkl', 'rb'))

# Recommendation function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
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

# Streamlit UI
st.title("🎬 Movie Recommendation System ")

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movies['title'].values
)

if st.button('Show Recommendation'):
    names, posters, ratings = recommend(selected_movie)
    
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.text(f"{names[i]} \n⭐ {ratings[i]}")
            st.image(posters[i])







