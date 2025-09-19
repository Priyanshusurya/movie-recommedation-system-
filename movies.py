import requests
import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔹 Step 1: Fetch movies from OMDb and build dataset
def fetch_movies_from_omdb(movie_titles):
    api_key = "f3d4e762"  # your OMDb API key
    movies_data = []
    for title in movie_titles:
        url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
        response = requests.get(url).json()
        if response.get("Response") == "True":
            movies_data.append({
                "title": response["Title"],
                "tags": f"{response.get('Genre', '')} {response.get('Director', '')} {response.get('Actors', '')} {response.get('Plot', '')}"
            })
    return pd.DataFrame(movies_data)

# Example list of movies to pull from OMDb (you can add 100s here)
movie_titles = [
    "Inception", "The Dark Knight", "Interstellar",
    "Avatar", "Titanic", "Avengers: Endgame", "Iron Man",
    "The Matrix", "Joker", "Forrest Gump", "Shutter Island"
]

# Fetch and save dataset
movies = fetch_movies_from_omdb(movie_titles)
movies.to_csv("new_df.csv", index=False)

# 🔹 Step 2: Load movies dataset
movies = pd.read_csv("new_df.csv")

# Create feature vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Compute similarity
similarity = cosine_similarity(vectors)

# Save similarity matrix
with open("movies.pkl", "wb") as f:
    pickle.dump(similarity, f)

# Load similarity
similarity = pickle.load(open('movies.pkl', 'rb'))

# 🔹 Step 3: Fetch poster and rating using OMDb
def fetch_poster_and_rating(movie_title):
    api_key = "f3d4e762"  # your OMDb API key
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    
    poster = data.get("Poster") if data.get("Poster") and data["Poster"] != "N/A" else "https://via.placeholder.com/300x450.png?text=No+Image"
    rating = data.get("imdbRating") if data.get("imdbRating") and data["imdbRating"] != "N/A" else "No Rating"
    
    return poster, rating

# 🔹 Step 4: Recommendation function
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

# 🔹 Step 5: Streamlit UI
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
