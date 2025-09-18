import requests
import streamlit as st
import pickle
import pandas as pd

# Function to fetch poster using OMDb API
def fetch_poster(movie_title):
    api_key = "f3d4e762"
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    if data.get("Poster") and data["Poster"] != "N/A":
        return data["Poster"]
    else:
        return "https://via.placeholder.com/300x450.png?text=No+Image"  # fallback image

# Load movie data
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Recommendation function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distance = similarity[movie_index]
    movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    recommended_movies_poster = []
    
    for i in movie_list:
        title = movies.iloc[i[0]].title
        recommended_movies.append(title)
        recommended_movies_poster.append(fetch_poster(title))
    
    return recommended_movies, recommended_movies_poster

# Streamlit UI
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movies['title'].values
)

if st.button('Show Recommendation'):
    names, posters = recommend(selected_movie)
    
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.text(names[i])
            st.image(posters[i])
