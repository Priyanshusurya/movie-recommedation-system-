import streamlit as st
import pickle
import pandas as pd

# Load movie dictionary and similarity matrix
with open('movie_dict.pkl', 'rb') as f:
    movie_dict = pickle.load(f)

with open('similarity.pkl', 'rb') as f:
    similarity = pickle.load(f)

# Load movies CSV
movies = pd.read_csv('movies.csv')

def recommend(movie):
    index = movie_dict[movie]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

st.title("Movie Recommender System")
selected_movie = st.selectbox("Choose a movie:", movies['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    for rec in recommendations:
        st.write(rec)
