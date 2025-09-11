import streamlit as st 
import pickle
import pandas as pd 
def fetch_poster(movie):
    url = ""
def recommend(movie):
    movie_index = movies[movies['title']==movie].index[0]
    distance = similarity[movie_index]
    movie_list = sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]
    recommended_movies = []
    recommended_movies_poster = []

    for i in movie_list :
        recommended_movies_poster.append(fetch_poster(movie_list))
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies

movies_dict = pickle.load(open('movie_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl','rb'))


st.title("Movies Recommmender System")
option = st.selectbox('How would you like to be contacted ?',movies['title'].values)

if st.button('Recommend'):
    recommendations = recommend(option)
    for i in recommendations:
        st.write(i)