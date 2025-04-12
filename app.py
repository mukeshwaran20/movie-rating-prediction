import streamlit as st
import pandas as pd
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity

# Load data and model
movie_dict = pickle.load(open('movie_dict.pkl', 'rb'))
new_df = pd.DataFrame(movie_dict)

model = pickle.load(open('rating_model.pkl', 'rb'))
vectors = pickle.load(open('vectorized_tags.pkl', 'rb'))

st.title("🎬 Movie Rating Predictor")
st.write("Predict movie ratings using machine learning regression on movie tags.")

# Movie selection
selected_movie = st.selectbox("Choose a movie", new_df['title'].values)

# Function to fetch poster from TMDB
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=27ce57411bffad61ee03407b5c1939dd&language=en-US"
        response = requests.get(url)
        data = response.json()
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except:
        return "https://via.placeholder.com/150x225.png?text=No+Image"

if st.button("Predict Rating"):
    try:
        index = new_df[new_df['title'] == selected_movie].index[0]
        movie_vector = vectors[index]
        predicted_rating = model.predict(movie_vector.reshape(1, -1))[0]

        st.success(f"⭐ Predicted Rating for '{selected_movie}': **{predicted_rating:.2f}/10**")

        # Show actual TMDB rating from dataset
        actual_vote = new_df.loc[index, 'vote_average']
        st.info(f"🎯 Actual TMDB Rating: **{actual_vote}/10**")

        # Show poster of selected movie
        movie_id = new_df.iloc[index]['movie_id']
        poster_url = fetch_poster(movie_id)
        st.image(poster_url, width=300, caption=f"{selected_movie} Poster")

        # Find similar movies based on cosine similarity
        similarity = cosine_similarity(vectors)
        similar_movies = sorted(list(enumerate(similarity[index])), key=lambda x: x[1], reverse=True)[1:6]

        # Show top 5 similar movies with posters
        st.subheader("🎥 Top 5 Similar Movies")
        names = []
        posters = []

        for sim in similar_movies:
            try:
                movie_idx = sim[0]
                similar_title = new_df.iloc[movie_idx]['title']
                similar_id = new_df.iloc[movie_idx]['movie_id']
                similar_poster = fetch_poster(similar_id)
            except:
                similar_title = "N/A"
                similar_poster = "https://via.placeholder.com/150x225.png?text=No+Image"

            names.append(similar_title)
            posters.append(similar_poster)

        # Display in horizontal layout
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.text(names[0])
            st.image(posters[0], use_container_width=True)

        with col2:
            st.text(names[1])
            st.image(posters[1], use_container_width=True)

        with col3:
            st.text(names[2])
            st.image(posters[2], use_container_width=True)

        with col4:
            st.text(names[3])
            st.image(posters[3], use_container_width=True)

        with col5:
            st.text(names[4])
            st.image(posters[4], use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
