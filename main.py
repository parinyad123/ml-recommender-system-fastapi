import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import pandas as pd

from typing import Union
from fastapi import FastAPI
from typing import Optional 

# EMBEDDING_SIZE = 50
model_file = 'recommander_system_model.keras'
movies_csv = "https://raw.githubusercontent.com/lukkiddd-tdg/movielens-small/main/movies.csv"
ratings_csv = "https://raw.githubusercontent.com/lukkiddd-tdg/movielens-small/main/ratings.csv"

# @keras.saving.register_keras_serializable(package="MyLayers")
class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        # Add all the components (including bias)
        x = dot_user_movie + user_bias + movie_bias
        # The sigmoid activation forces the rating to be between 0 and 11
        return tf.nn.sigmoid(x)


reconstructed_model = keras.models.load_model(
    model_file,
    custom_objects={"RecommenderNet": RecommenderNet},
)

# print(reconstructed_model.summary())


ratings_df = pd.read_csv(ratings_csv)

# Drop Duplicated values
ratings_df = ratings_df.drop_duplicates()

# remove rating >5 and <0
ratings_df = ratings_df[(ratings_df['rating']<=5) & (ratings_df['rating']>0)]

# Drop Nan values
ratings_df = ratings_df.dropna()

# convert type of userId and  movieId from float to int
ratings_df['userId'] = ratings_df['userId'].astype(int)
ratings_df['movieId'] = ratings_df['movieId'].astype(int)

ratings_df["rating"] = ratings_df["rating"].values.astype(np.float32)

ratings_df = ratings_df.reset_index(drop=True)

user_ids = ratings_df["userId"].unique().tolist()
# print("User_ids", user_ids)
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded2user = {i: x for i, x in enumerate(user_ids)}

movie_ids = ratings_df["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}


ratings_df["user"] = ratings_df["userId"].map(user2user_encoded)
ratings_df["movie"] = ratings_df["movieId"].map(movie2movie_encoded)

movie_df = pd.read_csv(movies_csv)

def str_to_bool(s="false"):
    truthy_values = {"true", "1", "yes"}
    # falsy_values = {"false", "0", "no"}
    s_lower = s.lower()
    if s_lower in truthy_values:
        return True
    else:
        return False

app = FastAPI()

@app.get("/")
def read_root():
    return {"Poject": "ML recommender System FastAPI"}

# @app.get("/feature/")
@app.get("/feature/{user_id}")
async def read_histories(user_id: int):
    movies_watched_by_user = ratings_df[ratings_df.userId == user_id]
    return {"feature": [
        {
            "histories": sorted(list(movies_watched_by_user.movieId), reverse=True)
        }
    ]}


@app.get("/recommendations/{user_id}")
# async def read_item(user_id: int = 0, returnMetadata: Optional[str] = "false"):
async def read_item(user_id: int, returnMetadata: Union[str, None] = None):
    if returnMetadata == None:
        returnMetadata = 'false'

    # print("returnMetadata = ", type(returnMetadata), returnMetadata)

    returnMetadata = str_to_bool(returnMetadata)
    # print("returnMetadata = ", type(returnMetadata), returnMetadata)

    n_recommended_movie = 10
    movies_watched_by_user = ratings_df[ratings_df.userId == user_id]
    movies_not_watched = movie_df[~movie_df['movieId'].isin(movies_watched_by_user.movieId.values)]['movieId']

    movies_not_watched = list(set(movies_not_watched).intersection(set(movie2movie_encoded.keys())))

    movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]

    user_encoder = user2user_encoded.get(user_id)

    user_movie_array = np.hstack(
        ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
    )

    ratings = reconstructed_model.predict(user_movie_array).flatten()
    top_ratings_indices = ratings.argsort()[-n_recommended_movie:][::-1]
    recommended_movie_ids = [
        movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
    ]

    # print("returnMetadata = ", returnMetadata)

    if returnMetadata:
        recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
        return {
            "items": [{"id": x , 
            "title": recommended_movies[recommended_movies.movieId == x].title.values[0],
            "genres":recommended_movies[recommended_movies.movieId == x].genres.values[0].split('|')
            } for x in recommended_movie_ids]
        }
    else:
        return {
            "items": [
                {"id": str(item)} for item in recommended_movie_ids
            ]
        }
        
        