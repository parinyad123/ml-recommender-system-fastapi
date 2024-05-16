# Recommender System
Recommendation systems are algorithms and techniques used in information filtering and decision making to suggest items or actions to users. They are widely employed in various domains, including e-commerce, streaming services, social media platforms, and more. The primary goal of recommendation systems is to predict the "preference" or "likeliness" of a user toward a particular item or action, based on their past behavior, preferences, and other contextual information.

Our recommender system employs Collaborative Filtering with [the MovieLens small dataset](https://github.com/lukkiddd-tdg/movielens-small) to suggest movies to users. Within the MovieLens ratings dataset, users have provided ratings for various movies. Our objective is to forecast ratings for movies that a user hasn't viewed yet. By predicting these ratings, we can identify movies with the highest anticipated ratings, subsequently recommending them to the user.


# Model Development
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parinyad123/ml-recommender-system-fastapi/blob/main/Develop_Model.ipynb)
Our recommender system model was developed in **Colab** using the popular *Keras* library. The training data comprises encoded `userId` and `movieId`, with normalized `ratings` as the target, all sourced from the [ratings.csv](https://raw.githubusercontent.com/lukkiddd-tdg/movielens-small/main/ratings.csv) file of the MovieLens small datasets.

After training the model, it was saved with the .keras extension as *recommender_system_model.keras*. For predictions, we input the `userId` of the user and the `movieId` of movies the user has not yet watched. The model predicts the `ratings` for each movie, then the system sorts these predicted ratings from highest to lowest, displaying the top 10 movies with the highest ratings as recommendations.

# Movie Recommendation System API

 Movie recommendation system API is built using FastAPI, a modern, fast (high-performance) web framework for building APIs with Python based on standard Python.

 How to Use

 #### 1. Clone the Project

 Use the command `git clone https://github.com/parinyad123/ml-recommender-system-fastapi.git` to clone the project.

 #### 2. Install Libraries

 Use the command `pip install -r requirements.txt` to install the required libraries.

 #### 3. Run the Server

`fastapi dev main.py` Run the server in development mode.
or
`uvicorn main:app --reload` : Run the server in normal mode.

#### 4. GET requests can be sent as follows


- `http://127.0.0.1:8000/recommendations/<user id>` *e.g.* `http://127.0.0.1:8000/recommendations/19`

- `127.0.0.1:8000/recommendations/<user id>?returnMetadata=true` *e.g.* `127.0.0.1:8000/recommendations/19?returnMetadata=true`

- `http://127.0.0.1:8000/feature/<user id>` : *e.g.* `http://127.0.0.1:8000/feature/19`

# Model Deploy
I deployed the model and *FastAPI* application on an *AWS EC2 instance*, making it publicly accessible using *Nginx* as a reverse proxy. The FastAPI application runs on localhost:8000 and can be accessed via the public IP address of the AWS EC2 instance: http://13.229.211.246.

##### GET requests can be sent as follows:

- `http://13.229.211.246/recommendations/<user id>` : *e.g.* http://13.229.211.246/recommendations/19
    ![Reference Image recommendation user](/img/recom_user.png)

- `http://13.229.211.246/recommendations/<user id>?returnMetadata=true` : *e.g.* http://13.229.211.246/recommendations/19?returnMetadata=true

    ![Reference Image recommendation user metadata](/img/recom_meta.png)

- `http://13.229.211.246/feature/19` : *e.g.* http://13.229.211.246/feature/19
    ![Reference Image feature user](/img/feature.png)

# Improve in the Future

To improve the RecommenderNet model, we can consider several approaches:

```python
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
```

#### 1. Increase Model Capacity
We can experiment with increasing the size of the embeddings or adding more layers to the model. Increasing the capacity can help the model learn more complex patterns in the data.

#### 2. Regularization 
While the model already includes L2 regularization on the embeddings, we might experiment with other regularization techniques such as dropout or L1 regularization. These techniques can help prevent overfitting and improve generalization.

#### 3. Learning Rate Tuning
Experiment with different learning rates for the optimizer. Learning rate tuning can significantly impact the training process and the final performance of the model. 

#### 4. Hyperparameter Tuning
Perform systematic hyperparameter tuning using techniques like grid search, random search, or Bayesian optimization to find the optimal combination of hyperparameters for your model.

#### 5.Increase Feature
Increasing both the timing and genres of movies, the model can provide timely and relevant recommendations tailored to users' current interests, thereby improving the overall quality of recommendations.

