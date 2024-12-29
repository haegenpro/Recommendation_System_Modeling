import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load data
ratings = pd.read_csv('Recommendation_System_Modeling/test/ratings.csv')
movies = pd.read_csv('Recommendation_System_Modeling/test/movies.csv')

# Limit to top 10,000 active users and top 5,000 popular movies
top_users = ratings['userId'].value_counts().head(10000).index
top_movies = ratings['movieId'].value_counts().head(2000).index

ratings = ratings[ratings['userId'].isin(top_users) & ratings['movieId'].isin(top_movies)]

# Map userId and movieId to sequential indices
ratings['userId'] = ratings['userId'].astype('category').cat.codes
ratings['movieId'] = ratings['movieId'].astype('category').cat.codes

# Create a sparse user-item matrix
user_item_matrix_sparse = csr_matrix(
    (ratings['rating'], (ratings['userId'], ratings['movieId']))
)

# SVD
U, sigma, Vt = svds(user_item_matrix_sparse, k=20)
sigma = np.diag(sigma)

# Predicted ratings
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(
    predicted_ratings, 
    index=np.arange(user_item_matrix_sparse.shape[0]), 
    columns=np.arange(user_item_matrix_sparse.shape[1])
)

# RMSE Evaluation
actual_ratings = user_item_matrix_sparse.toarray().flatten()
predicted_ratings_flat = predicted_ratings.flatten()
mask = actual_ratings != 0
rmse = sqrt(mean_squared_error(actual_ratings[mask], predicted_ratings_flat[mask]))
print(f"RMSE: {rmse}")

# Recommend movies
def recommend_movies(user_id, num_recommendations=10):
    if user_id not in predicted_ratings_df.index:
        return f"User {user_id} not found in the dataset."
    
    user_ratings = predicted_ratings_df.loc[user_id]
    rated_movies = ratings[ratings['userId'] == user_id]['movieId']
    recommendations = user_ratings[~user_ratings.index.isin(rated_movies)].sort_values(ascending=False).head(num_recommendations)
    
    recommended_movies = movies[movies['movieId'].isin(recommendations.index)]
    return recommended_movies[['movieId', 'title', 'genres']].merge(recommendations.rename('predicted_rating'), left_on='movieId', right_index=True)

if __name__ == '__main__':
    print(recommend_movies(1))
