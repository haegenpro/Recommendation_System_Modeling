import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import numpy as np

# Load data with train-test split
def load_data():
    ratings = pd.read_csv('Recommendation_System_Modeling/test/ratings.csv')
    movies = pd.read_csv('Recommendation_System_Modeling/test/movies.csv')

    top_users = ratings['userId'].value_counts().nlargest(25000).index
    top_movies = ratings['movieId'].value_counts().nlargest(5000).index
    ratings = ratings[(ratings['userId'].isin(top_users)) & (ratings['movieId'].isin(top_movies))]

    # Map userId and movieId to sequential indices
    ratings['userId'] = ratings['userId'].astype('category').cat.codes
    ratings['movieId'] = ratings['movieId'].astype('category').cat.codes

    user_activity = ratings['userId'].value_counts()
    low_activity_users = user_activity[user_activity <= 200].index
    medium_activity_users = user_activity[(user_activity > 200) & (user_activity <= 500)].index
    high_activity_users = user_activity[user_activity > 500].index

    # Split data into train and test sets (80/20 split)
    test_fraction = 0.2
    test_indices = np.random.choice(ratings.index, size=int(len(ratings) * test_fraction), replace=False)
    test_data = ratings.loc[test_indices]
    train_data = ratings.drop(test_indices)

    return train_data, test_data, movies, low_activity_users, medium_activity_users, high_activity_users

# Create the recommendation system
def build_recommendation_system(ratings):
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

    return predicted_ratings_df

# Recommend movies for a user
def recommend_movies_with_watched(user_id, predicted_ratings_df, ratings, movies, num_recommendations=10):
    # Check if the user exists
    if user_id not in predicted_ratings_df.index:
        return f"User {user_id} not found in the dataset.", []

    # Display watched movies
    watched_movies_ids = ratings[ratings['userId'] == user_id]['movieId'].unique()
    watched_movies = movies[movies['movieId'].isin(watched_movies_ids)][['movieId', 'title']]

    # Generate recommendations
    user_ratings = predicted_ratings_df.loc[user_id]
    recommendations_ids = user_ratings[~user_ratings.index.isin(watched_movies_ids)].sort_values(ascending=False).head(num_recommendations).index
    recommendations = movies[movies['movieId'].isin(recommendations_ids)][['movieId', 'title']]
    
    return watched_movies, recommendations

