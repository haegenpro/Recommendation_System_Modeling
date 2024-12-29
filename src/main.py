from recommendation import load_data, build_recommendation_system, recommend_movies_with_watched
from evaluation import calculate_rmse_mae
from scipy.sparse import csr_matrix
import numpy as np

if __name__ == '__main__':
    # Load data
    train_data, test_data, movies, low_activity_users, medium_activity_users, high_activity_users = load_data()

    # Build recommendation system using training data
    predicted_ratings_df = build_recommendation_system(train_data)

    # Prepare test user-item matrix
    test_user_item_matrix = csr_matrix(
        (test_data['rating'], (test_data['userId'], test_data['movieId'])),
        shape=(predicted_ratings_df.shape[0], predicted_ratings_df.shape[1])
    ).toarray()

    # Cluster movies based on rating counts
    movie_rating_counts = train_data['movieId'].value_counts()

    # Define clusters
    top_10_percent = movie_rating_counts.quantile(0.9)
    moderate_40_percent = movie_rating_counts.quantile(0.5)

    popular_movies = movie_rating_counts[movie_rating_counts >= top_10_percent].index
    moderate_movies = movie_rating_counts[(movie_rating_counts < top_10_percent) & (movie_rating_counts >= moderate_40_percent)].index
    niche_movies = movie_rating_counts[movie_rating_counts < moderate_40_percent].index

    # Function to filter ratings by movie clusters
    def filter_movies_by_cluster(cluster_movies, test_data):
        mask = test_data['movieId'].isin(cluster_movies)
        return test_data[mask]

    # Calculate RMSE and MAE for each movie cluster
    for cluster_name, cluster in [('Popular Movies', popular_movies),
                                  ('Moderate Movies', moderate_movies),
                                  ('Niche Movies', niche_movies)]:
        filtered_test_data = filter_movies_by_cluster(cluster, test_data)

        if filtered_test_data.empty:
            print(f"No test data for {cluster_name}. Skipping.")
            continue

        filtered_test_matrix = csr_matrix(
            (filtered_test_data['rating'], (filtered_test_data['userId'], filtered_test_data['movieId'])),
            shape=(predicted_ratings_df.shape[0], predicted_ratings_df.shape[1])
        ).toarray()

        # Calculate RMSE and MAE
        rmse, mae = calculate_rmse_mae(filtered_test_matrix, predicted_ratings_df.to_numpy())
        print(f"{cluster_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

