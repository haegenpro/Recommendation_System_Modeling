import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

def calculate_rmse_mae(actual, predicted):
    """
    Calculates RMSE and MAE given actual and predicted ratings.
    """
    # Filter for non-zero entries (only consider rated items)
    mask = actual.nonzero()
    actual_values = actual[mask]
    predicted_values = predicted[mask]

    # Calculate metrics
    rmse = sqrt(mean_squared_error(actual_values, predicted_values))
    mae = mean_absolute_error(actual_values, predicted_values)

    return rmse, mae

def precision_recall_f1_at_k(predicted_ratings, actual_ratings, k):
    """
    Calculates Precision@K, Recall@K, and F1@K.
    """
    precision_scores = []
    recall_scores = []

    for user_idx in range(len(actual_ratings)):
        # Get top-k recommendations for the user
        top_k_predicted = np.argsort(predicted_ratings[user_idx])[-k:]
        relevant_items = np.where(actual_ratings[user_idx] > 0)[0]  # Items the user rated

        # Calculate relevant items in top-k
        recommended_and_relevant = len(set(top_k_predicted) & set(relevant_items))
        precision = recommended_and_relevant / k
        recall = recommended_and_relevant / len(relevant_items) if len(relevant_items) > 0 else 0

        precision_scores.append(precision)
        recall_scores.append(recall)

    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    return avg_precision, avg_recall, f1_score

def ndcg_at_k(predicted_ratings, actual_ratings, k):
    """
    Calculates nDCG@K.
    """
    ndcg_scores = []

    for user_idx in range(len(actual_ratings)):
        top_k_predicted = np.argsort(predicted_ratings[user_idx])[-k:][::-1]
        actual_relevance = actual_ratings[user_idx][top_k_predicted]

        # Calculate DCG
        dcg = np.sum((2 ** actual_relevance - 1) / np.log2(np.arange(1, k + 1) + 1))

        # Calculate IDCG
        sorted_actual_relevance = np.sort(actual_ratings[user_idx])[::-1][:k]
        idcg = np.sum((2 ** sorted_actual_relevance - 1) / np.log2(np.arange(1, k + 1) + 1))

        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores)

def mean_reciprocal_rank(predicted_ratings, actual_ratings):
    """
    Calculates MRR.
    """
    reciprocal_ranks = []

    for user_idx in range(len(actual_ratings)):
        sorted_indices = np.argsort(predicted_ratings[user_idx])[::-1]
        relevant_items = np.where(actual_ratings[user_idx] > 0)[0]

        for rank, item_idx in enumerate(sorted_indices, start=1):
            if item_idx in relevant_items:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)

    return np.mean(reciprocal_ranks)
