# Recommendation System Modeling

This repository demonstrates a Simple Recommender System implemented in Python, focusing on a single approach: **Singular Value Decomposition (SVD)**.

### Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Data](#data)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Evaluation](#evaluation)
7. [Project Structure](#project-structure)
8. [License](#license)

---

## Overview
The goal of this project is to build and evaluate a recommendation system that leverages SVD for rating prediction. The core elements include:
- Resorting to SVD to factorize a user-item rating matrix.
- Generating item recommendations for users based on reconstructed predictions from the factorized matrices.
- Analyzing results via RMSE, MAE, and other ranking metrics like nDCG, MRR, and Precision/Recall@k.

---

### Features
- **Loading and Preprocessing:** Handles reading CSV data for ratings and movies, maps IDs to consecutive indices, and performs train-test splits.
- **SVD-based Prediction:** Evaluates user-item ratings with a specified low-rank approximation (default: k=20).
- **Recommendation Generation:** Provides recommendations for a specific user while also listing the user’s watched items.
- **Evaluation Metrics:** 
  - RMSE and MAE to compare reconstructed ratings with actual ratings.
  - Precision, Recall, F1 at Top-K, nDCG@K, and MRR for ranking-based performance.

---

### Data
The datasets used here include:
1. **movies.csv** and **ratings.csv** – Basic files containing movie metadata and user ratings. This dataset can be found in https://grouplens.org/datasets/movielens/.
2. The data is filtered to manage large scales, limiting the top users and movies. The filtering method implemented here is the selection of the top 20000 users and top 5000 movies, for time efficiency purposes.
---

### Installation & How to Run
1. Clone this repository or download the ZIP.
2. Install the required Python packages:
   Pandas, Matplotlib, Seaborn, Numpy, Scikit-learn, Scipy.
3. Run the main.py file using the current parameters implemented, or change the parameters before running to acquire a different result.

### Contributors
Developer: Haegen Quinston

### License
This repository is licensed under the rules of the MIT License.