import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
movies = pd.read_csv("Recommendation_System_Modeling/test/movies.csv")
ratings = pd.read_csv("Recommendation_System_Modeling/test/ratings.csv")

top_users = ratings['userId'].value_counts().nlargest(10000).index
top_movies = ratings['movieId'].value_counts().nlargest(2000).index
ratings = ratings[(ratings['userId'].isin(top_users)) & (ratings['movieId'].isin(top_movies))]

# Plot 1: Histogram of Ratings Distribution
plt.figure(figsize=(8, 6))
plt.hist(ratings['rating'], bins=5, edgecolor='black', alpha=0.7)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

import matplotlib.pyplot as plt

# Number of ratings per user
user_interactions = ratings['userId'].value_counts()

plt.figure(figsize=(8, 6))
plt.hist(user_interactions, bins=50, edgecolor='black')
plt.title("Distribution of Ratings per User")
plt.xlabel("Number of Ratings")
plt.ylabel("Number of Users")
plt.grid(axis='y')
plt.show()


