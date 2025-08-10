import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import joblib

print("Loading and preprocessing data...")

# Load data
df_books = pd.read_csv(
    'BX-Books.csv',
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    'BX-Book-Ratings.csv',
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

print("Filtering data...")
# Filter users with < 200 ratings and books with < 100 ratings
user_counts = df_ratings['user'].value_counts()
book_counts = df_ratings['isbn'].value_counts()

df_ratings = df_ratings[df_ratings['user'].isin(user_counts[user_counts >= 200].index)]
df_ratings = df_ratings[df_ratings['isbn'].isin(book_counts[book_counts >= 100].index)]

print("Creating pivot table...")
df = pd.merge(df_ratings, df_books, on='isbn')
book_pivot = df.pivot_table(index='title', columns='user', values='rating').fillna(0)
book_sparse = csr_matrix(book_pivot.values)

print("Training model...")
model = NearestNeighbors(algorithm='brute', metric='cosine')
model.fit(book_sparse)

print("Saving model...")
joblib.dump(model, 'model.h5')
joblib.dump(book_pivot, 'book_pivot.pkl')

print("Model trained and saved successfully!")