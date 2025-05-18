import pandas as pd
import numpy as np
import os
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- CONFIGURATION ----------
DATA_DIR = './data'
# or 'ratings.csv' (for larger dataset) or 'ratings.csv' (for smaller dataset)
META_FILE = os.path.join(DATA_DIR, 'movies_metadata.csv')
RATINGS_FILE = os.path.join(DATA_DIR, 'ratings_small.csv')

# ---------- 1. DATA LOADING & CLEANING ----------


def load_and_clean_metadata(path):
    """
    Loads movie metadata from a CSV file, cleans and preprocesses the data.

    Parameters:
        path (str): The file path to the metadata CSV file.

    Returns:
        pandas.DataFrame: A cleaned DataFrame with:
            - Only rows with valid numeric 'id' values.
            - 'id' column converted to integers.
            - 'release_date' parsed as datetime objects (invalid dates set to NaT).
            - Missing values in 'overview' filled with empty strings.
    """
    # Use low_memory=False to avoid dtype warning
    df = pd.read_csv(path, low_memory=False)
    # Keep only valid numeric IDs
    df = df[df['id'].str.isdigit()]
    df['id'] = df['id'].astype(int)
    # Parse dates
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    # Fill missing overview
    df['overview'] = df['overview'].fillna('')
    return df


def load_ratings(path):
    """
    Loads movie ratings from a CSV file.

    Args:
        path (str): The file path to the CSV file containing ratings data.

    Returns:
        pandas.DataFrame: A DataFrame containing the loaded ratings data.
    """
    return pd.read_csv(path)

# ---------- 2. COLLABORATIVE FILTERING ----------


def prepare_user_item_matrix(ratings_df):
    """
    Creates a user-item matrix from a DataFrame of movie ratings.

    This function pivots the input DataFrame so that each row represents a user,
    each column represents a movie, and the values are the corresponding ratings.
    Missing ratings are filled with 0.

    Args:
        ratings_df (pandas.DataFrame): DataFrame containing at least 'userId', 'movieId', and 'rating' columns.

    Returns:
        pandas.DataFrame: A user-item matrix with users as rows, movies as columns, and ratings as values.
    """
    user_item_matrix = ratings_df.pivot(
        index='userId', columns='movieId', values='rating').fillna(0)
    return user_item_matrix


def train_collaborative(user_item_matrix, n_components=50, model_path='model_svd.pkl'):
    """
    Trains a collaborative filtering model using Truncated Singular Value Decomposition (SVD) on the provided user-item matrix.

    Args:
        user_item_matrix (array-like or sparse matrix): The user-item interaction matrix to decompose.
        n_components (int, optional): Number of singular values and vectors to compute. Defaults to 50.
        model_path (str, optional): File path to save the trained SVD model. Defaults to 'model_svd.pkl'.

    Returns:
        TruncatedSVD: The trained TruncatedSVD model.

    Side Effects:
        Saves the trained SVD model to the specified file path using pickle.
        Prints a message indicating where the model was saved.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(user_item_matrix)
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(svd, f)
    print(f"Saved collaborative model to {model_path}")
    return svd


def recommend_for_user_collab(svd, user_id, user_item_matrix, movies_df, n=10):
    """
    Generate movie recommendations for a user using collaborative filtering with SVD.

    Args:
        svd (sklearn.decomposition.TruncatedSVD): Trained SVD model for collaborative filtering.
        user_id (int or str): The ID of the user for whom to generate recommendations.
        user_item_matrix (pd.DataFrame): User-item matrix with users as rows and movies as columns.
        movies_df (pd.DataFrame): DataFrame containing movie metadata, including 'id', 'title', and 'release_date'.
        n (int, optional): Number of top recommendations to return. Defaults to 10.

    Returns:
        pd.DataFrame: DataFrame containing the recommended movies with their titles and release dates.

    Raises:
        ValueError: If the specified user_id is not found in the user-item matrix.
    """
    if user_id not in user_item_matrix.index:
        raise ValueError(f"User ID {user_id} not found in user-item matrix.")
    user_ratings = user_item_matrix.loc[user_id].values.reshape(1, -1)
    user_latent = svd.transform(user_ratings)
    reconstructed_ratings = np.dot(user_latent, svd.components_)
    rated_movies = user_item_matrix.columns[user_item_matrix.loc[user_id] > 0]
    recommendations = {
        movie: score for movie, score in zip(user_item_matrix.columns, reconstructed_ratings.flatten())
        if movie not in rated_movies
    }
    top_movies = sorted(recommendations.items(),
                        key=lambda x: x[1], reverse=True)[:n]
    top_movie_ids = [movie_id for movie_id, _ in top_movies]
    return movies_df[movies_df['id'].isin(top_movie_ids)][['title', 'release_date']]

# ---------- 3. CONTENT-BASED FILTERING ----------


def train_content_based(meta_df, tfidf_path='tfidf.pkl'):
    """
    Trains a content-based movie recommendation model using TF-IDF vectorization on movie overviews.

    Args:
        meta_df (pd.DataFrame): DataFrame containing movie metadata, must include an 'overview' column and an 'id' column.
        tfidf_path (str, optional): File path to save the trained TF-IDF vectorizer, matrix, and movie IDs. Defaults to 'tfidf.pkl'.

    Returns:
        tuple: A tuple containing:
            - tfidf (TfidfVectorizer): The trained TF-IDF vectorizer.
            - tfidf_matrix (scipy.sparse.csr_matrix): The TF-IDF feature matrix for the movie overviews.
            - list: List of movie IDs corresponding to the rows in the TF-IDF matrix.

    Side Effects:
        Saves the trained TF-IDF vectorizer, matrix, and movie IDs to the specified file path using pickle.
    """
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(meta_df['overview'])
    with open(tfidf_path, 'wb') as f:
        pickle.dump((tfidf, tfidf_matrix, meta_df['id'].tolist()), f)
    print(f"Saved TF-IDF matrix to {tfidf_path}")
    return tfidf, tfidf_matrix, meta_df['id'].tolist()


def recommend_by_movie(title, meta_df, tfidf, tfidf_matrix, n=10):
    """
    Recommend movies similar to a given movie title based on TF-IDF cosine similarity.

    Args:
        title (str): The title of the movie to find recommendations for.
        meta_df (pd.DataFrame): DataFrame containing movie metadata, including 'title' and 'genres' columns.
        tfidf (TfidfVectorizer): Fitted TF-IDF vectorizer (not used directly in this function).
        tfidf_matrix (scipy.sparse matrix): TF-IDF feature matrix for all movies.
        n (int, optional): Number of similar movies to recommend. Defaults to 10.

    Returns:
        pd.DataFrame: DataFrame containing the recommended movies' titles and genres.

    Raises:
        ValueError: If the specified movie title is not found in the metadata DataFrame.
    """
    try:
        idx = meta_df.index[meta_df['title'] == title][0]
    except IndexError:
        raise ValueError(f"Movie title '{title}' not found.")
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    indices = [i for i, score in sim_scores]
    results = meta_df.iloc[indices][['title', 'genres']].copy()
    # Format genres as comma-separated string
    results['genres'] = results['genres'].apply(
        lambda g: ', '.join([d['name'] for d in eval(g)]
                            ) if isinstance(g, str) else g
    )
    return results


# ---------- 4. MAIN EXECUTION ----------
if __name__ == '__main__':
    print("Loading and cleaning data...")
    meta = load_and_clean_metadata(META_FILE)
    ratings = load_ratings(RATINGS_FILE)

    print("Preparing user-item matrix...")
    user_item_matrix = prepare_user_item_matrix(ratings)

    print("Training collaborative filtering model...")
    svd_model = train_collaborative(user_item_matrix)

    print("Training content-based model...")
    tfidf, tfidf_matrix, id_map = train_content_based(meta)

    # Example recommendations:
    search_title = "The Matrix Reloaded"
    print(f"\nTop 10 movie recommendations for '{search_title}':")
    print(recommend_by_movie(search_title, meta, tfidf, tfidf_matrix, n=10))
