"""Offline training pipeline for all 5 recommenders.

Run once (or whenever the data changes):
    source .venv/bin/activate
    pip install -r requirements-train.txt   # adds TensorFlow, training-only
    python scripts/train_models.py

Writes all artifacts to models/. The deployed app only ever *reads* these
artifacts - see requirements.txt (no TensorFlow) vs requirements-train.txt.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import data_utils
from src.recommenders import clustering, collaborative, content_based, neural, popularity

EMBED_DIM = 16
EPOCHS = 8
BATCH_SIZE = 256


def build_and_train_neural(movies, ratings, epochs: int = EPOCHS) -> dict:
    """Trains the Keras model and returns the exportable weights dict
    (does not touch disk) - reused by both the production training run and
    scripts/evaluate_models.py's held-out evaluation."""
    from tensorflow import keras
    from tensorflow.keras import layers

    genres = data_utils.all_genres(movies)
    genre_idx = {g: i for i, g in enumerate(genres)}

    matrix, user_ids, movie_ids = data_utils.user_item_matrix(ratings)
    user_pos = {u: i for i, u in enumerate(user_ids)}
    movie_pos = {m: i for i, m in enumerate(movie_ids)}

    genre_matrix = np.zeros((len(movie_ids), len(genres)), dtype=np.float32)
    movies_by_id = movies.set_index("movieId")["genre_list"]
    for i, mid in enumerate(movie_ids):
        for g in movies_by_id.get(mid, []):
            genre_matrix[i, genre_idx[g]] = 1.0

    u_idx = ratings["userId"].map(user_pos).values.astype(np.int32)
    m_idx = ratings["movieId"].map(movie_pos).values.astype(np.int32)
    g_feat = genre_matrix[m_idx]
    y = ratings["rating"].values.astype(np.float32)

    rng = np.random.default_rng(42)
    n = len(y)
    perm = rng.permutation(n)
    split = int(n * 0.9)
    train_idx, val_idx = perm[:split], perm[split:]

    n_users, n_movies, n_genres = len(user_ids), len(movie_ids), len(genres)

    user_input = keras.Input(shape=(1,), name="user")
    movie_input = keras.Input(shape=(1,), name="movie")
    genre_input = keras.Input(shape=(n_genres,), name="genre")

    # L2 regularization keeps embeddings for sparsely-rated movies/users close
    # to zero instead of overfitting to a single noisy data point - without
    # it, a movie with exactly 1 rating can get an embedding that looks
    # "confidently amazing" to every user (see README for the failure mode
    # this fixes).
    reg = keras.regularizers.l2(1e-6)
    user_embedding = layers.Embedding(n_users, EMBED_DIM, embeddings_regularizer=reg, name="user_embedding")(user_input)
    movie_embedding = layers.Embedding(n_movies, EMBED_DIM, embeddings_regularizer=reg, name="movie_embedding")(movie_input)
    user_vec = layers.Flatten()(user_embedding)
    movie_vec = layers.Flatten()(movie_embedding)

    x = layers.Concatenate()([user_vec, movie_vec, genre_input])
    x = layers.Dense(32, activation="relu", name="dense_1")(x)
    x = layers.Dense(16, activation="relu", name="dense_2")(x)
    out = layers.Dense(1, activation="linear", name="dense_out")(x)

    model = keras.Model(inputs=[user_input, movie_input, genre_input], outputs=out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    model.fit(
        {"user": u_idx[train_idx], "movie": m_idx[train_idx], "genre": g_feat[train_idx]},
        y[train_idx],
        validation_data=(
            {"user": u_idx[val_idx], "movie": m_idx[val_idx], "genre": g_feat[val_idx]},
            y[val_idx],
        ),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    W1, b1 = model.get_layer("dense_1").get_weights()
    W2, b2 = model.get_layer("dense_2").get_weights()
    W3, b3 = model.get_layer("dense_out").get_weights()

    return {
        "user_emb": model.get_layer("user_embedding").get_weights()[0],
        "movie_emb": model.get_layer("movie_embedding").get_weights()[0],
        "genre_matrix": genre_matrix,
        "W1": W1, "b1": b1,
        "W2": W2, "b2": b2,
        "W3": W3, "b3": b3,
        "user_ids": user_ids,
        "movie_ids": movie_ids,
    }


def train_neural(movies, ratings):
    weights = build_and_train_neural(movies, ratings)
    neural.save_weights(weights)
    print("Saved neural model weights ->", neural.ARTIFACT)


def main():
    t0 = time.time()
    print("Loading data...")
    movies = data_utils.load_movies()
    ratings = data_utils.load_ratings()
    tags = data_utils.load_tags()

    print("Training popularity table...")
    popularity.train_and_save(movies, ratings)

    print("Training content-based TF-IDF...")
    content_based.train_and_save(movies, tags)

    print("Training collaborative SVD...")
    collaborative.train_and_save(ratings)

    print("Training KMeans clustering...")
    clustering.train_and_save(movies, ratings)

    print("Training neural embedding model (TensorFlow, offline only)...")
    train_neural(movies, ratings)

    print(f"All models trained in {time.time() - t0:.1f}s. Artifacts in models/.")


if __name__ == "__main__":
    main()
