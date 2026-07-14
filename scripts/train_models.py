"""Offline training pipeline for all recommenders.

Run once (or whenever the data changes):
    source .venv/bin/activate
    pip install -r requirements-train.txt   # adds TensorFlow, training-only
    python scripts/train_models.py

Writes all artifacts to models/. The deployed app only ever *reads* these
artifacts - see requirements.txt (no TensorFlow) vs requirements-train.txt.
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import data_utils
from src.recommenders import clustering, collaborative, content_based, neural, popularity

EMBED_DIM = 16
LATENT_DIM = 16  # shared space the two towers project into for the dot product
EPOCHS = 8
BATCH_SIZE = 256
DEFAULT_NEURAL_SAMPLE_SIZE = 1_000_000


def _sample_ratings(ratings, max_rows: int | None, seed: int = 42):
    if not max_rows or len(ratings) <= max_rows:
        return ratings
    return ratings.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def build_and_train_neural(movies, ratings, epochs: int = EPOCHS) -> dict:
    """Trains a two-tower (Neural CF style) Keras model and returns the
    exportable weights dict (does not touch disk) - reused by both the
    production training run and scripts/evaluate_models.py's held-out
    evaluation.

    Architecture: user tower (user embedding -> dense) and item tower
    (movie embedding + genre features -> dense) each project into a shared
    LATENT_DIM space; prediction = dot(user_vec, item_vec) + user_bias +
    item_bias, trained on rating-minus-global-mean (so biases only need to
    learn a *deviation*, matching how the SVD model is mean-centered too).

    This replaced an earlier concat-everything-then-MLP design that turned
    out to lean heavily on item popularity rather than sharp personalization
    (see README/Model Comparison page) - forcing the interaction through an
    explicit dot product is the standard fix for that failure mode.
    """
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
    global_mean = float(ratings["rating"].mean())
    y = ratings["rating"].values.astype(np.float32) - global_mean  # train on the residual

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
    movie_bias_emb = layers.Embedding(n_movies, 1, embeddings_regularizer=reg, name="movie_bias")(movie_input)
    user_bias_emb = layers.Embedding(n_users, 1, embeddings_regularizer=reg, name="user_bias")(user_input)

    user_vec_in = layers.Flatten()(user_embedding)
    movie_vec_in = layers.Flatten()(movie_embedding)
    movie_bias = layers.Flatten()(movie_bias_emb)
    user_bias = layers.Flatten()(user_bias_emb)

    user_tower = layers.Dense(LATENT_DIM, activation="relu", name="user_tower")(user_vec_in)
    item_concat = layers.Concatenate()([movie_vec_in, genre_input])
    item_tower = layers.Dense(LATENT_DIM, activation="relu", name="item_tower")(item_concat)

    dot = layers.Dot(axes=1, name="dot")([user_tower, item_tower])
    out = layers.Add(name="add_biases")([dot, user_bias, movie_bias])

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

    Wi, bi = model.get_layer("item_tower").get_weights()

    return {
        # Only what's needed to serve NEW profiles at inference time (see
        # src/recommenders/neural.py) - the user tower/embedding/bias are
        # used during training but never at serve time, since every app
        # profile is a cold-start fold-in, never a "known" MovieLens user.
        "movie_emb": model.get_layer("movie_embedding").get_weights()[0],
        "movie_bias": model.get_layer("movie_bias").get_weights()[0].ravel(),
        "genre_matrix": genre_matrix,
        "Wi": Wi, "bi": bi,
        "global_mean": np.float32(global_mean),
        "movie_ids": movie_ids,
    }


def build_embedding_fallback_weights(collab_artifacts: dict, ratings) -> dict:
    """Build a serve-compatible embedding model without TensorFlow.

    The deployed numpy recommender expects an item-tower artifact. When the
    local training environment does not have TensorFlow, we export a pure
    embedding model from the collaborative SVD item factors. It keeps the same
    serving contract and produces personalized fold-in recommendations, while
    metrics.json records that this artifact was built from SVD factors.
    """
    movie_ids = collab_artifacts["movie_ids"]
    item_factors = collab_artifacts["item_factors"].astype(np.float32)
    norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
    item_unit = item_factors / np.maximum(norms, 1e-6)

    movie_emb = np.hstack([np.maximum(item_unit, 0), np.maximum(-item_unit, 0)]).astype(np.float32)
    movie_stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"]).reindex(movie_ids)
    global_mean = float(ratings["rating"].mean())
    counts = movie_stats["count"].fillna(0).to_numpy(dtype=np.float32)
    means = movie_stats["mean"].fillna(global_mean).to_numpy(dtype=np.float32)
    movie_bias = ((means - global_mean) * (counts / (counts + 25.0))).astype(np.float32)

    return {
        "movie_emb": movie_emb,
        "movie_bias": movie_bias,
        "genre_matrix": np.zeros((len(movie_ids), 0), dtype=np.float32),
        "Wi": np.eye(movie_emb.shape[1], dtype=np.float32),
        "bi": np.zeros(movie_emb.shape[1], dtype=np.float32),
        "global_mean": np.float32(global_mean),
        "movie_ids": movie_ids,
        "training_mode": np.array("svd_embedding_fallback"),
    }


def train_neural(movies, ratings, collab_artifacts: dict, mode: str, sample_size: int | None):
    if mode == "tensorflow":
        train_ratings = _sample_ratings(ratings, sample_size)
        if len(train_ratings) < len(ratings):
            print(f"Training TensorFlow neural model on a {len(train_ratings):,}-rating sample.")
        weights = build_and_train_neural(movies, train_ratings)
    else:
        print("Exporting latent taste embedding artifact from trained SVD factors.")
        weights = build_embedding_fallback_weights(collab_artifacts, ratings)
    neural.save_weights(weights)
    print("Saved neural model weights ->", neural.ARTIFACT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural-mode", choices=["embedding", "tensorflow"], default="embedding")
    parser.add_argument("--neural-sample-size", type=int, default=DEFAULT_NEURAL_SAMPLE_SIZE)
    args = parser.parse_args()

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
    collab_artifacts = collaborative.train_and_save(ratings)

    print("Training KMeans clustering...")
    clustering.train_and_save(movies, ratings)

    print("Training/exporting latent taste embedding model...")
    train_neural(movies, ratings, collab_artifacts, args.neural_mode, args.neural_sample_size)

    print(f"All models trained in {time.time() - t0:.1f}s. Artifacts in models/.")


if __name__ == "__main__":
    main()
