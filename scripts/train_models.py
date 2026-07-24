"""Offline training pipeline for all recommenders.

Run once (or whenever the data changes):
    source .venv/bin/activate
    python scripts/train_models.py

Writes all artifacts to models/. The deployed app only ever *reads* these
artifacts. The neural/taste-embedding model is trained here with NumPy and
served later with NumPy only; it is not derived from collaborative SVD factors.
"""
import sys
import time
import argparse
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import data_utils
from src.artifacts import atomic_replace, write_manifest
from src.recommenders.base import MODELS_DIR
from src.recommenders import clustering, collaborative, content_based, neural, popularity

EMBED_DIM = 16
LATENT_DIM = 16  # shared space the two towers project into for the dot product
EPOCHS = 8
BATCH_SIZE = 256
DEFAULT_NEURAL_SAMPLE_SIZE = 1_000_000
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-6


def _sample_ratings(ratings, max_rows: int | None, seed: int = 42):
    if not max_rows or len(ratings) <= max_rows:
        return ratings
    return ratings.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def _genre_matrix_for_movie_ids(movies, movie_ids) -> np.ndarray:
    genres = data_utils.all_genres(movies)
    genre_idx = {g: i for i, g in enumerate(genres)}
    genre_matrix = np.zeros((len(movie_ids), len(genres)), dtype=np.float32)
    movies_by_id = movies.set_index("movieId")["genre_list"]
    for i, mid in enumerate(movie_ids):
        for g in movies_by_id.get(mid, []):
            genre_matrix[i, genre_idx[g]] = 1.0
    return genre_matrix


def _adam_update(param, grad, state, key, step: int, lr: float):
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m_key, v_key = f"{key}_m", f"{key}_v"
    if m_key not in state:
        state[m_key] = np.zeros_like(param)
        state[v_key] = np.zeros_like(param)
    state[m_key] = beta1 * state[m_key] + (1.0 - beta1) * grad
    state[v_key] = beta2 * state[v_key] + (1.0 - beta2) * (grad * grad)
    m_hat = state[m_key] / (1.0 - beta1**step)
    v_hat = state[v_key] / (1.0 - beta2**step)
    param -= lr * m_hat / (np.sqrt(v_hat) + eps)


def build_and_train_neural(movies, ratings, epochs: int = EPOCHS, seed: int = 42) -> dict:
    """Train a genuine two-tower neural recommender with NumPy.

    The model learns user embeddings, movie embeddings, movie biases, and two
    dense towers from rating residuals. Only the item tower is exported because
    production users are new profiles folded into the learned item space.
    """

    _, user_ids, movie_ids = data_utils.user_item_matrix(ratings)
    user_pos = {u: i for i, u in enumerate(user_ids)}
    movie_pos = {m: i for i, m in enumerate(movie_ids)}

    genre_matrix = _genre_matrix_for_movie_ids(movies, movie_ids)

    u_idx = ratings["userId"].map(user_pos).values.astype(np.int32)
    m_idx = ratings["movieId"].map(movie_pos).values.astype(np.int32)
    global_mean = float(ratings["rating"].mean())
    y = ratings["rating"].values.astype(np.float32) - global_mean  # train on the residual

    rng = np.random.default_rng(seed)
    n = len(y)
    perm = rng.permutation(n)
    split = int(n * 0.9)
    train_idx, val_idx = perm[:split], perm[split:]

    n_users, n_movies, n_genres = len(user_ids), len(movie_ids), genre_matrix.shape[1]
    user_emb = rng.normal(0.0, 0.03, size=(n_users, EMBED_DIM)).astype(np.float32)
    movie_emb = rng.normal(0.0, 0.03, size=(n_movies, EMBED_DIM)).astype(np.float32)
    user_bias = np.zeros(n_users, dtype=np.float32)
    movie_bias = np.zeros(n_movies, dtype=np.float32)
    Wu = rng.normal(0.0, np.sqrt(2.0 / EMBED_DIM), size=(EMBED_DIM, LATENT_DIM)).astype(np.float32)
    bu = np.zeros(LATENT_DIM, dtype=np.float32)
    Wi = rng.normal(0.0, np.sqrt(2.0 / max(EMBED_DIM + n_genres, 1)), size=(EMBED_DIM + n_genres, LATENT_DIM)).astype(np.float32)
    bi = np.zeros(LATENT_DIM, dtype=np.float32)
    state = {}
    step = 0

    for epoch in range(epochs):
        epoch_idx = rng.permutation(train_idx)
        losses = []
        for start in range(0, len(epoch_idx), BATCH_SIZE):
            step += 1
            batch = epoch_idx[start:start + BATCH_SIZE]
            users = u_idx[batch]
            movies_batch = m_idx[batch]
            targets = y[batch]

            u_raw = user_emb[users]
            m_raw = movie_emb[movies_batch]
            item_input = np.concatenate([m_raw, genre_matrix[movies_batch]], axis=1)
            u_pre = u_raw @ Wu + bu
            i_pre = item_input @ Wi + bi
            u_vec = np.maximum(u_pre, 0.0)
            i_vec = np.maximum(i_pre, 0.0)
            pred = np.sum(u_vec * i_vec, axis=1) + user_bias[users] + movie_bias[movies_batch]
            err = pred - targets
            losses.append(float(np.mean(err * err)))
            grad = (2.0 / max(len(batch), 1)) * err

            du_vec = grad[:, None] * i_vec
            di_vec = grad[:, None] * u_vec
            du_pre = du_vec * (u_pre > 0.0)
            di_pre = di_vec * (i_pre > 0.0)

            grad_Wu = u_raw.T @ du_pre + WEIGHT_DECAY * Wu
            grad_bu = du_pre.sum(axis=0)
            grad_Wi = item_input.T @ di_pre + WEIGHT_DECAY * Wi
            grad_bi = di_pre.sum(axis=0)
            grad_user_raw = du_pre @ Wu.T + WEIGHT_DECAY * u_raw
            grad_item_input = di_pre @ Wi.T
            grad_movie_raw = grad_item_input[:, :EMBED_DIM] + WEIGHT_DECAY * m_raw
            grad_user_bias = grad
            grad_movie_bias = grad

            _adam_update(Wu, grad_Wu.astype(np.float32), state, "Wu", step, LEARNING_RATE)
            _adam_update(bu, grad_bu.astype(np.float32), state, "bu", step, LEARNING_RATE)
            _adam_update(Wi, grad_Wi.astype(np.float32), state, "Wi", step, LEARNING_RATE)
            _adam_update(bi, grad_bi.astype(np.float32), state, "bi", step, LEARNING_RATE)
            np.add.at(user_emb, users, -LEARNING_RATE * grad_user_raw.astype(np.float32))
            np.add.at(movie_emb, movies_batch, -LEARNING_RATE * grad_movie_raw.astype(np.float32))
            np.add.at(user_bias, users, -LEARNING_RATE * grad_user_bias.astype(np.float32))
            np.add.at(movie_bias, movies_batch, -LEARNING_RATE * grad_movie_bias.astype(np.float32))

        val_pred = _predict_training_residuals(
            u_idx[val_idx], m_idx[val_idx], user_emb, movie_emb, user_bias, movie_bias, Wu, bu, Wi, bi, genre_matrix
        )
        val_err = val_pred - y[val_idx]
        print(
            f"epoch {epoch + 1}/{epochs} loss={np.mean(losses):.4f} "
            f"val_rmse={np.sqrt(np.mean(val_err * val_err)):.4f}"
        )

    return {
        "movie_emb": movie_emb,
        "movie_bias": movie_bias,
        "genre_matrix": genre_matrix,
        "Wi": Wi, "bi": bi,
        "global_mean": np.float32(global_mean),
        "movie_ids": movie_ids,
        "training_mode": np.array("two_tower_neural"),
        "embedding_dim": np.int32(EMBED_DIM),
        "latent_dim": np.int32(LATENT_DIM),
        "epochs": np.int32(epochs),
        "validation_rmse_residual": np.float32(np.sqrt(np.mean(val_err * val_err))),
        "profile_baseline_strength": np.float32(collaborative.PROFILE_BASELINE_STRENGTH),
    }


def _predict_training_residuals(
    users, movies_batch, user_emb, movie_emb, user_bias, movie_bias, Wu, bu, Wi, bi, genre_matrix
) -> np.ndarray:
    preds = np.empty(len(users), dtype=np.float32)
    for start in range(0, len(users), 8192):
        stop = min(start + 8192, len(users))
        u_raw = user_emb[users[start:stop]]
        m_raw = movie_emb[movies_batch[start:stop]]
        item_input = np.concatenate([m_raw, genre_matrix[movies_batch[start:stop]]], axis=1)
        u_vec = np.maximum(u_raw @ Wu + bu, 0.0)
        i_vec = np.maximum(item_input @ Wi + bi, 0.0)
        preds[start:stop] = (
            np.sum(u_vec * i_vec, axis=1)
            + user_bias[users[start:stop]]
            + movie_bias[movies_batch[start:stop]]
        )
    return preds


def train_neural(movies, ratings, mode: str, sample_size: int | None, epochs: int, out_path=None):
    if mode != "two_tower":
        raise ValueError("Only the independently trained two_tower neural mode is supported.")
    train_ratings = _sample_ratings(ratings, sample_size)
    if len(train_ratings) < len(ratings):
        print(f"Training two-tower neural model on a {len(train_ratings):,}-rating sample.")
    weights = build_and_train_neural(movies, train_ratings, epochs=epochs)
    output = out_path or neural.ARTIFACT
    neural.save_weights(weights, output)
    print("Saved neural model weights ->", output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural-mode", choices=["two_tower"], default="two_tower")
    parser.add_argument("--neural-sample-size", type=int, default=DEFAULT_NEURAL_SAMPLE_SIZE)
    parser.add_argument("--neural-epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    t0 = time.time()
    print("Loading data...")
    movies = data_utils.load_movies()
    ratings = data_utils.load_ratings()
    tags = data_utils.load_tags()

    generation_id = str(uuid.uuid4())
    with tempfile.TemporaryDirectory(prefix="filmflicker-models-", dir=MODELS_DIR.parent) as temporary:
        staging = Path(temporary)
        print("Training popularity table...")
        popularity.train_and_save(movies, ratings, staging / popularity.ARTIFACT.name)

        print("Training content-based TF-IDF...")
        content_based.train_and_save(movies, tags, staging)

        print("Training collaborative SVD...")
        collab_artifacts = collaborative.train_and_save(
            ratings, out_path=staging / collaborative.ARTIFACT.name
        )

        print("Training KMeans clustering...")
        clustering.train_and_save(movies, ratings, staging / clustering.ARTIFACT.name)

        print("Training/exporting latent taste embedding model...")
        train_neural(
            movies,
            ratings,
            args.neural_mode,
            args.neural_sample_size,
            args.neural_epochs,
            staging / neural.ARTIFACT.name,
        )

        for path in staging.iterdir():
            atomic_replace(path, MODELS_DIR / path.name)

    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics_path.unlink()
    write_manifest(
        generation_id=generation_id,
        config={
            "collaborative_components": collaborative.N_COMPONENTS,
            "fold_in_regularization": collaborative.FOLD_IN_REGULARIZATION,
            "profile_baseline_strength": collaborative.PROFILE_BASELINE_STRENGTH,
            "neural_mode": args.neural_mode,
            "neural_epochs": args.neural_epochs,
            "neural_sample_size": args.neural_sample_size,
        },
    )

    print(f"All models trained in {time.time() - t0:.1f}s. Artifacts in models/.")


if __name__ == "__main__":
    main()
