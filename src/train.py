"""Training: fit the item-kNN baseline, the SVD retriever and the LightGBM reranker, evaluate them,
export the artifacts, and tune hyperparameters.

    python3 -m src.train             # fit, evaluate and write artifacts/ (skips if present)
    python3 -m src.train --force     # rebuild even if artifacts exist
    python3 -m src.train --tune all  # grid searches on the validation split (knn, svd, ranker)
"""

import argparse
import hashlib
import itertools
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from src.evaluate import KS, Splits, evaluate_model, evaluate_system, load_splits
from src.utils import (
    ARTIFACTS_DIR,
    GOLD_DIR,
    RELEVANT_RATING,
    RETRIEVAL_K,
    ArtifactError,
    atomic_dir,
    mask_scores,
    top_k,
)

logger = logging.getLogger(__name__)

# Chosen by the grid searches below (`--tune`) on the validation split.
ITEM_KNN_PARAMS = {"n_neighbors": 800, "shrinkage": 500.0, "power": 1.0}
SVD_PARAMS = {"dim": 32, "metric": "dot", "normalize_users": False}
RERANKER_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
}
ITEM_KNN_GRID = {
    "n_neighbors": [200, 400, 800],
    "shrinkage": [200, 500, 1000],
    "power": [1, 2, 3],
}
SVD_GRID = {"dim": [8, 16, 32, 48], "metric": ["dot", "cosine"], "normalize_users": [True, False]}
RERANKER_GRID = {
    "num_leaves": [7, 15, 31],
    "learning_rate": [0.02, 0.05],
    "min_child_samples": [50, 200],
}


# ----- item-kNN baseline -----------------------------------------------------


class ItemKNN:
    """Item-based collaborative filtering on user-mean-centred ratings with cosine similarity.

    score(u, j) = sum over rated items i of (r_ui - mean_u) * sim(i, j), using only each
    item's top `n_neighbors` positive-similarity neighbours. With `shrinkage` > 0, similarity is
    damped by n_common / (n_common + shrinkage) so pairs with few co-raters count for less.
    `power` > 1 raises similarity to that power, so strong neighbours dominate weak ones.
    """

    def __init__(self, n_neighbors: int = 50, shrinkage: float = 0.0, power: float = 1.0):
        self.n_neighbors = n_neighbors
        self.shrinkage = shrinkage
        self.power = power

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int) -> "ItemKNN":
        u = train["user_id_idx"].to_numpy()
        i = train["movie_id_idx"].to_numpy()
        r = train["rating"].to_numpy(float)
        user_mean = np.bincount(u, weights=r, minlength=n_users) / np.maximum(
            np.bincount(u, minlength=n_users), 1
        )
        self.history_ = sparse.csr_matrix((r - user_mean[u], (u, i)), shape=(n_users, n_items))

        item_vectors = self.history_.T.tocsr()
        nn = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, n_items), metric="cosine", algorithm="brute"
        ).fit(item_vectors)
        dist, nbr = nn.kneighbors(item_vectors)

        rows = np.repeat(np.arange(n_items), nbr.shape[1])
        cols, sim = nbr.ravel(), 1.0 - dist.ravel()
        keep = (cols != rows) & (sim > 0)
        rows, cols, sim = rows[keep], cols[keep], sim[keep]
        if self.shrinkage > 0:
            rated = sparse.csr_matrix(
                (np.ones(len(u), dtype=np.float32), (u, i)), shape=(n_users, n_items)
            )
            common = (rated.T @ rated).tocsr()
            n_common = np.asarray(common[rows, cols]).ravel()
            sim = sim * n_common / (n_common + self.shrinkage)
        sim = sim**self.power
        self.similarity_ = sparse.csr_matrix((sim, (rows, cols)), shape=(n_items, n_items))
        return self

    def scores(self, user_idxs: np.ndarray) -> np.ndarray:
        return (self.history_[user_idxs] @ self.similarity_).toarray()

    def explain(self, user_idx: int, movie_idx: int, n: int = 3) -> list[tuple[int, float]]:
        """Rated items that contribute most to recommending `movie_idx`: (item_idx, contribution)."""
        sims = self.similarity_[:, [movie_idx]].T
        contrib = self.history_[user_idx].multiply(sims).tocoo()
        order = np.argsort(-contrib.data)[:n]
        return [(int(contrib.col[o]), float(contrib.data[o])) for o in order if contrib.data[o] > 0]


# ----- SVD retriever ---------------------------------------------------------


class SVDEmbedding:
    """Factorise the binary user x item "liked" matrix X ~ U S V^T.

    Item vectors are the rows of V. A user vector is X_u V (equivalent to U S), which also
    "folds in" any user from their liked items. metric="dot" scores by reconstruction
    (popular items have longer vectors and rank higher); metric="cosine" scores by angle only.
    """

    def __init__(
        self, dim: int = 64, metric: str = "cosine", normalize_users: bool = True, seed: int = 0
    ):
        if metric not in ("dot", "cosine"):
            raise ValueError(f"metric must be 'dot' or 'cosine', got {metric!r}")
        self.dim = dim
        self.metric = metric
        self.normalize_users = normalize_users
        self.seed = seed

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int) -> "SVDEmbedding":
        likes = train[train["rating"] >= RELEVANT_RATING]
        x = sparse.csr_matrix(
            (
                np.ones(len(likes), dtype=np.float32),
                (likes["user_id_idx"].to_numpy(), likes["movie_id_idx"].to_numpy()),
            ),
            shape=(n_users, n_items),
        )
        if self.normalize_users:
            # Stops users with thousands of likes from dominating the factors.
            x = normalize(x, norm="l2", axis=1)

        svd = TruncatedSVD(n_components=self.dim, random_state=self.seed).fit(x)
        item_vectors = svd.components_.T
        user_vectors = x @ item_vectors
        if self.metric == "cosine":
            item_vectors = normalize(item_vectors)
            user_vectors = normalize(user_vectors)
        self.item_vectors_ = np.ascontiguousarray(item_vectors, dtype=np.float32)
        self.user_vectors_ = np.ascontiguousarray(user_vectors, dtype=np.float32)
        return self

    def scores(self, user_idxs: np.ndarray) -> np.ndarray:
        return self.user_vectors_[user_idxs] @ self.item_vectors_.T


# ----- reranker --------------------------------------------------------------


ITEM_PRIOR_COUNT = 10  # pseudo-ratings pulling rarely rated items toward the global mean


class FeatureStore:
    """Per-user, per-item and user-genre statistics computed from the train split only."""

    def fit(
        self,
        train: pd.DataFrame,
        users: pd.DataFrame,
        movies: pd.DataFrame,
        n_users: int,
        n_items: int,
    ) -> "FeatureStore":
        u = train["user_id_idx"].to_numpy()
        i = train["movie_id_idx"].to_numpy()
        r = train["rating"].to_numpy(float)
        liked = (r >= RELEVANT_RATING).astype(float)

        n_u = np.bincount(u, minlength=n_users).astype(float)
        safe_u = np.maximum(n_u, 1)
        mean_u = np.bincount(u, weights=r, minlength=n_users) / safe_u
        var_u = np.bincount(u, weights=r * r, minlength=n_users) / safe_u - mean_u**2
        like_u = np.bincount(u, weights=liked, minlength=n_users) / safe_u
        self.user_stats = np.column_stack([n_u, mean_u, np.sqrt(np.maximum(var_u, 0)), like_u])

        n_i = np.bincount(i, minlength=n_items).astype(float)
        sum_i = np.bincount(i, weights=r, minlength=n_items)
        mean_i = (sum_i + ITEM_PRIOR_COUNT * r.mean()) / (n_i + ITEM_PRIOR_COUNT)
        like_i = np.bincount(i, weights=liked, minlength=n_items) / np.maximum(n_i, 1)
        self.item_stats = np.column_stack([n_i, mean_i, like_i])

        user_cols = [c for c in users.columns if c not in ("user_id", "user_id_idx")]
        self.user_static = np.zeros((n_users, len(user_cols)))
        self.user_static[users["user_id_idx"].to_numpy()] = users[user_cols].to_numpy(float)

        known = movies[movies["movie_id_idx"] > 0]
        genre_cols = [c for c in movies.columns if c.startswith("genre_")]
        pos = known["movie_id_idx"].to_numpy()
        self.genres = np.zeros((n_items, len(genre_cols)))
        self.genres[pos] = known[genre_cols].to_numpy(float)
        self.item_static = np.zeros((n_items, 1 + len(genre_cols)))
        self.item_static[pos, 0] = known["release_year"].to_numpy(float)
        self.item_static[:, 1:] = self.genres

        # Mean centred rating and share of ratings per genre, per user.
        centred = sparse.csr_matrix((r - mean_u[u], (u, i)), shape=(n_users, n_items))
        rated = sparse.csr_matrix((np.ones(len(u)), (u, i)), shape=(n_users, n_items))
        genre_count = rated @ self.genres
        self.affinity = (centred @ self.genres) / np.maximum(genre_count, 1)
        self.share = genre_count / safe_u[:, None]

        self.feature_names = [
            "svd_score", "svd_rank", "knn_score",
            "user_n_ratings", "user_mean_rating", "user_std_rating", "user_like_rate",
            *user_cols,
            "item_n_ratings", "item_mean_rating", "item_like_rate",
            "release_year", *genre_cols,
            "genre_affinity", "genre_share",
        ]
        return self

    def features(self, users, items, svd_score, svd_rank, knn_score) -> np.ndarray:
        item_genres = self.genres[items]
        n_genres = np.maximum(item_genres.sum(axis=1), 1)
        affinity = (self.affinity[users] * item_genres).sum(axis=1) / n_genres
        share = (self.share[users] * item_genres).sum(axis=1) / n_genres
        return np.column_stack(
            [
                svd_score, svd_rank, knn_score,
                self.user_stats[users], self.user_static[users],
                self.item_stats[items], self.item_static[items],
                affinity, share,
            ]
        ).astype(np.float32)


def build_candidates(
    retriever, knn, store: FeatureStore, seen: sparse.csr_matrix, users: np.ndarray,
    k: int = RETRIEVAL_K,
    allowed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k unseen items per user from the retriever, with ranker features.

    Returns candidate item indices (users, k), features (users * k, n_features) and a mask of
    real candidates (users, k); with `allowed`, fewer than k candidates may exist.
    """
    masked = mask_scores(retriever.scores(users), seen[users], allowed)
    cand = top_k(masked, k)
    svd_score = np.take_along_axis(masked, cand, axis=1)
    valid = np.isfinite(svd_score)
    svd_score = np.where(valid, svd_score, 0.0)
    knn_score = np.take_along_axis(knn.scores(users), cand, axis=1)
    svd_rank = np.broadcast_to(np.arange(k), cand.shape)
    x = store.features(
        np.repeat(users, k), cand.ravel(), svd_score.ravel(), svd_rank.ravel(), knn_score.ravel()
    )
    return cand, x, valid


@dataclass(frozen=True)
class RankerData:
    x_train: np.ndarray
    y_train: np.ndarray
    group_train: np.ndarray
    x_valid: np.ndarray
    y_valid: np.ndarray
    group_valid: np.ndarray
    feature_names: list[str]


def build_ranker_data(
    retriever, knn, store: FeatureStore, seen: sparse.csr_matrix, relevant: sparse.csr_matrix,
    k: int = RETRIEVAL_K, valid_fraction: float = 0.2, seed: int = 0, batch_size: int = 1000,
) -> RankerData:
    """Label retrieved candidates with the val split; hold out whole users for early stopping."""
    users = np.flatnonzero(relevant.getnnz(axis=1) > 0)
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    n_valid = int(len(users) * valid_fraction)

    def build(subset: np.ndarray):
        xs, ys = [], []
        for start in range(0, len(subset), batch_size):
            batch = subset[start : start + batch_size]
            cand, x, _ = build_candidates(retriever, knn, store, seen, batch, k)
            labels = np.take_along_axis(relevant[batch].toarray(), cand, axis=1)
            xs.append(x)
            ys.append(labels.ravel().astype(np.int32))
        return np.vstack(xs), np.concatenate(ys), np.full(len(subset), k)

    x_va, y_va, g_va = build(users[:n_valid])
    x_tr, y_tr, g_tr = build(users[n_valid:])
    return RankerData(x_tr, y_tr, g_tr, x_va, y_va, g_va, store.feature_names)


def fit_ranker(data: RankerData, params: dict, seed: int = 0) -> lgb.LGBMRanker:
    ranker = lgb.LGBMRanker(
        objective="lambdarank", random_state=seed, n_jobs=4, verbose=-1, **params
    )
    ranker.fit(
        data.x_train, data.y_train, group=data.group_train,
        eval_X=data.x_valid, eval_y=data.y_valid, eval_group=[data.group_valid], eval_at=[10],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    ranker.set_params(n_jobs=1)  # single-row requests are faster without thread overhead
    return ranker


class TwoTier:
    """Retriever -> top-k candidates -> LightGBM rerank, exposed as a score matrix.

    Non-candidates score -inf, so it plugs into the same evaluation as the other models.
    """

    def __init__(self, retriever, knn, store, ranker, seen: sparse.csr_matrix, k: int = RETRIEVAL_K):
        self.retriever, self.knn, self.store = retriever, knn, store
        self.ranker, self.seen, self.k = ranker, seen, k

    def scores(self, user_idxs: np.ndarray, allowed: np.ndarray | None = None) -> np.ndarray:
        """Reranked scores for each user's candidates; everything else is -inf.

        `allowed` is an optional boolean mask over items (for example one genre).
        """
        cand, x, valid = build_candidates(
            self.retriever, self.knn, self.store, self.seen, user_idxs, self.k, allowed
        )
        predicted = np.where(valid, self.ranker.predict(x).reshape(cand.shape), -np.inf)
        out = np.full((len(user_idxs), self.seen.shape[1]), -np.inf, dtype=np.float32)
        np.put_along_axis(out, cand, predicted.astype(np.float32), axis=1)
        return out


@dataclass(frozen=True)
class System:
    """The fitted two-tier recommender: retriever, baseline, feature store and reranker."""

    knn: object
    svd: object
    store: FeatureStore
    ranker: object

    def two_tier(self, seen: sparse.csr_matrix, k: int = RETRIEVAL_K) -> TwoTier:
        return TwoTier(self.svd, self.knn, self.store, self.ranker, seen, k)


# ----- export and load -------------------------------------------------------


KNN_FILE = "item_knn.joblib"


SVD_FILE = "svd_retriever.joblib"


STORE_FILE = "feature_store.joblib"


RANKER_FILE = "reranker.txt"


SEEN_FILE = "seen.npz"


METRICS_FILE = "metrics.json"


MANIFEST_FILE = "manifest.json"


LIBRARIES = ("numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "joblib")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_map_fingerprint(users: pd.DataFrame, movies: pd.DataFrame) -> str:
    """Content hash of the id-to-index maps the models were built against."""
    digest = hashlib.sha256()
    for df in (users[["user_id", "user_id_idx"]], movies[["movie_id", "movie_id_idx"]]):
        digest.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def is_artifacts_present(artifacts_dir: Path = ARTIFACTS_DIR) -> bool:
    manifest = Path(artifacts_dir) / MANIFEST_FILE
    if not manifest.is_file():
        return False
    files = json.loads(manifest.read_text())["files"]
    return all((Path(artifacts_dir) / name).is_file() for name in files)


@dataclass(frozen=True)
class SingleThreadRanker:
    """Booster wrapper exposing `predict(x)`; one thread avoids overhead on 100-row requests."""

    booster: lgb.Booster

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.booster.predict(x, num_threads=1)


@dataclass(frozen=True)
class Artifacts:
    knn: object
    svd: object
    store: FeatureStore
    ranker: SingleThreadRanker
    seen: sparse.csr_matrix
    users: pd.DataFrame  # from gold
    movies: pd.DataFrame  # from gold, including genre columns
    metrics: dict
    manifest: dict

    def two_tier(self, seen: sparse.csr_matrix | None = None, k: int | None = None) -> TwoTier:
        """The recommender; `seen` defaults to the train and val history stored with the artifacts."""
        k = k or self.manifest["config"]["retrieval_k"]
        return TwoTier(self.svd, self.knn, self.store, self.ranker, self.seen if seen is None else seen, k)


def load_artifacts(
    artifacts_dir: Path = ARTIFACTS_DIR, gold_dir: Path = GOLD_DIR
) -> Artifacts:
    """Load models from artifacts and id maps and genre flags from gold, verifying both.

    Only load directories you produced: joblib uses pickle.
    """
    root = Path(artifacts_dir)
    manifest_path = root / MANIFEST_FILE
    if not manifest_path.is_file():
        raise ArtifactError(f"no manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file():
            raise ArtifactError(f"missing artifact file: {name}")
        if _sha256(path) != expected:
            raise ArtifactError(f"checksum mismatch: {name}")

    for lib, built_with in manifest["versions"].items():
        if lib != "python" and version(lib) != built_with:
            logger.warning("%s is %s but artifacts were built with %s", lib, version(lib), built_with)

    ranker = SingleThreadRanker(lgb.Booster(model_file=str(root / RANKER_FILE)))
    gold = Path(gold_dir)
    users = pd.read_parquet(gold / "users.parquet", columns=["user_id", "user_id_idx"])
    movies = pd.read_parquet(gold / "movies.parquet")
    if _id_map_fingerprint(users, movies) != manifest["gold_id_maps"]:
        raise ArtifactError(f"gold layer at {gold} does not match these artifacts; rebuild them")
    return Artifacts(
        knn=joblib.load(root / KNN_FILE),
        svd=joblib.load(root / SVD_FILE),
        store=joblib.load(root / STORE_FILE),
        ranker=ranker,
        seen=sparse.load_npz(root / SEEN_FILE).tocsr(),
        users=users,
        movies=movies,
        metrics=json.loads((root / METRICS_FILE).read_text()),
        manifest=manifest,
    )


def _verify(root: Path, gold_dir: Path, system: System, seen: sparse.csr_matrix, k: int) -> None:
    """The reloaded system must reproduce the in-memory system's scores."""
    users = np.arange(1, min(21, seen.shape[0]))
    expected = system.two_tier(seen, k).scores(users)
    actual = load_artifacts(root, gold_dir).two_tier(k=k).scores(users)
    if not np.allclose(expected, actual, rtol=1e-5, atol=1e-6):
        raise ArtifactError("reloaded artifacts do not reproduce the original scores")


def save_artifacts(
    system: System,
    metrics: pd.DataFrame,
    seen: sparse.csr_matrix,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    gold_dir: Path = GOLD_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    k: int = RETRIEVAL_K,
) -> None:
    """Write the serving models, verify they reload identically, then publish atomically.

    `users` and `movies` are the gold tables; only a fingerprint of their id maps is stored.
    """
    with atomic_dir(artifacts_dir) as tmp:
        joblib.dump(system.knn, tmp / KNN_FILE)
        joblib.dump(system.svd, tmp / SVD_FILE)
        joblib.dump(system.store, tmp / STORE_FILE)
        system.ranker.booster_.save_model(
            str(tmp / RANKER_FILE), num_iteration=system.ranker.best_iteration_
        )
        sparse.save_npz(tmp / SEEN_FILE, seen)
        (tmp / METRICS_FILE).write_text(
            json.dumps(json.loads(metrics.to_json(orient="index")), indent=2)
        )

        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "versions": {"python": sys.version.split()[0], **{lib: version(lib) for lib in LIBRARIES}},
            "config": {
                "relevant_rating": RELEVANT_RATING,
                "retrieval_k": k,
                "item_knn_params": ITEM_KNN_PARAMS,
                "svd_params": SVD_PARAMS,
                "reranker_params": RERANKER_PARAMS,
            },
            "n_users": seen.shape[0],
            "n_items": seen.shape[1],
            "reranker_trees": int(system.ranker.best_iteration_),
            "feature_names": system.store.feature_names,
            "gold_id_maps": _id_map_fingerprint(users, movies),
            "files": {p.name: _sha256(p) for p in sorted(tmp.iterdir())},
        }
        (tmp / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
        _verify(tmp, gold_dir, system, seen, k)
        # The swap replaces the directory, so carry over the tracked placeholder unchanged.
        placeholder = Path(artifacts_dir) / ".gitkeep"
        if placeholder.is_file():
            shutil.copy2(placeholder, tmp / ".gitkeep")
        else:
            (tmp / ".gitkeep").touch()
    logger.info("artifacts written to %s", artifacts_dir)


# ----- training, tuning and entry point --------------------------------------


def fit_system(sp: Splits) -> System:
    """Fit the retriever, baseline, feature store and reranker; only train and val are used."""
    knn = ItemKNN(**ITEM_KNN_PARAMS).fit(sp.train, sp.n_users, sp.n_items)
    svd = SVDEmbedding(**SVD_PARAMS).fit(sp.train, sp.n_users, sp.n_items)
    store = FeatureStore().fit(sp.train, sp.users, sp.movies, sp.n_users, sp.n_items)
    data = build_ranker_data(svd, knn, store, sp.seen("val"), sp.relevant("val"))
    return System(knn, svd, store, fit_ranker(data, RERANKER_PARAMS))


def tune(model_cls, grid: dict[str, list], sp: Splits, metric: str) -> pd.DataFrame:
    """Fit every grid combination on train, score on val, best first by `metric`."""
    relevant, seen = sp.relevant("val"), sp.seen("val")
    rows = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        model = model_cls(**params).fit(sp.train, sp.n_users, sp.n_items)
        rows.append({**params, **evaluate_model(model, relevant, seen, ks=KS)})
        logger.info("%s %s -> %s=%.4f", model_cls.__name__, params, metric, rows[-1][metric])
    return pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)


def tune_reranker(sp: Splits) -> pd.DataFrame:
    """Grid over LightGBM params, scored by NDCG@10 on held-out val users (not on test)."""
    knn = ItemKNN(**ITEM_KNN_PARAMS).fit(sp.train, sp.n_users, sp.n_items)
    svd = SVDEmbedding(**SVD_PARAMS).fit(sp.train, sp.n_users, sp.n_items)
    store = FeatureStore().fit(sp.train, sp.users, sp.movies, sp.n_users, sp.n_items)
    data = build_ranker_data(svd, knn, store, sp.seen("val"), sp.relevant("val"))
    rows = []
    for values in itertools.product(*RERANKER_GRID.values()):
        params = dict(zip(RERANKER_GRID, values))
        ranker = fit_ranker(data, {**RERANKER_PARAMS, **params})
        rows.append(
            {**params, "trees": ranker.best_iteration_,
             "valid_ndcg@10": ranker.best_score_["valid_0"]["ndcg@10"]}
        )
        logger.info("LGBMRanker %s -> %.4f", params, rows[-1]["valid_ndcg@10"])
    return pd.DataFrame(rows).sort_values("valid_ndcg@10", ascending=False).reset_index(drop=True)



def tune_main(target: str) -> None:
    sp = load_splits()
    if target in ("all", "knn"):
        # Item-kNN is a standalone recommender, so it is tuned on ranking quality.
        knn = tune(ItemKNN, ITEM_KNN_GRID, sp, "ndcg@10")
        print("\nItemKNN (val), top 5:\n", knn.head(5).round(4).to_string())
        print("best ItemKNN:", knn.iloc[0][list(ITEM_KNN_GRID)].to_dict())
    if target in ("all", "svd"):
        # The SVD model feeds the reranker, so it is tuned on how many liked items it retrieves.
        svd = tune(SVDEmbedding, SVD_GRID, sp, "recall@100")
        print("\nSVDEmbedding (val), top 5:\n", svd.head(5).round(4).to_string())
        print("best SVD:", svd.iloc[0][list(SVD_GRID)].to_dict())
    if target in ("all", "ranker"):
        ranker = tune_reranker(sp)
        print("\nLGBMRanker (held-out val users):\n", ranker.round(4).to_string())
        print("best LGBMRanker:", ranker.iloc[0][list(RERANKER_GRID)].to_dict())


def build(artifacts_dir: Path = ARTIFACTS_DIR, force: bool = False) -> pd.DataFrame | None:
    """Fit, evaluate and export; returns the evaluation table, or None if artifacts already exist."""
    if not force and is_artifacts_present(artifacts_dir):
        logger.info("artifacts already present at %s; use --force to rebuild", artifacts_dir)
        return None
    sp = load_splits()
    system = fit_system(sp)
    table = evaluate_system(sp, system)
    # Serving excludes everything already rated in train and val.
    save_artifacts(system, table, sp.seen("test"), sp.users, sp.movies, artifacts_dir=artifacts_dir)
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--force", action="store_true", help="rebuild even if artifacts exist")
    parser.add_argument("--tune", choices=["all", "knn", "svd", "ranker"],
                        help="run grid searches on the validation split instead of training")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if not args.tune else logging.WARNING, format="%(message)s")
    pd.set_option("display.width", 250, "display.max_columns", None)

    if args.tune:
        tune_main(args.tune)
        return 0
    table = build(force=args.force)
    if table is not None:
        print(table.round(4).T.to_string())
    return 0


if __name__ == "__main__":
    # Re-import so the models are pickled under `src.train`, not `__main__`; serving loads them from there.
    from src.train import main as _main

    sys.exit(_main())
