"""Small synthetic system shared by the artifact, inference and serving tests."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src import train as artifacts
from src.train import (
    FeatureStore,
    ItemKNN,
    RankerData,
    SVDEmbedding,
    System,
    fit_ranker,
)

N_USERS, N_ITEMS, K = 4, 6, 2

TRAIN = pd.DataFrame(
    [(1, 1, 5), (1, 2, 5), (1, 3, 1), (2, 1, 4), (2, 3, 2), (3, 2, 5), (3, 4, 4)],
    columns=["user_id_idx", "movie_id_idx", "rating"],
)
USERS = pd.DataFrame(
    {"user_id": [11, 12, 13], "user_id_idx": [1, 2, 3], "gender_F": [1, 0, 1], "gender_M": [0, 1, 0]}
)
MOVIES = pd.DataFrame(
    {
        "movie_id": [101, 102, 103, 104, 105],
        "title": ["A", "B", "C", "D", "E"],
        "movie_id_idx": [1, 2, 3, 4, 5],
        "genre_Action": [1, 1, 0, 0, 0],
        "genre_Drama": [0, 0, 1, 1, 1],
        "release_year": [0.0, 0.1, 0.2, 0.3, 0.4],
    }
)
METRICS = pd.DataFrame(
    {
        "test_ndcg@10": {"item_knn": 0.1, "svd+lightgbm": 0.2},
        "val_ndcg@10": {"item_knn": 0.3, "svd+lightgbm": np.nan},
    }
)


def build_system() -> System:
    store = FeatureStore().fit(TRAIN, USERS, MOVIES, N_USERS, N_ITEMS)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, len(store.feature_names))).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int32)
    data = RankerData(
        x[:80], y[:80], np.full(8, 10), x[80:], y[80:], np.full(4, 10), store.feature_names
    )
    ranker = fit_ranker(data, {"n_estimators": 10, "num_leaves": 4, "min_child_samples": 2})
    return System(
        ItemKNN().fit(TRAIN, N_USERS, N_ITEMS),
        SVDEmbedding(dim=2, metric="dot", normalize_users=False).fit(TRAIN, N_USERS, N_ITEMS),
        store,
        ranker,
    )


def build_seen() -> sparse.csr_matrix:
    rows, cols = TRAIN["user_id_idx"], TRAIN["movie_id_idx"]
    return sparse.csr_matrix((np.ones(len(TRAIN), bool), (rows, cols)), shape=(N_USERS, N_ITEMS))


def write_gold(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    USERS.to_parquet(path / "users.parquet", index=False)
    MOVIES.to_parquet(path / "movies.parquet", index=False)
    return path


def write_artifacts(root: Path, system: System | None = None) -> tuple[Path, Path]:
    """Write a gold folder and an artifacts folder under `root`; returns (artifacts, gold)."""
    gold = write_gold(root / "gold")
    out = root / "artifacts"
    artifacts.save_artifacts(
        system or build_system(), METRICS, build_seen(), USERS, MOVIES,
        gold_dir=gold, artifacts_dir=out, k=K,
    )
    return out, gold
