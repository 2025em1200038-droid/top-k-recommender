"""Evaluation: ranking metrics, per-request latency and a drift report.

    python3 -m src.evaluate            # score the saved artifacts on validation and test
    python3 -m src.evaluate --drift    # how user behaviour shifts between train and later periods
"""

import argparse
import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse, stats

from src.data_pipeline import load_gold, run
from src.utils import RELEVANT_RATING, mask_scores, top_k

logger = logging.getLogger(__name__)


def ranking_metrics(
    ranked: np.ndarray, relevant: np.ndarray, ks: tuple[int, ...]
) -> dict[str, np.ndarray]:
    """Per-user Precision@K, Recall@K and NDCG@K.

    ranked: (users, max_k) item indices, best first. relevant: (users, items) boolean matrix.
    Users with no relevant items get NaN so callers can exclude them from the mean.
    """
    hits = np.take_along_axis(relevant, ranked, axis=1)
    n_rel = relevant.sum(axis=1)
    discounts = 1.0 / np.log2(np.arange(2, ranked.shape[1] + 2))
    out: dict[str, np.ndarray] = {}
    for k in ks:
        h = hits[:, :k]
        n_hit = h.sum(axis=1)
        dcg = (h * discounts[:k]).sum(axis=1)
        ideal_hits = np.minimum(n_rel, k)
        idcg = np.cumsum(discounts[:k])[np.maximum(ideal_hits, 1) - 1]
        has_rel = n_rel > 0
        out[f"precision@{k}"] = np.where(has_rel, n_hit / k, np.nan)
        out[f"recall@{k}"] = np.where(has_rel, n_hit / np.maximum(n_rel, 1), np.nan)
        out[f"ndcg@{k}"] = np.where(has_rel, dcg / idcg, np.nan)
    return out


KS = (10, 20, 100)


def interaction_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> sparse.csr_matrix:
    data = np.ones(len(df), dtype=bool)
    return sparse.csr_matrix(
        (data, (df["user_id_idx"].to_numpy(), df["movie_id_idx"].to_numpy())),
        shape=(n_users, n_items),
    )


def relevant_items(df: pd.DataFrame, n_users: int, n_items: int) -> sparse.csr_matrix:
    """Liked items; rows with unseen-in-train movies (index 0) cannot be retrieved and are dropped."""
    liked = df[(df["rating"] >= RELEVANT_RATING) & (df["movie_id_idx"] > 0)]
    return interaction_matrix(liked, n_users, n_items)


def evaluate_model(
    model, relevant: sparse.csr_matrix, seen: sparse.csr_matrix, ks=KS, batch_size: int = 1000
) -> dict[str, float]:
    users = np.flatnonzero(relevant.getnnz(axis=1) > 0)
    per_user: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(users), batch_size):
        batch = users[start : start + batch_size]
        scores = mask_scores(model.scores(batch), seen[batch])
        ranked = top_k(scores, max(ks))
        for name, values in ranking_metrics(ranked, relevant[batch].toarray(), ks).items():
            per_user.setdefault(name, []).append(values)
    return {name: float(np.concatenate(v).mean()) for name, v in per_user.items()}


def measure_latency(
    model, users: np.ndarray, seen: sparse.csr_matrix, k: int = 100, warmup: int = 20
) -> dict[str, float]:
    """Warm, single-user request latency (score, mask, top-k) in milliseconds."""
    def request(u: int) -> None:
        top_k(mask_scores(model.scores(np.array([u])), seen[[u]]), k)

    for u in users[:warmup]:
        request(u)
    times = []
    for u in users:
        start = time.perf_counter()
        request(u)
        times.append((time.perf_counter() - start) * 1000)
    p50, p95, p99 = np.percentile(times, [50, 95, 99])
    return {"latency_p50_ms": p50, "latency_p95_ms": p95, "latency_p99_ms": p99}


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    users: pd.DataFrame
    movies: pd.DataFrame
    n_users: int
    n_items: int

    def seen(self, split: str) -> sparse.csr_matrix:
        """Items a user has already interacted with before `split` is predicted."""
        history = [self.train] if split == "val" else [self.train, self.val]
        return interaction_matrix(pd.concat(history), self.n_users, self.n_items)

    def relevant(self, split: str) -> sparse.csr_matrix:
        return relevant_items(getattr(self, split), self.n_users, self.n_items)


def load_splits() -> Splits:
    run()
    tables = load_gold()
    return Splits(
        train=tables["ratings_train"],
        val=tables["ratings_val"],
        test=tables["ratings_test"],
        users=tables["users"],
        movies=tables["movies"],
        n_users=int(tables["users"]["user_id_idx"].max()) + 1,
        n_items=int(tables["movies"]["movie_id_idx"].max()) + 1,
    )


def evaluate_system(
    sp: Splits, system: System, latency_users: int = 500, seed: int = 0
) -> pd.DataFrame:
    models = {
        "item_knn": lambda split: system.knn,
        "svd+lightgbm": lambda split: system.two_tier(sp.seen(split)),
    }
    rng = np.random.default_rng(seed)
    sample = rng.choice(
        np.flatnonzero(sp.relevant("test").getnnz(axis=1) > 0), size=latency_users, replace=False
    )

    rows = {}
    for name, make in models.items():
        row = {}
        for split in ("val", "test"):
            if name == "svd+lightgbm" and split == "val":
                continue  # the ranker is trained on val labels, so val scores would be optimistic
            metrics = evaluate_model(make(split), sp.relevant(split), sp.seen(split))
            row.update({f"{split}_{k}": v for k, v in metrics.items()})
        row.update(measure_latency(make("test"), sample, sp.seen("test")))
        rows[name] = row
    return pd.DataFrame(rows).T


# ----- drift -----------------------------------------------------------------


PSI_WARN = 0.10  # below: stable


PSI_ALERT = 0.25  # at or above: significant shift


KS_ALERT = 0.10  # KS statistic; p-values are ~0 on large samples, so use effect size


BINS = 10


def psi(reference, current, bins: int = BINS, eps: float = 1e-4) -> float:
    """Population Stability Index of `current` against `reference`.

    Continuous values use reference quantile bins; features with few distinct values (flags,
    one-hot columns) use one bin per reference value plus an "unseen" bin.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("psi needs at least one finite value in both samples")

    values = np.unique(ref)
    if len(values) <= bins:
        p = np.append([(ref == v).mean() for v in values], 0.0)
        q = np.append([(cur == v).mean() for v in values], (~np.isin(cur, values)).mean())
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
        p = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1) / len(ref)
        q = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1) / len(cur)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def drift_status(value: float) -> str:
    return "alert" if value >= PSI_ALERT else "warn" if value >= PSI_WARN else "ok"


def drift_table(reference: np.ndarray, current: np.ndarray, names: list[str]) -> pd.DataFrame:
    """PSI, KS statistic and status per column, most drifted first."""
    rows = []
    for i, name in enumerate(names):
        value = psi(reference[:, i], current[:, i])
        ks = stats.ks_2samp(reference[:, i], current[:, i]).statistic
        rows.append({"feature": name, "psi": value, "ks": ks, "status": drift_status(value),
                     "ks_alert": ks >= KS_ALERT})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)



def behaviour_matrix(ratings: pd.DataFrame, sp: Splits) -> tuple[np.ndarray, list[str]]:
    """Per rating: its value and the traits (popularity, mean rating, year, genres) of the movie rated."""
    ratings = ratings[ratings["movie_id_idx"] > 0]
    item = ratings["movie_id_idx"].to_numpy()
    train_item = sp.train["movie_id_idx"].to_numpy()
    counts = np.bincount(train_item, minlength=sp.n_items)
    mean = np.bincount(train_item, weights=sp.train["rating"].to_numpy(float), minlength=sp.n_items)
    mean = mean / np.maximum(counts, 1)

    movies = sp.movies[sp.movies["movie_id_idx"] > 0].set_index("movie_id_idx")
    genre_cols = [c for c in movies.columns if c.startswith("genre_")]
    year = np.zeros(sp.n_items)
    year[movies.index] = movies["release_year"]
    genres = np.zeros((sp.n_items, len(genre_cols)))
    genres[movies.index] = movies[genre_cols].to_numpy(float)

    matrix = np.column_stack(
        [ratings["rating"].to_numpy(float), np.log1p(counts[item]), mean[item], year[item], genres[item]]
    )
    return matrix, ["rating", "log_item_popularity", "item_mean_rating", "release_year", *genre_cols]


def drift_report(sp: Splits) -> dict[str, pd.DataFrame]:
    """PSI and KS of what users rate in validation and test against the training period."""
    reference, names = behaviour_matrix(sp.train, sp)
    pd.set_option("display.width", 160, "display.max_rows", None)
    tables = {}
    for split in ("val", "test"):
        current, _ = behaviour_matrix(getattr(sp, split), sp)
        table = drift_table(reference, current, names)
        tables[split] = table
        print(f"\nbehaviour, train vs {split} (top 8 of {len(names)}; PSI warn >= {PSI_WARN}, alert >= {PSI_ALERT}):")
        print(table.head(8).round(4).to_string(index=False))
        print(f"statuses: {table['status'].value_counts().to_dict()}")
    return tables


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--drift", action="store_true", help="print the behaviour drift report instead")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sp = load_splits()
    if args.drift:
        drift_report(sp)
        return
    from src.train import load_artifacts  # imported here because train imports this module

    table = evaluate_system(sp, load_artifacts())
    pd.set_option("display.width", 250, "display.max_columns", None)
    print(table.round(4).T.to_string())


if __name__ == "__main__":
    main()
