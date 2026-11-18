"""Data pipeline: bronze ingestion -> silver (clean, validate, leak-free split) -> gold (features).

Run `python3 -m src.data_pipeline`; each layer is built only if it is missing.
"""

import logging
import os
import shutil
import ssl
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import certifi
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import (
    BRONZE_DIR,
    CUTOFF_MONTHS,
    GOLD_DIR,
    SILVER_DIR,
    SPLIT_FRACTIONS,
    SPLITS,
    InvalidValueError,
    MissingColumnError,
    MissingValueError,
    ReferentialIntegrityError,
    atomic_dir,
)

logger = logging.getLogger(__name__)


# ----- feature transformers --------------------------------------------------


class IdIndexer(TransformerMixin, BaseEstimator):
    """Map ids to contiguous indices learned on train; unseen ids map to 0 (reserved)."""

    def fit(self, X, y=None):
        values = np.asarray(X).ravel()
        self.categories_ = np.sort(pd.unique(values))
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray(getattr(X, "columns", ["id"]), dtype=object)
        return self

    def transform(self, X):
        values = np.asarray(X).ravel()
        return (pd.Index(self.categories_).get_indexer(values) + 1).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        names = self.feature_names_in_ if input_features is None else input_features
        return np.array([f"{c}_idx" for c in names])


class GenreMultiHot(TransformerMixin, BaseEstimator):
    """Multi-hot encode a '|'-separated genre column; vocabulary is learned on train."""

    def __init__(self, sep: str = "|"):
        self.sep = sep

    def _split(self, X) -> pd.Series:
        return pd.Series(np.asarray(X).ravel()).str.split(self.sep, regex=False)

    def fit(self, X, y=None):
        self.genres_ = sorted({g for gs in self._split(X) for g in gs})
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        index = {g: i for i, g in enumerate(self.genres_)}
        out = np.zeros((len(X), len(self.genres_)), dtype=np.int8)
        for row, gs in enumerate(self._split(X)):
            for g in gs:
                if g in index:
                    out[row, index[g]] = 1
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"genre_{g}" for g in self.genres_])


class YearExtractor(TransformerMixin, BaseEstimator):
    """Extract the trailing '(YYYY)' release year from a title; NaN when absent."""

    def fit(self, X, y=None):
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        years = pd.Series(np.asarray(X).ravel()).str.extract(r"\((\d{4})\)\s*$")[0]
        return years.astype(float).to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["release_year"])


# ----- entity registry -------------------------------------------------------


RATINGS_COLS = ["user_id", "movie_id", "rating", "timestamp"]


MOVIES_COLS = ["movie_id", "title", "genres"]


USERS_COLS = ["user_id", "gender", "age", "occupation", "zip_code"]


def _wrap(transformer: ColumnTransformer) -> Pipeline:
    return Pipeline([("features", transformer)]).set_output(transform="pandas")


def _users_pipeline() -> Pipeline:
    return _wrap(
        ColumnTransformer(
            [
                ("raw", "passthrough", ["user_id"]),
                ("idx", IdIndexer(), ["user_id"]),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 ["gender", "age", "occupation"]),
            ],
            verbose_feature_names_out=False,
        )
    )


def _movies_pipeline() -> Pipeline:
    year = Pipeline(
        [("extract", YearExtractor()), ("impute", SimpleImputer(strategy="median")),
         ("scale", StandardScaler())]
    )
    return _wrap(
        ColumnTransformer(
            [
                ("raw", "passthrough", ["movie_id", "title"]),
                ("idx", IdIndexer(), ["movie_id"]),
                ("genres", GenreMultiHot(), ["genres"]),
                ("year", year, ["title"]),
            ],
            verbose_feature_names_out=False,
        )
    )


def _ratings_pipeline() -> Pipeline:
    return _wrap(
        ColumnTransformer(
            [
                ("raw", "passthrough", ["user_id", "movie_id", "rating"]),
                ("user_idx", IdIndexer(), ["user_id"]),
                ("movie_idx", IdIndexer(), ["movie_id"]),
                ("time", StandardScaler(), ["timestamp"]),
            ],
            verbose_feature_names_out=False,
        )
    )


def _check_rating(df: pd.DataFrame) -> None:
    bad = ~df["rating"].between(1, 5)
    if bad.any():
        raise InvalidValueError(f"ratings: {int(bad.sum())} values outside 1-5")


@dataclass(frozen=True)
class EntitySpec:
    name: str
    filename: str
    columns: list[str]
    key: list[str]
    build_pipeline: Callable[[], Pipeline]
    # column -> (parent entity, parent column)
    refs: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Sort by these then keep the last duplicate key (e.g. latest rating wins).
    dedupe_order: list[str] | None = None
    validators: tuple[Callable[[pd.DataFrame], None], ...] = ()
    # Set for time-split fact tables: split per group, ordered by time_column.
    split_group: str | None = None
    time_column: str | None = None
    # (fact entity, column): fit this entity only on rows its train partition references.
    train_ref: tuple[str, str] | None = None

    @property
    def partitioned(self) -> bool:
        return self.split_group is not None


ENTITIES: dict[str, EntitySpec] = {
    spec.name: spec
    for spec in (
        EntitySpec("users", "users.dat", USERS_COLS, ["user_id"], _users_pipeline,
                   train_ref=("ratings", "user_id")),
        EntitySpec("movies", "movies.dat", MOVIES_COLS, ["movie_id"], _movies_pipeline,
                   train_ref=("ratings", "movie_id")),
        EntitySpec(
            "ratings", "ratings.dat", RATINGS_COLS, ["user_id", "movie_id"], _ratings_pipeline,
            refs={"user_id": ("users", "user_id"), "movie_id": ("movies", "movie_id")},
            dedupe_order=["timestamp"], validators=(_check_rating,),
            split_group="user_id", time_column="timestamp",
        ),
    )
}


def processing_order() -> list[EntitySpec]:
    """Parents before children so references can be validated."""
    return sorted(ENTITIES.values(), key=lambda s: (len(s.refs), s.name))


# ----- bronze ----------------------------------------------------------------


URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


DATASET_DIRNAME = "ml-1m"


REQUIRED_FILES = tuple(spec.filename for spec in ENTITIES.values())


def is_data_present(bronze_dir: Path = BRONZE_DIR) -> bool:
    dataset_dir = Path(bronze_dir) / DATASET_DIRNAME
    return all((dataset_dir / name).is_file() for name in REQUIRED_FILES)


def _download_atomic(url: str, dest: Path) -> None:
    """Download to a temp file in the same directory, then rename into place."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(
            url, timeout=60, context=ssl.create_default_context(cafile=certifi.where())
        ) as resp:
            shutil.copyfileobj(resp, out)
        os.replace(tmp_name, dest)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _extract_atomic(zip_path: Path, bronze_dir: Path) -> None:
    """Extract into a temp dir (zip-slip safe), then rename into place."""
    final_dir = bronze_dir / DATASET_DIRNAME
    tmp_dir = Path(tempfile.mkdtemp(dir=bronze_dir, prefix=".extract-"))
    try:
        root = tmp_dir.resolve()
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if not (root / member).resolve().is_relative_to(root):
                    raise ValueError(f"Unsafe path in zip: {member}")
            zf.extractall(root)
        extracted = root / DATASET_DIRNAME
        if not all((extracted / n).is_file() for n in REQUIRED_FILES):
            raise FileNotFoundError(f"Archive is missing required files: {REQUIRED_FILES}")
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(extracted, final_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def ensure_bronze(bronze_dir: Path = BRONZE_DIR, url: str = URL) -> Path:
    bronze_dir = Path(bronze_dir)
    dataset_dir = bronze_dir / DATASET_DIRNAME
    if is_data_present(bronze_dir):
        logger.info("MovieLens 1M already present at %s", dataset_dir)
        return dataset_dir

    zip_path = bronze_dir / f"{DATASET_DIRNAME}.zip"
    logger.info("Downloading %s", url)
    _download_atomic(url, zip_path)
    try:
        _extract_atomic(zip_path, bronze_dir)
    finally:
        zip_path.unlink(missing_ok=True)
    logger.info("MovieLens 1M ready at %s", dataset_dir)
    return dataset_dir


def _read(path: Path, names: list[str]) -> pd.DataFrame:
    # The "::" separator requires the python engine; files are latin-1 encoded.
    return pd.read_csv(path, sep="::", engine="python", names=names, encoding="latin-1")


def load_bronze(bronze_dir: Path = BRONZE_DIR) -> dict[str, pd.DataFrame]:
    return {
        spec.name: _read(Path(bronze_dir) / DATASET_DIRNAME / spec.filename, spec.columns)
        for spec in ENTITIES.values()
    }


def run(
    bronze_dir: Path = BRONZE_DIR, silver_dir: Path = SILVER_DIR, gold_dir: Path = GOLD_DIR
) -> None:
    """Build only what is missing: a present gold skips everything, a present silver skips bronze."""
    if is_gold_present(gold_dir):
        logger.info("gold already present at %s; nothing to do", gold_dir)
        return
    if is_silver_present(silver_dir):
        logger.info("silver already present at %s; skipping bronze and silver", silver_dir)
    else:
        ensure_bronze(bronze_dir)
        build_silver(load_bronze(bronze_dir), silver_dir)
    build_gold(silver_dir, gold_dir)


# ----- silver ----------------------------------------------------------------


def silver_files(spec: EntitySpec) -> list[str]:
    if spec.partitioned:
        return [f"{spec.name}_{s}.parquet" for s in SPLITS]
    return [f"{spec.name}.parquet"]


def is_silver_present(silver_dir: Path = SILVER_DIR) -> bool:
    return all((Path(silver_dir) / f).is_file() for s in ENTITIES.values() for f in silver_files(s))


def clean_entity(spec: EntitySpec, df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in spec.columns if c not in df.columns]
    if missing:
        raise MissingColumnError(f"{spec.name}: missing columns {missing}")

    df = df[spec.columns]
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        raise MissingValueError(f"{spec.name}: null values {nulls.to_dict()}")

    before = len(df)
    df = df.drop_duplicates()
    if spec.dedupe_order:
        df = df.sort_values(spec.dedupe_order, kind="stable")
    df = df.drop_duplicates(subset=spec.key, keep="last" if spec.dedupe_order else "first")
    logger.info("silver[%s]: dropped %d duplicate rows (%d -> %d)",
                spec.name, before - len(df), before, len(df))

    for validate in spec.validators:
        validate(df)
    return df.reset_index(drop=True)


def check_references(spec: EntitySpec, df: pd.DataFrame, cleaned: dict[str, pd.DataFrame]) -> None:
    for col, (parent, parent_col) in spec.refs.items():
        orphans = ~df[col].isin(cleaned[parent][parent_col])
        if orphans.any():
            sample = df.loc[orphans, col].unique()[:5].tolist()
            raise ReferentialIntegrityError(
                f"{spec.name}.{col}: {int(orphans.sum())} rows reference missing "
                f"{parent}.{parent_col} (e.g. {sample})"
            )


def apply_cutoff(spec: EntitySpec, df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    now = now or pd.Timestamp.now()
    cutoff = int((now - pd.DateOffset(months=CUTOFF_MONTHS)).timestamp())
    keep = df[spec.time_column] <= cutoff
    logger.info("silver[%s]: cutoff %s dropped %d rows newer than cutoff",
                spec.name, pd.to_datetime(cutoff, unit="s").date(), int((~keep).sum()))
    return df[keep].reset_index(drop=True)


def split_by_group_time(spec: EntitySpec, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per group, oldest 70% -> train, next 15% -> val, newest 15% -> test."""
    df = df.sort_values([spec.split_group, spec.time_column, *spec.key], kind="stable")
    grouped = df.groupby(spec.split_group, sort=False)
    frac = grouped.cumcount() / grouped[spec.split_group].transform("size")
    bounds = np.cumsum(SPLIT_FRACTIONS)
    split_id = np.searchsorted(bounds, frac.to_numpy(), side="right").clip(0, len(SPLITS) - 1)
    return {
        name: df[split_id == i].reset_index(drop=True) for i, name in enumerate(SPLITS)
    }


def build_silver(raw: dict[str, pd.DataFrame], silver_dir: Path = SILVER_DIR) -> None:
    cleaned: dict[str, pd.DataFrame] = {}
    for spec in processing_order():
        df = clean_entity(spec, raw[spec.name])
        check_references(spec, df, cleaned)
        if spec.partitioned:
            df = apply_cutoff(spec, df)
        cleaned[spec.name] = df

    with atomic_dir(silver_dir) as tmp:
        for spec in processing_order():
            df = cleaned[spec.name]
            if spec.partitioned:
                for split, part in split_by_group_time(spec, df).items():
                    logger.info("silver[%s]: %s = %d rows", spec.name, split, len(part))
                    part.to_parquet(tmp / f"{spec.name}_{split}.parquet", index=False)
            else:
                df.to_parquet(tmp / f"{spec.name}.parquet", index=False)
    logger.info("silver layer ready at %s", silver_dir)


def load_silver(silver_dir: Path = SILVER_DIR) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame]]:
    out: dict = {}
    for spec in ENTITIES.values():
        if spec.partitioned:
            out[spec.name] = {
                s: pd.read_parquet(Path(silver_dir) / f"{spec.name}_{s}.parquet") for s in SPLITS
            }
        else:
            out[spec.name] = pd.read_parquet(Path(silver_dir) / f"{spec.name}.parquet")
    return out


# ----- gold ------------------------------------------------------------------


PIPELINES_FILE = "pipelines.joblib"


def gold_files() -> list[str]:
    return [*(f for spec in ENTITIES.values() for f in silver_files(spec)), PIPELINES_FILE]


def is_gold_present(gold_dir: Path = GOLD_DIR) -> bool:
    return all((Path(gold_dir) / f).is_file() for f in gold_files())


def _fit_frame(spec: EntitySpec, silver: dict) -> pd.DataFrame:
    """Train-only rows used to fit this entity's pipeline."""
    data = silver[spec.name]
    if spec.partitioned:
        return data["train"]
    if spec.train_ref is None:
        return data
    fact_name, col = spec.train_ref
    parent_col = ENTITIES[fact_name].refs[col][1]
    return data[data[parent_col].isin(silver[fact_name]["train"][col])]


def build_gold(silver_dir: Path = SILVER_DIR, gold_dir: Path = GOLD_DIR) -> None:
    silver = load_silver(silver_dir)
    pipelines = {}
    tables: dict[str, pd.DataFrame] = {}

    for spec in processing_order():
        pipe = spec.build_pipeline()
        fit_df = _fit_frame(spec, silver)
        pipe.fit(fit_df)
        pipelines[spec.name] = pipe
        logger.info("gold[%s]: fitted on %d train rows", spec.name, len(fit_df))

        if spec.partitioned:
            for split in SPLITS:
                tables[f"{spec.name}_{split}"] = pipe.transform(silver[spec.name][split])
        else:
            tables[spec.name] = pipe.transform(silver[spec.name])

    with atomic_dir(gold_dir) as tmp:
        for name, df in tables.items():
            df.to_parquet(tmp / f"{name}.parquet", index=False)
        joblib.dump(pipelines, tmp / PIPELINES_FILE)
    logger.info("gold layer ready at %s", gold_dir)


def load_gold(gold_dir: Path = GOLD_DIR) -> dict[str, pd.DataFrame]:
    """Every gold table by name, for example `ratings_train`, `users`, `movies`."""
    return {p.stem: pd.read_parquet(p) for p in sorted(Path(gold_dir).glob("*.parquet"))}


# ----- orchestration ---------------------------------------------------------


def print_stats(bronze_dir: Path = BRONZE_DIR) -> None:
    """Shape, memory, nulls, duplicates, dtypes and a summary for each raw table."""
    frames = load_bronze(bronze_dir)
    for name, df in frames.items():
        print(f"\n===== {name} =====")
        print(f"shape: {df.shape}, memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        print(f"duplicated rows: {df.duplicated().sum()}")
        print("nulls per column:\n" + df.isna().sum().to_string())
        print("dtypes:\n" + df.dtypes.to_string())
        print("describe:\n" + df.describe(include="all").T.to_string())

    ratings, movies, users = frames["ratings"], frames["movies"], frames["users"]
    n_users, n_items = ratings["user_id"].nunique(), ratings["movie_id"].nunique()
    per_user = ratings.groupby("user_id").size()
    print("\n===== interaction summary =====")
    print(f"users with ratings: {n_users} / {len(users)}")
    print(f"movies with ratings: {n_items} / {len(movies)}")
    print(f"density: {len(ratings) / (n_users * n_items):.4%}")
    print(f"ratings per user (min/median/max): {per_user.min()}/{per_user.median()}/{per_user.max()}")
    print(f"rating distribution:\n{ratings['rating'].value_counts().sort_index().to_string()}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
    if is_data_present():
        print_stats()


if __name__ == "__main__":
    # Re-import so fitted objects are pickled under `src.data_pipeline`, not `__main__`.
    from src.data_pipeline import main as _main

    _main()
