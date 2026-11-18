"""Shared constants, errors, file helpers and score-matrix helpers."""

import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

BRONZE_DIR = Path("data/bronze")
SILVER_DIR = Path("data/silver")
GOLD_DIR = Path("data/gold")
ARTIFACTS_DIR = Path("artifacts")

SPLITS = ("train", "val", "test")
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)

# Ratings newer than (now - CUTOFF_MONTHS) are dropped as a safeguard against bad/future timestamps.
CUTOFF_MONTHS = 6

# Minimum rating counted as relevant; ratings of 4 and 5 are positive feedback.
RELEVANT_RATING = 4
RETRIEVAL_K = 100


# ----- errors ----------------------------------------------------------------


class DataError(Exception):
    """Base class for data validation failures."""


class MissingColumnError(DataError):
    pass


class MissingValueError(DataError):
    pass


class ReferentialIntegrityError(DataError):
    pass


class InvalidValueError(DataError):
    pass


class ArtifactError(Exception):
    """Missing, corrupt or inconsistent model artifacts."""


class BadRequestError(InvalidValueError):
    """A request value the service cannot serve; carries the failing field and value."""

    def __init__(self, field: str, value, message: str):
        super().__init__(f"{field}: {message}")
        self.field, self.value, self.message = field, value, message


# ----- file helpers ----------------------------------------------------------


@contextmanager
def atomic_dir(final_dir: Path) -> Iterator[Path]:
    """Yield a temp dir beside final_dir; on success swap it into place, on failure discard it."""
    final_dir = Path(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=final_dir.parent, prefix=f".{final_dir.name}-tmp-"))
    try:
        yield tmp_dir
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(tmp_dir, final_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    """Write via a hidden temp file beside `path`, then rename, so readers never see a partial file."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.part")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# ----- score matrices --------------------------------------------------------


def genre_mask(movies: pd.DataFrame, genre: str, n_items: int) -> np.ndarray:
    """Boolean mask over item indices for movies in `genre` (gold movies table)."""
    col = f"genre_{genre}"
    if col not in movies.columns:
        raise InvalidValueError(f"unknown genre: {genre}")
    mask = np.zeros(n_items, dtype=bool)
    mask[movies.loc[movies[col] == 1, "movie_id_idx"].to_numpy()] = True
    mask[0] = False
    return mask


def mask_scores(
    scores: np.ndarray, seen: sparse.csr_matrix | None = None, allowed: np.ndarray | None = None
) -> np.ndarray:
    """Set scores of seen items, the reserved index 0 and non-allowed items to -inf (in place)."""
    scores[:, 0] = -np.inf
    if allowed is not None:
        scores[:, ~allowed] = -np.inf
    if seen is not None:
        coo = seen.tocoo()
        scores[coo.row, coo.col] = -np.inf
    return scores


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, best first."""
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)
