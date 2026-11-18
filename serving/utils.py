"""Inference shared by the online API, the batch script and the benchmark."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.train import Artifacts, load_artifacts
from src.utils import ARTIFACTS_DIR, GOLD_DIR, BadRequestError, genre_mask, mask_scores, top_k

logger = logging.getLogger("uvicorn.error")  # shows up in uvicorn's console output


CHUNK_SIZE = 1000


@dataclass(frozen=True)
class Item:
    movie_id: int
    title: str
    score: float


@dataclass(frozen=True)
class Result:
    user_id: int
    items: list[Item]
    degraded: bool = False  # True when the reranker failed and retriever order was used


class Recommender:
    """Top-K unseen movies per known user, optionally within a genre."""

    def __init__(self, artifacts: Artifacts):
        self.artifacts = artifacts
        self.max_k = artifacts.manifest["config"]["retrieval_k"]
        movies = artifacts.movies
        self.genres = sorted(c.removeprefix("genre_") for c in movies.columns if c.startswith("genre_"))
        self._n_items = artifacts.seen.shape[1]
        self._two_tier = artifacts.two_tier()
        self._user_idx = dict(zip(artifacts.users["user_id"], artifacts.users["user_id_idx"]))
        self._movie_ids = np.zeros(self._n_items, dtype=np.int64)
        self._titles = np.full(self._n_items, "", dtype=object)
        self._movie_ids[movies["movie_id_idx"]] = movies["movie_id"]
        self._titles[movies["movie_id_idx"]] = movies["title"]
        self._genre_masks: dict[str, np.ndarray] = {}

    @property
    def n_users(self) -> int:
        return len(self._user_idx)

    def has_user(self, user_id: int) -> bool:
        return user_id in self._user_idx

    def validate(self, k: int, genre: str | None) -> None:
        if not 1 <= k <= self.max_k:
            raise BadRequestError("k", k, f"k must be between 1 and {self.max_k}")
        if genre is not None and genre not in self.genres:
            raise BadRequestError("genre", genre, f"unknown genre; valid genres: {self.genres}")

    def _mask(self, genre: str) -> np.ndarray:
        if genre not in self._genre_masks:
            self._genre_masks[genre] = genre_mask(self.artifacts.movies, genre, self._n_items)
        return self._genre_masks[genre]

    def _score(self, idx: np.ndarray, allowed: np.ndarray | None) -> tuple[np.ndarray, bool]:
        """Reranked scores; if the reranker fails, fall back to the retriever's own ranking."""
        try:
            return self._two_tier.scores(idx, allowed), False
        except Exception:
            logger.exception("reranker failed; falling back to retriever order")
            retrieved = self.artifacts.svd.scores(idx)
            return mask_scores(retrieved, self.artifacts.seen[idx], allowed), True

    def recommend(
        self, user_ids: Sequence[int], k: int = 10, genre: str | None = None
    ) -> list[Result]:
        """One Result per user id, in order. Raises BadRequestError for a bad k, genre or user."""
        self.validate(k, genre)
        unknown = [u for u in user_ids if u not in self._user_idx]
        if unknown:
            raise BadRequestError("user_id", unknown[0], "unknown user id")
        allowed = self._mask(genre) if genre else None
        idx = np.array([self._user_idx[u] for u in user_ids], dtype=np.int64)

        results: list[Result] = []
        for start in range(0, len(idx), CHUNK_SIZE):
            scores, degraded = self._score(idx[start : start + CHUNK_SIZE], allowed)
            top = top_k(scores, k)
            top_scores = np.take_along_axis(scores, top, axis=1)
            for row in range(len(top)):
                keep = np.isfinite(top_scores[row])
                items = [
                    Item(int(self._movie_ids[i]), str(self._titles[i]), float(s))
                    for i, s in zip(top[row][keep], top_scores[row][keep])
                ]
                results.append(Result(int(user_ids[start + row]), items, degraded))
        return results


def load_recommender(
    artifacts_dir: Path = ARTIFACTS_DIR, gold_dir: Path = GOLD_DIR
) -> Recommender:
    return Recommender(load_artifacts(artifacts_dir, gold_dir))
