"""Model service: a FastAPI app (GET /health, POST /predict) and a batch scoring command.

    python3 -m uvicorn serving.app:app --host 0.0.0.0 --port 8000
    python3 -m serving.app batch --input users.csv --output recs.csv --k 10

API status codes: 200 ok; 400 unknown user, unknown genre or k out of range; 422 malformed
payload; 500 any server-side failure. 400 and 422 bodies name the failing field and value.

Batch input (CSV or JSON Lines): `user_id` is required; `record_id` and `genre` are optional.
Batch output (CSV, JSON Lines or Parquet): one row per recommended movie, or one status row for a
record that could not be scored, so every input record appears in the output.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from serving.schemas import (
    BatchOptions,
    BatchRecord,
    ErrorResponse,
    FieldError,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    Recommendation,
)
from serving.utils import Recommender, load_recommender
from src.utils import (
    ARTIFACTS_DIR,
    GOLD_DIR,
    ArtifactError,
    BadRequestError,
    InvalidValueError,
    atomic_write,
)

logger = logging.getLogger("uvicorn.error")  # shows up in uvicorn's console output

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (400, 422, 500)
}
VALUE_LIMIT = 100
HTTP_ERROR_CODES = {400: "bad_request", 404: "not_found", 405: "method_not_allowed"}


def _safe(value, limit: int = VALUE_LIMIT):
    """A rejected value that is always valid, short JSON (NaN, huge numbers and blobs included)."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, int) and abs(value) <= 10**15:
        return value
    text = value if isinstance(value, str) else (
        str(value) if isinstance(value, (int, float)) else json.dumps(value, default=str)
    )
    return text if len(text) <= limit else text[:limit] + "..."


def _error(status: int, error: str, message: str | None = None, details=()) -> JSONResponse:
    body = ErrorResponse(error=error, message=message, details=list(details))
    return JSONResponse(status_code=status, content=body.model_dump())


def create_app(artifacts_dir: Path | None = None, gold_dir: Path | None = None) -> FastAPI:
    artifacts_dir = Path(artifacts_dir or os.environ.get("ARTIFACTS_DIR", ARTIFACTS_DIR))
    gold_dir = Path(gold_dir or os.environ.get("GOLD_DIR", GOLD_DIR))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.recommender = None
        app.state.load_error = None
        try:
            app.state.recommender = load_recommender(artifacts_dir, gold_dir)
            logger.info("model loaded from %s", artifacts_dir)
        except Exception as exc:  # keep the process up so /health can report the reason
            logger.exception("could not load model artifacts")
            app.state.load_error = f"{type(exc).__name__}: {exc}"
        yield

    app = FastAPI(title="Top-K movie recommendations", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = []
        for err in exc.errors():
            loc = [str(p) for p in err["loc"][1:]]
            no_value = err["type"] in ("missing", "json_invalid")
            details.append(
                FieldError(
                    field="body" if err["type"] == "json_invalid" or not loc else ".".join(loc),
                    message=err["msg"],
                    value=None if no_value else _safe(err.get("input")),
                )
            )
        return _error(422, "invalid_request", "payload failed validation", details)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = HTTP_ERROR_CODES.get(exc.status_code, "http_error")
        return _error(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def server_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled error on %s", request.url.path, exc_info=exc)
        return _error(500, "internal_error", "internal server error")

    def not_loaded(request: Request) -> JSONResponse:
        reason = request.app.state.load_error or "model not loaded"
        return _error(500, "internal_error", f"model unavailable: {reason}")

    @app.get("/health", response_model=HealthResponse, responses=ERROR_RESPONSES)
    def health(request: Request):
        rec: Recommender | None = request.app.state.recommender
        if rec is None:
            return not_loaded(request)
        return HealthResponse(
            status="ok",
            model_version=rec.artifacts.manifest["created_at"],
            n_users=rec.n_users,
            genres=rec.genres,
        )

    @app.post("/predict", response_model=PredictResponse, responses=ERROR_RESPONSES)
    def predict(body: PredictRequest, request: Request):
        rec: Recommender | None = request.app.state.recommender
        if rec is None:
            return not_loaded(request)
        start = time.perf_counter()
        try:
            (result,) = rec.recommend([body.user_id], body.k, body.genre)
        except BadRequestError as exc:
            field = FieldError(field=exc.field, message=exc.message, value=_safe(exc.value))
            return _error(400, "bad_request", f"{exc.field}: {exc.message}", [field])
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "predict user_id=%d k=%d genre=%s items=%d degraded=%s latency_ms=%.1f",
            body.user_id, body.k, body.genre, len(result.items), result.degraded, latency_ms,
        )
        return PredictResponse(
            user_id=result.user_id,
            genre=body.genre,
            k=body.k,
            recommendations=[
                Recommendation(rank=r, movie_id=i.movie_id, title=i.title, score=i.score)
                for r, i in enumerate(result.items, start=1)
            ],
            degraded=result.degraded,
            latency_ms=round(latency_ms, 3),
            model_version=rec.artifacts.manifest["created_at"],
        )

    return app


app = create_app()

# ----- batch scoring ---------------------------------------------------------------------

batch_logger = logging.getLogger("serving.batch")

OUTPUT_COLUMNS = [
    "record_id", "user_id", "genre", "rank", "movie_id", "title", "score", "status", "detail",
]


STATUS_OK = "ok"


STATUS_OK_DEGRADED = "ok_degraded"  # reranker failed; retriever order used


STATUS_NO_CANDIDATES = "no_candidates"


STATUS_INVALID_USER = "invalid_user_id"


STATUS_UNKNOWN_USER = "unknown_user"


STATUS_INVALID_GENRE = "invalid_genre"


STATUS_INVALID_RECORD = "invalid_record"


FIELD_STATUS = {"user_id": STATUS_INVALID_USER, "genre": STATUS_INVALID_GENRE}


EXIT_OK, EXIT_FATAL, EXIT_INVALID_ROWS = 0, 1, 2


def read_input(path: Path) -> pd.DataFrame:
    """Read every field as text so malformed values can be reported per record."""
    if path.suffix == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    elif path.suffix in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True, dtype=False)
        df = df.astype(object).where(df.notna(), "").astype(str)
    else:
        raise ValueError(f"unsupported input format {path.suffix!r}; use .csv or .jsonl")
    df.columns = [c.strip() for c in df.columns]
    if "user_id" not in df.columns:
        raise ValueError(f"input needs a 'user_id' column, found {list(df.columns)}")
    return df


def write_output(df: pd.DataFrame, path: Path) -> None:
    writers = {
        ".csv": lambda p: df.to_csv(p, index=False),
        ".jsonl": lambda p: df.to_json(p, orient="records", lines=True),
        ".parquet": lambda p: df.to_parquet(p, index=False),
    }
    if path.suffix not in writers:
        raise ValueError(f"unsupported output format {path.suffix!r}; use .csv, .jsonl or .parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, writers[path.suffix])


def _describe(error: ValidationError) -> tuple[str, str]:
    """Status and a readable reason (field, message, rejected value) for a failed record."""
    first = error.errors()[0]
    status = FIELD_STATUS.get(str(first["loc"][0]), STATUS_INVALID_RECORD)
    parts = [f"{'.'.join(map(str, e['loc']))}: {e['msg']} (value {str(e['input'])[:50]!r})"
             for e in error.errors()]
    return status, "; ".join(parts)


def score_records(
    rec: Recommender, records: pd.DataFrame, k: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Score all records; invalid ones become status rows instead of aborting the run."""
    blank = pd.Series("", index=records.index)
    record_ids = records["record_id"].str.strip() if "record_id" in records.columns else blank
    record_ids = record_ids.where(record_ids != "", records.index.to_series().astype(str))
    users = records["user_id"]
    genres = records["genre"] if "genre" in records.columns else blank

    rows: list[list[dict]] = [[] for _ in range(len(records))]
    by_genre: dict[str | None, list[tuple[int, BatchRecord]]] = {}
    stats = {"invalid": 0, "degraded": 0}

    def reject(pos: int, status: str, detail: str, user_id=None, genre=None) -> None:
        rows[pos] = [{"record_id": record_ids.iloc[pos], "user_id": user_id, "genre": genre,
                      "status": status, "detail": detail}]
        stats["invalid"] += 1

    for pos in range(len(records)):
        raw = {"record_id": record_ids.iloc[pos], "user_id": users.iloc[pos], "genre": genres.iloc[pos]}
        try:
            record = BatchRecord.model_validate(raw)
        except ValidationError as exc:
            reject(pos, *_describe(exc))
        else:
            if not rec.has_user(record.user_id):
                fits = record.user_id < 2**63  # larger ids appear only in `detail`
                reject(pos, STATUS_UNKNOWN_USER, f"user_id: unknown user id (value {record.user_id})",
                       record.user_id if fits else None, record.genre)
            elif record.genre is not None and record.genre not in rec.genres:
                reject(pos, STATUS_INVALID_GENRE, f"genre: unknown genre (value {record.genre!r})",
                       record.user_id, record.genre)
            else:
                by_genre.setdefault(record.genre, []).append((pos, record))

    for genre, items in by_genre.items():
        results = rec.recommend([r.user_id for _, r in items], k, genre)
        for (pos, record), result in zip(items, results):
            base = {"record_id": record.record_id, "user_id": record.user_id, "genre": genre}
            stats["degraded"] += result.degraded
            if result.items:
                rows[pos] = [
                    {**base, "rank": r, "movie_id": i.movie_id, "title": i.title, "score": i.score,
                     "status": STATUS_OK_DEGRADED if result.degraded else STATUS_OK}
                    for r, i in enumerate(result.items, start=1)
                ]
            else:
                rows[pos] = [{**base, "status": STATUS_NO_CANDIDATES}]

    flat = [row for group in rows for row in group]
    # Built separately so large ids are not squeezed through float64.
    user_ids = pd.array([row.get("user_id") for row in flat], dtype="Int64")
    out = pd.DataFrame([{**row, "user_id": None} for row in flat], columns=OUTPUT_COLUMNS)
    out["user_id"] = user_ids
    return out.astype({"rank": "Int64", "movie_id": "Int64"}), stats



def batch_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10, help="movies per record (default 10)")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--gold-dir", type=Path, default=GOLD_DIR)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    parser.add_argument("--strict", action="store_true", help="exit 2 if any record was invalid")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    started = time.perf_counter()
    try:
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")
        batch_logger.info("reading %s", args.input)
        records = read_input(args.input)
        batch_logger.info("loading artifacts from %s and gold from %s", args.artifacts_dir, args.gold_dir)
        rec = load_recommender(args.artifacts_dir, args.gold_dir)
        BatchOptions(k=args.k)  # range check; the model's own limit is checked next
        rec.validate(args.k, None)
        if "record_id" not in records.columns:
            batch_logger.warning("no record_id column; using the row number as record id")

        batch_logger.info("scoring %d records with k=%d", len(records), args.k)
        scoring_started = time.perf_counter()
        result, stats = score_records(rec, records, args.k)
        scoring_seconds = time.perf_counter() - scoring_started
        write_output(result, args.output)
    except (OSError, ValueError, ArtifactError, InvalidValueError, ValidationError) as exc:
        batch_logger.error("batch failed: %s", exc)
        return EXIT_FATAL

    invalid = stats["invalid"]
    elapsed = time.perf_counter() - started
    batch_logger.info(
        "done: %d records, %d invalid or unknown, %d output rows -> %s",
        len(records), invalid, len(result), args.output,
    )
    batch_logger.info(
        "throughput: %.0f records/s scoring only (%.2fs), %.0f records/s end to end (%.2fs)",
        len(records) / max(scoring_seconds, 1e-9), scoring_seconds,
        len(records) / max(elapsed, 1e-9), elapsed,
    )
    if stats["degraded"]:
        batch_logger.warning("%d records used the retriever-only fallback", stats["degraded"])
    if invalid:
        batch_logger.warning("%d records could not be scored; see the status column", invalid)
        if args.strict:
            return EXIT_INVALID_ROWS
    return EXIT_OK


def cli(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] != ["batch"]:
        print(__doc__)
        print("usage: python3 -m serving.app batch --input FILE --output FILE [--k N]")
        return 2
    return batch_main(argv[1:])


if __name__ == "__main__":
    sys.exit(cli())
