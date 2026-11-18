# Machine Learning Model Engineering Assignment - Top-K Movie Recommendation

Two-tier recommender on MovieLens 1M. Given a user, optionally with a genre, it returns the top-K movies the user has not rated.

1. **Retrieval:** truncated-SVD embeddings, exact top-100 candidates per user (genre filter applied before ranking).
2. **Reranking:** LightGBM LambdaRank over the 100 candidates.
3. **Baseline:** interpretable item-based kNN ("because you rated X"), kept for comparison.

An online API and a batch command serve it from the saved artifacts.

## Quick start

Requires Python 3.14. Run from the project directory.

Windows (PowerShell):

```text
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.data_pipeline
python -m src.train
python -m uvicorn serving.app:app
```

macOS / Linux:

```text
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_pipeline
python -m src.train
python -m uvicorn serving.app:app
```

The data pipeline downloads MovieLens 1M into `data/` (internet access needed on first run) and the train step writes `artifacts/`; both are one-time. After that, only activate the venv and run the last command. To use another port, add `--port 8011` to the uvicorn command and change the URLs below. The API (FastAPI served by uvicorn, not Flask) is at <http://localhost:8000>; open `/docs` for interactive documentation. Stop with Ctrl+C.

### Run the tests

With the venv active (no server needed):

```text
python -m pytest tests
```

All 35 tests should pass.

### Check the server is working

With the server running, open a second terminal.

macOS / Linux:

```text
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"user_id": 1, "genre": "Comedy", "k": 5}'
```

Windows PowerShell (`curl` is an alias there, so use `curl.exe` and escape the inner quotes, or use `Invoke-RestMethod`):

```text
curl.exe http://localhost:8000/health
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{\"user_id\": 1, \"genre\": \"Comedy\", \"k\": 5}'

Invoke-RestMethod -Method Post http://localhost:8000/predict -ContentType application/json -Body '{"user_id": 1, "genre": "Comedy", "k": 5}'
```

`/health` returns `{"status": "ok", ...}`. `/predict` returns `recommendations`, each with `rank`, `movie_id`, `title` and `score`. `genre` is optional and `k` defaults to 10.

## API behaviour

Status codes: 200 ok; 400 unknown user, unknown genre or `k` out of range; 422 malformed payload; 500 server error. 400 and 422 bodies name the failing field and value. If the reranker fails, the response is still 200 with `degraded: true`.

## Batch mode

Input is CSV or JSON Lines with `user_id` (required), `record_id` and `genre` (optional). Every record appears in the output with a `status` and, if it was rejected, a `detail` naming the field and value.

```text
python -m serving.app batch --input users.csv --output out.csv --k 10
```

Add `--overwrite` to replace an existing output file and `--strict` to exit with code 2 if any record was invalid.

## Layout

```text
.
├── README.md  requirements.txt
├── .venv/                   virtual environment (local, not shared)
├── data/                    created by src.data_pipeline: bronze, silver, gold
├── artifacts/               created by src.train: models and manifest
├── src/
│   ├── data_pipeline.py     ingestion, cleaning, leak-free split, features
│   ├── train.py             models, tuning, export and loading
│   ├── evaluate.py          metrics, latency, drift
│   └── utils.py             constants, errors, file and score helpers
├── serving/
│   ├── app.py               FastAPI service and batch command
│   ├── schemas.py           pydantic schemas (HTTP and batch)
│   ├── benchmark.py         latency and throughput profiling
│   └── utils.py             Recommender shared by the above
└── tests/                   API and schema tests
```

## Commands

Model development commands require Python 3.14 and the dependencies in `requirements.txt`:

```text
python -m src.data_pipeline          # bronze -> silver -> gold, skips layers that already exist
python -m src.train [--force]        # fit, evaluate and write artifacts/
python -m src.train --tune all       # grid searches on the validation split
python -m src.evaluate [--drift]     # metrics and latency of the saved artifacts, or the drift report
python -m uvicorn serving.app:app    # online API: GET /health, POST /predict
python -m serving.app batch --input FILE --output FILE
python -m serving.benchmark [--url http://127.0.0.1:8000]
python -m pytest tests
```

## Key settings

`src/utils.py` holds `RELEVANT_RATING` (4, so ratings of 4 and 5 count as liked) and `RETRIEVAL_K`; the tuned model parameters are at the top of `src/train.py`.
