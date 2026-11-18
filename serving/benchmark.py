"""Latency and throughput benchmark, in-process by default or over HTTP with --url.

    python3 -m serving.benchmark                                  # model only, no network
    python3 -m serving.benchmark --profile                        # plus where the time goes
    python3 -m serving.benchmark --url http://localhost:8000 --concurrency 8
"""

import argparse
import cProfile
import pstats
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from serving.utils import load_recommender
from src.utils import ARTIFACTS_DIR, GOLD_DIR


def percentiles(times_ms: list[float]) -> dict[str, float]:
    p50, p95, p99 = np.percentile(times_ms, [50, 95, 99])
    return {"p50_ms": p50, "p95_ms": p95, "p99_ms": p99, "mean_ms": float(np.mean(times_ms)),
            "max_ms": float(np.max(times_ms))}


def bench_in_process(
    users: list[int], k: int, batch_sizes: list[int], artifacts_dir: Path, gold_dir: Path,
    profile: bool,
):
    rec = load_recommender(artifacts_dir, gold_dir)
    for u in users[:20]:  # warm-up
        rec.recommend([u], k)

    profiler = cProfile.Profile() if profile else None
    times = []
    for u in users:
        start = time.perf_counter()
        if profiler:
            profiler.enable()
        rec.recommend([u], k)
        if profiler:
            profiler.disable()
        times.append((time.perf_counter() - start) * 1000)
    if profiler:
        print("\nwhere single-request time goes (cumulative, top 12):")
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(12)

    throughput = {}
    for size in batch_sizes:
        batch = [int(u) for u in np.resize(users, size)]
        start = time.perf_counter()
        rec.recommend(batch, k)
        throughput[f"throughput_{size}_users_per_s"] = size / (time.perf_counter() - start)
    return times, throughput, 0


def bench_http(users: list[int], k: int, url: str, concurrency: int):
    with httpx.Client(
        base_url=url, timeout=10, limits=httpx.Limits(max_connections=concurrency)
    ) as client:
        client.get("/health").raise_for_status()

        def call(user: int) -> tuple[float, bool]:
            start = time.perf_counter()
            try:
                ok = client.post("/predict", json={"user_id": user, "k": k}).status_code == 200
            except httpx.HTTPError:
                ok = False
            return (time.perf_counter() - start) * 1000, ok

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(call, users[: max(20, 4 * concurrency)]))  # warm-up opens every connection
            started = time.perf_counter()
            results = list(pool.map(call, users))
            total = time.perf_counter() - started

    times = [t for t, _ in results]
    errors = sum(not ok for _, ok in results)
    return times, {"throughput_requests_per_s": len(users) / total}, errors


def main(argv: list[str] | None = None) -> pd.Series:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--requests", type=int, default=500, help="single-user requests to time")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[100, 1000, 6040],
                        help="user counts for the in-process batch throughput runs")
    parser.add_argument("--url", help="benchmark this running server instead of the model")
    parser.add_argument("--concurrency", type=int, default=1, help="parallel clients (HTTP only)")
    parser.add_argument("--profile", action="store_true", help="print a profile (in-process only)")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--gold-dir", type=Path, default=GOLD_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    all_users = pd.read_parquet(args.gold_dir / "users.parquet", columns=["user_id"])["user_id"]
    rng = np.random.default_rng(args.seed)
    users = [int(u) for u in rng.choice(all_users.to_numpy(), size=args.requests, replace=True)]

    if args.url:
        times, throughput, errors = bench_http(users, args.k, args.url, args.concurrency)
        mode = f"http {args.url}, {args.concurrency} client(s)"
    else:
        times, throughput, errors = bench_in_process(
            users, args.k, args.batch_sizes, args.artifacts_dir, args.gold_dir, args.profile
        )
        mode = "in-process"

    result = pd.Series({**percentiles(times), **throughput, "errors": errors})
    print(f"\nmode: {mode}; {args.requests} single-user requests, k={args.k}")
    print(result.round(2).to_string())
    return result


if __name__ == "__main__":
    main()
