"""Print one line per Locust run from its *_stats.csv: requests, RPS, p50/p95/p99, failures.

python loadtest/summary.py loadtest/results/baseline loadtest/results/omp1 ...
"""

import csv
import sys


def summarise(prefix: str) -> str:
    with open(f"{prefix}_stats.csv") as f:
        row = next(r for r in csv.DictReader(f) if r["Name"] == "/predict")
    return (
        f"{prefix.rsplit('/', 1)[-1]:24s} reqs={int(row['Request Count']):6d} "
        f"rps={float(row['Requests/s']):7.1f}  p50={row['50%']:>4s} ms  "
        f"p95={row['95%']:>4s} ms  p99={row['99%']:>4s} ms  fails={row['Failure Count']}"
    )


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(summarise(p))
