"""Pretty-print the anomaly_events table.

Examples:
    PYTHONPATH=. python tools/show_events.py
    PYTHONPATH=. python tools/show_events.py --record 208
    PYTHONPATH=. python tools/show_events.py --starts-only
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="logs/events.db")
    p.add_argument("--record", default=None,
                   help="filter to a specific record/patient_id")
    p.add_argument("--starts-only", action="store_true",
                   help="only show anomaly_start rows")
    p.add_argument("--limit", type=int, default=200)
    args = p.parse_args()

    if not Path(args.db).exists():
        print(f"No DB at {args.db}. Run the pipeline first.")
        return 1

    where = []
    params: list = []
    if args.record:
        where.append("patient_id = ?"); params.append(args.record)
    if args.starts_only:
        where.append("event_type = 'anomaly_start'")
    sql = (
        "SELECT id, event_type, patient_id, "
        "ROUND(record_offset_seconds, 2) AS t_rec, "
        "ROUND(residual, 4) AS res, "
        "ROUND(threshold, 4) AS thr, "
        "ts_utc "
        "FROM anomaly_events"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY id LIMIT {args.limit}"

    con = sqlite3.connect(args.db)
    rows = con.execute(sql, params).fetchall()
    if not rows:
        print("No matching events.")
        return 0

    cols = ["id", "event", "rec", "t_rec(s)", "residual", "threshold", "ts_utc"]
    widths = [max(len(c), max(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)))

    n_starts = con.execute(
        "SELECT COUNT(*) FROM anomaly_events WHERE event_type='anomaly_start'"
        + (" AND patient_id=?" if args.record else ""),
        ([args.record] if args.record else []),
    ).fetchone()[0]
    print(f"\nTotal anomaly_starts: {n_starts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
