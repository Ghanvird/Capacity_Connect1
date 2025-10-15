from __future__ import annotations
import datetime as dt
import pandas as pd
from typing import List, Tuple


def day_cols_for_weeks(weeks: List[str]) -> Tuple[List[dict], List[str]]:
    """Return DataTable columns and day ids spanning the given week Mondays (inclusive).
    weeks: list of ISO week-start dates (YYYY-MM-DD, Mondays)
    """
    # Normalize weeks list
    dts = [pd.to_datetime(w, errors="coerce") for w in weeks]
    dts = [pd.Timestamp(d) for d in dts if not pd.isna(d)]
    if not dts:
        # Fallback: single week starting today Monday
        today = dt.date.today()
        start = today - dt.timedelta(days=today.weekday())
        dts = [pd.Timestamp(start)]
    start = min(dts).date()
    end = (max(dts) + pd.Timedelta(days=6)).date()

    # Build day ids from start..end
    ids: List[str] = []
    cur = start
    while cur <= end:
        ids.append(cur.isoformat())
        cur += dt.timedelta(days=1)

    # Columns with Actual/Plan tags
    today = dt.date.today()
    cols: List[dict] = [{"name": "Metric", "id": "metric", "editable": False}]
    for d in ids:
        dd = pd.to_datetime(d).date()
        tag = "Actual" if dd <= today else "Plan"
        cols.append({"name": f"{tag}\n{dd.strftime('%m/%d/%y')}", "id": d})
    return cols, ids


def interval_cols_for_day(day: dt.date | None = None, ivl_min: int = 30) -> Tuple[List[dict], List[str]]:
    """Return DataTable columns and interval ids (HH:MM) for a single day.
    Intended as a compact interval view template.
    """
    if not isinstance(ivl_min, int) or ivl_min <= 0:
        ivl_min = 30
    if day is None:
        day = dt.date.today()
    # Build HH:MM slots
    ids: List[str] = []
    t = dt.datetime.combine(day, dt.time(0, 0))
    end = t + dt.timedelta(days=1)
    while t < end:
        ids.append(t.strftime("%H:%M"))
        t += dt.timedelta(minutes=ivl_min)

    cols: List[dict] = [{"name": "Metric", "id": "metric", "editable": False}]
    label_day = day.strftime("%a %Y-%m-%d") if isinstance(day, dt.date) else ""
    for slot in ids:
        # show date + time in header for clarity
        name = f"{label_day}\n{slot}" if label_day else slot
        cols.append({"name": name, "id": slot})
    return cols, ids
