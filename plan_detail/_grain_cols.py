from __future__ import annotations
import datetime as dt
import pandas as pd
from typing import List, Tuple, Optional


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


def interval_cols_for_day(
    day: dt.date | None = None,
    ivl_min: int = 30,
    start_hhmm: Optional[str] = None,
    end_hhmm: Optional[str] = None,
) -> Tuple[List[dict], List[str]]:
    """Return DataTable columns and interval ids (HH:MM) for a single day.
    - start_hhmm: optional start clock time (e.g., "08:00"). Defaults to 00:00.
    - end_hhmm: optional end clock time (exclusive). Defaults to 24:00 of the same day.
    """
    if not isinstance(ivl_min, int) or ivl_min <= 0:
        ivl_min = 30
    if day is None:
        day = dt.date.today()

    def _parse_hhmm(s: Optional[str], default_h: int, default_m: int) -> dt.time:
        if not s:
            return dt.time(default_h, default_m)
        try:
            hh, mm = s.strip().split(":", 1)
            h = max(0, min(23, int(hh)))
            m = max(0, min(59, int(mm)))
            return dt.time(h, m)
        except Exception:
            return dt.time(default_h, default_m)

    start_t = _parse_hhmm(start_hhmm, 0, 0)
    end_t   = _parse_hhmm(end_hhmm, 23, 59)

    # Build HH:MM slots from start_t up to and including the last slot that starts before end-of-window+1min
    ids: List[str] = []
    t = dt.datetime.combine(day, start_t)
    last_start = dt.datetime.combine(day, end_t)
    # If end < start, clamp to same-day end (23:59)
    if last_start <= t:
        last_start = dt.datetime.combine(day, dt.time(23, 59))
    # Emit slots while the start time is within the window and a full interval fits before midnight
    day_end = dt.datetime.combine(day, dt.time(23, 59))
    while t <= last_start and (t + dt.timedelta(minutes=ivl_min)) <= (day_end + dt.timedelta(minutes=1)):
        ids.append(t.strftime("%H:%M"))
        t += dt.timedelta(minutes=ivl_min)

    cols: List[dict] = [{"name": "Metric", "id": "metric", "editable": False}]
    label_day = day.strftime("%a %Y-%m-%d") if isinstance(day, dt.date) else ""
    for slot in ids:
        # show date + time in header for clarity
        name = f"{label_day}\n{slot}" if label_day else slot
        cols.append({"name": name, "id": slot})
    return cols, ids
