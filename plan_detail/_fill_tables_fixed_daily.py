from __future__ import annotations
import pandas as pd
import numpy as np
import datetime as dt
from typing import List, Tuple
from dash import dash_table, html

from plan_store import get_plan
from ._common import _week_span
from ._calc import _fill_tables_fixed
from ._grain_cols import day_cols_for_weeks


def _broadcast_weekly_to_daily(df: pd.DataFrame, day_ids: List[str]) -> pd.DataFrame:
    """Convert a weekly-matrix DataFrame (metric + week_id cols) to a daily-matrix
    by evenly distributing additive rows across the 7 days of each week.
    Percent/AHT/SUT-like rows are replicated to each day.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["metric"] + list(day_ids))

    # Identify weekly columns (ISO dates; Mondays)
    week_cols = [c for c in df.columns if c != "metric"]

    # Output skeleton
    out = pd.DataFrame({"metric": df["metric"].astype(str)})
    for d in day_ids:
        out[d] = 0.0

    def is_percent_like(name: str) -> bool:
        s = str(name).strip().lower()
        return (
            ("%" in s)
            or ("service level" in s)
            or ("occupancy" in s)
            or ("utilization" in s)
            or ("efficiency" in s)
            or ("aht/sut" in s)
        )

    # Map week -> its 7 days
    week_to_days = {}
    for w in week_cols:
        try:
            monday = pd.to_datetime(w).date()
        except Exception:
            continue
        week_to_days[w] = [(monday + dt.timedelta(days=i)).isoformat() for i in range(7)]

    # Fill rows
    for i, row in df.iterrows():
        name = str(row.get("metric", ""))
        replicate = is_percent_like(name)
        for w in week_cols:
            days = week_to_days.get(w, [])
            try:
                val = float(pd.to_numeric(row.get(w), errors="coerce"))
            except Exception:
                val = 0.0
            if not days:
                continue
            if replicate:
                for d in days:
                    if d in out.columns:
                        out.at[i, d] = val
            else:
                # even split across 7
                per = float(val) / float(len(days))
                for d in days:
                    if d in out.columns:
                        out.at[i, d] = per

    return out[["metric"] + list(day_ids)]


def _round_one_decimal(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        if c == "metric":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").round(1)
    return out


def _make_upper_table(upper_daily: pd.DataFrame, day_cols: List[dict]):
    if not isinstance(upper_daily, pd.DataFrame) or upper_daily.empty:
        upper_daily = pd.DataFrame({"metric": []})
    # Round to 1 decimal place for all numeric cells
    upper_daily = _round_one_decimal(upper_daily)
    return dash_table.DataTable(
        id="tbl-upper",
        data=upper_daily.to_dict("records"),
        columns=[{"name": "Metric", "id": "metric", "editable": False}] + [
            {"name": c["name"], "id": c["id"]} for c in day_cols if c["id"] != "metric"
        ],
        editable=False,
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={"whiteSpace": "pre"},
    )


def _fill_tables_fixed_daily(ptype, pid, _fw_cols_unused, _tick, whatif=None):
    """Daily view: derive daily grids by broadcasting weekly outputs to days.
    Keeps existing weekly logic intact by reusing _fill_tables_fixed, then reshaping.
    """
    p = get_plan(pid) or {}
    weeks = _week_span(p.get("start_week"), p.get("end_week"))
    day_cols, day_ids = day_cols_for_weeks(weeks)

    # Build weekly columns for reuse
    weekly_fw_cols = [
        {"name": "Metric", "id": "metric", "editable": False}
    ] + [{"name": w, "id": w} for w in weeks]

    # Run weekly engine once and reshape
    weekly = _fill_tables_fixed(ptype, pid, weekly_fw_cols, _tick, whatif=whatif, grain='week')
    (
        upper_w,
        fw_w, hc_w, att_w, shr_w, trn_w, rat_w, seat_w, bva_w, nh_w,
        roster_rec, bulk_files_rec, notes_rec,
    ) = weekly

    # Convert weekly dicts -> DataFrames
    def to_df(recs):
        try:
            return pd.DataFrame(recs or [])
        except Exception:
            return pd.DataFrame()

    fw_d   = _round_one_decimal(_broadcast_weekly_to_daily(to_df(fw_w),   day_ids))
    hc_d   = _round_one_decimal(_broadcast_weekly_to_daily(to_df(hc_w),   day_ids))
    att_d  = _round_one_decimal(_broadcast_weekly_to_daily(to_df(att_w),  day_ids))
    shr_d  = _round_one_decimal(_broadcast_weekly_to_daily(to_df(shr_w),  day_ids))
    trn_d  = _round_one_decimal(_broadcast_weekly_to_daily(to_df(trn_w),  day_ids))
    rat_d  = _round_one_decimal(_broadcast_weekly_to_daily(to_df(rat_w),  day_ids))
    seat_d = _round_one_decimal(_broadcast_weekly_to_daily(to_df(seat_w), day_ids))
    bva_d  = _round_one_decimal(_broadcast_weekly_to_daily(to_df(bva_w),  day_ids))
    nh_d   = _round_one_decimal(_broadcast_weekly_to_daily(to_df(nh_w),   day_ids))

    upper_df_w = to_df(getattr(upper_w, 'data', None))
    upper_d    = _broadcast_weekly_to_daily(upper_df_w, day_ids)
    upper_dtbl = _make_upper_table(_round_one_decimal(upper_d), day_cols)

    return (
        upper_dtbl,
        fw_d.to_dict("records"),
        hc_d.to_dict("records"),
        att_d.to_dict("records"),
        shr_d.to_dict("records"),
        trn_d.to_dict("records"),
        rat_d.to_dict("records"),
        seat_d.to_dict("records"),
        bva_d.to_dict("records"),
        nh_d.to_dict("records"),
        roster_rec,
        bulk_files_rec,
        notes_rec,
    )
