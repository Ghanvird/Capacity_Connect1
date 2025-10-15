from __future__ import annotations
import pandas as pd
import numpy as np
import datetime as dt
from typing import List, Tuple
from dash import dash_table

from plan_store import get_plan
from ._common import _week_span
from ._calc import _fill_tables_fixed
from ._grain_cols import interval_cols_for_day


def _broadcast_daily_to_intervals(df: pd.DataFrame, interval_ids: List[str]) -> pd.DataFrame:
    """Convert a daily-matrix DataFrame (metric + date cols) to an interval-matrix
    for a single representative day by uniformly distributing additive rows across intervals.
    Percent/AHT/SUT-like rows are replicated per interval.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["metric"] + list(interval_ids))

    # Identify daily columns (ISO dates)
    day_cols = [c for c in df.columns if c != "metric"]
    if not day_cols:
        return pd.DataFrame(columns=["metric"] + list(interval_ids))

    # Use the first day as representative
    rep_day = day_cols[0]

    out = pd.DataFrame({"metric": df["metric"].astype(str)})
    for ivl in interval_ids:
        out[ivl] = 0.0

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

    n = max(1, len(interval_ids))
    for i, row in df.iterrows():
        name = str(row.get("metric", ""))
        try:
            val = float(pd.to_numeric(row.get(rep_day), errors="coerce"))
        except Exception:
            val = 0.0
        if is_percent_like(name):
            for ivl in interval_ids:
                out.at[i, ivl] = val
        else:
            per = float(val) / float(n)
            for ivl in interval_ids:
                out.at[i, ivl] = per

    return out[["metric"] + list(interval_ids)]


def _round_one_decimal(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        if c == "metric":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").round(1)
    return out


def _make_upper_table(upper_ivl: pd.DataFrame, ivl_cols: List[dict]):
    if not isinstance(upper_ivl, pd.DataFrame) or upper_ivl.empty:
        upper_ivl = pd.DataFrame({"metric": []})
    # Round to 1 decimal place for all numeric cells
    upper_ivl = _round_one_decimal(upper_ivl)
    return dash_table.DataTable(
        id="tbl-upper",
        data=upper_ivl.to_dict("records"),
        columns=[{"name": "Metric", "id": "metric", "editable": False}] + [
            {"name": c["name"], "id": c["id"]} for c in ivl_cols if c["id"] != "metric"
        ],
        editable=False,
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={"whiteSpace": "pre"},
    )


def _fill_tables_fixed_interval(ptype, pid, _fw_cols_unused, _tick, whatif=None, ivl_min: int = 30):
    """Interval view (representative day):
    1) Reuse weekly engine then daily reshape
    2) Split first-day values across HH:MM intervals uniformly (percent-like rows replicated)
    """
    p = get_plan(pid) or {}
    weeks = _week_span(p.get("start_week"), p.get("end_week"))

    # Build weekly, daily first
    weekly_fw_cols = [{"name": "Metric", "id": "metric", "editable": False}] + [{"name": w, "id": w} for w in weeks]
    weekly = _fill_tables_fixed(ptype, pid, weekly_fw_cols, _tick, whatif=whatif, grain='week')
    (
        upper_w,
        fw_w, hc_w, att_w, shr_w, trn_w, rat_w, seat_w, bva_w, nh_w,
        roster_rec, bulk_files_rec, notes_rec,
    ) = weekly

    # Weekly -> daily (single representative day = first week Monday)
    monday = pd.to_datetime(weeks[0]).date() if weeks else dt.date.today()
    rep_day = monday.isoformat()
    # Build simple daily frames by copying weekly Monday values
    def weekly_col_to_day(recs):
        df = pd.DataFrame(recs or [])
        out = pd.DataFrame({"metric": df.get("metric", pd.Series(dtype="object"))})
        if rep_day not in out.columns:
            out[rep_day] = 0.0
        if not isinstance(df, pd.DataFrame) or df.empty:
            return out[["metric", rep_day]]
        if weeks:
            w0 = weeks[0]
        else:
            w0 = rep_day
        if w0 in df.columns:
            out[rep_day] = pd.to_numeric(df[w0], errors="coerce").fillna(0.0)
        else:
            # copy first numeric col
            wn = [c for c in df.columns if c != "metric"]
            if wn:
                out[rep_day] = pd.to_numeric(df[wn[0]], errors="coerce").fillna(0.0)
        return out[["metric", rep_day]]

    fw_d   = weekly_col_to_day(fw_w)
    hc_d   = weekly_col_to_day(hc_w)
    att_d  = weekly_col_to_day(att_w)
    shr_d  = weekly_col_to_day(shr_w)
    trn_d  = weekly_col_to_day(trn_w)
    rat_d  = weekly_col_to_day(rat_w)
    seat_d = weekly_col_to_day(seat_w)
    bva_d  = weekly_col_to_day(bva_w)
    nh_d   = weekly_col_to_day(nh_w)

    upper_df_w = pd.DataFrame(getattr(upper_w, 'data', None) or [])
    upper_d    = weekly_col_to_day(upper_df_w.to_dict("records"))

    # Daily -> intervals
    ivl_cols, ivl_ids = interval_cols_for_day(monday, ivl_min=ivl_min)
    fw_i   = _round_one_decimal(_broadcast_daily_to_intervals(fw_d,   ivl_ids))
    hc_i   = _round_one_decimal(_broadcast_daily_to_intervals(hc_d,   ivl_ids))
    att_i  = _round_one_decimal(_broadcast_daily_to_intervals(att_d,  ivl_ids))
    shr_i  = _round_one_decimal(_broadcast_daily_to_intervals(shr_d,  ivl_ids))
    trn_i  = _round_one_decimal(_broadcast_daily_to_intervals(trn_d,  ivl_ids))
    rat_i  = _round_one_decimal(_broadcast_daily_to_intervals(rat_d,  ivl_ids))
    seat_i = _round_one_decimal(_broadcast_daily_to_intervals(seat_d, ivl_ids))
    bva_i  = _round_one_decimal(_broadcast_daily_to_intervals(bva_d,  ivl_ids))
    nh_i   = _round_one_decimal(_broadcast_daily_to_intervals(nh_d,   ivl_ids))

    upper_tbl = _make_upper_table(_round_one_decimal(_broadcast_daily_to_intervals(upper_d, ivl_ids)), ivl_cols)

    return (
        upper_tbl,
        fw_i.to_dict("records"),
        hc_i.to_dict("records"),
        att_i.to_dict("records"),
        shr_i.to_dict("records"),
        trn_i.to_dict("records"),
        rat_i.to_dict("records"),
        seat_i.to_dict("records"),
        bva_i.to_dict("records"),
        nh_i.to_dict("records"),
        roster_rec,
        bulk_files_rec,
        notes_rec,
    )
