from __future__ import annotations
import pandas as pd
import numpy as np
import datetime as dt
from typing import List, Tuple
from dash import dash_table, html

from plan_store import get_plan
from cap_store import resolve_settings
from capacity_core import required_fte_daily
from ._common import _week_span, _scope_key, _assemble_voice, _assemble_bo, _assemble_ob, _assemble_chat
from cap_db import load_df
from common import summarize_shrinkage_bo, summarize_shrinkage_voice
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

    # Build a simple daily upper grid from raw daily requirement calculators
    p = get_plan(pid) or {}
    ba = p.get("vertical"); sba = p.get("sub_ba"); ch0 = (p.get("channel") or p.get("lob") or "").split(",")[0].strip()
    settings = resolve_settings(ba=ba, subba=sba, lob=ch0)
    sk = _scope_key(ba, sba, ch0)
    # Assemble daily inputs (forecast vs actual)
    vF = _assemble_voice(sk, "forecast"); bF = _assemble_bo(sk, "forecast"); oF = _assemble_ob(sk, "forecast")
    vA = _assemble_voice(sk, "actual");   bA = _assemble_bo(sk, "actual");   oA = _assemble_ob(sk, "actual")
    use_v = vA if isinstance(vA, pd.DataFrame) and not vA.empty else vF
    use_b = bA if isinstance(bA, pd.DataFrame) and not bA.empty else bF
    use_o = oA if isinstance(oA, pd.DataFrame) and not oA.empty else oF

    req_daily_A = required_fte_daily(use_v, use_b, use_o, settings)
    req_daily_F = required_fte_daily(vF, bF, oF, settings)

    def _req_map(df: pd.DataFrame) -> dict:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date.astype(str)
        g = d.groupby("date", as_index=True)["total_req_fte"].sum()
        return {k: float(v) for k, v in g.to_dict().items()}

    mA = _req_map(req_daily_A); mF = _req_map(req_daily_F)
    upper_d = pd.DataFrame({"metric": ["FTE Required @ Forecast Volume", "FTE Required @ Actual Volume"]})
    for dcol in day_ids:
        upper_d[dcol] = 0.0
    if mF:
        for k, v in mF.items():
            if k in upper_d.columns:
                upper_d.loc[upper_d["metric"].eq("FTE Required @ Forecast Volume"), k] = v
    if mA:
        for k, v in mA.items():
            if k in upper_d.columns:
                upper_d.loc[upper_d["metric"].eq("FTE Required @ Actual Volume"), k] = v
    upper_dtbl = _make_upper_table(_round_one_decimal(upper_d), day_cols)

    # Build Shrinkage daily table from raw uploads when available (BO/Voice)
    def _planned_shr_pct(settings: dict, ch_key: str) -> float:
        try:
            if ch_key.lower() in ("back office","bo"):
                val = settings.get("bo_shrinkage_pct", settings.get("shrinkage_pct", 0.0))
            elif ch_key.lower() == "voice":
                val = settings.get("voice_shrinkage_pct", settings.get("shrinkage_pct", 0.0))
            else:
                val = settings.get("shrinkage_pct", 0.0)
            x = float(val or 0.0)
            return (x if x > 1.0 else x*100.0)
        except Exception:
            return 0.0

    def _safe_filter(df: pd.DataFrame, p: dict) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        d = df.copy()
        L = {str(c).strip().lower(): c for c in d.columns}
        def col(*names):
            for n in names:
                c = L.get(str(n).strip().lower())
                if c: return c
            return None
        c_ba  = col("business area","journey","ba")
        c_sba = col("sub business area","level_3","sub_ba")
        c_ch  = col("channel","lob")
        c_site= col("site","location","country")
        m = pd.Series(True, index=d.index)
        if c_ba and p.get("vertical"): m &= d[c_ba].astype(str).str.strip().str.lower().eq(str(p.get("vertical")).strip().lower())
        if c_sba and p.get("sub_ba"): m &= d[c_sba].astype(str).str.strip().str.lower().eq(str(p.get("sub_ba")).strip().lower())
        if c_ch and ch0: m &= d[c_ch].astype(str).str.strip().str.lower().str.contains(str(ch0).strip().lower()[:5], na=False)
        if c_site and (p.get("site") or p.get("location") or p.get("country")):
            target = str(p.get("site") or p.get("location") or p.get("country")).strip().lower()
            loc_l = d[c_site].astype(str).str.strip().str.lower()
            if loc_l.eq(target).any(): m &= loc_l.eq(target)
        return d.loc[m]

    ch_key = (ch0 or '').strip().lower()
    shr_daily = pd.DataFrame({"metric": [
        "OOO Shrink Hours (#)",
        "In-Office Shrink Hours (#)",
        "OOO Shrinkage %",
        "In-Office Shrinkage %",
        "Overall Shrinkage %",
        "Planned Shrinkage %",
    ]})
    for dcol in day_ids:
        shr_daily[dcol] = np.nan
    try:
        if ch_key in ("back office","bo"):
            dfraw = _safe_filter(load_df("shrinkage_raw_backoffice"), p)
            daily = summarize_shrinkage_bo(dfraw)
            if isinstance(daily, pd.DataFrame) and not daily.empty:
                d = daily.copy(); d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date.astype(str)
                ooo = d.groupby("date")["OOO Hours"].sum().to_dict()
                ino = d.groupby("date")["In Office Hours"].sum().to_dict()
                base= d.groupby("date")["Base Hours"].sum().to_dict()
                ttw = d.get("TTW Hours"); ttw = ttw.groupby(d["date"]).sum().to_dict() if isinstance(ttw, pd.Series) else {}
                for k in day_ids:
                    b = float(base.get(k, 0.0)); i = float(ino.get(k, 0.0)); o = float(ooo.get(k, 0.0)); t = float(ttw.get(k, b))
                    shr_daily.loc[shr_daily["metric"].eq("OOO Shrink Hours (#)"),       k] = o
                    shr_daily.loc[shr_daily["metric"].eq("In-Office Shrink Hours (#)"), k] = i
                    ooo_pct = (100.0 * o / b) if b > 0 else 0.0
                    ino_pct = (100.0 * i / t) if t > 0 else 0.0
                    shr_daily.loc[shr_daily["metric"].eq("OOO Shrinkage %"),            k] = ooo_pct
                    shr_daily.loc[shr_daily["metric"].eq("In-Office Shrinkage %"),       k] = ino_pct
                    shr_daily.loc[shr_daily["metric"].eq("Overall Shrinkage %"),         k] = (ooo_pct + ino_pct)
        elif ch_key == "voice":
            dfraw = _safe_filter(load_df("shrinkage_raw_voice"), p)
            daily = summarize_shrinkage_voice(dfraw)
            if isinstance(daily, pd.DataFrame) and not daily.empty:
                d = daily.copy(); d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date.astype(str)
                ooo = d.groupby("date")["OOO Hours"].sum().to_dict()
                ino = d.groupby("date")["In Office Hours"].sum().to_dict()
                base= d.groupby("date")["Base Hours"].sum().to_dict()
                for k in day_ids:
                    b = float(base.get(k, 0.0)); i = float(ino.get(k, 0.0)); o = float(ooo.get(k, 0.0))
                    shr_daily.loc[shr_daily["metric"].eq("OOO Shrink Hours (#)"),       k] = o
                    shr_daily.loc[shr_daily["metric"].eq("In-Office Shrink Hours (#)"), k] = i
                    ooo_pct = (100.0 * o / b) if b > 0 else 0.0
                    ino_pct = (100.0 * i / b) if b > 0 else 0.0
                    shr_daily.loc[shr_daily["metric"].eq("OOO Shrinkage %"),            k] = ooo_pct
                    shr_daily.loc[shr_daily["metric"].eq("In-Office Shrinkage %"),       k] = ino_pct
                    shr_daily.loc[shr_daily["metric"].eq("Overall Shrinkage %"),         k] = (ooo_pct + ino_pct)
        # Planned shrink row (same for both)
        plan_pct = _planned_shr_pct(settings, ch_key)
        for k in day_ids:
            shr_daily.loc[shr_daily["metric"].eq("Planned Shrinkage %"), k] = plan_pct
    except Exception:
        pass
    shr_d = _round_one_decimal(shr_daily)

    # --- FW daily rows from native time series (no weekly drilling) ---
    def _series_from_df(df: pd.DataFrame, val_col: str, agg: str = "sum") -> dict:
        if not isinstance(df, pd.DataFrame) or df.empty or val_col not in df.columns:
            return {}
        d = df.copy(); d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date.astype(str)
        if agg == "sum":
            s = d.groupby("date")[val_col].sum()
        elif agg == "mean":
            s = d.groupby("date")[val_col].mean()
        else:
            s = d.groupby("date")[val_col].sum()
        return {k: float(v) for k, v in s.to_dict().items()}

    def _weighted_aht(df: pd.DataFrame, vol_col: str, aht_col: str) -> dict:
        if not isinstance(df, pd.DataFrame) or df.empty or vol_col not in df.columns:
            return {}
        d = df.copy(); d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date.astype(str)
        vol = pd.to_numeric(d[vol_col], errors="coerce").fillna(0.0)
        aht = pd.to_numeric(d.get(aht_col), errors="coerce").fillna(np.nan)
        d["_num"] = vol * aht
        g_vol = d.groupby("date")[vol_col].sum()
        g_num = d.groupby("date")["_num"].sum()
        out = {}
        for k in g_vol.index:
            v = float(g_vol.loc[k])
            out[k] = float(g_num.loc[k] / v) if v > 0 else 0.0
        return out

    def _write_row(df: pd.DataFrame, row_name: str, mapping: dict):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df
        mser = df["metric"].astype(str).str.strip()
        if row_name not in mser.values:
            return df
        for dcol in day_ids:
            if dcol in df.columns:
                val = mapping.get(dcol)
                if val is not None:
                    df.loc[mser == row_name, dcol] = float(val)
        return df

    # Fill per channel
    ch_key = (ch0 or '').strip().lower()
    # Voice
    if ch_key == "voice":
        vF = _assemble_voice(sk, "forecast"); vA = _assemble_voice(sk, "actual"); vT = _assemble_voice(sk, "tactical")
        for _df in (vF, vA, vT):
            if isinstance(_df, pd.DataFrame) and not _df.empty:
                _df["date"] = pd.to_datetime(_df["date"], errors="coerce").dt.date
        # Daily volume = sum of interval volume per day
        volF = _series_from_df(vF, "volume")
        volA = _series_from_df(vA, "volume")
        volT = _series_from_df(vT, "volume")
        # Weighted AHT per day
        ahtF = _weighted_aht(vF, "volume", "aht_sec")
        ahtA = _weighted_aht(vA, "volume", "aht_sec")
        fw_d = _write_row(fw_d, "Forecast", volF)
        fw_d = _write_row(fw_d, "Tactical Forecast", volT)
        fw_d = _write_row(fw_d, "Actual Volume", volA)
        fw_d = _write_row(fw_d, "Forecast AHT/SUT", ahtF)
        fw_d = _write_row(fw_d, "Actual AHT/SUT", ahtA)
    # Back Office
    elif ch_key in ("back office", "bo"):
        bF = _assemble_bo(sk, "forecast"); bA = _assemble_bo(sk, "actual"); bT = _assemble_bo(sk, "tactical")
        volF = _series_from_df(bF, "items")
        volA = _series_from_df(bA, "items")
        volT = _series_from_df(bT, "items")
        ahtF = _series_from_df(bF, "aht_sec", agg="mean")
        ahtA = _series_from_df(bA, "aht_sec", agg="mean")
        fw_d = _write_row(fw_d, "Forecast", volF)
        fw_d = _write_row(fw_d, "Tactical Forecast", volT)
        fw_d = _write_row(fw_d, "Actual Volume", volA)
        fw_d = _write_row(fw_d, "Forecast AHT/SUT", ahtF)
        fw_d = _write_row(fw_d, "Actual AHT/SUT", ahtA)
    # Chat
    elif ch_key == "chat":
        cF = _assemble_chat(sk, "forecast"); cA = _assemble_chat(sk, "actual"); cT = _assemble_chat(sk, "tactical")
        volF = _series_from_df(cF, "items"); volA = _series_from_df(cA, "items"); volT = _series_from_df(cT, "items")
        ahtF = _series_from_df(cF, "aht_sec", agg="mean"); ahtA = _series_from_df(cA, "aht_sec", agg="mean")
        fw_d = _write_row(fw_d, "Forecast", volF)
        fw_d = _write_row(fw_d, "Tactical Forecast", volT)
        fw_d = _write_row(fw_d, "Actual Volume", volA)
        fw_d = _write_row(fw_d, "Forecast AHT/SUT", ahtF)
        fw_d = _write_row(fw_d, "Actual AHT/SUT", ahtA)
    # Outbound
    elif ch_key in ("outbound", "ob"):
        oF = _assemble_ob(sk, "forecast"); oA = _assemble_ob(sk, "actual"); oT = _assemble_ob(sk, "tactical")
        volF = _series_from_df(oF, "opc") or _series_from_df(oF, "items")
        volA = _series_from_df(oA, "opc") or _series_from_df(oA, "items")
        volT = _series_from_df(oT, "opc") or _series_from_df(oT, "items")
        ahtF = _series_from_df(oF, "aht_sec", agg="mean"); ahtA = _series_from_df(oA, "aht_sec", agg="mean")
        fw_d = _write_row(fw_d, "Forecast", volF)
        fw_d = _write_row(fw_d, "Tactical Forecast", volT)
        fw_d = _write_row(fw_d, "Actual Volume", volA)
        fw_d = _write_row(fw_d, "Forecast AHT/SUT", ahtF)
        fw_d = _write_row(fw_d, "Actual AHT/SUT", ahtA)


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
