from __future__ import annotations
import math
import re
import pandas as pd
import numpy as np
import datetime as dt
from typing import List, Tuple, Optional
from dash import dash_table, html
from plan_store import get_plan
from cap_store import load_headcount, resolve_settings
from capacity_core import required_fte_daily, voice_requirements_interval, min_agents, _ivl_minutes_from_str
from ._common import _load_ts_with_fallback, _week_span, _scope_key, _assemble_voice, _assemble_bo, _assemble_ob, _assemble_chat
from cap_db import load_df, save_df
from cap_store import load_roster_long
from ._calc import _fill_tables_fixed
from ._grain_cols import day_cols_for_weeks

def _hhmm_to_minutes(x) -> float:
    if pd.isna(x): return 0.0
    s = str(x).strip()
    if not s: return 0.0
    # allow "HH:MM", "H:MM", "MM", "H.MM" etc.
    m = None
    if ":" in s:
        parts = s.split(":")
        if len(parts) >= 2:
            try:
                h = int(parts[0]); mm = int(parts[1])
                return float(h * 60 + mm)
            except Exception:
                pass
    try:
        # fallback: numeric minutes
        return float(s)
    except Exception:
        return 0.0

def _hc_lookup():
    """Return simple dict lookups from headcount: BRID→{lm_name, site, city, country, journey, level_3}"""
    try:
        hc = load_headcount()
    except Exception:
        hc = pd.DataFrame()
    if not isinstance(hc, pd.DataFrame) or hc.empty:
        return {}
    L = {c.lower(): c for c in hc.columns}
    def col(name):
        return L.get(name, name)
    out = {}
    for _, r in hc.iterrows():
        brid = str(r.get(col("brid"), "")).strip()
        if not brid: 
            continue
        out[brid] = dict(
            lm_name = r.get(col("line_manager_full_name")),
            site    = r.get(col("position_location_building_description")),
            city    = r.get(col("position_location_city")),
            country = r.get(col("position_location_country")),
            journey = r.get(col("journey")),
            level_3 = r.get(col("level_3")),
        )
    return out

def _parse_date_series(series: pd.Series) -> pd.Series:
    """Parse dates robustly without warnings.
    Heuristic:
      - If ISO (YYYY-MM-DD) → explicit format
      - If slash form → decide mm/dd vs dd/mm by values, then explicit format
      - If dash form with 2-digit first part → decide similarly
      - Else → default parser (no dayfirst) to avoid mm/dd + dayfirst warnings
    """
    s = pd.Series(series)
    # Already datetime-like
    try:
        if np.issubdtype(s.dtype, np.datetime64):
            return pd.to_datetime(s, errors="coerce").dt.date
    except Exception:
        pass

    sample = s.dropna().astype(str).str.strip()
    if sample.empty:
        return pd.to_datetime(s, errors="coerce").dt.date

    # ISO 8601: 2025-09-30
    iso_mask = sample.str.match(r"^\d{4}-\d{1,2}-\d{1,2}$")
    if iso_mask.any() and iso_mask.mean() > 0.5:
        return pd.to_datetime(s, errors="coerce", format="%Y-%m-%d").dt.date

    # Slash separated: 09/30/2025 or 30/09/2025
    slash_mask = sample.str.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$")
    if slash_mask.any() and slash_mask.mean() > 0.5:
        parts = sample[slash_mask].str.extract(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$")
        first = pd.to_numeric(parts[0], errors="coerce")
        if (first > 12).any():
            fmt = "%d/%m/%Y"
        else:
            fmt = "%m/%d/%Y"
        return pd.to_datetime(s, errors="coerce", format=fmt).dt.date

    # Dash separated ambiguous: 01-02-2025 or 30-09-2025
    dash_mask = sample.str.match(r"^\d{1,2}-\d{1,2}-\d{2,4}$")
    if dash_mask.any() and dash_mask.mean() > 0.5:
        parts = sample[dash_mask].str.extract(r"^(\d{1,2})-(\d{1,2})-(\d{2,4})$")
        first = pd.to_numeric(parts[0], errors="coerce")
        if (first > 12).any():
            fmt = "%d-%m-%Y"
        else:
            fmt = "%m-%d-%Y"
        return pd.to_datetime(s, errors="coerce", format=fmt).dt.date

    # Fallback: default parser
    return pd.to_datetime(s, errors="coerce").dt.date

def _bo_bucket(activity: str) -> str:
    try:
        if isinstance(activity, str):
            s = activity
        elif pd.isna(activity):
            s = ""
        else:
            s = str(activity)
    except Exception:
        s = ""
    s = s.strip().lower()
    # flexible matching
    if "divert" in s: return "diverted"
    if "down" in s or s == "downtime": return "downtime"
    if "staff complement" in s or s == "staff complement": return "staff_complement"
    if "flex" in s or s == "flexitime": return "flextime"
    if "lend" in s or s == "lend staff": return "lend_staff"
    if "borrow" in s or s == "borrowed staff": return "borrowed_staff"
    if "overtime" in s or s=="ot" or s == "overtime": return "overtime"
    if "core time" in s or s=="core": return "core_time"
    if "time worked" in s: return "time_worked"
    if "work out" in s or "workout" in s: return "work_out"
    return "other"


def summarize_shrinkage_bo(dff: pd.DataFrame) -> pd.DataFrame:
    """Daily BO summary in hours with buckets needed for new shrinkage formula.
    Buckets come from `_bo_bucket` applied to the free-text `activity` field.
    Returns per-day rows including:
      - "OOO Hours"      := Downtime
      - "In Office Hours": Diverted Time
      - "Base Hours"     := Staff Complement
      - "TTW Hours"      := Staff Complement - Downtime + Flexi + Overtime + Borrowed - Lend
    """
    if dff is None or dff.empty:
        return pd.DataFrame()
    d = dff.copy()
    d["date"] = pd.to_datetime(d.get("date"), errors="coerce").dt.date

    # Derive explicit buckets (robust if 'activity' column is missing)
    if "activity" in d.columns:
        d["bucket"] = d["activity"].map(_bo_bucket)
    else:
        # default to 'other' bucket to avoid crashes; upstream should normalize first
        d["bucket"] = pd.Series([_bo_bucket("")]*len(d), index=d.index)

    keys = ["date", "journey", "sub_business_area", "channel"]
    if "country" in d.columns:
        keys.append("country")
    if "site" in d.columns:
        keys.append("site")

    # Use hour granularity directly if present
    if "time_hours" in d.columns:
        val_col = "time_hours"
        factor = 1.0
    else:
        val_col = "duration_seconds"
        factor = 1.0 / 3600.0

    agg = (
        d.groupby(keys + ["bucket"], dropna=False)[val_col]
         .sum()
         .reset_index()
    )
    pivot = agg.pivot_table(index=keys, columns="bucket", values=val_col, fill_value=0.0).reset_index()

    def _col(frame: pd.DataFrame, names: list[str]) -> pd.Series:
        for nm in names:
            if nm in frame.columns:
                return frame[nm]
        return pd.Series(0.0, index=frame.index)

    sc  = _col(pivot, ["staff_complement"]) * factor
    dwn = _col(pivot, ["downtime"]) * factor
    flx = _col(pivot, ["flextime"]) * factor
    ot  = _col(pivot, ["overtime"]) * factor
    bor = _col(pivot, ["borrowed_staff", "borrowed"]) * factor
    lnd = _col(pivot, ["lend_staff", "lend"]) * factor
    div = _col(pivot, ["diverted"]) * factor

    ttw = sc - dwn + flx + ot + bor - lnd

    pivot["OOO Hours"] = dwn
    pivot["In Office Hours"] = div
    pivot["Base Hours"] = sc
    pivot["TTW Hours"] = ttw

    pivot = pivot.rename(columns={
        "journey": "Business Area",
        "sub_business_area": "Sub Business Area",
        "channel": "Channel",
        "country": "Country",
        "site": "Site",
    })

    keep_keys = [c for c in ["date", "Business Area", "Sub Business Area", "Channel", "Country", "Site"] if c in pivot.columns]
    keep = keep_keys + ["OOO Hours", "In Office Hours", "Base Hours", "TTW Hours"]
    return pivot[keep].sort_values(keep_keys)

def normalize_shrinkage_voice(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    L = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n.lower() in L: return L[n.lower()]
        return None
    col_date = pick("Date")
    col_state = pick("Superstate")
    col_hours = pick("Hours")
    col_brid  = pick("BRID","AgentID(BRID)","Employee Id","EmployeeID")
    if not (col_date and col_state and col_hours and col_brid):
        return pd.DataFrame()

    dff = df.copy()
    dff.rename(columns={col_date:"date", col_state:"superstate", col_hours:"hours_raw", col_brid:"brid"}, inplace=True)
    dff["date"] = _parse_date_series(dff["date"])  # robust date parsing
    dff["brid"] = dff["brid"].astype(str).str.strip()
    # convert HH:MM -> minutes, then to hours (as per spec they divide by 60)
    mins = dff["hours_raw"].map(_hhmm_to_minutes).fillna(0.0)
    dff["hours"] = mins/60.0

    # enrich from headcount
    hc = _hc_lookup()
    dff["TL Name"] = dff["brid"].map(lambda x: (hc.get(x) or {}).get("lm_name"))
    dff["Site"]    = dff["brid"].map(lambda x: (hc.get(x) or {}).get("site"))
    dff["City"]    = dff["brid"].map(lambda x: (hc.get(x) or {}).get("city"))
    dff["Country"] = dff["brid"].map(lambda x: (hc.get(x) or {}).get("country"))
    dff["Business Area"] = dff.get("Business Area", pd.Series(index=dff.index)).fillna(dff["brid"].map(lambda x: (hc.get(x) or {}).get("journey")))
    dff["Sub Business Area"] = dff.get("Sub Business Area", pd.Series(index=dff.index)).fillna(dff["brid"].map(lambda x: (hc.get(x) or {}).get("level_3")))
    if "Channel" not in dff.columns:
        dff["Channel"] = "Voice"

    # defaults so the pivot in summarize_shrinkage_voice never drops rows
    for col, default in [("Business Area", "All"), ("Sub Business Area", "All"), ("Country", "All")]:
        if col not in dff.columns:
            dff[col] = default
        else:
            dff[col] = dff[col].replace("", np.nan).fillna(default)
    dff["Channel"] = dff["Channel"].replace("", np.nan).fillna("Voice")
    return dff

def summarize_shrinkage_voice(dff: pd.DataFrame) -> pd.DataFrame:
    if dff is None or dff.empty:
        return pd.DataFrame()
    d = dff.copy()

    keys = ["date", "Business Area", "Sub Business Area", "Channel"]
    if "Country" in d.columns and d["Country"].notna().any():
        keys.append("Country")

    piv = d.pivot_table(index=keys, columns="superstate", values="hours", aggfunc="sum", fill_value=0.0).reset_index()

    def _series(name: str) -> pd.Series:
        return piv[name] if name in piv.columns else pd.Series(0.0, index=piv.index)

    ooo_codes = [
        "SC_ABSENCE_TOTAL",
        "SC_A_Sick_Long_Term",
        "SC_HOLIDAY",
        "SC_VACATION",
        "SC_LEAVE",
        "SC_UNPAID",
    ]
    ino_codes = [
        "SC_TRAINING_TOTAL",
        "SC_BREAKS",
        "SC_SYSTEM_EXCEPTION",
        "SC_MEETING",
        "SC_COACHING",
    ]

    piv["OOO Hours"] = sum((_series(code) for code in ooo_codes))
    piv["In Office Hours"] = sum((_series(code) for code in ino_codes))
    piv["Base Hours"] = _series("SC_INCLUDED_TIME")

    keep = keys + ["OOO Hours", "In Office Hours", "Base Hours"]
    return piv[keep].sort_values(keys)

def _broadcast_weekly_to_daily(
    df: pd.DataFrame,
    day_ids: List[str],
    *,
    pid: Optional[int] = None,
    channel: Optional[str] = None,
    tab: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a daily-matrix DataFrame from weekly (base), with channel-aware overrides for FW when interval/daily series exist.

    - For Voice/Chat/Outbound (tab == 'fw'):
      Use uploaded interval series to roll up to daily (sum for volumes, volume-weighted for AHT). If only daily series exist,
      use them as-is (no weekly/7 split).
    - For Back Office (tab == 'fw'):
      Use uploaded daily series as-is (no weekly/7 split).
    - For all other tabs or when data missing:
      Fall back to weekly broadcast: additive rows evenly split across 7 days; percent-like rows replicated.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["metric"] + list(day_ids))

    # Identify weekly columns (ISO dates; Mondays)
    week_cols = [c for c in df.columns if c != "metric"]

    # Output skeleton (single allocation to avoid fragmentation)
    out = pd.DataFrame({"metric": df["metric"].astype(str).tolist()})
    zeros = pd.DataFrame(0.0, index=out.index, columns=day_ids)
    out = pd.concat([out, zeros], axis=1)

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

    # Fill rows (baseline weekly->daily broadcast)
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

    # Channel-aware overrides only for FW grid
    ch = (channel or "").strip().lower()
    if (tab or "").strip().lower() != "fw" or not pid or ch == "":
        return out[["metric"] + list(day_ids)]

    try:
        p = get_plan(pid) or {}
    except Exception:
        p = {}
    sk = _scope_key(p.get("vertical"), p.get("sub_ba"), ch)

    # Helpers for daily maps
    def _is_interval_level(dfin: pd.DataFrame) -> bool:
        if not isinstance(dfin, pd.DataFrame) or dfin.empty:
            return False
        L = {str(c).strip().lower(): c for c in dfin.columns}
        c_ivl = L.get("interval") or L.get("time") or L.get("interval_start") or L.get("start_time") or L.get("slot")
        return bool(c_ivl and (c_ivl in dfin.columns) and dfin[c_ivl].notna().any())

    def _day_sum_map(dfin: pd.DataFrame, val_col: str, pick_first_if_daily: bool = True) -> dict:
        if not isinstance(dfin, pd.DataFrame) or dfin.empty:
            return {}
        d = dfin.copy()
        L = {str(c).strip().lower(): c for c in d.columns}
        c_date = L.get("date") or L.get("day")
        if not c_date or c_date not in d.columns:
            return {}
        d[c_date] = pd.to_datetime(d[c_date], errors="coerce").dt.date.astype(str)
        if _is_interval_level(d):
            s = d.groupby(c_date)[val_col].sum()
            return {k: float(v) for k, v in s.to_dict().items()}
        # Day-level data: pick first non-null per day to "display as uploaded"
        if pick_first_if_daily:
            outm = {}
            for k, sub in d.groupby(c_date):
                vals = pd.to_numeric(sub.get(val_col), errors="coerce")
                first = vals.dropna().iloc[0] if not vals.dropna().empty else np.nan
                outm[k] = float(first) if pd.notna(first) else 0.0
            return outm
        # else fallback to sum
        s = d.groupby(c_date)[val_col].sum()
        return {k: float(v) for k, v in s.to_dict().items()}

    def _weighted_aht_map(dfin: pd.DataFrame, vol_col: str, aht_col: str) -> dict:
        if not isinstance(dfin, pd.DataFrame) or dfin.empty:
            return {}
        d = dfin.copy()
        L = {str(c).strip().lower(): c for c in d.columns}
        c_date = L.get("date") or L.get("day")
        if not c_date or c_date not in d.columns:
            return {}
        d[c_date] = pd.to_datetime(d[c_date], errors="coerce").dt.date.astype(str)
        if _is_interval_level(d):
            vol = pd.to_numeric(d.get(vol_col), errors="coerce").fillna(0.0)
            aht = pd.to_numeric(d.get(aht_col), errors="coerce").fillna(np.nan)
            d["_num"] = vol * aht
            g_vol = d.groupby(c_date)[vol_col].sum()
            g_num = d.groupby(c_date)["_num"].sum()
            outm = {}
            for k in g_vol.index:
                v = float(g_vol.loc[k])
                outm[k] = float(g_num.loc[k] / v) if v > 0 else 0.0
            return outm
        # Day-level: pick first available
        outm = {}
        vals = pd.to_numeric(d.get(aht_col), errors="coerce") if aht_col in d.columns else pd.Series(dtype=float)
        for k, sub in d.groupby(c_date):
            s = pd.to_numeric(sub.get(aht_col), errors="coerce") if aht_col in sub.columns else pd.Series(dtype=float)
            first = s.dropna().iloc[0] if not s.dropna().empty else np.nan
            outm[k] = float(first) if pd.notna(first) else 0.0
        return outm

    def _write_row_local(df_out: pd.DataFrame, row_name: str, mapping: dict):
        if not isinstance(df_out, pd.DataFrame) or df_out.empty or not isinstance(mapping, dict) or not mapping:
            return
        mser = df_out["metric"].astype(str).str.strip()
        if row_name not in mser.values:
            return
        # Clear existing values for the row to avoid weekly/7 remnants
        df_out.loc[mser == row_name, day_ids] = [[0.0 for _ in day_ids]]
        set_cols = [d for d in day_ids if (d in df_out.columns) and (d in mapping) and (mapping.get(d) is not None)]
        if set_cols:
            df_out.loc[mser == row_name, set_cols] = [[float(mapping[d]) for d in set_cols]]

    try:
        m = out["metric"].astype(str).str.strip().tolist()
        if ch == "voice":
            vF = _assemble_voice(sk, "forecast"); vA = _assemble_voice(sk, "actual"); vT = _assemble_voice(sk, "tactical")
            volF = _day_sum_map(vF, "volume"); volA = _day_sum_map(vA, "volume"); volT = _day_sum_map(vT, "volume")
            ahtF = _weighted_aht_map(vF, "volume", "aht_sec"); ahtA = _weighted_aht_map(vA, "volume", "aht_sec")
            _write_row_local(out, "Forecast", volF)
            _write_row_local(out, "Tactical Forecast", volT)
            _write_row_local(out, "Actual Volume", volA)
            if "Forecast AHT/SUT" in m:
                _write_row_local(out, "Forecast AHT/SUT", ahtF)
            if "Actual AHT/SUT" in m:
                _write_row_local(out, "Actual AHT/SUT", ahtA)
            if ("AHT/SUT" in m) and ("Forecast AHT/SUT" not in m and "Actual AHT/SUT" not in m):
                _write_row_local(out, "AHT/SUT", ahtF or ahtA)
        elif ch == "chat":
            cF = _assemble_chat(sk, "forecast"); cA = _assemble_chat(sk, "actual"); cT = _assemble_chat(sk, "tactical")
            volF = _day_sum_map(cF, "items", pick_first_if_daily=True); volA = _day_sum_map(cA, "items", pick_first_if_daily=True); volT = _day_sum_map(cT, "items", pick_first_if_daily=True)
            ahtF = _weighted_aht_map(cF, "items", "aht_sec"); ahtA = _weighted_aht_map(cA, "items", "aht_sec")
            _write_row_local(out, "Forecast", volF)
            _write_row_local(out, "Tactical Forecast", volT)
            _write_row_local(out, "Actual Volume", volA)
            if "Forecast AHT/SUT" in m:
                _write_row_local(out, "Forecast AHT/SUT", ahtF)
            if "Actual AHT/SUT" in m:
                _write_row_local(out, "Actual AHT/SUT", ahtA)
            if ("AHT/SUT" in m) and ("Forecast AHT/SUT" not in m and "Actual AHT/SUT" not in m):
                _write_row_local(out, "AHT/SUT", ahtF or ahtA)
        elif ch in ("outbound", "ob"):
            oF = _assemble_ob(sk, "forecast"); oA = _assemble_ob(sk, "actual"); oT = _assemble_ob(sk, "tactical")
            # OPC/dials/calls
            colF = "opc" if "opc" in (oF.columns if isinstance(oF, pd.DataFrame) else []) else "items"
            colA = "opc" if "opc" in (oA.columns if isinstance(oA, pd.DataFrame) else []) else "items"
            colT = "opc" if "opc" in (oT.columns if isinstance(oT, pd.DataFrame) else []) else "items"
            volF = _day_sum_map(oF, colF, pick_first_if_daily=True); volA = _day_sum_map(oA, colA, pick_first_if_daily=True); volT = _day_sum_map(oT, colT, pick_first_if_daily=True)
            ahtF = _weighted_aht_map(oF, colF, "aht_sec"); ahtA = _weighted_aht_map(oA, colA, "aht_sec")
            _write_row_local(out, "Forecast", volF)
            _write_row_local(out, "Tactical Forecast", volT)
            _write_row_local(out, "Actual Volume", volA)
            if "Forecast AHT/SUT" in m:
                _write_row_local(out, "Forecast AHT/SUT", ahtF)
            if "Actual AHT/SUT" in m:
                _write_row_local(out, "Actual AHT/SUT", ahtA)
            if ("AHT/SUT" in m) and ("Forecast AHT/SUT" not in m and "Actual AHT/SUT" not in m):
                _write_row_local(out, "AHT/SUT", ahtF or ahtA)
        elif ch in ("back office", "bo"):
            bF = _assemble_bo(sk, "forecast"); bA = _assemble_bo(sk, "actual"); bT = _assemble_bo(sk, "tactical")
            volF = _day_sum_map(bF, "items", pick_first_if_daily=True); volA = _day_sum_map(bA, "items", pick_first_if_daily=True); volT = _day_sum_map(bT, "items", pick_first_if_daily=True)
            ahtF = _day_sum_map(bF, "aht_sec", pick_first_if_daily=True); ahtA = _day_sum_map(bA, "aht_sec", pick_first_if_daily=True)
            _write_row_local(out, "Forecast", volF)
            _write_row_local(out, "Tactical Forecast", volT)
            _write_row_local(out, "Actual Volume", volA)
            if "Forecast AHT/SUT" in m:
                _write_row_local(out, "Forecast AHT/SUT", ahtF)
            if "Actual AHT/SUT" in m:
                _write_row_local(out, "Actual AHT/SUT", ahtA)
            if ("AHT/SUT" in m) and ("Forecast AHT/SUT" not in m and "Actual AHT/SUT" not in m):
                _write_row_local(out, "AHT/SUT", ahtF or ahtA)
    except Exception:
        # swallow and keep baseline when data missing
        pass

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
    ch0 = (p.get("channel") or p.get("lob") or "").split(",")[0].strip().lower()
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

    # FW daily: start with a blank daily matrix and fill only from uploaded daily/interval series
    _fw_week_df = to_df(fw_w)
    _fw_metrics = _fw_week_df.get("metric", pd.Series(dtype="object")).astype(str).tolist() if isinstance(_fw_week_df, pd.DataFrame) else []
    fw_d = pd.DataFrame({"metric": _fw_metrics})
    for d in day_ids:
        fw_d[d] = 0.0
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

    # Build upper using weekly upper spec/values, then override FTE rows with interval-first daily FTE rollups (V/Ch/OB)
    upper_week_df = pd.DataFrame(getattr(upper_w, 'data', None) or [])
    if not isinstance(upper_week_df, pd.DataFrame) or upper_week_df.empty:
        # Fallback to just the two FTE rows if weekly upper is unavailable
        upper_d = pd.concat([
            pd.DataFrame({"metric": ["FTE Required @ Forecast Volume", "FTE Required @ Actual Volume"]}),
            pd.DataFrame(0.0, index=range(2), columns=day_ids)
        ], axis=1)
    else:
        weekly_ids = [c for c in upper_week_df.columns if c != 'metric']
        # Map each day to its week's Monday and copy the weekly value
        def _week_of_day(d):
            try:
                t = pd.to_datetime(d).date()
                monday = (t - dt.timedelta(days=t.weekday())).isoformat()
                return monday
            except Exception:
                return None
        day_to_week = {d: _week_of_day(d) for d in day_ids}
        # Initialize upper_d with all weekly metrics
        upper_d = pd.DataFrame({"metric": upper_week_df["metric"].astype(str).tolist()})
        for d in day_ids:
            upper_d[d] = 0.0
        for _, row in upper_week_df.iterrows():
            name = str(row.get('metric',''))
            for d in day_ids:
                w = day_to_week.get(d)
                if w and w in weekly_ids:
                    try:
                        v = float(pd.to_numeric(row.get(w), errors='coerce'))
                    except Exception:
                        v = 0.0
                    upper_d.loc[upper_d["metric"].eq(name), d] = v
    # Ensure rows used later exist
    try:
        need = ["Projected Handling Capacity (#)", "Projected Service Level",
                "FTE Required @ Forecast Volume", "FTE Required @ Actual Volume"]
        have = set(upper_d["metric"].astype(str).tolist())
        missing = [nm for nm in need if nm not in have]
        if missing:
            add = pd.DataFrame({"metric": missing})
            for d in day_ids:
                add[d] = 0.0
            upper_d = pd.concat([upper_d, add], ignore_index=True)
    except Exception:
        pass
    # Compute interval → day FTE rollups for Voice/Chat/Outbound
    hrs_per_fte = float(settings.get("hours_per_fte", 8.0) or 8.0)
    def _fte_from_agents_df(df: pd.DataFrame, ivl_col: str = "interval", agents_col: str = "agents") -> float:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return 0.0
        d = df.copy()
        d[ivl_col] = d[ivl_col].astype(str)
        ivm = d[ivl_col].map(lambda s: _ivl_minutes_from_str(s, int(float(settings.get("interval_minutes", 30) or 30)))).fillna(30).astype(float)
        agents = pd.to_numeric(d.get(agents_col), errors="coerce").fillna(0.0)
        staff_sec = (agents * (ivm * 60.0)).sum()
        return float(staff_sec / max(1.0, hrs_per_fte * 3600.0))

    fte_F: dict[str,float] = {}
    fte_A: dict[str,float] = {}

    ch_low = (ch0 or '').strip().lower()
    for dd in day_ids:
        try:
            day_dt = pd.to_datetime(dd).date()
        except Exception:
            continue
        if ch_low == "voice":
            vF_day = _assemble_voice(sk, "forecast"); vA_day = _assemble_voice(sk, "actual")
            def _agents_voice(src: pd.DataFrame) -> float:
                try:
                    iv = voice_requirements_interval(src, settings)
                    iv = iv.copy(); iv["date"] = pd.to_datetime(iv["date"], errors="coerce").dt.date
                    iv = iv[iv["date"].eq(day_dt)].rename(columns={"agents_req":"agents"})
                    return _fte_from_agents_df(iv[["interval","agents"]])
                except Exception:
                    return 0.0
            fte_F[dd] = _agents_voice(vF_day)
            fte_A[dd] = _agents_voice(vA_day if isinstance(vA_day, pd.DataFrame) and not vA_day.empty else vF_day)
        elif ch_low == "chat":
            conc = float(settings.get("chat_concurrency", 1.5) or 1.0)
            target_sl = float(settings.get("chat_target_sl", settings.get("target_sl", 0.8)) or 0.8)
            T_sec = float(settings.get("chat_sl_seconds", settings.get("sl_seconds", 20)) or 20.0)
            occ_cap = float(settings.get("occupancy_cap_chat", settings.get("util_chat", settings.get("occupancy_cap_voice", 0.85))) or 0.85)
            def _agents_chat(vol_df: pd.DataFrame, aht_df: pd.DataFrame) -> float:
                if not isinstance(vol_df, pd.DataFrame) or vol_df.empty:
                    return 0.0
                d = vol_df.copy(); L = {str(c).strip().lower(): c for c in d.columns}
                c_date = L.get("date") or L.get("day"); c_ivl = L.get("interval") or L.get("time")
                c_itm  = L.get("items") or L.get("volume")
                d["date"] = pd.to_datetime(d[c_date] if c_date else d.get("date"), errors="coerce").dt.date
                d = d[d["date"].eq(day_dt)]
                if c_ivl is None or c_ivl not in d.columns:
                    return 0.0
                # merge aht
                if isinstance(aht_df, pd.DataFrame) and not aht_df.empty:
                    ah = aht_df.copy(); LA = {str(c).strip().lower(): c for c in ah.columns}
                    ad = LA.get("date") or LA.get("day"); ai = LA.get("interval") or LA.get("time")
                    aht_c = LA.get("aht_sec") or LA.get("aht") or LA.get("avg_aht")
                    if aht_c:
                        if ad: ah[ad] = pd.to_datetime(ah[ad], errors="coerce").dt.date
                        join = [c for c in (ad, ai) if c]
                        if join:
                            d = d.merge(ah[[*join, aht_c]], on=join, how="left").rename(columns={aht_c:"aht_sec"})
                if "aht_sec" not in d.columns:
                    d["aht_sec"] = float(settings.get("chat_aht_sec", settings.get("target_aht", 240)) or 240.0)
                # per interval agents
                agents = []
                labs = d[c_ivl].astype(str)
                ivmins = []
                for i, r in d.iterrows():
                    items = float(pd.to_numeric(r.get(c_itm), errors="coerce") or 0.0)
                    aht = float(pd.to_numeric(r.get("aht_sec"), errors="coerce") or 0.0) / max(0.1, conc)
                    ivm = _ivl_minutes_from_str(str(r[c_ivl]), int(float(settings.get("chat_interval_minutes", settings.get("interval_minutes", 30)) or 30)))
                    ivmins.append(int(ivm))
                    N, *_ = min_agents(items, aht, int(ivm), target_sl, T_sec, occ_cap)
                    agents.append(float(N))
                tmp = pd.DataFrame({"interval": labs.values, "agents": agents, "ivm": ivmins})
                return float((pd.to_numeric(tmp["agents"], errors="coerce") * (pd.to_numeric(tmp["ivm"], errors="coerce") * 60.0)).sum() / (hrs_per_fte * 3600.0))
            volF = _load_ts_with_fallback("chat_forecast_volume", sk); ahtF = _load_ts_with_fallback("chat_forecast_aht", sk)
            volA = _load_ts_with_fallback("chat_actual_volume", sk);   ahtA = _load_ts_with_fallback("chat_actual_aht", sk)
            fte_F[dd] = _agents_chat(volF, ahtF)
            fte_A[dd] = _agents_chat(volA if isinstance(volA, pd.DataFrame) and not volA.empty else volF, ahtA if isinstance(ahtA, pd.DataFrame) and not ahtA.empty else ahtF)
        elif ch_low in ("outbound","ob"):
            target_sl = float(settings.get("ob_target_sl", settings.get("target_sl", 0.8)) or 0.8)
            T_sec = float(settings.get("ob_sl_seconds", settings.get("sl_seconds", 20)) or 20.0)
            occ_cap = float(settings.get("occupancy_cap_ob", settings.get("util_ob", settings.get("occupancy_cap_voice", 0.85))) or 0.85)
            def _expected_calls_df(df):
                if not isinstance(df, pd.DataFrame) or df.empty:
                    return pd.DataFrame(columns=["interval","calls","aht","ivm"])
                d = df.copy(); L = {str(c).strip().lower(): c for c in d.columns}
                c_date = L.get("date") or L.get("day"); c_ivl = L.get("interval") or L.get("time")
                opc = L.get("opc") or L.get("dials") or L.get("calls") or L.get("volume")
                d["date"] = pd.to_datetime(d[c_date] if c_date else d.get("date"), errors="coerce").dt.date
                d = d[d["date"].eq(day_dt)]
                if c_ivl is None or c_ivl not in d.columns:
                    return pd.DataFrame(columns=["interval","calls","aht","ivm"])
                calls = []
                ivmins = []
                ahts = []
                for i, r in d.iterrows():
                    v_opc = float(pd.to_numeric(r.get(opc), errors="coerce") or 0.0)
                    conn = r.get(L.get("connect_rate")) if L.get("connect_rate") else r.get("connect_rate")
                    rpcr = r.get(L.get("rpc_rate")) if L.get("rpc_rate") else r.get("rpc_rate")
                    rpc_ct = float(pd.to_numeric(r.get(L.get("rpc") or "rpc"), errors="coerce") or 0.0)
                    try: conn = float(conn)
                    except Exception: conn = None
                    try: rpcr = float(rpcr)
                    except Exception: rpcr = None
                    if rpc_ct and rpc_ct > 0: c = rpc_ct
                    elif (conn is not None) and (rpcr is not None): c = v_opc * conn * rpcr
                    elif (conn is not None): c = v_opc * conn
                    else: c = v_opc
                    calls.append(float(c))
                    ivmins.append(_ivl_minutes_from_str(str(r[c_ivl]), int(float(settings.get("ob_interval_minutes", settings.get("interval_minutes", 30)) or 30))))
                    ahts.append(float(pd.to_numeric(r.get(L.get("aht") or "aht"), errors="coerce") or (settings.get("ob_aht_sec", settings.get("target_aht", 240)) or 240.0)))
                return pd.DataFrame({"interval": d[c_ivl].astype(str).values, "calls": calls, "aht": ahts, "ivm": ivmins})
            volF = _load_ts_with_fallback("ob_forecast_opc", sk)
            if (not isinstance(volF, pd.DataFrame)) or volF.empty:
                tmp = _load_ts_with_fallback("outbound_forecast_opc", sk) or _load_ts_with_fallback("ob_forecast_dials", sk) or _load_ts_with_fallback("outbound_forecast_dials", sk) or _load_ts_with_fallback("ob_forecast_calls", sk)
                volF = tmp if isinstance(tmp, pd.DataFrame) else volF
            volA = _load_ts_with_fallback("ob_actual_opc", sk)
            if (not isinstance(volA, pd.DataFrame)) or volA.empty:
                tmp = _load_ts_with_fallback("outbound_actual_opc", sk) or _load_ts_with_fallback("ob_actual_dials", sk) or _load_ts_with_fallback("outbound_actual_dials", sk) or _load_ts_with_fallback("ob_actual_calls", sk)
                volA = tmp if isinstance(tmp, pd.DataFrame) else volA
            dfF = _expected_calls_df(volF)
            dfA = _expected_calls_df(volA if isinstance(volA, pd.DataFrame) and not volA.empty else volF)
            def _fte_from_expected(df) -> float:
                if not isinstance(df, pd.DataFrame) or df.empty:
                    return 0.0
                agents = []
                for i, r in df.iterrows():
                    N, *_ = min_agents(float(r.get("calls", 0.0) or 0.0), float(r.get("aht", 0.0) or 0.0), int(float(r.get("ivm", 30) or 30)), target_sl, T_sec, occ_cap)
                    agents.append(float(N))
                sec = (pd.Series(agents) * (pd.to_numeric(df.get("ivm"), errors="coerce").fillna(0.0) * 60.0)).sum()
                return float(sec / (hrs_per_fte * 3600.0))
            target_sl = float(settings.get("ob_target_sl", settings.get("target_sl", 0.8)) or 0.8)
            T_sec     = float(settings.get("ob_sl_seconds", settings.get("sl_seconds", 20)) or 20.0)
            occ_cap   = float(settings.get("occupancy_cap_ob", settings.get("util_ob", settings.get("occupancy_cap_voice", 0.85))) or 0.85)
            fte_F[dd] = _fte_from_expected(dfF)
            fte_A[dd] = _fte_from_expected(dfA)

    # Override FTE rows with interval-first daily values (fallback to original if empty)
    if fte_F:
        for k, v in fte_F.items():
            if k in upper_d.columns:
                upper_d.loc[upper_d["metric"].eq("FTE Required @ Forecast Volume"), k] = float(v)
    if fte_A:
        for k, v in fte_A.items():
            if k in upper_d.columns:
                upper_d.loc[upper_d["metric"].eq("FTE Required @ Actual Volume"), k] = float(v)

    # Interval → Day rollups for Projected Handling Capacity and Projected Service Level
    def _parse_hhmm_to_min(hhmm: str) -> int:
        try:
            h, m = hhmm.split(":", 1)
            return int(h) * 60 + int(m)
        except Exception:
            return 0

    def _staff_by_slot_for_day(day: dt.date) -> dict[str, float]:
        try:
            rl = load_roster_long()
        except Exception:
            return {}
        if not isinstance(rl, pd.DataFrame) or rl.empty:
            return {}
        df = rl.copy()
        # Coarse scope filters using plan fields
        def _col(opts):
            for c in opts:
                if c in df.columns:
                    return c
            return None
        c_ba  = _col(["Business Area","business area","vertical"]) 
        c_sba = _col(["Sub Business Area","sub business area","sub_ba"]) 
        c_lob = _col(["LOB","lob","Channel","channel"]) 
        c_site= _col(["Site","site","Location","location","Country","country"]) 
        BA  = p.get("vertical"); SBA = p.get("sub_ba"); LOB = (p.get("channel") or p.get("lob") or "").split(",")[0].strip()
        SITE= (p.get("site") or p.get("location") or p.get("country") or "").strip()
        def _match(series, val):
            if not val or not isinstance(series, pd.Series):
                return pd.Series([True]*len(series))
            s = series.astype(str).str.strip().str.lower()
            return s.eq(str(val).strip().lower())
        msk = pd.Series([True]*len(df))
        if c_ba:  msk &= _match(df[c_ba], BA)
        if c_sba and (SBA not in (None, "")): msk &= _match(df[c_sba], SBA)
        if c_lob: msk &= _match(df[c_lob], LOB)
        if c_site and (SITE not in (None, "")): msk &= _match(df[c_site], SITE)
        df = df[msk]
        if "is_leave" in df.columns:
            df = df[~df["is_leave"].astype(bool)]
        if "date" not in df.columns or "entry" not in df.columns:
            return {}
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df = df[df["date"].eq(day)]
        if df.empty:
            return {}
        # Build slot labels from 08:00 by default and 30m intervals unless settings override later
        start_hhmm = "08:00"
        ivl_min = 30
        slots: dict[str,float] = {}
        t = dt.datetime.combine(day, dt.time(8,0))
        end = t + dt.timedelta(hours=24)
        labels: list[str] = []
        while t < end:
            lab = t.strftime("%H:%M"); labels.append(lab); slots[lab] = 0.0; t += dt.timedelta(minutes=ivl_min)
        cov_start_min = _parse_hhmm_to_min(start_hhmm)
        for _, r in df.iterrows():
            try:
                sft = str(r.get("entry", "")).strip()
                m = re.match(r"^(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})$", sft)
                if not m:
                    continue
                sh, sm, eh, em = map(int, m.groups())
                sh = min(23, max(0, sh)); eh = min(24, max(0, eh))
                start_min = sh*60 + sm
                end_min   = eh*60 + em
                if end_min <= start_min:
                    end_min += 24*60
                for idx2, lab2 in enumerate(labels):
                    slot_abs = cov_start_min + idx2*ivl_min
                    slot_rel = slot_abs
                    if slot_rel < start_min:
                        slot_rel += 24*60
                    if start_min <= slot_rel < end_min:
                        slots[lab2] = slots.get(lab2, 0.0) + 1.0
            except Exception:
                continue
        return slots

    def _slot_map(df: pd.DataFrame, day: dt.date, val_col: str, label_col: str = "interval") -> dict[str, float]:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}
        d = df.copy(); d["date"] = pd.to_datetime(d.get("date"), errors="coerce").dt.date
        d = d[d["date"].eq(day)]
        if label_col not in d.columns:
            return {}
        labs = d[label_col].astype(str).str.slice(0,5)
        vals = pd.to_numeric(d.get(val_col), errors="coerce").fillna(0.0).tolist()
        return dict(zip(labs, vals))

    # Ensure rows exist
    if "Projected Handling Capacity (#)" not in upper_d["metric"].astype(str).values:
        upper_d = pd.concat([upper_d, pd.DataFrame({"metric":["Projected Handling Capacity (#)"]})], ignore_index=True)
        for dd in day_ids:
            if dd not in upper_d.columns: upper_d[dd] = 0.0
    if "Projected Service Level" not in upper_d["metric"].astype(str).values:
        upper_d = pd.concat([upper_d, pd.DataFrame({"metric":["Projected Service Level"]})], ignore_index=True)
        for dd in day_ids:
            if dd not in upper_d.columns: upper_d[dd] = 0.0

    if ch0 == "voice":
        for dd in day_ids:
            try:
                day_dt = pd.to_datetime(dd).date()
            except Exception:
                continue
            staff = _staff_by_slot_for_day(day_dt)
            vF_day = _assemble_voice(sk, "forecast")
            vols = _slot_map(vF_day, day_dt, "volume")
            ahts = _slot_map(vF_day, day_dt, "aht_sec")
            ivl_sec = max(60, int(float(settings.get("interval_minutes", 30) or 30)) * 60)
            T_sec = int(float(settings.get("sl_seconds", 20) or 20))
            target = float(settings.get("target_sl", 0.8) or 0.8)
            occ_cap = float(settings.get("occupancy_cap_voice", 0.85) or 0.85)
            def Ec(A, N):
                if N <= 0: return 1.0
                if A <= 0: return 0.0
                if A >= N: return 1.0
                term = 1.0; ssum = term
                for k in range(1, N): term *= A/k; ssum += term
                term *= A/N; last = term*(N/(N-A)); den = ssum + last
                if den <= 0: return 1.0
                return last * (1.0/den)
            def SL(calls, aht, ag):
                if aht<=0 or ivl_sec<=0 or ag<=0: return 0.0
                if calls<=0: return 1.0
                A=(calls*aht)/ivl_sec
                pw=Ec(A, int(max(1, math.floor(ag))))
                return max(0.0, min(1.0, 1.0 - pw * math.exp(-max(0.0,(ag-A))*(T_sec/max(1.0,aht)))))
            def CAP(ag, aht):
                if ag<=0 or aht<=0: return 0.0
                lo, hi = 0, max(1, int((ag*ivl_sec)/aht))
                if SL(hi, aht, ag) < target: cap_hi = hi
                else:
                    while SL(hi, aht, ag) >= target and hi < 10000000:
                        lo = hi; hi *= 2
                    while lo < hi:
                        mid = (lo + hi + 1)//2
                        if SL(mid, aht, ag) >= target: lo = mid
                        else: hi = mid - 1
                    cap_hi = lo
                occ_calls_cap = (occ_cap*ag*ivl_sec)/max(1.0,aht)
                return float(min(cap_hi, occ_calls_cap))
            phc = 0.0; tot_vol = 0.0
            for lab, v in vols.items():
                aht = float(ahts.get(lab, ahts.get(next(iter(ahts), lab), 300.0)) or 300.0)
                ag  = float(staff.get(lab, 0.0) or 0.0)
                phc += CAP(ag, aht)
                tot_vol += float(v or 0.0)
            psl = 0.0
            if tot_vol > 0:
                num = 0.0
                for lab, v in vols.items():
                    aht = float(ahts.get(lab, ahts.get(next(iter(ahts), lab), 300.0)) or 300.0)
                    ag  = float(staff.get(lab, 0.0) or 0.0)
                    num += float(v or 0.0) * SL(float(v or 0.0), aht, ag)
                psl = 100.0 * (num / tot_vol)
            upper_d.loc[upper_d["metric"].eq("Projected Handling Capacity (#)") , dd] = phc
            upper_d.loc[upper_d["metric"].eq("Projected Service Level")         , dd] = psl
    elif ch0 == "chat":
        # Chat: concurrency-aware Erlang rollup
        for dd in day_ids:
            try:
                day_dt = pd.to_datetime(dd).date()
            except Exception:
                continue
            staff = _staff_by_slot_for_day(day_dt)
            cF_day = _assemble_chat(sk, "forecast")
            vols = _slot_map(cF_day, day_dt, "items")
            aht_day = _load_ts_with_fallback("chat_forecast_aht", sk)
            ahts = _slot_map(aht_day, day_dt, "aht_sec") if isinstance(aht_day, pd.DataFrame) else {}
            ivl_sec = max(60, int(float(settings.get("chat_interval_minutes", settings.get("interval_minutes", 30)) or 30)) * 60)
            T_sec = int(float(settings.get("chat_sl_seconds", settings.get("sl_seconds", 20)) or 20))
            target = float(settings.get("chat_target_sl", settings.get("target_sl", 0.8)) or 0.8)
            occ = float(settings.get("occupancy_cap_chat", settings.get("util_chat", settings.get("occupancy_cap_voice", 0.85))) or 0.85)
            conc = float(settings.get("chat_concurrency", 1.5) or 1.0)
            def Ec(A,N):
                if N<=0: return 1.0
                if A<=0: return 0.0
                if A>=N: return 1.0
                t=1.0; s=t
                for k in range(1,N): t*=A/k; s+=t
                t*=A/N; last=t*(N/(N-A)); den=s+last
                if den<=0: return 1.0
                return last*(1.0/den)
            def SL(calls,aht,ag):
                if aht<=0 or ivl_sec<=0 or ag<=0: return 0.0
                if calls<=0: return 1.0
                A=(calls*aht)/ivl_sec
                pw=Ec(A,int(max(1,math.floor(ag))))
                return max(0.0,min(1.0,1.0-pw*math.exp(-max(0.0,(ag-A))*(T_sec/max(1.0,aht)))))
            def CAP(ag,aht):
                if ag<=0 or aht<=0: return 0.0
                lo,hi=0,max(1,int((ag*ivl_sec)/aht))
                if SL(hi,aht,ag)<target: cap_hi=hi
                else:
                    while SL(hi,aht,ag)>=target and hi<10000000:
                        lo=hi; hi*=2
                    while lo<hi:
                        mid=(lo+hi+1)//2
                        if SL(mid,aht,ag)>=target: lo=mid
                        else: hi=mid-1
                    cap_hi=lo
                occ_cap_calls=(occ*ag*ivl_sec)/max(1.0,aht)
                return float(min(cap_hi,occ_cap_calls))
            phc=0.0; tot_vol=0.0
            for lab,v in vols.items():
                aht=float(ahts.get(lab,ahts.get(next(iter(ahts), lab),240.0)) or 240.0)/max(0.1,conc)
                ag=float(staff.get(lab,0.0) or 0.0)
                phc+=CAP(ag,aht); tot_vol+=float(v or 0.0)
            psl=0.0
            if tot_vol>0:
                num=0.0
                for lab,v in vols.items():
                    aht=float(ahts.get(lab,ahts.get(next(iter(ahts), lab),240.0)) or 240.0)/max(0.1,conc)
                    ag=float(staff.get(lab,0.0) or 0.0)
                    num+=float(v or 0.0)*SL(float(v or 0.0),aht,ag)
                psl=100.0*(num/tot_vol)
            upper_d.loc[upper_d["metric"].eq("Projected Handling Capacity (#)") , dd] = phc
            upper_d.loc[upper_d["metric"].eq("Projected Service Level")         , dd] = psl
    elif ch0 in ("outbound","ob"):
        for dd in day_ids:
            try:
                day_dt = pd.to_datetime(dd).date()
            except Exception:
                continue
            staff = _staff_by_slot_for_day(day_dt)
            oF_day = _assemble_ob(sk, "forecast")
            # expected calls map per slot
            def _calls_map(df):
                if not isinstance(df, pd.DataFrame) or df.empty:
                    return {}
                d2=df.copy(); d2["date"]=pd.to_datetime(d2.get("date"), errors="coerce").dt.date
                d2=d2[d2["date"].eq(day_dt)]
                if "interval" not in d2.columns: return {}
                labs=d2["interval"].astype(str).str.slice(0,5)
                def cr(row):
                    opc=float(row.get("opc",0.0) or 0.0)
                    rpc_ct=float(row.get("rpc",0.0) or 0.0)
                    conn=row.get("connect_rate",None)
                    rpcr=row.get("rpc_rate",None)
                    try: conn=float(conn)
                    except Exception: conn=None
                    try: rpcr=float(rpcr)
                    except Exception: rpcr=None
                    if rpc_ct and rpc_ct>0: return rpc_ct
                    if (conn is not None) and (rpcr is not None): return opc*conn*rpcr
                    if conn is not None: return opc*conn
                    return opc
                calls=d2.apply(cr, axis=1).astype(float).tolist()
                return dict(zip(labs,calls))
            calls=_calls_map(oF_day)
            ivl_sec=max(60,int(float(settings.get("ob_interval_minutes", settings.get("interval_minutes",30)) or 30))*60)
            T_sec=int(float(settings.get("ob_sl_seconds", settings.get("sl_seconds",20)) or 20))
            target=float(settings.get("ob_target_sl", settings.get("target_sl",0.8)) or 0.8)
            occ=float(settings.get("occupancy_cap_ob", settings.get("util_ob", settings.get("occupancy_cap_voice",0.85))) or 0.85)
            def Ec(A,N):
                if N<=0: return 1.0
                if A<=0: return 0.0
                if A>=N: return 1.0
                t=1.0; s=t
                for k in range(1,N): t*=A/k; s+=t
                t*=A/N; last=t*(N/(N-A)); den=s+last
                if den<=0: return 1.0
                return last*(1.0/den)
            def SL(calls,aht,ag):
                if aht<=0 or ivl_sec<=0 or ag<=0: return 0.0
                if calls<=0: return 1.0
                A=(calls*aht)/ivl_sec
                pw=Ec(A,int(max(1,math.floor(ag))))
                return max(0.0,min(1.0,1.0 - pw*math.exp(-max(0.0,(ag-A))*(T_sec/max(1.0,aht)))))
            def CAP(ag,aht):
                if ag<=0 or aht<=0: return 0.0
                lo,hi=0,max(1,int((ag*ivl_sec)/aht))
                if SL(hi,aht,ag)<target: cap_hi=hi
                else:
                    while SL(hi,aht,ag)>=target and hi<10000000:
                        lo=hi; hi*=2
                    while lo<hi:
                        mid=(lo+hi+1)//2
                        if SL(mid,aht,ag)>=target: lo=mid
                        else: hi=mid-1
                    cap_hi=lo
                occ_cap_calls=(occ*ag*ivl_sec)/max(1.0,aht)
                return float(min(cap_hi,occ_cap_calls))
            phc=0.0; tot_vol=0.0
            # derive AHT per slot if present
            aht_map={}
            if isinstance(oF_day, pd.DataFrame) and not oF_day.empty and "aht" in oF_day.columns:
                d3=oF_day.copy(); d3=d3[d3["date"].eq(day_dt) & d3["interval"].notna()]
                aht_map=dict(zip(d3["interval"].astype(str).str.slice(0,5), pd.to_numeric(d3.get("aht"), errors="coerce").fillna(0.0).tolist()))
            for lab,v in calls.items():
                aht=float(aht_map.get(lab, aht_map.get(next(iter(aht_map), lab), 240.0)) or 240.0)
                ag=float(staff.get(lab,0.0) or 0.0)
                phc+=CAP(ag,aht); tot_vol+=float(v or 0.0)
            psl=0.0
            if tot_vol>0:
                num=0.0
                for lab,v in calls.items():
                    aht=float(aht_map.get(lab, aht_map.get(next(iter(aht_map), lab), 240.0)) or 240.0)
                    ag=float(staff.get(lab,0.0) or 0.0)
                    num+=float(v or 0.0)*SL(float(v or 0.0), aht, ag)
                psl=100.0*(num/tot_vol)
            upper_d.loc[upper_d["metric"].eq("Projected Handling Capacity (#)") , dd]=phc
            upper_d.loc[upper_d["metric"].eq("Projected Service Level")         , dd]=psl

    # Persist rollups for reuse: date, metric, value
    try:
        vals = []
        for dd in day_ids:
            for mname in ("Projected Handling Capacity (#)", "Projected Service Level"):
                try:
                    v = float(pd.to_numeric(upper_d.loc[upper_d["metric"].eq(mname), dd], errors="coerce").fillna(0.0).iloc[0])
                except Exception:
                    v = 0.0
                vals.append({"date": dd, "metric": mname, "value": v, "channel": ch0})
        if vals:
            save_df(f"plan_{pid}_upper_daily_rollups", pd.DataFrame(vals))
    except Exception:
        pass

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
    _shr_rows = [
        "OOO Shrink Hours (#)",
        "In-Office Shrink Hours (#)",
        "OOO Shrinkage %",
        "In-Office Shrinkage %",
        "Overall Shrinkage %",
        "Planned Shrinkage %",
    ]
    shr_daily = pd.concat([
        pd.DataFrame({"metric": _shr_rows}),
        pd.DataFrame(np.nan, index=range(len(_shr_rows)), columns=day_ids)
    ], axis=1)
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
                arr_ooo  = [float(ooo.get(k, 0.0))  for k in day_ids]
                arr_ino  = [float(ino.get(k, 0.0))  for k in day_ids]
                arr_base = [float(base.get(k, 0.0)) for k in day_ids]
                arr_ttw  = [float(ttw.get(k, arr_base[i])) for i, k in enumerate(day_ids)]
                arr_ooo_pct = [(100.0 * o / b) if b > 0 else 0.0 for o, b in zip(arr_ooo, arr_base)]
                arr_ino_pct = [(100.0 * i / t) if t > 0 else 0.0 for i, t in zip(arr_ino, arr_ttw)]
                arr_ovr_pct = [o + i for o, i in zip(arr_ooo_pct, arr_ino_pct)]
                mask = shr_daily["metric"].eq("OOO Shrink Hours (#)")
                shr_daily.loc[mask, day_ids] = [arr_ooo]
                mask = shr_daily["metric"].eq("In-Office Shrink Hours (#)")
                shr_daily.loc[mask, day_ids] = [arr_ino]
                mask = shr_daily["metric"].eq("OOO Shrinkage %")
                shr_daily.loc[mask, day_ids] = [arr_ooo_pct]
                mask = shr_daily["metric"].eq("In-Office Shrinkage %")
                shr_daily.loc[mask, day_ids] = [arr_ino_pct]
                mask = shr_daily["metric"].eq("Overall Shrinkage %")
                shr_daily.loc[mask, day_ids] = [arr_ovr_pct]
        elif ch_key == "voice":
            dfraw = _safe_filter(load_df("shrinkage_raw_voice"), p)
            daily = summarize_shrinkage_voice(dfraw)
            if isinstance(daily, pd.DataFrame) and not daily.empty:
                d = daily.copy(); d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date.astype(str)
                ooo = d.groupby("date")["OOO Hours"].sum().to_dict()
                ino = d.groupby("date")["In Office Hours"].sum().to_dict()
                base= d.groupby("date")["Base Hours"].sum().to_dict()
                arr_ooo  = [float(ooo.get(k, 0.0))  for k in day_ids]
                arr_ino  = [float(ino.get(k, 0.0))  for k in day_ids]
                arr_base = [float(base.get(k, 0.0)) for k in day_ids]
                arr_ooo_pct = [(100.0 * o / b) if b > 0 else 0.0 for o, b in zip(arr_ooo, arr_base)]
                arr_ino_pct = [(100.0 * i / b) if b > 0 else 0.0 for i, b in zip(arr_ino, arr_base)]
                arr_ovr_pct = [o + i for o, i in zip(arr_ooo_pct, arr_ino_pct)]
                mask = shr_daily["metric"].eq("OOO Shrink Hours (#)")
                shr_daily.loc[mask, day_ids] = [arr_ooo]
                mask = shr_daily["metric"].eq("In-Office Shrink Hours (#)")
                shr_daily.loc[mask, day_ids] = [arr_ino]
                mask = shr_daily["metric"].eq("OOO Shrinkage %")
                shr_daily.loc[mask, day_ids] = [arr_ooo_pct]
                mask = shr_daily["metric"].eq("In-Office Shrinkage %")
                shr_daily.loc[mask, day_ids] = [arr_ino_pct]
                mask = shr_daily["metric"].eq("Overall Shrinkage %")
                shr_daily.loc[mask, day_ids] = [arr_ovr_pct]
        # Planned shrink row (same for both)
        plan_pct = _planned_shr_pct(settings, ch_key)
        shr_daily.loc[shr_daily["metric"].eq("Planned Shrinkage %"), day_ids] = [[plan_pct for _ in day_ids]]
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

    def _is_interval_level(df: pd.DataFrame) -> bool:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return False
        L = {str(c).strip().lower(): c for c in df.columns}
        c_ivl = L.get("interval") or L.get("time")
        if not c_ivl or c_ivl not in df.columns:
            return False
        return df[c_ivl].notna().any()

    def _series_day_aware(df: pd.DataFrame, val_col: str, default_agg: str = "sum", no_agg_if_daylevel: bool = True) -> dict:
        """Return date->value mapping.
        - If interval-level data present and default_agg provided: aggregate by date using default_agg.
        - If day-level (no interval labels) and no_agg_if_daylevel: pick first non-null per date (no aggregation).
        """
        if not isinstance(df, pd.DataFrame) or df.empty or val_col not in df.columns:
            return {}
        d = df.copy(); d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date.astype(str)
        if _is_interval_level(d):
            return _series_from_df(d, val_col, agg=default_agg)
        # Day-level: no aggregation (take first per date)
        d = d.dropna(subset=["date"])  # keep clean
        out = {}
        for k, sub in d.groupby("date"):
            # first non-null value in val_col
            vals = pd.to_numeric(sub[val_col], errors="coerce")
            first = vals.dropna().iloc[0] if not vals.dropna().empty else np.nan
            out[k] = float(first) if pd.notna(first) else 0.0
        return out

    def _aht_day_aware(df: pd.DataFrame, vol_col: str, aht_col: str, default_agg: str = "mean") -> dict:
        """AHT mapping with interval-aware behavior.
        - If interval-level present: use volume-weighted AHT.
        - If day-level: no aggregation; take first available AHT per day (assumes provided per-day values).
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}
        if _is_interval_level(df):
            # Weighted across intervals
            return _weighted_aht(df, vol_col, aht_col)
        # Day-level: pick first available value per date
        d = df.copy(); d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date.astype(str)
        vals = pd.to_numeric(d.get(aht_col), errors="coerce") if aht_col in d.columns else pd.Series(dtype=float)
        out = {}
        for k, sub in d.groupby("date"):
            s = pd.to_numeric(sub.get(aht_col), errors="coerce") if aht_col in sub.columns else pd.Series(dtype=float)
            first = s.dropna().iloc[0] if not s.dropna().empty else np.nan
            out[k] = float(first) if pd.notna(first) else 0.0
        return out

    def _write_row(df: pd.DataFrame, row_name: str, mapping: dict):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df
        mser = df["metric"].astype(str).str.strip()
        if row_name not in mser.values:
            return df
        # Vectorized assignment: set only columns present in mapping
        target_cols = [d for d in day_ids if (d in df.columns) and (d in mapping) and (mapping.get(d) is not None)]
        if not target_cols:
            return df
        if target_cols:
            values = [[float(mapping[d]) for d in target_cols]]
            df.loc[mser == row_name, target_cols] = values
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
        # Fallback for plans with a single AHT/SUT row
        mvals = fw_d["metric"].astype(str)
        if "AHT/SUT" in mvals.values and not any(x in mvals.values for x in ["Forecast AHT/SUT","Actual AHT/SUT"]):
            use_map = ahtF or ahtA or {}
            fw_d = _write_row(fw_d, "AHT/SUT", use_map)
    # Back Office
    elif ch_key in ("back office", "bo"):
        bF = _assemble_bo(sk, "forecast"); bA = _assemble_bo(sk, "actual"); bT = _assemble_bo(sk, "tactical")
        # No aggregation for BO: treat data as day-level and take first value per day
        volF = _series_day_aware(bF, "items", default_agg="sum", no_agg_if_daylevel=True)
        volA = _series_day_aware(bA, "items", default_agg="sum", no_agg_if_daylevel=True)
        volT = _series_day_aware(bT, "items", default_agg="sum", no_agg_if_daylevel=True)
        ahtF = _aht_day_aware(bF, "items", "aht_sec", default_agg="mean")
        ahtA = _aht_day_aware(bA, "items", "aht_sec", default_agg="mean")
        fw_d = _write_row(fw_d, "Forecast", volF)
        fw_d = _write_row(fw_d, "Tactical Forecast", volT)
        fw_d = _write_row(fw_d, "Actual Volume", volA)
        fw_d = _write_row(fw_d, "Forecast AHT/SUT", ahtF)
        fw_d = _write_row(fw_d, "Actual AHT/SUT", ahtA)
        mvals = fw_d["metric"].astype(str)
        if "AHT/SUT" in mvals.values and not any(x in mvals.values for x in ["Forecast AHT/SUT","Actual AHT/SUT"]):
            use_map = ahtF or ahtA or {}
            fw_d = _write_row(fw_d, "AHT/SUT", use_map)
    # Chat
    elif ch_key == "chat":
        cF = _assemble_chat(sk, "forecast"); cA = _assemble_chat(sk, "actual"); cT = _assemble_chat(sk, "tactical")
        # Interval → day sum; if day-level provided, no aggregation (first per day)
        volF = _series_day_aware(cF, "items", default_agg="sum", no_agg_if_daylevel=True)
        volA = _series_day_aware(cA, "items", default_agg="sum", no_agg_if_daylevel=True)
        volT = _series_day_aware(cT, "items", default_agg="sum", no_agg_if_daylevel=True)
        ahtF = _aht_day_aware(cF, "items", "aht_sec", default_agg="mean")
        ahtA = _aht_day_aware(cA, "items", "aht_sec", default_agg="mean")
        fw_d = _write_row(fw_d, "Forecast", volF)
        fw_d = _write_row(fw_d, "Tactical Forecast", volT)
        fw_d = _write_row(fw_d, "Actual Volume", volA)
        fw_d = _write_row(fw_d, "Forecast AHT/SUT", ahtF)
        fw_d = _write_row(fw_d, "Actual AHT/SUT", ahtA)
        mvals = fw_d["metric"].astype(str)
        if "AHT/SUT" in mvals.values and not any(x in mvals.values for x in ["Forecast AHT/SUT","Actual AHT/SUT"]):
            use_map = ahtF or ahtA or {}
            fw_d = _write_row(fw_d, "AHT/SUT", use_map)
    # Outbound
    elif ch_key in ("outbound", "ob"):
        oF = _assemble_ob(sk, "forecast"); oA = _assemble_ob(sk, "actual"); oT = _assemble_ob(sk, "tactical")
        volF = _series_day_aware(oF, "opc", default_agg="sum", no_agg_if_daylevel=True) or _series_day_aware(oF, "items", default_agg="sum", no_agg_if_daylevel=True)
        volA = _series_day_aware(oA, "opc", default_agg="sum", no_agg_if_daylevel=True) or _series_day_aware(oA, "items", default_agg="sum", no_agg_if_daylevel=True)
        volT = _series_day_aware(oT, "opc", default_agg="sum", no_agg_if_daylevel=True) or _series_day_aware(oT, "items", default_agg="sum", no_agg_if_daylevel=True)
        ahtF = _aht_day_aware(oF, "opc", "aht_sec", default_agg="mean"); ahtA = _aht_day_aware(oA, "opc", "aht_sec", default_agg="mean")
        fw_d = _write_row(fw_d, "Forecast", volF)
        fw_d = _write_row(fw_d, "Tactical Forecast", volT)
        fw_d = _write_row(fw_d, "Actual Volume", volA)
        fw_d = _write_row(fw_d, "Forecast AHT/SUT", ahtF)
        fw_d = _write_row(fw_d, "Actual AHT/SUT", ahtA)
        mvals = fw_d["metric"].astype(str)
        if "AHT/SUT" in mvals.values and not any(x in mvals.values for x in ["Forecast AHT/SUT","Actual AHT/SUT"]):
            use_map = ahtF or ahtA or {}
            fw_d = _write_row(fw_d, "AHT/SUT", use_map)


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
