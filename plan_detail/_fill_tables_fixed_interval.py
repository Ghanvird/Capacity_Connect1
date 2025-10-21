
from __future__ import annotations
import math
import re
import datetime as dt
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from dash import dash_table

from plan_store import get_plan
from cap_store import resolve_settings, load_roster_long
from ._grain_cols import interval_cols_for_day
from ._common import (
    _scope_key,
    _assemble_voice,
    _assemble_chat,
    _assemble_ob,
    _load_ts_with_fallback,
)


def _pick_ivl_col(df: pd.DataFrame) -> Optional[str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    low = {str(c).strip().lower(): c for c in df.columns}
    for k in ("interval", "time", "interval_start", "start_time", "slot"):
        c = low.get(k)
        if c and c in df.columns:
            return c
    return None


def _slot_series_for_day(df: pd.DataFrame, day: dt.date, val_col: str) -> Dict[str, float]:
    if not isinstance(df, pd.DataFrame) or df.empty or val_col not in df.columns:
        return {}
    d = df.copy()
    L = {str(c).strip().lower(): c for c in d.columns}
    c_date = L.get("date") or L.get("day")
    c_ivl = _pick_ivl_col(d)
    if not c_ivl:
        return {}
    if c_date:
        d[c_date] = pd.to_datetime(d[c_date], errors="coerce").dt.date
        d = d[d[c_date].eq(day)]
    if d.empty:
        return {}
    labs = d[c_ivl].astype(str).str.slice(0, 5)
    vals = pd.to_numeric(d.get(val_col), errors="coerce").fillna(0.0)
    g = pd.DataFrame({"lab": labs, "val": vals}).groupby("lab", as_index=True)["val"].sum()
    return {str(k): float(v) for k, v in g.to_dict().items()}


def _infer_start_hhmm(plan: dict, day: dt.date, ch: str, sk: str) -> str:
    def _earliest_from(df: pd.DataFrame) -> Optional[str]:
        try:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return None
            d = df.copy(); L = {str(c).strip().lower(): c for c in d.columns}
            c_date = L.get("date") or L.get("day"); ivc = _pick_ivl_col(d)
            if not ivc:
                return None
            if c_date:
                d[c_date] = pd.to_datetime(d[c_date], errors="coerce").dt.date
                d = d[d[c_date].eq(day)]
            if d.empty:
                return None
            labs = d[ivc].astype(str).str.slice(0, 5)
            labs = labs[labs.str.match(r"^\d{2}:\d{2}$")]
            return None if labs.empty else str(labs.min())
        except Exception:
            return None

    start = None
    try:
        if ch == "voice":
            for df in (_assemble_voice(sk, "forecast"), _assemble_voice(sk, "actual")):
                start = start or _earliest_from(df)
        elif ch == "chat":
            for key in ("chat_forecast_volume", "chat_actual_volume"):
                df = _load_ts_with_fallback(key, sk)
                start = start or _earliest_from(df)
        elif ch in ("outbound", "ob"):
            for key in (
                "ob_forecast_opc","outbound_forecast_opc","ob_actual_opc","outbound_actual_opc",
                "ob_forecast_dials","outbound_forecast_dials","ob_actual_dials","outbound_actual_dials",
                "ob_forecast_calls","outbound_forecast_calls","ob_actual_calls","outbound_actual_calls",
            ):
                df = _load_ts_with_fallback(key, sk)
                start = start or _earliest_from(df)
    except Exception:
        start = None
    return start or "08:00"


def _staff_by_slot_for_day(plan: dict, day: dt.date, ivl_ids: List[str], start_hhmm: str, ivl_min: int) -> Dict[str, float]:
    try:
        rl = load_roster_long()
    except Exception:
        return {lab: 0.0 for lab in ivl_ids}
    if not isinstance(rl, pd.DataFrame) or rl.empty:
        return {lab: 0.0 for lab in ivl_ids}
    df = rl.copy()
    def _col(opts):
        for c in opts:
            if c in df.columns:
                return c
        return None
    c_ba  = _col(["Business Area","business area","vertical"]) 
    c_sba = _col(["Sub Business Area","sub business area","sub_ba"]) 
    c_lob = _col(["LOB","lob","Channel","channel"]) 
    c_site= _col(["Site","site","Location","location","Country","country"]) 
    BA  = plan.get("vertical"); SBA = plan.get("sub_ba"); LOB = (plan.get("channel") or plan.get("lob") or "").split(",")[0].strip()
    SITE= (plan.get("site") or plan.get("location") or plan.get("country") or "").strip()
    def _match(series, val):
        if not val or not isinstance(series, pd.Series):
            return pd.Series(True, index=series.index)
        s = series.astype(str).str.strip().str.lower()
        return s.eq(str(val).strip().lower())
    msk = pd.Series(True, index=df.index)
    if c_ba:  msk &= _match(df[c_ba], BA)
    if c_sba and (SBA not in (None, "")): msk &= _match(df[c_sba], SBA)
    if c_lob: msk &= _match(df[c_lob], LOB)
    if c_site and (SITE not in (None, "")): msk &= _match(df[c_site], SITE)
    df = df[msk]
    if "is_leave" in df.columns:
        df = df[~df["is_leave"].astype(bool)]
    if "date" not in df.columns or "entry" not in df.columns:
        return {lab: 0.0 for lab in ivl_ids}
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df[df["date"].eq(day)]
    slots = {lab: 0.0 for lab in ivl_ids}
    if df.empty:
        return slots
    def _parse_hhmm_to_min(hhmm: str) -> int:
        try:
            h, m = hhmm.split(":", 1)
            return int(h) * 60 + int(m)
        except Exception:
            return 0
    cov_start_min = _parse_hhmm_to_min(start_hhmm)
    for _, rr in df.iterrows():
        try:
            sft = str(rr.get("entry", "")).strip()
            m = re.match(r"^(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})$", sft)
            if not m:
                continue
            sh, sm, eh, em = map(int, m.groups())
            sh = min(23, max(0, sh)); eh = min(24, max(0, eh))
            start_min = sh*60 + sm
            end_min   = eh*60 + em
            if end_min <= start_min:
                end_min += 24*60
            for idx2, lab2 in enumerate(ivl_ids):
                slot_abs = cov_start_min + idx2*ivl_min
                slot_rel = slot_abs
                if slot_rel < start_min:
                    slot_rel += 24*60
                if start_min <= slot_rel < end_min:
                    slots[lab2] = slots.get(lab2, 0.0) + 1.0
        except Exception:
            continue
    return slots


def _erlang_c(A: float, N: int) -> float:
    if N <= 0:
        return 1.0
    if A <= 0:
        return 0.0
    if A >= N:
        return 1.0
    term = 1.0; ssum = term
    for k in range(1, N):
        term *= A / k
        ssum += term
    term *= A / N
    last = term * (N / (N - A))
    denom = ssum + last
    if denom <= 0:
        return 1.0
    p0 = 1.0 / denom
    return last * p0


def _erlang_sl(calls: float, aht: float, agents: float, ivl_sec: float, T_sec: float) -> float:
    if aht <= 0 or ivl_sec <= 0 or agents <= 0:
        return 0.0
    if calls <= 0:
        return 1.0
    A = (calls * aht) / ivl_sec
    pw = _erlang_c(A, int(max(1, math.floor(agents))))
    return max(0.0, min(1.0, 1.0 - pw * math.exp(-max(0.0, (agents - A)) * (T_sec / max(1.0, aht)))))


def _make_upper_table(df: pd.DataFrame, ivl_cols: List[dict]):
    if not isinstance(df, pd.DataFrame) or df.empty:
        df = pd.DataFrame({"metric": []})
    return dash_table.DataTable(
        id="tbl-upper",
        data=df.to_dict("records"),
        columns=[{"name": "Metric", "id": "metric", "editable": False}] + [
            {"name": c["name"], "id": c["id"]} for c in ivl_cols if c["id"] != "metric"
        ],
        editable=False,
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={"whiteSpace": "pre"},
    )


def _fill_tables_fixed_interval(ptype, pid, _fw_cols_unused, _tick, whatif=None, ivl_min: int = 30, sel_date: Optional[str] = None):
    """Interval view (data-first):
    - Render FW intervals exactly as uploaded for the selected date
    - Compute Upper (PHC/SL) and FW Occupancy via Erlang using uploaded intervals + roster
    - Other grids are left empty (or can be loaded from persistence by callers)
    """
    plan = get_plan(pid) or {}
    # pick representative date
    if sel_date:
        try:
            ref_day = pd.to_datetime(sel_date).date()
        except Exception:
            ref_day = dt.date.today()
    else:
        ref_day = dt.date.today()

    ch = (plan.get("channel") or plan.get("lob") or "").split(",")[0].strip().lower()
    sk = _scope_key(plan.get("vertical"), plan.get("sub_ba"), ch)
    settings = resolve_settings(ba=plan.get("vertical"), subba=plan.get("sub_ba"), lob=ch)

    start_hhmm = _infer_start_hhmm(plan, ref_day, ch, sk)
    ivl_cols, ivl_ids = interval_cols_for_day(ref_day, ivl_min=ivl_min, start_hhmm=start_hhmm)

    # FW metrics
    try:
        fw_saved = _load_ts_with_fallback(f"plan_{pid}_fw", "")  # not a real scope; try cap_db
        fw_metrics = []
    except Exception:
        fw_saved = None
        fw_metrics = []
    if isinstance(fw_saved, pd.DataFrame) and not fw_saved.empty and "metric" in fw_saved.columns:
        fw_metrics = fw_saved["metric"].astype(str).tolist()
    if not fw_metrics:
        fw_metrics = [
            "Forecast", "Tactical Forecast", "Actual Volume",
            "Forecast AHT/SUT", "Actual AHT/SUT", "Occupancy",
        ]
    fw_i = pd.DataFrame({"metric": fw_metrics})
    for lab in ivl_ids:
        fw_i[lab] = np.nan

    # Upper rows
    upper = pd.DataFrame({"metric": [
        "FTE Required @ Forecast Volume",
        "FTE Required @ Actual Volume",
        "Projected Handling Capacity (#)",
        "Projected Service Level",
    ]})
    for lab in ivl_ids:
        upper[lab] = 0.0

    ivl_sec = max(60, int(ivl_min) * 60)
    T_sec = float(settings.get("sl_seconds", 20) or 20.0)
    target_sl = float(settings.get("target_sl", 0.8) or 0.8)

    # Channel-specific fills
    if ch == "voice":
        vF = _assemble_voice(sk, "forecast"); vA = _assemble_voice(sk, "actual")
        volF = _slot_series_for_day(vF, ref_day, "volume")
        volA = _slot_series_for_day(vA if isinstance(vA, pd.DataFrame) and not vA.empty else vF, ref_day, "volume")
        ahtF = _slot_series_for_day(vF, ref_day, "aht_sec")
        ahtA = _slot_series_for_day(vA, ref_day, "aht_sec")
        mser = fw_i["metric"].astype(str)
        for lab in ivl_ids:
            if lab in volF and "Forecast" in mser.values:
                fw_i.loc[mser == "Forecast", lab] = float(volF[lab])
            if lab in volA and "Actual Volume" in mser.values:
                fw_i.loc[mser == "Actual Volume", lab] = float(volA[lab])
            if lab in ahtF and "Forecast AHT/SUT" in mser.values:
                fw_i.loc[mser == "Forecast AHT/SUT", lab] = float(ahtF[lab])
            if lab in ahtA and "Actual AHT/SUT" in mser.values:
                fw_i.loc[mser == "Actual AHT/SUT", lab] = float(ahtA[lab])
        # Staffing and Erlang rollups
        staff = _staff_by_slot_for_day(plan, ref_day, ivl_ids, start_hhmm, ivl_min)
        occ_cap = float(settings.get("occupancy_cap_voice", 0.85) or 0.85)
        for lab in ivl_ids:
            ag = float(staff.get(lab, 0.0) or 0.0)
            calls = float(volF.get(lab, 0.0))
            aht = float(ahtF.get(lab, ahtF.get(next(iter(ahtF), lab), 300.0)) or 300.0)
            cap = 0.0; sl = 0.0
            if aht > 0 and ivl_sec > 0:
                # capacity at target SL via search with occupancy cap
                # simple monotone search up to occupancy cap-limited calls
                # occupancy-limited calls:
                occ_calls = (occ_cap * ag * ivl_sec) / max(1.0, aht)
                # binary search for SL target
                lo, hi = 0, int(max(1, (ag * ivl_sec) / max(1.0, aht)))
                if _erlang_sl(hi, aht, ag, ivl_sec, T_sec) < target_sl:
                    cap = float(min(hi, occ_calls))
                else:
                    while _erlang_sl(hi, aht, ag, ivl_sec, T_sec) >= target_sl and hi < 10_000_000:
                        lo = hi; hi *= 2
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if _erlang_sl(mid, aht, ag, ivl_sec, T_sec) >= target_sl:
                            lo = mid
                        else:
                            hi = mid - 1
                    cap = float(min(lo, occ_calls))
                sl = 100.0 * _erlang_sl(calls, aht, ag, ivl_sec, T_sec)
            upper.loc[upper["metric"].eq("Projected Handling Capacity (#)"), lab] = cap
            upper.loc[upper["metric"].eq("Projected Service Level"), lab] = sl
            # FW Occupancy
            A = (calls * aht) / ivl_sec if aht > 0 and ivl_sec > 0 else 0.0
            occ = 100.0 * min(occ_cap, (A / max(ag, 1e-6)) if ag > 0 else 0.0)
            if "Occupancy" in mser.values:
                fw_i.loc[mser == "Occupancy", lab] = occ

    elif ch == "chat":
        volF = _load_ts_with_fallback("chat_forecast_volume", sk)
        volA = _load_ts_with_fallback("chat_actual_volume", sk)
        ahtF = _load_ts_with_fallback("chat_forecast_aht", sk)
        volF_map = _slot_series_for_day(volF, ref_day, "items") or _slot_series_for_day(volF, ref_day, "volume")
        volA_map = _slot_series_for_day(volA if isinstance(volA, pd.DataFrame) and not volA.empty else volF, ref_day, "items")
        aht_map  = _slot_series_for_day(ahtF, ref_day, "aht_sec") or _slot_series_for_day(ahtF, ref_day, "aht")
        mser = fw_i["metric"].astype(str)
        for lab in ivl_ids:
            if lab in volF_map and "Forecast" in mser.values:
                fw_i.loc[mser == "Forecast", lab] = float(volF_map[lab])
            if lab in volA_map and "Actual Volume" in mser.values:
                fw_i.loc[mser == "Actual Volume", lab] = float(volA_map[lab])
            if lab in aht_map and "Forecast AHT/SUT" in mser.values:
                fw_i.loc[mser == "Forecast AHT/SUT", lab] = float(aht_map[lab])
        staff = _staff_by_slot_for_day(plan, ref_day, ivl_ids, start_hhmm, ivl_min)
        ivl_sec = max(60, int(ivl_min) * 60)
        T_sec = float(settings.get("chat_sl_seconds", settings.get("sl_seconds", 20)) or 20.0)
        target_sl = float(settings.get("chat_target_sl", settings.get("target_sl", 0.8)) or 0.8)
        occ_cap = float(settings.get("occupancy_cap_chat", settings.get("util_chat", settings.get("occupancy_cap_voice", 0.85))) or 0.85)
        conc = float(settings.get("chat_concurrency", 1.5) or 1.0)
        for lab in ivl_ids:
            ag = float(staff.get(lab, 0.0) or 0.0)
            items = float(volF_map.get(lab, 0.0))
            aht = float(aht_map.get(lab, aht_map.get(next(iter(aht_map), lab), 240.0)) or 240.0) / max(0.1, conc)
            # capacity search (as voice)
            occ_calls = (occ_cap * ag * ivl_sec) / max(1.0, aht)
            lo, hi = 0, int(max(1, (ag * ivl_sec) / max(1.0, aht)))
            if _erlang_sl(hi, aht, ag, ivl_sec, T_sec) < target_sl:
                cap = float(min(hi, occ_calls))
            else:
                while _erlang_sl(hi, aht, ag, ivl_sec, T_sec) >= target_sl and hi < 10_000_000:
                    lo = hi; hi *= 2
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if _erlang_sl(mid, aht, ag, ivl_sec, T_sec) >= target_sl:
                        lo = mid
                    else:
                        hi = mid - 1
                cap = float(min(lo, occ_calls))
            sl = 100.0 * _erlang_sl(items, aht, ag, ivl_sec, T_sec)
            upper.loc[upper["metric"].eq("Projected Handling Capacity (#)"), lab] = cap
            upper.loc[upper["metric"].eq("Projected Service Level"), lab] = sl
            A = (items * aht) / ivl_sec if aht > 0 and ivl_sec > 0 else 0.0
            occ = 100.0 * min(occ_cap, (A / max(ag, 1e-6)) if ag > 0 else 0.0)
            if "Occupancy" in mser.values:
                fw_i.loc[mser == "Occupancy", lab] = occ

    elif ch in ("outbound", "ob"):
        # Try expected calls per interval from any available series (best-effort)
        def _first_non_empty(keys: List[str]) -> pd.DataFrame:
            for k in keys:
                df = _load_ts_with_fallback(k, sk)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
            return pd.DataFrame()
        vF = _first_non_empty(["ob_forecast_opc","outbound_forecast_opc","ob_forecast_dials","outbound_forecast_dials","ob_forecast_calls"]) 
        vA = _first_non_empty(["ob_actual_opc","outbound_actual_opc","ob_actual_dials","outbound_actual_dials","ob_actual_calls"]) 
        aF = _load_ts_with_fallback("ob_forecast_aht", sk)
        volF = _slot_series_for_day(vF, ref_day, "opc") or _slot_series_for_day(vF, ref_day, "dials") or _slot_series_for_day(vF, ref_day, "calls") or _slot_series_for_day(vF, ref_day, "volume")
        volA = _slot_series_for_day(vA, ref_day, "opc") or _slot_series_for_day(vA, ref_day, "dials") or _slot_series_for_day(vA, ref_day, "calls") or _slot_series_for_day(vA, ref_day, "volume")
        aht_map = _slot_series_for_day(aF, ref_day, "aht_sec") or _slot_series_for_day(aF, ref_day, "aht")
        mser = fw_i["metric"].astype(str)
        for lab in ivl_ids:
            if lab in volF and "Forecast" in mser.values:
                fw_i.loc[mser == "Forecast", lab] = float(volF[lab])
            if lab in volA and "Actual Volume" in mser.values:
                fw_i.loc[mser == "Actual Volume", lab] = float(volA[lab])
            if lab in aht_map and "Forecast AHT/SUT" in mser.values:
                fw_i.loc[mser == "Forecast AHT/SUT", lab] = float(aht_map[lab])
        staff = _staff_by_slot_for_day(plan, ref_day, ivl_ids, start_hhmm, ivl_min)
        ivl_sec = max(60, int(ivl_min) * 60)
        T_sec = float(settings.get("ob_sl_seconds", settings.get("sl_seconds", 20)) or 20.0)
        target_sl = float(settings.get("ob_target_sl", settings.get("target_sl", 0.8)) or 0.8)
        occ_cap = float(settings.get("occupancy_cap_ob", settings.get("util_ob", settings.get("occupancy_cap_voice", 0.85))) or 0.85)
        for lab in ivl_ids:
            ag = float(staff.get(lab, 0.0) or 0.0)
            calls = float(volF.get(lab, 0.0))
            aht = float(aht_map.get(lab, aht_map.get(next(iter(aht_map), lab), 240.0)) or 240.0)
            # capacity & SL
            occ_calls = (occ_cap * ag * ivl_sec) / max(1.0, aht)
            lo, hi = 0, int(max(1, (ag * ivl_sec) / max(1.0, aht)))
            if _erlang_sl(hi, aht, ag, ivl_sec, T_sec) < target_sl:
                cap = float(min(hi, occ_calls))
            else:
                while _erlang_sl(hi, aht, ag, ivl_sec, T_sec) >= target_sl and hi < 10_000_000:
                    lo = hi; hi *= 2
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if _erlang_sl(mid, aht, ag, ivl_sec, T_sec) >= target_sl:
                        lo = mid
                    else:
                        hi = mid - 1
                cap = float(min(lo, occ_calls))
            sl = 100.0 * _erlang_sl(calls, aht, ag, ivl_sec, T_sec)
            upper.loc[upper["metric"].eq("Projected Handling Capacity (#)"), lab] = cap
            upper.loc[upper["metric"].eq("Projected Service Level"), lab] = sl
            A = (calls * aht) / ivl_sec if aht > 0 and ivl_sec > 0 else 0.0
            occ = 100.0 * min(occ_cap, (A / max(ag, 1e-6)) if ag > 0 else 0.0)
            if "Occupancy" in mser.values:
                fw_i.loc[mser == "Occupancy", lab] = occ

    # Upper table component
    upper_tbl = _make_upper_table(upper, ivl_cols)

    # Other tabs: leave empty (callers can persist/load as needed)
    empty = []
    return (
        upper_tbl,
        fw_i.to_dict("records"),
        empty, empty, empty, empty, empty, empty, empty, empty,
        empty, empty, empty,
    )
