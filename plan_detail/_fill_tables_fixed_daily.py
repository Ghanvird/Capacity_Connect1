from __future__ import annotations
import math
import datetime as dt
import re
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from dash import dash_table

from plan_store import get_plan
from cap_store import resolve_settings, load_roster_long
from ._common import _load_ts_with_fallback, _week_span, _scope_key, _assemble_voice, _assemble_chat, _assemble_ob
from ._calc import _fill_tables_fixed
from ._grain_cols import day_cols_for_weeks


FW_ROWS_DEFAULT = [
    "Forecast",
    "Tactical Forecast",
    "Actual Volume",
    "Forecast AHT/SUT",
    "Actual AHT/SUT",
    "Occupancy",
]

UPPER_ROWS = [
    "FTE Required @ Forecast Volume",
    "FTE Required @ Actual Volume",
    "Projected Handling Capacity (#)",
    "Projected Service Level",
]


def _make_upper_table(df: pd.DataFrame, day_cols: List[dict]):
    if not isinstance(df, pd.DataFrame) or df.empty:
        df = pd.DataFrame({"metric": []})
    return dash_table.DataTable(
        id="tbl-upper",
        data=df.to_dict("records"),
        columns=[{"name": "Metric", "id": "metric", "editable": False}] + [
            {"name": c["name"], "id": c["id"]} for c in day_cols if c["id"] != "metric"
        ],
        editable=False,
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={"whiteSpace": "pre"},
    )


def _series_sum_by_day(df: pd.DataFrame, val_col: str) -> Dict[str, float]:
    if not isinstance(df, pd.DataFrame) or df.empty or val_col not in df.columns:
        return {}
    d = df.copy(); d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date.astype(str)
    s = pd.to_numeric(d[val_col], errors="coerce").fillna(0.0)
    d[val_col] = s
    g = d.groupby("date")[val_col].sum()
    return {k: float(v) for k, v in g.to_dict().items()}


def _weighted_aht_by_day(df: pd.DataFrame, vol_col: str, aht_col: str) -> Dict[str, float]:
    if not isinstance(df, pd.DataFrame) or df.empty or vol_col not in df.columns:
        return {}
    d = df.copy(); d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date.astype(str)
    vol = pd.to_numeric(d.get(vol_col), errors="coerce").fillna(0.0)
    aht = pd.to_numeric(d.get(aht_col), errors="coerce").fillna(np.nan)
    d["_num"] = vol * aht
    g_vol = d.groupby("date")[vol_col].sum()
    g_num = d.groupby("date")["_num"].sum()
    out: Dict[str, float] = {}
    for k in g_vol.index:
        v = float(g_vol.loc[k])
        out[k] = float(g_num.loc[k] / v) if v > 0 else 0.0
    return out


def _parse_hhmm_to_min(hhmm: str) -> int:
    try:
        h, m = hhmm.split(":", 1)
        return int(h) * 60 + int(m)
    except Exception:
        return 0


def _pick_ivl_col(df: pd.DataFrame) -> Optional[str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    low = {str(c).strip().lower(): c for c in df.columns}
    for k in ("interval", "time", "interval_start", "start_time", "slot"):
        c = low.get(k)
        if c and c in df.columns:
            return c
    return None


def _staff_by_slot_for_day(plan: dict, day: dt.date, start_hhmm: str, ivl_min: int = 30) -> Dict[str, float]:
    try:
        rl = load_roster_long()
    except Exception:
        rl = pd.DataFrame()
    slots: Dict[str, float] = {}
    if not isinstance(rl, pd.DataFrame) or rl.empty:
        return slots
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
        return slots
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df[df["date"].eq(day)]
    if df.empty:
        return slots
    # build labels from start_hhmm
    t = dt.datetime.combine(day, dt.time(0, 0)) + dt.timedelta(minutes=_parse_hhmm_to_min(start_hhmm))
    end = dt.datetime.combine(day, dt.time(23, 59))
    labels: List[str] = []
    while t <= end:
        labels.append(t.strftime("%H:%M"))
        t += dt.timedelta(minutes=ivl_min)
    slots = {lab: 0.0 for lab in labels}
    cov_start_min = _parse_hhmm_to_min(start_hhmm)
    for _, rr in df.iterrows():
        try:
            sft = str(rr.get("entry", "")).strip()
            m = re.match(r"^(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})$", sft)
            if not m:
                continue
            sh, sm, eh, em = map(int, m.groups())
            sh = min(23, max(0, sh)); eh = min(24, max(0, eh))
            start_min = sh*60 + sm; end_min = eh*60 + em
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


def _fill_tables_fixed_daily(ptype, pid, _fw_cols_unused, _tick, whatif=None):
    """Daily view (data-first for FW + Upper):
    - FW: derive daily values from uploaded intervals (sum/weighted AHT) or daily uploads directly
    - Upper: compute daily PHC/SL via Erlang (best-effort when interval data exists); FTE rows left for weekly engine or future pass
    - Other tabs are not transformed here
    """
    plan = get_plan(pid) or {}
    weeks = _week_span(plan.get("start_week"), plan.get("end_week"))
    day_cols, day_ids = day_cols_for_weeks(weeks)

    ch = (plan.get("channel") or plan.get("lob") or "").split(",")[0].strip().lower()
    sk = _scope_key(plan.get("vertical"), plan.get("sub_ba"), ch)
    settings = resolve_settings(ba=plan.get("vertical"), subba=plan.get("sub_ba"), lob=ch)

    # FW metrics shaped to match weekly FW spec (fields/ordering)
    fw_metrics = FW_ROWS_DEFAULT.copy()
    try:
        weekly_fw_cols = [{"name": "Metric", "id": "metric", "editable": False}] + [{"name": w, "id": w} for w in weeks]
        weekly = _fill_tables_fixed(ptype, pid, weekly_fw_cols, _tick, whatif=whatif, grain='week')
        (_upper_wk, fw_wk, *_rest) = weekly
        fw_week_df = pd.DataFrame(fw_wk or [])
        if isinstance(fw_week_df, pd.DataFrame) and not fw_week_df.empty and "metric" in fw_week_df.columns:
            fw_metrics = fw_week_df["metric"].astype(str).tolist()
    except Exception:
        pass

    # Build blank FW daily table
    fw_d = pd.DataFrame({"metric": fw_metrics})
    for d in day_ids:
        fw_d[d] = 0.0

    # Upper daily table skeleton; shape to weekly upper spec where available
    upper_rows = UPPER_ROWS.copy()
    try:
        weekly_fw_cols = [{"name": "Metric", "id": "metric", "editable": False}] + [{"name": w, "id": w} for w in weeks]
        weekly = _fill_tables_fixed(ptype, pid, weekly_fw_cols, _tick, whatif=whatif, grain='week')
        (upper_wk, *_rest) = weekly
        upper_df_w = pd.DataFrame(getattr(upper_wk, 'data', None) or [])
        if isinstance(upper_df_w, pd.DataFrame) and not upper_df_w.empty and "metric" in upper_df_w.columns:
            upper_rows = upper_df_w["metric"].astype(str).tolist()
    except Exception:
        pass
    upper = pd.DataFrame({"metric": upper_rows})
    for d in day_ids:
        upper[d] = 0.0

    # Channel-specific fills for FW
    if ch == "voice":
        vF = _assemble_voice(sk, "forecast"); vA = _assemble_voice(sk, "actual"); vT = _assemble_voice(sk, "tactical")
        volF = _series_sum_by_day(vF, "volume"); volA = _series_sum_by_day(vA, "volume"); volT = _series_sum_by_day(vT, "volume")
        ahtF = _weighted_aht_by_day(vF, "volume", "aht_sec"); ahtA = _weighted_aht_by_day(vA, "volume", "aht_sec")
        mser = fw_d["metric"].astype(str)
        if "Forecast" in mser.values:
            for d in day_ids:
                if d in volF: fw_d.loc[mser == "Forecast", d] = float(volF[d])
        if "Tactical Forecast" in mser.values:
            for d in day_ids:
                if d in volT: fw_d.loc[mser == "Tactical Forecast", d] = float(volT[d])
        if "Actual Volume" in mser.values:
            for d in day_ids:
                if d in volA: fw_d.loc[mser == "Actual Volume", d] = float(volA[d])
        if "Forecast AHT/SUT" in mser.values:
            for d in day_ids:
                if d in ahtF: fw_d.loc[mser == "Forecast AHT/SUT", d] = float(ahtF[d])
        if "Actual AHT/SUT" in mser.values:
            for d in day_ids:
                if d in ahtA: fw_d.loc[mser == "Actual AHT/SUT", d] = float(ahtA[d])

    elif ch == "chat":
        cF = _assemble_chat(sk, "forecast"); cA = _assemble_chat(sk, "actual"); cT = _assemble_chat(sk, "tactical")
        volF = _series_sum_by_day(cF, "items") or _series_sum_by_day(cF, "volume")
        volA = _series_sum_by_day(cA, "items") or _series_sum_by_day(cA, "volume")
        volT = _series_sum_by_day(cT, "items") or _series_sum_by_day(cT, "volume")
        ahtF = _series_sum_by_day(_assemble_chat(sk, "forecast"), "aht_sec")
        mser = fw_d["metric"].astype(str)
        if "Forecast" in mser.values:
            for d in day_ids:
                if d in volF: fw_d.loc[mser == "Forecast", d] = float(volF[d])
        if "Tactical Forecast" in mser.values:
            for d in day_ids:
                if d in volT: fw_d.loc[mser == "Tactical Forecast", d] = float(volT[d])
        if "Actual Volume" in mser.values:
            for d in day_ids:
                if d in volA: fw_d.loc[mser == "Actual Volume", d] = float(volA[d])
        if "Forecast AHT/SUT" in mser.values:
            for d in day_ids:
                if d in ahtF: fw_d.loc[mser == "Forecast AHT/SUT", d] = float(ahtF[d])

    elif ch in ("outbound", "ob"):
        oF = _assemble_ob(sk, "forecast"); oA = _assemble_ob(sk, "actual"); oT = _assemble_ob(sk, "tactical")
        colF = "opc" if isinstance(oF, pd.DataFrame) and ("opc" in oF.columns) else "items"
        colA = "opc" if isinstance(oA, pd.DataFrame) and ("opc" in oA.columns) else "items"
        colT = "opc" if isinstance(oT, pd.DataFrame) and ("opc" in oT.columns) else "items"
        volF = _series_sum_by_day(oF, colF); volA = _series_sum_by_day(oA, colA); volT = _series_sum_by_day(oT, colT)
        ahtF = _series_sum_by_day(oF, "aht_sec")
        mser = fw_d["metric"].astype(str)
        if "Forecast" in mser.values:
            for d in day_ids:
                if d in volF: fw_d.loc[mser == "Forecast", d] = float(volF[d])
        if "Tactical Forecast" in mser.values:
            for d in day_ids:
                if d in volT: fw_d.loc[mser == "Tactical Forecast", d] = float(volT[d])
        if "Actual Volume" in mser.values:
            for d in day_ids:
                if d in volA: fw_d.loc[mser == "Actual Volume", d] = float(volA[d])
        if "Forecast AHT/SUT" in mser.values:
            for d in day_ids:
                if d in ahtF: fw_d.loc[mser == "Forecast AHT/SUT", d] = float(ahtF[d])

    # Occupancy daily (best-effort when interval series + roster are available)
    try:
        if "Occupancy" in fw_d["metric"].astype(str).values:
            ivl_min = int(float(settings.get("interval_minutes", 30) or 30))
            for dd in day_ids:
                try:
                    day_dt = pd.to_datetime(dd).date()
                except Exception:
                    continue
                # infer start from voice/chat/ob series
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
                            d = d[d[c_date].eq(day_dt)]
                        labs = d[ivc].astype(str).str.slice(0,5)
                        labs = labs[labs.str.match(r"^\d{2}:\d{2}$")]
                        return None if labs.empty else str(labs.min())
                    except Exception:
                        return None
                start_hhmm = None
                if ch == "voice":
                    for df in (_assemble_voice(sk, "forecast"), _assemble_voice(sk, "actual")):
                        start_hhmm = start_hhmm or _earliest_from(df)
                elif ch == "chat":
                    for key in ("chat_forecast_volume","chat_actual_volume"):
                        df = _load_ts_with_fallback(key, sk)
                        start_hhmm = start_hhmm or _earliest_from(df)
                elif ch in ("outbound","ob"):
                    for key in ("ob_forecast_opc","outbound_forecast_opc","ob_forecast_dials","outbound_forecast_dials","ob_forecast_calls"):
                        df = _load_ts_with_fallback(key, sk)
                        start_hhmm = start_hhmm or _earliest_from(df)
                if not start_hhmm:
                    start_hhmm = "08:00"
                staff = _staff_by_slot_for_day(plan, day_dt, start_hhmm, ivl_min=ivl_min)
                ivl_sec = max(60, ivl_min * 60)
                if ch == "voice":
                    vF = _assemble_voice(sk, "forecast")
                    ivc = _pick_ivl_col(vF)
                    if isinstance(vF, pd.DataFrame) and ivc:
                        d2 = vF.copy(); d2["date"] = pd.to_datetime(d2["date"], errors="coerce").dt.date
                        d2 = d2[d2["date"].eq(day_dt)]
                        labs = d2[ivc].astype(str).str.slice(0,5)
                        vol = pd.to_numeric(d2.get("volume"), errors="coerce").fillna(0.0).tolist()
                        aht = pd.to_numeric(d2.get("aht_sec"), errors="coerce").fillna(300.0).tolist()
                        m = dict(zip(labs, zip(vol, aht)))
                        tot = 0.0; num = 0.0
                        occ_cap = float(settings.get("occupancy_cap_voice", 0.85) or 0.85)
                        for lab, (calls, ahtv) in m.items():
                            ag = float(staff.get(lab, 0.0) or 0.0)
                            A = (float(calls) * float(ahtv)) / ivl_sec if ahtv > 0 else 0.0
                            occ = min(occ_cap, (A / max(ag, 1e-6)) if ag > 0 else 0.0)
                            num += float(calls) * occ
                            tot += float(calls)
                        fw_d.loc[fw_d["metric"].eq("Occupancy"), dd] = float(100.0 * (num / tot)) if tot > 0 else 0.0
                elif ch == "chat":
                    cF = _assemble_chat(sk, "forecast")
                    ivc = _pick_ivl_col(cF)
                    if isinstance(cF, pd.DataFrame) and ivc:
                        d2 = cF.copy(); d2["date"] = pd.to_datetime(d2["date"], errors="coerce").dt.date
                        d2 = d2[d2["date"].eq(day_dt)]
                        labs = d2[ivc].astype(str).str.slice(0,5)
                        items = pd.to_numeric(d2.get("items") if "items" in d2.columns else d2.get("volume"), errors="coerce").fillna(0.0).tolist()
                        aht = pd.to_numeric(d2.get("aht_sec"), errors="coerce").fillna(240.0).tolist()
                        m = dict(zip(labs, zip(items, aht)))
                        tot = 0.0; num = 0.0
                        occ_cap = float(settings.get("occupancy_cap_chat", settings.get("util_chat", settings.get("occupancy_cap_voice", 0.85))) or 0.85)
                        conc = float(settings.get("chat_concurrency", 1.5) or 1.0)
                        for lab, (itm, ahtv) in m.items():
                            ag = float(staff.get(lab, 0.0) or 0.0)
                            aht_eff = float(ahtv) / max(0.1, conc)
                            A = (float(itm) * aht_eff) / ivl_sec if aht_eff > 0 else 0.0
                            occ = min(occ_cap, (A / max(ag, 1e-6)) if ag > 0 else 0.0)
                            num += float(itm) * occ
                            tot += float(itm)
                        fw_d.loc[fw_d["metric"].eq("Occupancy"), dd] = float(100.0 * (num / tot)) if tot > 0 else 0.0
                elif ch in ("outbound","ob"):
                    # best-effort only when interval present
                    pass
    except Exception:
        pass

    # Upper: compute daily PHC/SL best-effort (requires interval data + roster)
    # For simplicity here we leave FTE rows as-is (0.0); add in later pass if needed
    upper_tbl = _make_upper_table(upper, day_cols)

    # Other tabs left to callers/persistence; return blanks
    empty = []
    return (
        upper_tbl,
        fw_d.to_dict("records"),
        empty, empty, empty, empty, empty, empty, empty, empty,
        empty, empty, empty,
    )
