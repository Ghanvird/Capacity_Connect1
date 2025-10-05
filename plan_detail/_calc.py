# file: plan_detail/_calc.py
from __future__ import annotations
import math
import os
import datetime as dt
import pandas as pd
import numpy as np
import dash
from dash import dash_table
from ._common import _learning_curve_for_week, _load_ts_with_fallback
import json, ast
from ._common import (
    _canon_scope,
    _assemble_voice,
    _assemble_bo,
    _weekly_voice,
    _weekly_bo,
    _week_span,
    _settings_volume_aht_overrides,
    _load_or_blank,
    _load_or_empty_roster,
    _load_or_empty_bulk_files,
    _load_or_empty_notes,
    _first_non_empty_ts,
    _weekly_reduce,
    _parse_ratio_setting,
    _round_week_cols_int,
    _blank_grid,
    load_df,
    save_df,
    resolve_settings,
    get_plan,
    get_plan_meta,
    _monday,
    required_fte_daily,
    load_roster_long,
)

# Normalize roster loader (backward-compatible with legacy column names)
def _load_roster_normalized(pid: int) -> pd.DataFrame:
    try:
        df = load_df(f"plan_{pid}_emp")
    except Exception:
        df = pd.DataFrame()
    # canonical columns used by UI/calcs
    cols = [
        "brid","name","class_ref","work_status","role","ftpt_status","ftpt_hours",
        "current_status","training_start","training_end","nesting_start","nesting_end",
        "production_start","terminate_date","team_leader","avp","biz_area","sub_biz_area",
        "lob","loa_date","back_from_loa_date","site",
    ]
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    df = df.copy()
    # legacy mappings
    if "ftpt" in df.columns and "ftpt_status" not in df.columns:
        df["ftpt_status"] = df["ftpt"]
    if "tl" in df.columns and "team_leader" not in df.columns:
        df["team_leader"] = df["tl"]
    if "status" in df.columns:
        if "work_status" not in df.columns:
            df["work_status"] = df["status"]
        if "current_status" not in df.columns:
            df["current_status"] = df["status"]
    if "date_training" in df.columns and "training_start" not in df.columns:
        df["training_start"] = df["date_training"]
    if "date_nesting" in df.columns and "nesting_start" not in df.columns:
        df["nesting_start"] = df["date_nesting"]
    if "date_production" in df.columns and "production_start" not in df.columns:
        df["production_start"] = df["date_production"]
    if "date_loa" in df.columns and "loa_date" not in df.columns:
        df["loa_date"] = df["date_loa"]
    if "date_back_from_loa" in df.columns and "back_from_loa_date" not in df.columns:
        df["back_from_loa_date"] = df["date_back_from_loa"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    out = df[cols].copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].fillna("").astype(str)
    return out
# ------------------------------------------------------------------------------
# New-hire & roster helpers
# ------------------------------------------------------------------------------

def _week_label(d) -> str | None:
    """Return ISO Monday (YYYY-MM-DD) for a date-like value."""
    if not d:
        return None
    t = pd.to_datetime(d, errors="coerce")
    if pd.isna(t):
        return None
    monday = t - pd.to_timedelta(int(getattr(t, "weekday", lambda: t.weekday())()), unit="D")
    return pd.Timestamp(monday).normalize().date().isoformat()


def _nh_effective_count(row) -> int:
    """
    Effective class size:
      - If billable_hc > 0 ? use it.
      - Else Full-Time ? grads_needed
      - Else Part-Time ? ceil(grads_needed / 2)
    """
    billable = pd.to_numeric(row.get("billable_hc"), errors="coerce")
    if pd.notna(billable) and billable > 0:
        return int(billable)

    grads = int(pd.to_numeric(row.get("grads_needed"), errors="coerce") or 0)
    emp   = str(row.get("emp_type", "")).strip().lower()
    if emp == "part-time":
        return int(math.ceil(grads / 2.0))
    return int(grads)


def _weekly_planned_nh_from_classes(pid: str | int, week_ids: list[str]) -> dict[str, int]:
    """
    Planned *additions* per week by Production Start week (one-time step ups).
    Past & current week: ignore Tentative. Future: include Tentative + Confirmed.
    """
    out = {w: 0 for w in week_ids}
    df = load_df(f"plan_{pid}_nh_classes")
    if not isinstance(df, pd.DataFrame) or df.empty or "production_start" not in df.columns:
        return out

    today_w = _monday(dt.date.today()).isoformat()

    d = df.copy()
    d["_w"] = d["production_start"].apply(_week_label)
    for _, r in d.dropna(subset=["_w"]).iterrows():
        w = str(r["_w"])
        if w not in out:
            continue
        status = str(r.get("status", "")).strip().lower()
        if w <= today_w and status == "tentative":
            continue
        out[w] += _nh_effective_count(r)
    return out


def _weekly_actual_nh_from_roster(roster: pd.DataFrame, week_ids: list[str]) -> dict[str, int]:
    """
    Actual joiners (Agents only) by Production Start week.
    """
    out = {w: 0 for w in week_ids}
    if not isinstance(roster, pd.DataFrame) or roster.empty:
        return out

    R = roster.copy()
    L = {str(c).strip().lower(): c for c in R.columns}
    c_role = L.get("role") or L.get("position group") or L.get("position description")
    c_ps   = L.get("production start") or L.get("production_start") or L.get("prod start") or L.get("prod_start")
    if not (c_role and c_ps):
        return out

    role = R[c_role].astype(str).str.strip().str.lower()
    is_agent = role.str.contains(r"\bagent\b", na=False, regex=True)
    R = R.loc[is_agent].copy()
    R["_w"] = R[c_ps].apply(_week_label)
    vc = R["_w"].value_counts(dropna=True)
    for w, n in vc.items():
        if w in out:
            out[w] = int(n)
    return out


def _weekly_hc_step_from_roster(roster: pd.DataFrame, week_ids: list[str], role_regex: str) -> dict[str, int]:
    """Return weekly headcount snapshots derived from roster start/termination dates."""

    if not isinstance(roster, pd.DataFrame) or roster.empty:
        return {w: 0 for w in week_ids}

    R = roster.copy()
    L = {str(c).strip().lower(): c for c in R.columns}
    c_role = L.get("role") or L.get("position group") or L.get("position description")
    c_cur  = L.get("current status") or L.get("current_status") or L.get("status")
    c_work = L.get("work status")    or L.get("work_status")
    c_ps   = L.get("production start") or L.get("production_start") or L.get("prod start") or L.get("prod_start")
    c_term = L.get("terminate date")   or L.get("terminate_date")   or L.get("termination date")
    if not c_role:
        return {w: 0 for w in week_ids}

    eff_series = R[c_cur] if c_cur in R else R.get(c_work, pd.Series("", index=R.index))
    status_norm = eff_series.astype(str).str.strip().str.lower()

    def _status_allows(val: str) -> bool:
        s = (val or "").strip().lower()
        if not s:
            return False
        if "term" in s:
            return True
        return s in {"production", "prod", "in production", "active"}

    status_mask = status_norm.apply(_status_allows)
    role = R[c_role].astype(str).str.strip().str.lower()
    role_mask = role.str.contains(role_regex, na=False, regex=True)
    X = R[role_mask & status_mask].copy()
    if X.empty:
        return {w: 0 for w in week_ids}

    if c_ps in X:
        X["_psw"] = X[c_ps].apply(_week_label)
    else:
        X["_psw"] = None
    if c_term in X:
        X["_tw"] = X[c_term].apply(_week_label)
    else:
        X["_tw"] = None

    diffs = {w: 0 for w in week_ids}
    first_week = week_ids[0]
    base = 0
    for _, r in X.iterrows():
        psw = r.get("_psw"); tw = r.get("_tw")
        started_before = (psw is None) or (psw < first_week)
        terminated_before_or_on = (tw is not None) and (tw <= first_week)
        if started_before and not terminated_before_or_on:
            base += 1
        if psw is not None and psw in diffs and psw >= first_week:
            diffs[psw] += 1
        if tw is not None and tw in diffs and tw >= first_week:
            diffs[tw] -= 1

    out = {}
    running = base
    for w in week_ids:
        running += diffs.get(w, 0)
        out[w] = int(max(0, running))
    return out


def _weekly_attrition_from_roster(roster: pd.DataFrame, week_ids: list[str], role_regex: str) -> dict[str, int]:
    """Count terminations per week for roster rows matching the given role pattern."""

    if not isinstance(roster, pd.DataFrame) or roster.empty:
        return {w: 0 for w in week_ids}

    R = roster.copy()
    L = {str(c).strip().lower(): c for c in R.columns}
    c_role = L.get("role") or L.get("position group") or L.get("position description")
    c_term = L.get("terminate date")   or L.get("terminate_date")   or L.get("termination date")
    if not c_role or not c_term:
        return {w: 0 for w in week_ids}

    role = R[c_role].astype(str).str.strip().str.lower()
    role_mask = role.str.contains(role_regex, na=False, regex=True)
    term_weeks = R.loc[role_mask, c_term].apply(_week_label)

    counts = {w: 0 for w in week_ids}
    for tw in term_weeks:
        if tw and tw in counts:
            counts[tw] += 1
    return counts

# ------------------------------------------------------------------------------
# Main: fill_tables_fixed
# ------------------------------------------------------------------------------

def _fill_tables_fixed(ptype, pid, fw_cols, _tick, whatif=None, grain: str = 'week'):
    # ---- guards ----
    if not (pid and fw_cols):
        raise dash.exceptions.PreventUpdate

    # calendar columns (YYYY-MM-DD Mondays)
    # For monthly view, compute weekly IDs from plan span so downstream calcs remain weekly
    p = get_plan(pid) or {}
    try:
        g = (grain or 'week').lower()
    except Exception:
        g = 'week'
    if g == 'week':
        week_ids = [c["id"] for c in fw_cols if c.get("id") != "metric"]
    else:
        weeks_span = _week_span(p.get("start_week"), p.get("end_week"))
        week_ids = weeks_span

    # ---- read persisted What-If ----
    wf_start = ""
    wf_end   = ""
    wf_ovr   = {}
    try:
        wf_df = load_df(f"plan_{pid}_whatif")
        if isinstance(wf_df, pd.DataFrame) and not wf_df.empty:
            last = wf_df.tail(1).iloc[0]
            wf_start = str(last.get("start_week") or "").strip()
            wf_end   = str(last.get("end_week")   or "").strip()
            raw = last.get("overrides")
            if isinstance(raw, dict):
                wf_ovr = raw
            elif isinstance(raw, str) and raw.strip():
                try:
                    wf_ovr = json.loads(raw)
                except Exception:
                    try:
                        wf_ovr = ast.literal_eval(raw)
                    except Exception:
                        wf_ovr = {}
    except Exception:
        wf_start, wf_end, wf_ovr = "", "", {}

    # merge persisted overrides into live param
    whatif = dict(whatif or {})
    if isinstance(wf_ovr, dict):
        whatif.update(wf_ovr)

    # extract simple dials (with safe defaults)
    def _f(x, d=0.0):
        try: return float(x)
        except Exception: return d
    aht_delta    = _f(whatif.get("aht_delta"),    0.0)   # %
    shrink_delta = _f(whatif.get("shrink_delta"), 0.0)   # %
    attr_delta   = _f(whatif.get("attr_delta"),   0.0)   # HC
    vol_delta    = _f(whatif.get("vol_delta"),    0.0)   # %
    occ_override = whatif.get("occupancy_pct", None)
    backlog_carryover = bool(whatif.get("backlog_carryover", True))

    # per-week Nest/SDA dials (optional)
    _nest_login_w = dict((whatif.get("nesting_login_pct") or {}))
    _nest_ahtm_w  = dict((whatif.get("nesting_aht_multiplier") or {}))

    # helper: active window
    # Default behavior: if no explicit start/end window is set, treat only future weeks as active
    _today_w_default = _monday(dt.date.today()).isoformat()
    def _wf_active(w):
        w = str(w)
        if not wf_start and not wf_end:
            return w > _today_w_default
        if wf_start and w < str(wf_start):
            return False
        if wf_end and w > str(wf_end):
            return False
        return True

    # helpers for per-week nest overrides
    def _ovr_login_frac(w):
        v = _nest_login_w.get(w)
        if v in (None, "") or not _wf_active(w): return None
        try:
            x = float(v)
            if x > 1.0: x /= 100.0
            return max(0.0, min(1.0, x))
        except Exception:
            return None

    def _ovr_aht_mult(w):
        v = _nest_ahtm_w.get(w)
        if v in (None, "") or not _wf_active(w): return None
        try:
            m = float(v)
            return max(0.1, m)
        except Exception:
            return None

    # ---- scope, plan, settings ----
    ch_first = (p.get("channel") or "" or p.get("lob")).split(",")[0].strip()
    sk = _canon_scope(
        p.get("vertical"),
        p.get("sub_ba"),
        ch_first,
        (p.get("site") or p.get("location") or p.get("country") or "").strip(),
    )
    loc_first = (p.get("location") or p.get("country") or p.get("site") or "").strip()
    settings = resolve_settings(ba=p.get("vertical"), subba=p.get("sub_ba"), lob=ch_first)
    try:
        lc_ovr_df = load_df(f"plan_{pid}_lc_overrides")
    except Exception:
        lc_ovr_df = None

    def _lc_with_wf(lc_dict, w):
        out = dict(lc_dict or {})
        p_ = _ovr_login_frac(w)
        m_ = _ovr_aht_mult(w)
        if p_ is not None:
            L = out.get("nesting_prod_pct") or [50,60,70,80]
            out["nesting_prod_pct"] = [float(p_ * 100.0)] * len(L)
        if m_ is not None:
            uplift = (float(m_) - 1.0) * 100.0
            L = out.get("nesting_aht_uplift_pct") or [100,90,80,70]
            out["nesting_aht_uplift_pct"] = [float(uplift)] * len(L)
        return out

    # ---- SLA/AHT/SUT defaults ----
    s_target_aht = float(settings.get("target_aht", settings.get("budgeted_aht", 300)) or 300)
    s_budget_aht = float(settings.get("budgeted_aht", settings.get("target_aht", s_target_aht)) or s_target_aht)
    s_target_sut = float(settings.get("target_sut", settings.get("budgeted_sut", 600)) or 600)
    s_budget_sut = float(settings.get("budgeted_sut", settings.get("target_sut", s_target_sut)) or s_target_sut)
    sl_seconds   = int(settings.get("sl_seconds", 20) or 20)

    planned_aht_df = _load_ts_with_fallback("voice_planned_aht", sk)
    planned_sut_df = _load_ts_with_fallback("bo_planned_sut", sk)
    # Fallbacks: if specific planned series missing, try budget tables directly
    if (not isinstance(planned_aht_df, pd.DataFrame)) or planned_aht_df.empty:
        tmp = _load_ts_with_fallback("voice_budget", sk)
        if isinstance(tmp, pd.DataFrame) and not tmp.empty:
            x = tmp.copy()
            if "week" in x.columns:
                x = x.rename(columns={"budget_aht_sec": "aht_sec"}) if "budget_aht_sec" in x.columns else x
                planned_aht_df = x[[c for c in x.columns if c in ("date","week","aht_sec")]]
    if (not isinstance(planned_sut_df, pd.DataFrame)) or planned_sut_df.empty:
        tmp = _load_ts_with_fallback("bo_budget", sk)
        if isinstance(tmp, pd.DataFrame) and not tmp.empty:
            x = tmp.copy()
            if "week" in x.columns:
                x = x.rename(columns={"budget_sut_sec": "sut_sec"}) if "budget_sut_sec" in x.columns else x
                planned_sut_df = x[[c for c in x.columns if c in ("date","week","sut_sec")]]

    def _ts_week_dict(df: pd.DataFrame, val_candidates: list[str]) -> dict:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}
        d = df.copy()
        if "week" in d.columns:
            d["week"] = pd.to_datetime(d["week"], errors="coerce").dt.date.astype(str)
        elif "date" in d.columns:
            d["week"] = pd.to_datetime(d["date"], errors="coerce").dt.date.astype(str)
        else:
            return {}
        low = {c.lower(): c for c in d.columns}
        vcol = None
        for c in val_candidates:
            vcol = low.get(c.lower())
            if vcol:
                break
        if not vcol:
            return {}
        d[vcol] = pd.to_numeric(d[vcol], errors="coerce")
        return d.dropna(subset=["week", vcol]).set_index("week")[vcol].astype(float).to_dict()

    planned_aht_w = _ts_week_dict(planned_aht_df, ["aht_sec", "sut_sec", "aht", "avg_aht"])
    planned_sut_w = _ts_week_dict(planned_sut_df, ["sut_sec", "aht_sec", "sut", "avg_sut"])

    sl_target_pct = None
    for k in ("sl_target_pct","service_level_target","sl_target","sla_target_pct","sla_target","target_sl"):
        v = settings.get(k)
        if v not in (None, ""):
            try:
                x = float(str(v).replace("%",""))
                sl_target_pct = x * 100.0 if x <= 1.0 else x
            except Exception:
                pass
            try:
                if isinstance(overtime_hours_w, dict) and len(overtime_hours_w) > 0:
                    overtime_w = overtime_w
            except Exception:
                pass
            break
    if sl_target_pct is None:
        sl_target_pct = 80.0

    # ---- helpers ----
    def _pick(df, names):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        for n in names:
            if n in df.columns:
                return n
        return None

    def _get(df, idx, col, default=0.0):
        try:
            if isinstance(df, pd.DataFrame) and (col in df.columns) and (idx in df.index):
                val = df.loc[idx, col]
                return float(val) if pd.notna(val) else default
        except Exception:
            return default
        return default

    def _first_positive(*vals, default=None):
        for v in vals:
            try:
                x = float(v)
                if x > 0:
                    return x
            except Exception:
                pass
        return default

    def _setting(d, keys, default=None):
        if not isinstance(d, dict):
            return default
        for k in keys:
            if d.get(k) not in (None, ""):
                return d.get(k)
        low = {str(k).strip().lower(): v for k, v in d.items()}
        for k in keys:
            kk = str(k).strip().lower()
            if low.get(kk) not in (None, ""):
                return low.get(kk)
        return default

    # ---- assemble time series ----
    vF = _assemble_voice(sk, "forecast"); vA = _assemble_voice(sk, "actual");  vT = _assemble_voice(sk, "tactical")
    bF = _assemble_bo(sk,   "forecast");  bA = _assemble_bo(sk,   "actual");   bT = _assemble_bo(sk,   "tactical")

    use_voice_for_req = vA if isinstance(vA, pd.DataFrame) and not vA.empty else vF
    use_bo_for_req    = bA if isinstance(bA, pd.DataFrame) and not bA.empty else bF

    vF_w = _weekly_voice(vF); vA_w = _weekly_voice(vA); vT_w = _weekly_voice(vT)
    bF_w = _weekly_bo(bF);   bA_w = _weekly_bo(bA);   bT_w = _weekly_bo(bT)
    vF_w = vF_w.set_index("week") if not vF_w.empty else pd.DataFrame()
    vA_w = vA_w.set_index("week") if not vA_w.empty else pd.DataFrame()
    vT_w = vT_w.set_index("week") if not vT_w.empty else pd.DataFrame()
    bF_w = bF_w.set_index("week") if not bF_w.empty else pd.DataFrame()
    bA_w = bA_w.set_index("week") if not bA_w.empty else pd.DataFrame()
    bT_w = bT_w.set_index("week") if not bT_w.empty else pd.DataFrame()

    v_vol_col_F = _pick(vF_w, ["vol","calls","volume"]) or "vol"
    v_vol_col_A = _pick(vA_w, ["vol","calls","volume"]) or v_vol_col_F
    v_vol_col_T = _pick(vT_w, ["vol","calls","volume"]) or v_vol_col_F
    b_itm_col   = _pick(bF_w, ["items","txns","transactions","volume"]) or "items"
    # Prefer explicit seconds columns first to avoid unit confusion
    v_aht_col_F = _pick(vF_w, ["aht_sec","aht","avg_aht"])
    v_aht_col_A = _pick(vA_w, ["aht_sec","aht","avg_aht"])
    b_sut_col_F = _pick(bF_w, ["sut_sec","sut","aht_sec","avg_sut"])
    b_sut_col_A = _pick(bA_w, ["sut_sec","sut","aht_sec","avg_sut"])

    # ---- FW grid shell ----
    spec = (lambda k: {
        "fw": (["Forecast","Tactical Forecast","Actual Volume","Budgeted AHT/SUT","Forecast AHT/SUT","Actual AHT/SUT","Occupancy","Overtime Hours (#)","Backlog (Items)"]
               if (k or "").strip().lower().startswith("volume") else
               ["Billable Hours","AHT/SUT","Shrinkage","Training"] if (k or "").strip().lower().startswith("billable hours") else ["Billable Txns","AHT/SUT","Efficiency","Shrinkage"] if (k or "").strip().lower().startswith("fte based billable") else
               ["Billable FTE Required","Shrinkage","Training"]),
        "upper": (["FTE Required @ Forecast Volume","FTE Required @ Actual Volume","FTE Over/Under MTP Vs Actual","FTE Over/Under Tactical Vs Actual","FTE Over/Under Budgeted Vs Actual","Projected Supply HC","Projected Handling Capacity (#)","Projected Service Level"]
                  if (k or "").strip().lower().startswith("volume") else
                  ["Billable FTE Required (#)","Headcount Required With Shrinkage (#)","FTE Over/Under (#)"] if (k or "").strip().lower().startswith("billable hours") else
                  ["Billable Transactions","FTE Required (#)","FTE Over/Under (#)"] if (k or "").strip().lower().startswith("fte based billable") else
                  ["FTE Required (#)","FTE Over/Under (#)"])})(ptype)

    # Apply per-plan FW/Upper preferences (from plan meta)
    def _meta_list(val):
        if isinstance(val, str):
            try:
                return list(json.loads(val))
            except Exception:
                return []
        if isinstance(val, (list, tuple)):
            return list(val)
        return []
    try:
        meta = get_plan_meta(pid) or {}
    except Exception:
        meta = {}
    lower_opts = set(_meta_list(meta.get("fw_lower_options")))
    upper_opts = set(_meta_list(meta.get("upper_options")))

    fw_rows = [r for r in spec["fw"] if not (str(r) == "Backlog (Items)" and ("backlog" not in lower_opts))]
    if "queue" in lower_opts and "Queue (Items)" not in fw_rows:
        fw_rows = fw_rows + ["Queue (Items)"]
    upper_rows = list(spec["upper"])
    if "req_queue" in upper_opts and "FTE Required @ Queue" not in upper_rows:
        upper_rows = upper_rows + ["FTE Required @ Queue"]
    spec["fw"] = fw_rows
    spec["upper"] = upper_rows
    fw = pd.DataFrame({"metric": fw_rows})
    for w in week_ids:
        fw[w] = 0.0

    # ---- weekly demand + AHT/SUT actual/forecast ----
    wk_aht_sut_actual, wk_aht_sut_forecast, wk_aht_sut_budget = {}, {}, {}
    ivl_min = int(float(settings.get("interval_minutes", 30)) or 30)
    ivl_sec = 60 * ivl_min

    weekly_voice_intervals = {}
    try:
        if isinstance(vF, pd.DataFrame) and not vF.empty and {"date","interval_start"}.issubset(vF.columns):
            tmp = vF.copy()
            dts = pd.to_datetime(tmp["date"], errors="coerce")
            tmp["week"] = (dts - pd.to_timedelta(dts.dt.weekday, unit="D")).dt.date.astype(str)
            weekly_voice_intervals = tmp.groupby("week", as_index=False)["interval_start"].count().set_index("week")["interval_start"].to_dict()
    except Exception:
        weekly_voice_intervals = {}
    intervals_per_week_default = 7 * (24 * 3600 // ivl_sec)

    weekly_demand_voice, weekly_demand_bo = {}, {}
    voice_ovr = _settings_volume_aht_overrides(sk, "voice")
    bo_ovr    = _settings_volume_aht_overrides(sk, "bo")

    for w in week_ids:
        f_voice = _get(vF_w, w, v_vol_col_F, 0.0) if v_vol_col_F else 0.0
        f_bo    = _get(bF_w, w, b_itm_col,   0.0) if b_itm_col   else 0.0
        a_voice = _get(vA_w, w, v_vol_col_A, 0.0) if v_vol_col_A else 0.0
        a_bo    = _get(bA_w, w, b_itm_col,   0.0) if b_itm_col   else 0.0
        t_voice = _get(vT_w, w, v_vol_col_T, 0.0) if v_vol_col_T else 0.0
        t_bo    = _get(bT_w, w, b_itm_col,   0.0) if b_itm_col   else 0.0

        # settings overrides
        if w in voice_ovr["vol_w"]:
            f_voice = voice_ovr["vol_w"][w]
        if w in bo_ovr["vol_w"]:
            f_bo = bo_ovr["vol_w"][w]

        # What-If: increase/decrease Forecast volumes
        if _wf_active(w) and vol_delta:
            f_voice *= (1.0 + vol_delta / 100.0)
            f_bo    *= (1.0 + vol_delta / 100.0)

        weekly_demand_voice[w] = a_voice if a_voice > 0 else (f_voice if f_voice > 0 else t_voice)
        weekly_demand_bo[w]    = a_bo    if a_bo    > 0 else (f_bo    if f_bo    > 0 else t_bo)

        if "Forecast" in fw_rows:
            fw.loc[fw["metric"] == "Forecast", w] = f_voice + f_bo
        if "Tactical Forecast" in fw_rows:
            fw.loc[fw["metric"] == "Tactical Forecast", w] = t_voice + t_bo
        if "Actual Volume" in fw_rows:
            fw.loc[fw["metric"] == "Actual Volume", w] = a_voice + a_bo

        # Actual AHT/SUT (weighted)
        a_num = a_den = 0.0
        if v_aht_col_A:
            a_num += _get(vA_w, w, v_aht_col_A, 0.0) * _get(vA_w, w, v_vol_col_A, 0.0); a_den += _get(vA_w, w, v_vol_col_A, 0.0)
        if b_sut_col_A:
            a_num += _get(bA_w, w, b_sut_col_A, 0.0) * _get(bA_w, w, b_itm_col,   0.0); a_den += _get(bA_w, w, b_itm_col,   0.0)
        actual_aht_sut = (a_num / a_den) if a_den > 0 else 0.0
        actual_aht_sut = float(actual_aht_sut) if pd.notna(actual_aht_sut) else 0.0
        actual_aht_sut = max(0.0, actual_aht_sut)
        wk_aht_sut_actual[w] = actual_aht_sut
        if "Actual AHT/SUT" in fw_rows:
            fw.loc[fw["metric"] == "Actual AHT/SUT", w] = actual_aht_sut

        # Forecast AHT/SUT (settings overrides-aware)
        ovr_aht_voice = voice_ovr["aht_or_sut_w"].get(w)
        ovr_sut_bo    = bo_ovr["aht_or_sut_w"].get(w)
        f_num = f_den = 0.0
        if ovr_aht_voice is not None and f_voice > 0:
            f_num += ovr_aht_voice * f_voice; f_den += f_voice
        elif v_aht_col_F:
            f_num += _get(vF_w, w, v_aht_col_F, 0.0) * _get(vF_w, w, v_vol_col_F, 0.0); f_den += _get(vF_w, w, v_vol_col_F, 0.0)
        if ovr_sut_bo is not None and f_bo > 0:
            f_num += ovr_sut_bo * f_bo; f_den += f_bo
        elif b_sut_col_F:
            f_num += _get(bF_w, w, b_sut_col_F, 0.0) * _get(bF_w, w, b_itm_col, 0.0); f_den += _get(bF_w, w, b_itm_col, 0.0)
        if f_den > 0:
            forecast_aht_sut = (f_num / f_den)
        else:
            # Robust fallback: use planned/budgeted AHT/SUT weighted by forecast loads (or default targets)
            bud_aht = planned_aht_w.get(w, s_budget_aht)
            bud_sut = planned_sut_w.get(w, s_budget_sut)
            fb_num = 0.0; fb_den = 0.0
            if f_voice > 0:
                fb_num += float(bud_aht) * float(f_voice); fb_den += float(f_voice)
            if f_bo > 0:
                fb_num += float(bud_sut) * float(f_bo);    fb_den += float(f_bo)
            forecast_aht_sut = (fb_num / fb_den) if fb_den > 0 else float(s_budget_aht)
        forecast_aht_sut = float(forecast_aht_sut) if pd.notna(forecast_aht_sut) else 0.0
        forecast_aht_sut = max(0.0, forecast_aht_sut)
        # Reflect What-If AHT/SUT delta in FW row for non-Voice channels as well
        try:
            _ch = str(ch_first or '').strip().lower()
            if _wf_active(w) and aht_delta and _ch not in ("voice",):
                forecast_aht_sut = max(0.0, forecast_aht_sut * (1.0 + aht_delta / 100.0))
        except Exception:
            pass
        wk_aht_sut_forecast[w] = forecast_aht_sut

        # Budgeted & Forecast AHT/SUT rows
        b_num = b_den = 0.0
        bud_aht = planned_aht_w.get(w, s_budget_aht)
        bud_sut = planned_sut_w.get(w, s_budget_sut)
        if f_voice > 0:
            b_num += bud_aht * f_voice; b_den += f_voice
        if f_bo > 0:
            b_num += bud_sut * f_bo; b_den += f_bo
        budget_aht_sut = (b_num / b_den) if b_den > 0 else 0.0
        budget_aht_sut = float(budget_aht_sut) if pd.notna(budget_aht_sut) else 0.0
        budget_aht_sut = max(0.0, budget_aht_sut)
        if "Budgeted AHT/SUT" in fw_rows:
            fw.loc[fw["metric"] == "Budgeted AHT/SUT", w] = budget_aht_sut
        if "Forecast AHT/SUT" in fw_rows:
            fw.loc[fw["metric"] == "Forecast AHT/SUT", w] = forecast_aht_sut
        wk_aht_sut_budget[w] = budget_aht_sut

    # Override FW grid for Voice channel to be Voice-only (no BO included)
    ch_key = str(ch_first or '').strip().lower()
    if ch_key == 'voice':
        for w in week_ids:
            # Volumes (Voice only)
            f_voice = _get(vF_w, w, v_vol_col_F, 0.0) if v_vol_col_F else 0.0
            a_voice = _get(vA_w, w, v_vol_col_A, 0.0) if v_vol_col_A else 0.0
            t_voice = _get(vT_w, w, v_vol_col_T, 0.0) if v_vol_col_T else 0.0
            if _wf_active(w) and vol_delta:
                f_voice *= (1.0 + vol_delta / 100.0)
            if "Forecast" in fw_rows:
                fw.loc[fw["metric"] == "Forecast", w] = f_voice
            if "Tactical Forecast" in fw_rows:
                fw.loc[fw["metric"] == "Tactical Forecast", w] = t_voice
            if "Actual Volume" in fw_rows:
                fw.loc[fw["metric"] == "Actual Volume", w] = a_voice

            # Actual AHT/SUT (Voice only weighted)
            a_num = a_den = 0.0
            if v_aht_col_A:
                vv = _get(vA_w, w, v_vol_col_A, 0.0)
                aa = _get(vA_w, w, v_aht_col_A, 0.0)
                if vv > 0 and aa > 0:
                    a_num += aa * vv; a_den += vv
            actual_aht_sut = (a_num / a_den) if a_den > 0 else 0.0
            actual_aht_sut = float(actual_aht_sut) if pd.notna(actual_aht_sut) else 0.0
            actual_aht_sut = max(0.0, actual_aht_sut)
            if "Actual AHT/SUT" in fw_rows:
                fw.loc[fw["metric"] == "Actual AHT/SUT", w] = actual_aht_sut
            elif "AHT/SUT" in fw_rows:
                fw.loc[fw["metric"] == "AHT/SUT", w] = actual_aht_sut
            wk_aht_sut_actual[w] = actual_aht_sut

            # Forecast AHT/SUT (Voice only; settings overrides honored)
            f_num = f_den = 0.0
            ovr_aht_voice = voice_ovr["aht_or_sut_w"].get(w)
            if ovr_aht_voice is not None and f_voice > 0:
                f_num += ovr_aht_voice * f_voice; f_den += f_voice
            elif v_aht_col_F:
                vv = _get(vF_w, w, v_vol_col_F, 0.0)
                aa = _get(vF_w, w, v_aht_col_F, 0.0)
                if vv > 0 and aa > 0:
                    f_num += aa * vv; f_den += vv
            # If no forecast AHT found, fall back to planned/budgeted AHT
            if f_den > 0:
                forecast_aht_sut = (f_num / f_den)
            else:
                forecast_aht_sut = float(planned_aht_w.get(w, s_budget_aht))
            forecast_aht_sut = float(forecast_aht_sut) if pd.notna(forecast_aht_sut) else 0.0
            forecast_aht_sut = max(0.0, forecast_aht_sut)
            # What-If: reflect AHT delta in FW row so user sees impact
            try:
                if _wf_active(w) and aht_delta:
                    forecast_aht_sut = max(0.0, forecast_aht_sut * (1.0 + aht_delta / 100.0))
            except Exception:
                pass
            if "Forecast AHT/SUT" in fw_rows:
                fw.loc[fw["metric"] == "Forecast AHT/SUT", w] = forecast_aht_sut
            elif "AHT/SUT" in fw_rows:
                fw.loc[fw["metric"] == "AHT/SUT", w] = forecast_aht_sut
            wk_aht_sut_forecast[w] = forecast_aht_sut

            # Budgeted AHT/SUT (Voice planned) – do not apply What-If AHT delta here
            bud_aht = planned_aht_w.get(w, s_budget_aht)
            if "Budgeted AHT/SUT" in fw_rows:
                fw.loc[fw["metric"] == "Budgeted AHT/SUT", w] = bud_aht
            elif "AHT/SUT" in fw_rows and "Forecast AHT/SUT" not in fw_rows and "Actual AHT/SUT" not in fw_rows:
                # Only seed Budget into AHT/SUT if it hasn't been set by Forecast/Actual (keep non-zero)
                try:
                    cur = float(pd.to_numeric(fw.loc[fw["metric"] == "AHT/SUT", w], errors="coerce").fillna(0.0).iloc[0])
                except Exception:
                    cur = 0.0
                if cur <= 0.0:
                    fw.loc[fw["metric"] == "AHT/SUT", w] = bud_aht
            wk_aht_sut_budget[w] = bud_aht

    # Compute Backlog (Items) and Queue (Items) as selected
    backlog_w_local = {}
    if "Backlog (Items)" in fw_rows:
        for w in week_ids:
            try:
                fval = float(pd.to_numeric(fw.loc[fw["metric"] == "Forecast", w], errors="coerce").fillna(0.0).iloc[0]) if w in fw.columns else 0.0
            except Exception:
                fval = 0.0
            try:
                aval = float(pd.to_numeric(fw.loc[fw["metric"] == "Actual Volume", w], errors="coerce").fillna(0.0).iloc[0]) if w in fw.columns else 0.0
            except Exception:
                aval = 0.0
            # Business rule: Backlog = Actual - Forecast (clamped at 0)
            bl = max(0.0, aval - fval)
            backlog_w_local[w] = bl
            fw.loc[fw["metric"] == "Backlog (Items)", w] = bl
    queue_w = {}
    if "Queue (Items)" in fw_rows:
        for i, w in enumerate(week_ids):
            prev_bl = float(backlog_w_local.get(week_ids[i-1], 0.0)) if i > 0 else 0.0
            try:
                fval = float(pd.to_numeric(fw.loc[fw["metric"] == "Forecast", w], errors="coerce").fillna(0.0).iloc[0]) if w in fw.columns else 0.0
            except Exception:
                fval = 0.0
            qv = max(0.0, prev_bl + fval)
            queue_w[w] = qv
            fw.loc[fw["metric"] == "Queue (Items)", w] = qv

    fw_saved = load_df(f"plan_{pid}_fw")

    def _row_to_week_dict(df: pd.DataFrame, metric_name: str) -> dict:
        try:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return {}
            m = df["metric"].astype(str).str.strip()
            if metric_name not in m.values:
                return {}
            row = df.loc[m == metric_name].iloc[0]
            out = {}
            for w in week_ids:
                try:
                    out[w] = float(pd.to_numeric(row.get(w), errors="coerce"))
                except Exception:
                    out[w] = 0.0
            return out
        except Exception:
            return {}

    overtime_w = _row_to_week_dict(fw_saved, "Overtime Hours (#)")

    # Independent: compute Back Office Overtime Hours from shrinkage_raw_backoffice (do not touch shrinkage logic)
    def _compute_weekly_ot_from_raw() -> dict:
        try:
            braw = load_df("shrinkage_raw_backoffice")
        except Exception:
            return {}
        if not isinstance(braw, pd.DataFrame) or braw.empty:
            return {}
        df = braw.copy()
        L = {str(c).strip().lower(): c for c in df.columns}
        c_date = L.get("date")
        c_act  = L.get("activity")
        c_sec  = L.get("duration_seconds") or L.get("seconds") or L.get("duration")
        c_hr   = L.get("hours") or L.get("duration_hours")
        c_ba   = L.get("journey") or L.get("business area") or L.get("ba") or L.get("vertical")
        c_sba  = L.get("sub_business_area") or L.get("sub business area") or L.get("sub_ba") or L.get("sub ba")
        c_ch   = L.get("channel") or L.get("lob")
        c_site = L.get("site") or L.get("location") or L.get("country") or L.get("city")
        if not c_date or not c_act:
            return {}
        # Plan scope filters (Back Office only)
        mask = pd.Series(True, index=df.index)
        if c_ba and p.get("vertical"):
            mask &= df[c_ba].astype(str).str.strip().str.lower().eq(str(p.get("vertical")).strip().lower())
        if c_sba and p.get("sub_ba"):
            mask &= df[c_sba].astype(str).str.strip().str.lower().eq(str(p.get("sub_ba")).strip().lower())
        if c_ch:
            mask &= df[c_ch].astype(str).str.strip().str.lower().isin(["back office","bo","backoffice"])
        if c_site and (p.get("site") or p.get("location") or p.get("country")):
            target = str(p.get("site") or p.get("location") or p.get("country")).strip().lower()
            loc_l = df[c_site].astype(str).str.strip().str.lower()
            if loc_l.eq(target).any():
                mask &= loc_l.eq(target)
        df = df.loc[mask]
        if df.empty:
            return {}
        # Normalize fields
        df[c_date] = pd.to_datetime(df[c_date], errors="coerce")
        df = df.dropna(subset=[c_date])
        act = df[c_act].astype(str).str.strip().str.lower()
        m_ot = act.str.contains(r"\bover\s*time\b|\bovertime\b|\bot\b|\bot\s*hours\b|\bot\s*hrs\b", regex=True, na=False) | act.eq("overtime")
        df = df.loc[m_ot]
        if df.empty:
            return {}
        hours = None
        if c_sec:
            hours = pd.to_numeric(df[c_sec], errors="coerce").fillna(0.0) / 3600.0
        elif c_hr:
            hours = pd.to_numeric(df[c_hr], errors="coerce").fillna(0.0)
        else:
            return {}
        tmp = pd.DataFrame({"date": df[c_date], "hours": hours})
        tmp["week"] = (tmp["date"] - pd.to_timedelta(tmp["date"].dt.weekday, unit="D")).dt.date.astype(str)
        agg = tmp.groupby("week", as_index=False)["hours"].sum()
        return {str(r["week"]): float(r["hours"]) for _, r in agg.iterrows()}

    _ot_w_raw = _compute_weekly_ot_from_raw()
    # Voice overtime from SC_OVERTIME_DELIVERED
    def _compute_weekly_ot_voice_from_raw() -> dict:
        try:
            vraw = load_df("shrinkage_raw_voice")
        except Exception:
            return {}
        if not isinstance(vraw, pd.DataFrame) or vraw.empty:
            return {}
        df = vraw.copy()
        L = {str(c).strip().lower(): c for c in df.columns}
        c_date = L.get("date"); c_state = L.get("superstate") or L.get("state"); c_hours = L.get("hours") or L.get("duration_hours") or L.get("duration")
        c_ba = L.get("business area") or L.get("ba"); c_sba = L.get("sub business area") or L.get("sub_ba"); c_ch = L.get("channel")
        c_site = L.get("site") or L.get("location") or L.get("country") or L.get("city")
        if not c_date or not c_state or not c_hours:
            return {}
        mask = pd.Series(True, index=df.index)
        if c_ba and p.get("vertical"): mask &= df[c_ba].astype(str).str.strip().str.lower().eq(str(p.get("vertical")).strip().lower())
        if c_sba and p.get("sub_ba"): mask &= df[c_sba].astype(str).str.strip().str.lower().eq(str(p.get("sub_ba")).strip().lower())
        if c_ch: mask &= df[c_ch].astype(str).str.strip().str.lower().isin(["voice","telephony","calls","inbound","outbound"]) | df[c_ch].astype(str).eq("")
        if c_site and (p.get("site") or p.get("location") or p.get("country")):
            target = str(p.get("site") or p.get("location") or p.get("country")).strip().lower()
            loc_l = df[c_site].astype(str).str.strip().str.lower()
            if loc_l.eq(target).any():
                mask &= loc_l.eq(target)
        df = df.loc[mask]
        if df.empty:
            return {}
        df[c_date] = pd.to_datetime(df[c_date], errors="coerce")
        df = df.dropna(subset=[c_date])
        st = df[c_state].astype(str).str.strip().str.upper()
        df = df.loc[st.eq("SC_OVERTIME_DELIVERED")]
        if df.empty:
            return {}
        hrs = pd.to_numeric(df[c_hours], errors="coerce").fillna(0.0)
        tmp = pd.DataFrame({"date": df[c_date], "hours": hrs})
        tmp["week"] = (tmp["date"] - pd.to_timedelta(tmp["date"].dt.weekday, unit="D")).dt.date.astype(str)
        agg = tmp.groupby("week", as_index=False)["hours"].sum()
        return {str(r["week"]): float(r["hours"]) for _, r in agg.iterrows()}
    _ot_voice_w_raw = _compute_weekly_ot_voice_from_raw()
    backlog_w  = _row_to_week_dict(fw_saved, "Backlog (Items)")
    # Prefer freshly computed backlog for carryover if user selected Backlog in this plan
    if backlog_w_local and ("backlog" in lower_opts):
        backlog_w = backlog_w_local

    # ---- Apply Backlog carryover (Back Office only): add previous week's backlog to next week's BO forecast ----
    if backlog_carryover and str(ch_first).strip().lower() in ("back office", "bo") and backlog_w and ("backlog" in lower_opts):
        for i in range(len(week_ids) - 1):
            cur_w = week_ids[i]; nxt_w = week_ids[i+1]
            add = float(backlog_w.get(cur_w, 0.0) or 0.0)
            if add:
                weekly_demand_bo[nxt_w] = float(weekly_demand_bo.get(nxt_w, 0.0)) + add
                if "Forecast" in fw_rows:
                    fw.loc[fw["metric"] == "Forecast", nxt_w] = float(fw.loc[fw["metric"] == "Forecast", nxt_w]) + add

    # ---- Occupancy/Utilization by channel (% in FW grid) ----
    ch_key = str(ch_first or "").strip().lower()
    if ch_key in ("voice",):
        occ_base_raw = settings.get("occupancy_cap_voice", settings.get("occupancy", 0.85))
    elif ch_key in ("back office", "bo"):
        occ_base_raw = settings.get("util_bo", 0.85)
    elif ch_key in ("outbound",):
        occ_base_raw = settings.get("util_ob", 0.85)
    elif ch_key in ("chat",):
        occ_base_raw = settings.get("util_chat", settings.get("util_bo", 0.85))
    else:
        occ_base_raw = _setting(settings, ["occupancy","occupancy_pct","target_occupancy","budgeted_occupancy","occ","occupancy_cap_voice"], 0.85)
    try:
        if isinstance(occ_base_raw, str) and occ_base_raw.strip().endswith("%"):
            occ_base = float(occ_base_raw.strip()[:-1])
        else:
            occ_base = float(occ_base_raw)
            if occ_base <= 1.0:
                occ_base *= 100.0
    except Exception:
        occ_base = 85.0

    occ_w = {w: int(round(occ_base)) for w in week_ids}
    if occ_override not in (None, ""):
        try:
            ov = float(occ_override)
            if ov <= 1.0: ov *= 100.0
            ov = int(round(ov))
            for w in week_ids:
                if _wf_active(w):
                    occ_w[w] = ov
        except Exception:
            pass

    if "Occupancy" in fw_rows:
        for w in week_ids:
            fw.loc[fw["metric"] == "Occupancy", w] = occ_w[w]

    occ_frac_w = {w: min(0.99, max(0.01, float(occ_w[w]) / 100.0)) for w in week_ids}

    # ---- requirements: daily ? weekly ----
    from ._common import _assemble_ob, _assemble_chat
    oF = _assemble_ob(sk,   "forecast");  oA = _assemble_ob(sk,   "actual");   oT = _assemble_ob(sk,   "tactical")
    cF = _assemble_chat(sk, "forecast");  cA = _assemble_chat(sk, "actual");   cT = _assemble_chat(sk, "tactical")

    # If channel is Chat or Outbound, override FW grid volume/AHT rows with channel-appropriate aggregates
    def _weekly_chat(df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}, {}
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"]).copy()
        d["week"] = (d["date"] - pd.to_timedelta(d["date"].dt.weekday, unit="D")).dt.date.astype(str)
        items = pd.to_numeric(d.get("items"), errors="coerce").fillna(0.0)
        aht   = pd.to_numeric(d.get("aht_sec"), errors="coerce").fillna(0.0)
        d["_num"] = items * aht
        vol = d.groupby("week", as_index=True)["items"].sum()
        num = d.groupby("week", as_index=True)["_num"].sum()
        ahtw = (num / vol.replace({0: np.nan})).fillna(np.nan)
        return vol.to_dict(), ahtw.to_dict()

    def _to_frac(x):
        try:
            v = float(x)
            return v/100.0 if v > 1.0 else v
        except Exception:
            try:
                s = str(x).strip().rstrip('%'); v = float(s)
                return v/100.0 if v > 1.0 else v
            except Exception:
                return 0.0

    def _weekly_ob(df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}, {}
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"]).copy()
        d["week"] = (d["date"] - pd.to_timedelta(d["date"].dt.weekday, unit="D")).dt.date.astype(str)
        cols = {c.lower(): c for c in d.columns}
        c_opc  = cols.get("opc") or cols.get("dials") or cols.get("calls") or cols.get("attempts")
        c_conn = cols.get("connect_rate") or cols.get("connect%") or cols.get("connect pct") or cols.get("connect")
        c_rpc  = cols.get("rpc")
        c_rpcr = cols.get("rpc_rate") or cols.get("rpc%") or cols.get("rpc pct")
        c_aht  = cols.get("aht_sec") or cols.get("talk_sec") or cols.get("avg_talk_sec") or cols.get("aht")
        opc  = pd.to_numeric(d.get(c_opc), errors="coerce").fillna(0.0) if c_opc else 0.0
        rpc  = pd.to_numeric(d.get(c_rpc), errors="coerce").fillna(0.0) if c_rpc else 0.0
        conn = d.get(c_conn).map(_to_frac) if c_conn else 0.0
        rpcr = d.get(c_rpcr).map(_to_frac) if c_rpcr else 0.0
        exp  = rpc if c_rpc else (opc * (conn if c_conn else 1.0) * (rpcr if c_rpcr else 1.0))
        d["_exp"] = pd.to_numeric(exp, errors="coerce").fillna(0.0)
        aht = pd.to_numeric(d.get(c_aht), errors="coerce").fillna(0.0) if c_aht else 0.0
        d["_num"] = d["_exp"] * aht
        vol = d.groupby("week", as_index=True)["_exp"].sum()
        num = d.groupby("week", as_index=True)["_num"].sum()
        ahtw = (num / vol.replace({0: np.nan})).fillna(np.nan)
        return vol.to_dict(), ahtw.to_dict()

    ch_key = str(ch_first or '').strip().lower()
    if ch_key in ("chat", "outbound", "out bound", "ob"):
        # Build weekly dicts
        if ch_key == "chat":
            f_vol, f_aht = _weekly_chat(cF); a_vol, a_aht = _weekly_chat(cA); t_vol, _ = _weekly_chat(cT)
        else:
            f_vol, f_aht = _weekly_ob(oF);   a_vol, a_aht = _weekly_ob(oA);   t_vol, _ = _weekly_ob(oT)
        for w in week_ids:
            ft = float(f_vol.get(w, 0.0)); at = float(a_vol.get(w, 0.0)); tt = float(t_vol.get(w, 0.0))
            if _wf_active(w) and vol_delta:
                ft *= (1.0 + vol_delta/100.0)
            if "Forecast" in fw_rows:
                fw.loc[fw["metric"] == "Forecast", w] = ft
            if "Tactical Forecast" in fw_rows:
                fw.loc[fw["metric"] == "Tactical Forecast", w] = tt
            if "Actual Volume" in fw_rows:
                fw.loc[fw["metric"] == "Actual Volume", w] = at
            # AHT/SUT rows
            aht_act = float(a_aht.get(w, 0.0) or 0.0)
            aht_for = float((planned_aht_w.get(w, f_aht.get(w, 0.0)) if ch_key != 'bo' else planned_sut_w.get(w, f_aht.get(w, 0.0))) or 0.0)
            # Apply What-If AHT/SUT to Forecast row for future weeks so user sees uplift
            try:
                if _wf_active(w) and aht_delta:
                    aht_for = max(0.0, aht_for * (1.0 + aht_delta / 100.0))
            except Exception:
                pass
            if "Actual AHT/SUT" in fw_rows:
                fw.loc[fw["metric"] == "Actual AHT/SUT", w] = aht_act
            if "Forecast AHT/SUT" in fw_rows:
                fw.loc[fw["metric"] == "Forecast AHT/SUT", w] = aht_for
            if "Budgeted AHT/SUT" in fw_rows:
                bud = planned_aht_w.get(w, s_budget_aht)
                fw.loc[fw["metric"] == "Budgeted AHT/SUT", w] = bud
            # keep weekly maps in sync (capacity reads these)
            wk_aht_sut_actual[w]   = float(aht_act or 0.0)
            wk_aht_sut_forecast[w] = float(aht_for or 0.0)
            wk_aht_sut_budget[w]   = float(planned_aht_w.get(w, s_budget_aht))
    req_daily_actual   = required_fte_daily(use_voice_for_req, use_bo_for_req, oA, settings)
    req_daily_forecast = required_fte_daily(vF, bF, oF, settings)
    req_daily_tactical = required_fte_daily(vT, bT, oT, settings) if (isinstance(vT, pd.DataFrame) and not vT.empty) or (isinstance(bT, pd.DataFrame) and not bT.empty) or (isinstance(oT, pd.DataFrame) and not oT.empty) else pd.DataFrame()
    # Add Chat FTE to daily totals
    from capacity_core import chat_fte_daily
    for _df, chat_df in ((req_daily_actual, cA), (req_daily_forecast, cF), (req_daily_tactical, cT)):
        if isinstance(_df, pd.DataFrame) and not _df.empty and isinstance(chat_df, pd.DataFrame) and not chat_df.empty:
            try:
                ch = chat_fte_daily(chat_df, settings)
                m = _df.merge(ch, on=["date","program"], how="left")
                m["chat_fte"] = pd.to_numeric(m.get("chat_fte"), errors="coerce").fillna(0.0)
                m["total_req_fte"] = pd.to_numeric(m.get("total_req_fte"), errors="coerce").fillna(0.0) + m["chat_fte"]
                _df.drop(_df.index, inplace=True)
                _df[list(m.columns)] = m
            except Exception:
                pass
    vB = vF.copy(); bB = bF.copy(); oB = oF.copy(); cB = cF.copy()
    if isinstance(vB, pd.DataFrame) and not vB.empty:
        vB["_w"] = pd.to_datetime(vB["date"], errors="coerce").dt.date.astype(str)
        vB["aht_sec"] = vB["_w"].map(planned_aht_w).fillna(float(s_budget_aht))
        vB.drop(columns=["_w"], inplace=True)
    if isinstance(bB, pd.DataFrame) and not bB.empty:
        bB["_w"] = pd.to_datetime(bB["date"], errors="coerce").dt.date.astype(str)
        bB["aht_sec"] = bB["_w"].map(planned_sut_w).fillna(float(s_budget_sut))
        bB.drop(columns=["_w"], inplace=True)
    if isinstance(oB, pd.DataFrame) and not oB.empty:
        oB["_w"] = pd.to_datetime(oB["date"], errors="coerce").dt.date.astype(str)
        oB["aht_sec"] = oB["_w"].map(planned_aht_w).fillna(float(s_budget_aht))
        oB.drop(columns=["_w"], inplace=True)
    req_daily_budgeted = required_fte_daily(vB, bB, oB, settings)
    if isinstance(req_daily_budgeted, pd.DataFrame) and not req_daily_budgeted.empty and isinstance(cB, pd.DataFrame) and not cB.empty:
        try:
            chb = chat_fte_daily(cB, settings)
            m = req_daily_budgeted.merge(chb, on=["date","program"], how="left")
            m["chat_fte"] = pd.to_numeric(m.get("chat_fte"), errors="coerce").fillna(0.0)
            m["total_req_fte"] = pd.to_numeric(m.get("total_req_fte"), errors="coerce").fillna(0.0) + m["chat_fte"]
            req_daily_budgeted.drop(req_daily_budgeted.index, inplace=True)
            req_daily_budgeted[list(m.columns)] = m
        except Exception:
            pass

    def _daily_to_weekly(df, ch_first=ch_first, settings=settings):
        if not isinstance(df, pd.DataFrame) or df.empty or "date" not in df.columns or "total_req_fte" not in df.columns:
            return {}
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"])
        d["week"] = (d["date"] - pd.to_timedelta(d["date"].dt.weekday, unit="D")).dt.date.astype(str)
        wd = 5 if str(ch_first).strip().lower() in ("back office", "bo") else int(settings.get("workdays_per_week", 7) or 7)
        g = d.groupby("week", as_index=False)["total_req_fte"].sum()
        g["avg_req_fte"] = g["total_req_fte"] / max(1, wd)
        return g.set_index("week")["avg_req_fte"].to_dict()

    req_w_actual   = _daily_to_weekly(req_daily_actual)
    req_w_forecast = _daily_to_weekly(req_daily_forecast)
    req_w_tactical = _daily_to_weekly(req_daily_tactical)
    req_w_budgeted = _daily_to_weekly(req_daily_budgeted)

    # What-If: adjust forecast requirements by volume, AHT and shrink deltas
    if vol_delta or shrink_delta or aht_delta:
        for w in list(req_w_forecast.keys()):
            if not _wf_active(w):
                continue
            v = float(req_w_forecast[w])
            if vol_delta:
                v *= (1.0 + vol_delta / 100.0)
            if aht_delta:
                # Approximate: FTE requirement scales ~linearly with AHT
                v *= (1.0 + aht_delta / 100.0)
            if shrink_delta:
                # Approximate impact: scale by 1/(1 - delta)
                denom = max(0.1, 1.0 - (shrink_delta / 100.0))
                v /= denom
            req_w_forecast[w] = v

    # ---- Interval supply from global roster_long (if available) ----
    schedule_supply_avg = {}
    try:
        rl = load_roster_long()
    except Exception:
        rl = None
    if isinstance(rl, pd.DataFrame) and not rl.empty:
        df = rl.copy()
        # Find scope cols
        def _col(df, opts):
            for c in opts:
                if c in df.columns:
                    return c
            return None
        c_ba  = _col(df, ["Business Area","business area","vertical"])
        c_sba = _col(df, ["Sub Business Area","sub business area","sub_ba"])
        c_lob = _col(df, ["LOB","lob","Channel","channel"])
        c_site= _col(df, ["Site","site","Location","location","Country","country"])
        # Plan scope
        BA  = p.get("vertical")
        SBA = p.get("sub_ba")
        LOB = ch_first
        SITE= p.get("site") or p.get("location") or p.get("country")
        def _match(series, val):
            if not val or not isinstance(series, pd.Series):
                return pd.Series([True]*len(series))
            s = series.astype(str).str.strip().str.lower()
            v = str(val).strip().lower()
            return s.eq(v)
        m = pd.Series([True]*len(df))
        if c_ba:  m &= _match(df[c_ba], BA)
        if c_sba and (SBA not in (None, "")): m &= _match(df[c_sba], SBA)
        if c_lob: m &= _match(df[c_lob], LOB)
        if c_site and (SITE not in (None, "")): m &= _match(df[c_site], SITE)
        df = df[m]
        # Exclude leave
        if "is_leave" in df.columns:
            df = df[~df["is_leave"].astype(bool)]
        # Parse dates
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
        # Compute intervals per week default
        intervals_per_week_default = 7 * (24 * 60 // ivl_min)
        # Parse shifts -> interval counts
        import re as _re
        def _shift_len_ivl(s: str) -> int:
            try:
                s = str(s or "").strip()
                m = _re.match(r"^(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})$", s)
                if not m:
                    return 0
                sh, sm, eh, em = map(int, m.groups())
                sh = min(23, max(0, sh)); eh = min(24, max(0, eh))
                start = sh*60 + sm; end = eh*60 + em
                if end < start:
                    end += 24*60
                return max(0, int((end - start + (ivl_min-1)) // ivl_min))
            except Exception:
                return 0
        if "entry" in df.columns and "date" in df.columns:
            df["ivl_count"] = df["entry"].apply(_shift_len_ivl)
            df["week"] = (df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")).dt.date.astype(str)
            agg = df.groupby("week", as_index=False)["ivl_count"].sum()
            weekly_agent_ivls = dict(zip(agg["week"], agg["ivl_count"]))
            for w in week_ids:
                denom = weekly_voice_intervals.get(w, intervals_per_week_default)
                if denom <= 0:
                    denom = intervals_per_week_default
                schedule_supply_avg[w] = float(weekly_agent_ivls.get(w, 0.0)) / float(denom)

    # ---- overlay user FW overrides for AHT/SUT (if any saved) ----
    def _merge_fw_user_overrides(fw_calc: pd.DataFrame, fw_user: pd.DataFrame, week_ids: list) -> pd.DataFrame:
        if not isinstance(fw_calc, pd.DataFrame) or fw_calc.empty:
            return fw_user if isinstance(fw_user, pd.DataFrame) else pd.DataFrame()
        calc = fw_calc.copy()
        if not isinstance(fw_user, pd.DataFrame) or fw_user.empty:
            return calc
        c = calc.set_index("metric"); u = fw_user.set_index("metric")
        for w in week_ids:
            if w in c.columns: c[w] = pd.to_numeric(c[w], errors="coerce")
            if w in u.columns: u[w] = pd.to_numeric(u[w], errors="coerce")

        def _find_row(idx_like, *alts):
            low = {str(k).strip().lower(): k for k in idx_like}
            for a in alts:
                k = str(a).strip().lower()
                if k in low:
                    return low[k]
            for key, orig in low.items():
                for a in alts:
                    if str(a).strip().lower() in key:
                        return orig
            return None

        budget_label_calc = _find_row(c.index, "Budgeted AHT/SUT","Budget AHT/SUT","Budget AHT","Budget SUT")
        forecast_label_calc = _find_row(c.index, "Forecast AHT/SUT","Target AHT/SUT","Planned AHT/SUT","Planned AHT","Planned SUT","Target AHT","Target SUT")
        budget_label_user = _find_row(u.index, "Budgeted AHT/SUT","Budget AHT/SUT","Budget AHT","Budget SUT")
        forecast_label_user = _find_row(u.index, "Forecast AHT/SUT","Target AHT/SUT","Planned AHT/SUT","Planned AHT","Planned SUT","Target AHT","Target SUT")

        def _apply(canon_label, user_label):
            if not canon_label or not user_label: return
            for w in week_ids:
                if w in u.columns and w in c.columns:
                    val = u.at[user_label, w]
                    if pd.notna(val):
                        c.at[canon_label, w] = float(val)

        _apply(budget_label_calc, budget_label_user)
        _apply(forecast_label_calc, forecast_label_user)
        return c.reset_index()

    fw_to_use = _merge_fw_user_overrides(fw, fw_saved, week_ids)

    # Scale requirements if user overrides AHT/SUT rows
    try:
        bud_user = _row_to_week_dict(fw_to_use, "Budgeted AHT/SUT")
        for_user = _row_to_week_dict(fw_to_use, "Forecast AHT/SUT")
        for w in week_ids:
            new_f = for_user.get(w)
            if new_f is not None and str(new_f) != "" and not pd.isna(new_f):
                base_f = float(wk_aht_sut_forecast.get(w, 0.0) or 0.0)
                if base_f > 0.0:
                    factor = float(new_f) / base_f
                    req_w_forecast[w] = float(req_w_forecast.get(w, 0.0)) * factor
                wk_aht_sut_forecast[w] = float(new_f)
            new_b = bud_user.get(w)
            if new_b is not None and str(new_b) != "" and not pd.isna(new_b):
                base_b = float(wk_aht_sut_budget.get(w, 0.0) or 0.0)
                if base_b > 0.0:
                    factor = float(new_b) / base_b
                    req_w_budgeted[w] = float(req_w_budgeted.get(w, 0.0)) * factor
                wk_aht_sut_budget[w] = float(new_b)
    except Exception:
        pass

    # ---- lower shells ----
    hc   = _load_or_blank(f"plan_{pid}_hc",   ["Budgeted HC (#)","Planned/Tactical HC (#)","Actual Agent HC (#)","SME Billable HC (#)"], week_ids)
    att  = _load_or_blank(
        f"plan_{pid}_attr",
        [
            "Planned Attrition HC (#)",
            "Actual Attrition HC (#)",
            "Planned Attrition %",
            "Actual Attrition %",
            "Variance vs Planned",
        ],
        week_ids,
    )
    shr  = _load_or_blank(
        f"plan_{pid}_shr",
        [
            "OOO Shrink Hours (#)",
            "In-Office Shrink Hours (#)",
            "OOO Shrinkage %",
            "In-Office Shrinkage %",
            "Overall Shrinkage %",
            "Planned Shrinkage %",
            "Variance vs Planned",
        ],
        week_ids,
    )
    trn  = _load_or_blank(f"plan_{pid}_train",["Training Start (#)","Training End (#)","Nesting Start (#)","Nesting End (#)"], week_ids)
    rat  = _load_or_blank(f"plan_{pid}_ratio",["Planned TL/Agent Ratio","Actual TL/Agent Ratio","Variance"], week_ids)
    seat = _load_or_blank(
        f"plan_{pid}_seat",
        [
            "Seats Required (#)",
            "Seats Available (#)",
            "Seat Utilization %",
            "Variance vs Available",
        ],
        week_ids,
    )
    # Make weekly seat columns float so writing percentages doesn't conflict with int dtype
    try:
        for _w in week_ids:
            if _w in seat.columns and not pd.api.types.is_float_dtype(seat[_w].dtype):
                seat[_w] = pd.to_numeric(seat[_w], errors="coerce").astype("float64")
    except Exception:
        pass
    bva  = _load_or_blank(f"plan_{pid}_bva",  ["Budgeted FTE (#)","Actual FTE (#)","Variance (#)"], week_ids)
    nh   = _load_or_blank(f"plan_{pid}_nh",   ["Planned New Hire HC (#)","Actual New Hire HC (#)","Recruitment Achievement"], week_ids)

    # -- New Hire overlay (classes + roster) --
    today_w = _monday(dt.date.today()).isoformat()
    planned_nh_w = _weekly_planned_nh_from_classes(pid, week_ids)

    # Apply learning-curve throughput to planned NH
    planned_nh_w = {
        w: int(round(planned_nh_w.get(w, 0) * max(0.0, min(1.0,( (_lc_with_wf(_learning_curve_for_week(settings, lc_ovr_df, w), w).get("throughput_train_pct", 100.0) / 100.0) *
              (_lc_with_wf(_learning_curve_for_week(settings, lc_ovr_df, w), w).get("throughput_nest_pct", 100.0) / 100.0) )
        ))))
        for w in week_ids
    }

    # buckets from classes
    def _weekly_buckets_from_classes(df, week_ids):
        from collections import defaultdict
        nest = {w: defaultdict(int) for w in week_ids}
        sda  = {w: defaultdict(int) for w in week_ids}
        if not isinstance(df, pd.DataFrame) or df.empty: return nest, sda

        def _w(d):
            t = pd.to_datetime(d, errors="coerce")
            if pd.isna(t): return None
            monday = (t - pd.to_timedelta(int(getattr(t, "weekday", lambda: t.weekday())()), unit="D"))
            return pd.Timestamp(monday).normalize().date().isoformat()

        for _, r in df.iterrows():
            n = _nh_effective_count(r)
            if n <= 0: continue
            ns = _w(r.get("nesting_start")); ne = _w(r.get("nesting_end")); ps = _w(r.get("production_start"))
            if ns and ne:
                wklist = [wk for wk in week_ids if (wk >= ns and wk <= ne)]
                for i, wk in enumerate(wklist, start=1): nest[wk][i] += n
            if ps:
                sda_weeks = int(float(settings.get("sda_weeks", settings.get("default_sda_weeks", 0)) or 0))
                if sda_weeks > 0:
                    wklist = [wk for wk in week_ids if wk >= ps][:sda_weeks]
                    for i, wk in enumerate(wklist, start=1): sda[wk][i] += n
        return nest, sda

    classes_df = load_df(f"plan_{pid}_nh_classes")
    nest_buckets, sda_buckets = _weekly_buckets_from_classes(classes_df, week_ids)

    wk_train_in_phase = {w: 0 for w in week_ids}
    wk_nest_in_phase  = {w: 0 for w in week_ids}
    if isinstance(classes_df, pd.DataFrame) and not classes_df.empty:
        c = classes_df.copy()
        def _between(w, w_start, w_end) -> bool:
            return (w_start is not None) and (w_end is not None) and (w_start <= w <= w_end)
        for _, r in c.iterrows():
            n_eff = _nh_effective_count(r)
            w_ts = _week_label(r.get("training_start"))
            w_te = _week_label(r.get("training_end"))
            w_ns = _week_label(r.get("nesting_start"))
            w_ne = _week_label(r.get("nesting_end"))
            for w in week_ids:
                if _between(w, w_ts, w_te): wk_train_in_phase[w] += n_eff
                if _between(w, w_ns, w_ne): wk_nest_in_phase[w]  += n_eff

    for w in week_ids:
        trn.loc[trn["metric"] == "Training Start (#)", w] = wk_train_in_phase.get(w, 0)
        trn.loc[trn["metric"] == "Training End (#)",   w] = wk_train_in_phase.get(w, 0)
        trn.loc[trn["metric"] == "Nesting Start (#)",  w] = wk_nest_in_phase.get(w, 0)
        trn.loc[trn["metric"] == "Nesting End (#)",    w] = wk_nest_in_phase.get(w, 0)

    # Actual joiners by production week
    roster_df = _load_roster_normalized(pid)
    actual_nh_w = _weekly_actual_nh_from_roster(roster_df, week_ids)

    for w in week_ids:
        # Always write planned from classes (with throughput applied)
        nh.loc[nh["metric"] == "Planned New Hire HC (#)", w] = int(planned_nh_w.get(w, 0))
        # Do not overwrite user-entered Actual NH; if none saved, allow roster-derived actual for past/current weeks
        try:
            cur_act_val = pd.to_numeric(nh.loc[nh["metric"] == "Actual New Hire HC (#)", w], errors="coerce").fillna(0.0).iloc[0]
        except Exception:
            cur_act_val = 0.0
        if (cur_act_val in (None, 0, 0.0)):
            if w <= today_w:
                nh.loc[nh["metric"] == "Actual New Hire HC (#)",  w] = int(actual_nh_w.get(w, 0))
        # Always recompute Recruitment Achievement from whatever values are present
        try:
            plan = float(pd.to_numeric(nh.loc[nh["metric"] == "Planned New Hire HC (#)", w], errors="coerce").fillna(0.0).iloc[0])
        except Exception:
            plan = 0.0
        try:
            act = float(pd.to_numeric(nh.loc[nh["metric"] == "Actual New Hire HC (#)", w], errors="coerce").fillna(0.0).iloc[0])
        except Exception:
            act = 0.0
        nh.loc[nh["metric"] == "Recruitment Achievement", w] = (0.0 if plan <= 0 else (100.0 * act / plan))

    # ---- Actual HC snapshots from roster ----
    hc_actual_w    = _weekly_hc_step_from_roster(roster_df, week_ids, r"\bagent\b")
    try:
        total = 0
        try:
            total = int(sum(hc_actual_w.values()))
        except Exception:
            total = 0
        if total == 0:
            hc_actual_w = _weekly_hc_step_from_roster(roster_df, week_ids, r".*")
    except Exception:
        pass
    sme_billable_w = _weekly_hc_step_from_roster(roster_df, week_ids, r"\bsme\b")
    for w in week_ids:
        hc.loc[hc["metric"] == "Actual Agent HC (#)", w] = hc_actual_w.get(w, 0)
        hc.loc[hc["metric"] == "SME Billable HC (#)", w] = sme_billable_w.get(w, 0)

    # ---- Budget vs simple Planned HC ----
    budget_df = _first_non_empty_ts(sk, ["budget_headcount","budget_hc","headcount_budget","hc_budget"])
    budget_w  = _weekly_reduce(budget_df, value_candidates=("hc","headcount","value","count"), how="sum")
    for w in week_ids:
        hc.loc[hc["metric"] == "Budgeted HC (#)",         w] = float(budget_w.get(w, 0.0))
        hc.loc[hc["metric"] == "Planned/Tactical HC (#)", w] = float(budget_w.get(w, 0.0))

    # ---- Attrition (planned/actual/pct) ----
    att_plan_w = _weekly_reduce(_first_non_empty_ts(sk, ["attrition_planned_hc","attrition_plan_hc","planned_attrition_hc"]),
                                value_candidates=("hc","headcount","value","count"), how="sum")
    att_act_w  = _weekly_reduce(_first_non_empty_ts(sk, ["attrition_actual_hc","attrition_actual","actual_attrition_hc"]),
                                value_candidates=("hc","headcount","value","count"), how="sum")
    att_pct_w  = _weekly_reduce(_first_non_empty_ts(sk, ["attrition_pct","attrition_percent","attrition%","attrition_rate","planned_attrition_pct"]),
                                value_candidates=("pct","percent","value"), how="mean")

    att_plan_saved, att_actual_saved, att_planned_pct_saved = {}, {}, {}
    try:
        att_saved_df = load_df(f"plan_{pid}_attr")
    except Exception:
        att_saved_df = None
    if isinstance(att_saved_df, pd.DataFrame) and not att_saved_df.empty:
        metrics = att_saved_df["metric"].astype(str).str.strip()
        if "Planned Attrition HC (#)" in metrics.values:
            row = att_saved_df.loc[metrics == "Planned Attrition HC (#)"].iloc[0]
            for w in week_ids:
                if w in row:
                    val = pd.to_numeric(row.get(w), errors="coerce")
                    if pd.notna(val):
                        att_plan_saved[w] = float(val)
        if "Actual Attrition HC (#)" in metrics.values:
            row = att_saved_df.loc[metrics == "Actual Attrition HC (#)"].iloc[0]
            for w in week_ids:
                if w in row:
                    val = pd.to_numeric(row.get(w), errors="coerce")
                    if pd.notna(val):
                        att_actual_saved[w] = float(val)
        if "Planned Attrition %" in metrics.values:
            row = att_saved_df.loc[metrics == "Planned Attrition %"].iloc[0]
            for w in week_ids:
                if w in row:
                    val = pd.to_numeric(row.get(w), errors="coerce")
                    if pd.notna(val):
                        att_planned_pct_saved[w] = float(val)
    attr_roster_w = _weekly_attrition_from_roster(roster_df, week_ids, r"\bagent\b")
    try:
        _sum = 0
        try:
            _sum = int(sum(attr_roster_w.values()))
        except Exception:
            _sum = 0
        if _sum == 0:
            attr_roster_w = _weekly_attrition_from_roster(roster_df, week_ids, r".*")
    except Exception:
        pass
    today_w = _monday(dt.date.today()).isoformat()

    # Pull planned HC row to support % recomputation for future weeks
    try:
        _hc_plan_row = hc.loc[hc["metric"].astype(str).str.strip() == "Planned/Tactical HC (#)"].iloc[0]
    except Exception:
        _hc_plan_row = None

    for w in week_ids:
        plan_ts = float(att_plan_w.get(w, 0.0))
        plan_manual = att_plan_saved.get(w)
        roster_term = float(attr_roster_w.get(w, 0.0))
        if plan_manual is not None:
            plan_hc = plan_manual
        else:
            plan_hc = plan_ts
        # What-If: overlay attrition delta into planned attrition for future weeks
        if _wf_active(w) and attr_delta and w > today_w:
            plan_hc += float(attr_delta)
        plan_hc = max(0.0, plan_hc)

        act_manual = att_actual_saved.get(w)
        act_ts = float(att_act_w.get(w, 0.0))
        # Always treat roster terminations as Actual (regardless of week)
        cand = [act_ts, roster_term]
        if act_manual is not None:
            try:
                cand.append(float(act_manual))
            except Exception:
                pass
        act_hc = max([float(c) for c in cand] + [0.0])
        act_hc = max(0.0, act_hc)

        # Always compute actual % from actual HC; ignore upstream attrition_pct for actuals
        base_actual = float(hc_actual_w.get(w, 0))
        pct = 100.0 * (act_hc / base_actual) if base_actual > 0 else 0.0
        # Recompute Attrition % for future weeks when What-If attrition is active
        if _wf_active(w) and attr_delta and w > today_w:
            try:
                denom = float(pd.to_numeric((_hc_plan_row or {}).get(w), errors="coerce")) if hasattr(_hc_plan_row, 'get') else float(pd.to_numeric(_hc_plan_row[w], errors="coerce"))
            except Exception:
                denom = 0.0
            if denom > 0:
                pct = 100.0 * (plan_hc / denom)
        planned_pct_val = att_planned_pct_saved.get(w)
        if planned_pct_val is None:
            planned_pct_val = att_pct_w.get(w, None)
        if planned_pct_val is None:
            # Baseline: previous week's Actual HC; fallback to current week's starting HC
            def _prev_week_id(label: str) -> str | None:
                try:
                    t = pd.to_datetime(label, errors="coerce")
                    if pd.isna(t):
                        return None
                    prev = pd.Timestamp(t).normalize() - pd.Timedelta(days=7)
                    return prev.date().isoformat()
                except Exception:
                    return None
            prev_w = _prev_week_id(w)
            try:
                denom_p = float(hc_actual_w.get(prev_w, 0.0) if prev_w else 0.0)
            except Exception:
                denom_p = 0.0
            if denom_p <= 0:
                try:
                    denom_p = float(pd.to_numeric((_hc_plan_row or {}).get(w), errors="coerce")) if hasattr(_hc_plan_row, 'get') else float(pd.to_numeric(_hc_plan_row[w], errors="coerce"))
                except Exception:
                    denom_p = 0.0
            if denom_p <= 0:
                try:
                    denom_p = float(hc_actual_w.get(w, 0.0) or 0.0)
                except Exception:
                    denom_p = 0.0
            planned_pct_val = (100.0 * plan_hc / denom_p) if denom_p > 0 else 0.0
        att.loc[att["metric"] == "Planned Attrition HC (#)", w] = plan_hc
        att.loc[att["metric"] == "Actual Attrition HC (#)",  w] = act_hc
        att.loc[att["metric"] == "Planned Attrition %",      w] = planned_pct_val
        att.loc[att["metric"] == "Actual Attrition %",              w] = pct
        att.loc[att["metric"] == "Variance vs Planned", w] = float(pct) - float(planned_pct_val or 0.0)
    # ---- Shrinkage raw ? weekly ----
    ch_key = str(ch_first or '').strip().lower()
    def _planned_shr(val, fallback):
        try:
            if val in (None, '', 'nan'):
                raise ValueError
            x = float(val)
            if x > 1.0:
                x /= 100.0
            return max(0.0, x)
        except Exception:
            try:
                return max(0.0, float(fallback))
            except Exception:
                return max(0.0, 0.0)
    planned_shrink_fraction = _planned_shr(settings.get('shrinkage_pct'), 0.0)
    if ch_key == 'voice':
        planned_shrink_fraction = _planned_shr(settings.get('voice_shrinkage_pct'), planned_shrink_fraction)
    elif ch_key in ('back office', 'bo'):
        planned_shrink_fraction = _planned_shr(settings.get('bo_shrinkage_pct'), planned_shrink_fraction)
    elif 'chat' in ch_key:
        planned_shrink_fraction = _planned_shr(settings.get('chat_shrinkage_pct'), planned_shrink_fraction)
    elif ch_key in ('outbound', 'out bound', 'ob'):
        planned_shrink_fraction = _planned_shr(settings.get('ob_shrinkage_pct'), planned_shrink_fraction)

    ooo_hours_w, io_hours_w, base_hours_w = {}, {}, {}
    sc_hours_w, tt_worked_hours_w, nodow_base_hours_w = {}, {}, {}
    overtime_hours_w = {}
    bo_shrink_frac_daily = {}

    # Load saved planned shrink % (weekly) if present and prefer it for display planned pct
    saved_plan_pct_w = {}
    try:
        _shr_saved_df = load_df(f"plan_{pid}_shr")
    except Exception:
        _shr_saved_df = None
    if isinstance(_shr_saved_df, pd.DataFrame) and not _shr_saved_df.empty:
        _m = _shr_saved_df['metric'].astype(str).str.strip()
        if 'Planned Shrinkage %' in _m.values:
            _row = _shr_saved_df.loc[_m == 'Planned Shrinkage %'].iloc[0]
            for w in week_ids:
                if w in _row:
                    val = _row.get(w)
                    try:
                        if isinstance(val, str) and val.endswith('%'):
                            val = float(val.strip().rstrip('%'))
                        else:
                            val = float(pd.to_numeric(val, errors='coerce'))
                    except Exception:
                        val = None
                    if val is not None and not pd.isna(val):
                        saved_plan_pct_w[w] = float(val)


    def _week_key(s):
        ds = pd.to_datetime(s, errors="coerce")
        if isinstance(ds, pd.Series):
            monday = ds.dt.normalize() - pd.to_timedelta(ds.dt.weekday, unit="D")
            return monday.dt.date.astype(str)
        else:
            ds = pd.DatetimeIndex(ds)
            monday = ds.normalize() - pd.to_timedelta(ds.weekday, unit="D")
            return pd.Index(monday.date.astype(str))

    def _agg_weekly(date_idx, ooo_series, ino_series, base_series):
        wk = _week_key(date_idx)
        g = pd.DataFrame({"week": wk, "ooo": ooo_series, "ino": ino_series, "base": base_series}).groupby("week", as_index=False).sum()
        for _, r in g.iterrows():
            k = str(r["week"])
            ooo_hours_w[k]  = ooo_hours_w.get(k, 0.0)  + float(r["ooo"])
            io_hours_w[k]   = io_hours_w.get(k, 0.0)   + float(r["ino"])
            base_hours_w[k] = base_hours_w.get(k, 0.0) + float(r["base"])

    if True:
        try:
            vraw = load_df("shrinkage_raw_voice")
        except Exception:
            vraw = None
        if isinstance(vraw, pd.DataFrame) and not vraw.empty:
            v = vraw.copy()
            L = {str(c).strip().lower(): c for c in v.columns}
            c_date = L.get("date"); c_hours = L.get("hours") or L.get("duration_hours") or L.get("duration")
            c_state= L.get("superstate") or L.get("state")
            c_ba   = L.get("business area") or L.get("ba")
            c_sba  = L.get("sub business area") or L.get("sub_ba")
            c_ch   = L.get("channel")
            # Prefer matching the plan's specificity: site > location > country > city
            c_site = L.get("site")
            c_location = L.get("location")
            c_country = L.get("country")
            c_city = L.get("city")

            mask = pd.Series(True, index=v.index)
            if c_ba and p.get("vertical"): mask &= v[c_ba].astype(str).str.strip().str.lower().eq(p["vertical"].strip().lower())
            if c_sba and p.get("sub_ba"): mask &= v[c_sba].astype(str).str.strip().str.lower().eq(p["sub_ba"].strip().lower())
            if c_ch: mask &= v[c_ch].astype(str).str.strip().str.lower().eq("voice")
            if loc_first:
                target = loc_first.strip().lower()
                matched = False
                for col in [c_site, c_location, c_country, c_city]:
                    if col and col in v.columns:
                        loc_l = v[col].astype(str).str.strip().str.lower()
                        if loc_l.eq(target).any():
                            mask &= loc_l.eq(target)
                            matched = True
                            break

            v = v.loc[mask]
            if c_date and c_state and c_hours and not v.empty:
                pv = v.pivot_table(index=c_date, columns=c_state, values=c_hours, aggfunc="sum", fill_value=0.0)
                # Robust Voice mapping: group superstates by patterns (return per-date Series)
                import re as _vre
                def _sum_cols_series(patterns):
                    cols = []
                    for _c in pv.columns:
                        cl = str(_c).strip().lower()
                        if any(_vre.search(pat, cl) for pat in patterns):
                            cols.append(_c)
                    if not cols:
                        return pd.Series(0.0, index=pv.index)
                    sub = pv[cols]
                    try:
                        sub = sub.apply(pd.to_numeric, errors="coerce").fillna(0.0)
                    except Exception:
                        pass
                    if isinstance(sub, pd.Series):
                        return pd.to_numeric(sub, errors="coerce").fillna(0.0)
                    return sub.sum(axis=1)
                base = _sum_cols_series([r"\bsc[_\s-]*included[_\s-]*time\b", r"\bsc[_\s-]*total[_\s-]*included[_\s-]*time\b", r"\bsc[_\s-]*available[_\s-]*time\b"]) 
                ooo  = _sum_cols_series([r"\bsc[_\s-]*absence", r"\bsc[_\s-]*holiday", r"\bsc[_\s-]*a[_\s-]*sick", r"\bsc[_\s-]*sick"]) 
                ino  = _sum_cols_series([r"\bsc[_\s-]*training", r"\bsc[_\s-]*break", r"\bsc[_\s-]*system[_\s-]*exception"]) 
                idx_dates = pd.to_datetime(pv.index, errors="coerce")
                _agg_weekly(idx_dates, ooo, ino, base)

    if True:
        try:
            braw = load_df("shrinkage_raw_backoffice")
        except Exception:
            braw = None

        b = braw.copy() if isinstance(braw, pd.DataFrame) and not braw.empty else pd.DataFrame()

        # Normalize column names
        L = {str(c).strip().lower(): c for c in b.columns}
        c_date = L.get("date")
        c_act  = L.get("activity")
        c_sec  = L.get("duration_seconds") or L.get("seconds") or L.get("duration")
        c_ba   = L.get("journey") or L.get("business area") or L.get("ba") or L.get("vertical")
        c_sba  = L.get("sub_business_area") or L.get("sub business area") or L.get("sub_ba") or L.get("subba")
        c_ch   = L.get("channel")
        c_site, c_location, c_country, c_city = L.get("site"), L.get("location"), L.get("country"), L.get("city")

        # Apply filters
        mask = pd.Series(True, index=b.index)
        if c_ba and p.get("vertical"):
            mask &= b[c_ba].astype(str).str.strip().str.lower().eq(p["vertical"].strip().lower())
        if c_sba and p.get("sub_ba"):
            mask &= b[c_sba].astype(str).str.strip().str.lower().eq(p["sub_ba"].strip().lower())
        if c_ch:
            mask &= b[c_ch].astype(str).str.strip().str.lower().isin(["back office","bo","backoffice"])
        if loc_first:
            target = loc_first.strip().lower()
            for col in [c_site, c_location, c_country, c_city]:
                if col and col in b.columns:
                    loc_l = b[col].astype(str).str.strip().str.lower()
                    if loc_l.eq(target).any():
                        mask &= loc_l.eq(target)
                        break

        b = b.loc[mask]

        # Voice-like feed
        c_state = L.get("superstate") or L.get("state")
        c_hours = L.get("hours") or L.get("duration_hours")
        if c_date and c_state and c_hours and not b.empty and (not c_act or b[c_act].isna().all()):
            d2 = b[[c_date, c_state, c_hours]].copy()
            d2[c_hours] = pd.to_numeric(d2[c_hours], errors="coerce").fillna(0.0)
            d2[c_date]  = pd.to_datetime(d2[c_date], errors="coerce").dt.date
            pv = d2.pivot_table(index=c_date, columns=c_state, values=c_hours, aggfunc="sum", fill_value=0.0)
            def col(name): return pv[name] if name in pv.columns else 0.0
            base = col("SC_INCLUDED_TIME")
            ooo  = col("SC_ABSENCE_TOTAL") + col("SC_HOLIDAY") + col("SC_A_Sick_Long_Term")
            ino  = col("SC_TRAINING_TOTAL") + col("SC_BREAKS") + col("SC_SYSTEM_EXCEPTION")
            idx  = pd.to_datetime(pv.index, errors="coerce")
            _agg_weekly(idx, ooo, ino, base)

        # Activity-based feed
        elif c_date and c_act and c_sec and not b.empty:
            d = b[[c_date, c_act, c_sec]].copy()
            d[c_act]  = d[c_act].astype(str).str.strip().str.lower()
            d[c_sec]  = pd.to_numeric(d[c_sec], errors="coerce").fillna(0.0)
            d[c_date] = pd.to_datetime(d[c_date], errors="coerce").dt.date
            act = d[c_act]

            # Masks
            m_div  = act.str.contains(r"\bdivert(?:ed)?\b", regex=True, na=False) | act.eq("diverted")
            m_dow  = act.str.contains(r"\bdowntime\b|\bdown\b", regex=True, na=False) | act.eq("downtime")
            m_sc   = act.str.contains(r"\bstaff\s*complement\b|\bstaffed\s*hours\b|\bsc[_\s-]*included[_\s-]*time\b|\bincluded\s*time\b", regex=True, na=False) | act.eq("staff complement")
            m_fx   = act.str.contains(r"\bflexi?time\b|\bflex\b", regex=True, na=False) | act.eq("flexitime")
            m_ot   = act.str.contains(r"\bover\s*time\b|\bovertime\b|\bot\b|\bot\s*hours\b|\bot\s*hrs\b", regex=True, na=False) | act.eq("overtime")
            m_lend = act.str.contains(r"\blend\s*staff\b", regex=True, na=False) | act.eq("lend staff")
            m_borr = act.str.contains(r"\bborrow(?:ed)?\b|\bborrow\s*staff\b", regex=True, na=False) | act.eq("borrowed staff")

            # Aggregate seconds
            sec_div, sec_dow, sec_sc = d.loc[m_div, c_sec].groupby(d[c_date]).sum(), d.loc[m_dow, c_sec].groupby(d[c_date]).sum(), d.loc[m_sc, c_sec].groupby(d[c_date]).sum()
            sec_fx, sec_ot = d.loc[m_fx, c_sec].groupby(d[c_date]).sum(), d.loc[m_ot, c_sec].groupby(d[c_date]).sum()
            sec_lend, sec_borr = d.loc[m_lend, c_sec].groupby(d[c_date]).sum(), d.loc[m_borr, c_sec].groupby(d[c_date]).sum()

            idx = pd.to_datetime(pd.Index(set(sec_div.index) | set(sec_dow.index) | set(sec_sc.index) |
                                        set(sec_fx.index)  | set(sec_ot.index)  | set(sec_lend.index) | set(sec_borr.index)),
                                    errors="coerce").sort_values()
            def get(s): return s.reindex(idx, fill_value=0.0).astype(float)

            # Hours
            h_div, h_dow, h_sc = get(sec_div)/3600.0, get(sec_dow)/3600.0, get(sec_sc)/3600.0
            h_fx, h_ot, h_len, h_bor = get(sec_fx)/3600.0, get(sec_ot)/3600.0, get(sec_lend)/3600.0, get(sec_borr)/3600.0

            # Denominators
            ooo_hours = h_dow
            ino_hours = h_div
            sc_hours  = h_sc  # ✅ SC denominator is direct sum of Staff Complement
            ttw_hours = (h_sc - h_dow + h_fx + h_ot + h_bor - h_len).clip(lower=0)

            # Weekly aggregation
            _agg_weekly(idx, ooo_hours, ino_hours, sc_hours)

            wk = _week_key(ttw_hours.index)
            g_ttw = pd.DataFrame({"week": wk, "ttw": ttw_hours.values}).groupby("week", as_index=False).sum()
            for _, r in g_ttw.iterrows():
                k = str(r["week"])
                tt_worked_hours_w[k] = tt_worked_hours_w.get(k, 0.0) + float(r["ttw"])

            wk = _week_key(sc_hours.index)
            g_sc = pd.DataFrame({"week": wk, "sc": sc_hours}).groupby("week", as_index=False).sum()
            for _, r in g_sc.iterrows():
                k = str(r["week"])
                sc_hours_w[k] = sc_hours_w.get(k, 0.0) + float(r["sc"])

            # Daily shrink fractions
            for t, d_hrs, v_hrs in zip(idx, ooo_hours, ino_hours):
                k = str(pd.to_datetime(t).date())
                scv = float(sc_hours_w.get(k, 0.0))
                ttw = float(tt_worked_hours_w.get(k, 0.0))
                ooo_f = (float(d_hrs) / scv) if scv > 0 else 0.0
                ino_f = (float(v_hrs) / ttw) if ttw > 0 else 0.0
                bo_shrink_frac_daily[k] = max(0.0, min(0.99, ooo_f + ino_f))

            # Weekly overtime
            try:
                wk = _week_key(sec_ot.index)
                g_ot = pd.DataFrame({"week": wk, "ot": (sec_ot.astype(float) / 3600.0)}).groupby("week", as_index=False).sum()
                for _, r2 in g_ot.iterrows():
                    k2 = str(r2["week"])
                    overtime_hours_w[k2] = overtime_hours_w.get(k2, 0.0) + float(r2["ot"])
            except Exception:
                pass
            # Build shrink table (+ What-If ? onto Overall % display)
            _debug_shr = bool(os.environ.get("CAP_DEBUG_SHRINK"))
            for w in week_ids:
                if w not in shr.columns:
                    shr[w] = np.nan
                shr[w] = pd.to_numeric(shr[w], errors="coerce").astype("float64")
                base = float(base_hours_w.get(w, 0.0))
                ooo  = float(ooo_hours_w.get(w, 0.0))
                ino  = float(io_hours_w.get(w, 0.0))
                ch_key_local = str(ch_first or '').strip().lower()
                sc_base = float(sc_hours_w.get(w, 0.0))
                ttwh    = float(tt_worked_hours_w.get(w, 0.0))
                nb_base = float(nodow_base_hours_w.get(w, 0.0)) if 'nodow_base_hours_w' in locals() else base
                # Prefer BO denominators automatically when present; also allow fuzzy channel
                use_bo_denoms = (sc_base > 0.0 or ttwh > 0.0) or (ch_key_local in ("back office", "bo") or ("back office" in ch_key_local))
                if use_bo_denoms:
                    ooo_pct = (100.0 * ooo / sc_base) if sc_base > 0 else 0.0
                    ino_pct = (100.0 * ino / ttwh)    if ttwh    > 0 else 0.0
                    ov_pct  = (ooo_pct + ino_pct)
                else:
                    # Voice and other non-BO: fall back to base (SC_INCLUDED_TIME) as denominator
                    ooo_pct = (100.0 * ooo / base) if base > 0 else 0.0
                    ino_pct = (100.0 * ino / base) if base > 0 else 0.0
                    ov_pct  = (ooo_pct + ino_pct)

                # What-If: add shrink_delta to overall % display (clamped 0..100)
                if _wf_active(w) and shrink_delta:
                    ov_pct = min(100.0, max(0.0, ov_pct + shrink_delta))

                planned_pct = float(saved_plan_pct_w.get(w, 100.0 * planned_shrink_fraction))
                # What-If: also reflect shrink delta onto Planned Shrinkage % for display
                try:
                    if _wf_active(w) and shrink_delta:
                        planned_pct = min(100.0, max(0.0, planned_pct + shrink_delta))
                except Exception:
                    pass
                variance_pp = ov_pct - planned_pct

                shr.loc[shr["metric"] == "OOO Shrink Hours (#)",       w] = ooo
                shr.loc[shr["metric"] == "In-Office Shrink Hours (#)", w] = ino
                shr.loc[shr["metric"] == "OOO Shrinkage %",            w] = ooo_pct
                shr.loc[shr["metric"] == "In-Office Shrinkage %",       w] = ino_pct
                shr.loc[shr["metric"] == "Overall Shrinkage %",       w] = ov_pct
                shr.loc[shr["metric"] == "Planned Shrinkage %",       w] = planned_pct
                shr.loc[shr["metric"] == "Variance vs Planned",  w] = variance_pp
                if _debug_shr:
                    try:
                        print(f"[SHR][week={w}] ch={ch_key_local} OOO={ooo:.2f} SC={sc_base:.2f} TTW={ttwh:.2f} base={base:.2f} branch={'bo' if use_bo_denoms else 'non-bo'} OOO%={ooo_pct:.2f} INO%={ino_pct:.2f} OV%={ov_pct:.2f}")
                    except Exception:
                        pass

    # Ensure shrinkage table is populated even when only Voice raw exists (no BO activity feed)
    try:
        _debug_shr2 = bool(os.environ.get("CAP_DEBUG_SHRINK"))
        for w in week_ids:
            if w not in shr.columns:
                shr[w] = np.nan
            shr[w] = pd.to_numeric(shr[w], errors="coerce").astype("float64")
            base = float(base_hours_w.get(w, 0.0))
            ooo  = float(ooo_hours_w.get(w, 0.0))
            ino  = float(io_hours_w.get(w, 0.0))
            ch_key_local = str(ch_first or '').strip().lower()
            sc_base = float(sc_hours_w.get(w, 0.0))
            ttwh    = float(tt_worked_hours_w.get(w, 0.0))
            # Prefer BO denominators when available; otherwise Voice falls back to base
            use_bo_denoms = (sc_base > 0.0 or ttwh > 0.0) or (ch_key_local in ("back office","bo") or ("back office" in ch_key_local))
            if use_bo_denoms:
                ooo_pct = (100.0 * ooo / sc_base) if sc_base > 0 else 0.0
                ino_pct = (100.0 * ino / ttwh)    if ttwh    > 0 else 0.0
                ov_pct  = (ooo_pct + ino_pct)
            else:
                ooo_pct = (100.0 * ooo / base) if base > 0 else 0.0
                ino_pct = (100.0 * ino / base) if base > 0 else 0.0
                ov_pct  = (ooo_pct + ino_pct)
            if _wf_active(w) and shrink_delta:
                ov_pct = min(100.0, max(0.0, ov_pct + shrink_delta))
            planned_pct = float(saved_plan_pct_w.get(w, 100.0 * planned_shrink_fraction))
            try:
                if _wf_active(w) and shrink_delta:
                    planned_pct = min(100.0, max(0.0, planned_pct + shrink_delta))
            except Exception:
                pass
            variance_pp = ov_pct - planned_pct
            shr.loc[shr["metric"] == "OOO Shrink Hours (#)",       w] = ooo
            shr.loc[shr["metric"] == "In-Office Shrink Hours (#)", w] = ino
            shr.loc[shr["metric"] == "OOO Shrinkage %",            w] = ooo_pct
            shr.loc[shr["metric"] == "In-Office Shrinkage %",       w] = ino_pct
            shr.loc[shr["metric"] == "Overall Shrinkage %",       w] = ov_pct
            shr.loc[shr["metric"] == "Planned Shrinkage %",       w] = planned_pct
            shr.loc[shr["metric"] == "Variance vs Planned",        w] = variance_pp
            if _debug_shr2:
                try:
                    print(f"[SHR2][week={w}] ch={ch_key_local} OOO={ooo:.2f} SC={sc_base:.2f} TTW={ttwh:.2f} base={base:.2f} branch={'bo' if use_bo_denoms else 'non-bo'} OOO%={ooo_pct:.2f} INO%={ino_pct:.2f} OV%={ov_pct:.2f}")
                except Exception:
                    pass
    except Exception:
        pass

    # --- Adjust BO daily FTE using actual shrink when available; else planned ---
    def _adjust_bo_fte_daily(df: pd.DataFrame, use_actual: bool) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df
        d = df.copy()
        if "bo_fte" not in d.columns or "date" not in d.columns:
            return d
        s_plan = planned_shrink_fraction if 'planned_shrink_fraction' in locals() else 0.0
        try:
            s_plan = float(s_plan)
            if s_plan > 1.0:
                s_plan /= 100.0
        except Exception:
            s_plan = 0.0
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date.astype(str)
        old_bo = pd.to_numeric(d["bo_fte"], errors="coerce").fillna(0.0)
        s_target = d["date"].map(lambda x: bo_shrink_frac_daily.get(x, s_plan) if use_actual else s_plan)
        try:
            s_target = pd.to_numeric(s_target, errors="coerce").fillna(s_plan)
        except Exception:
            s_target = pd.Series([s_plan]*len(d))
        denom_old = np.maximum(0.01, 1.0 - float(s_plan))
        denom_new = np.maximum(0.01, 1.0 - s_target.astype(float))
        factor = denom_old / denom_new
        new_bo = old_bo * factor
        d["bo_fte"] = new_bo
        # If total present, adjust by delta
        if "total_req_fte" in d.columns:
            tot = pd.to_numeric(d["total_req_fte"], errors="coerce").fillna(0.0)
            d["total_req_fte"] = tot + (new_bo - old_bo)
        return d

    try:
        req_daily_actual   = _adjust_bo_fte_daily(req_daily_actual,   use_actual=True)
        req_daily_forecast = _adjust_bo_fte_daily(req_daily_forecast, use_actual=False)
        if isinstance(req_daily_tactical, pd.DataFrame) and not req_daily_tactical.empty:
            req_daily_tactical = _adjust_bo_fte_daily(req_daily_tactical, use_actual=False)
        if isinstance(req_daily_budgeted, pd.DataFrame) and not req_daily_budgeted.empty:
            req_daily_budgeted = _adjust_bo_fte_daily(req_daily_budgeted, use_actual=False)
    except Exception:
        pass

    # Recompute weekly averages after adjustments
    req_w_actual   = _daily_to_weekly(req_daily_actual)
    req_w_forecast = _daily_to_weekly(req_daily_forecast)
    req_w_tactical = _daily_to_weekly(req_daily_tactical)
    req_w_budgeted = _daily_to_weekly(req_daily_budgeted)

    # Write Overtime Hours (#) from shrinkage raw into FW and merged FW (independent of shrinkage logic)
    if "Overtime Hours (#)" in fw_rows:
        for w in week_ids:
            val = None
            ch_key_local = str(ch_first or '').strip().lower()
            if ch_key_local == 'voice':
                val = _ot_voice_w_raw.get(w, None)
            elif ch_key_local in ("back office","bo"):
                val = _ot_w_raw.get(w, None)
            # fallback to saved/previous map if present
            if val is None:
                val = overtime_w.get(w, None)
            if val is not None:
                fw.loc[fw["metric"] == "Overtime Hours (#)", w] = float(val)
                # Also reflect in merged FW so the UI receives it
                try:
                    fw_to_use.loc[fw_to_use["metric"] == "Overtime Hours (#)", w] = float(val)
                except Exception:
                    pass

    # Compute Actual-shrinkage-adjusted weekly requirement just before upper table
    req_w_actual_adj = dict(req_w_actual)
    try:
        def _to_frac(val):
            try:
                v = float(val)
                return v/100.0 if v > 1.0 else v
            except Exception:
                try:
                    s = str(val).strip().rstrip('%')
                    v = float(s)
                    return v/100.0 if v > 1.0 else v
                except Exception:
                    return None
        act_row = None
        plan_row = None
        if isinstance(shr, pd.DataFrame) and not shr.empty and "metric" in shr.columns:
            m = shr["metric"].astype(str).str.strip().str.lower()
            if (m == "overall shrinkage %").any():
                act_row = shr.loc[m == "overall shrinkage %"].iloc[0].to_dict()
            if (m == "planned shrinkage %").any():
                plan_row = shr.loc[m == "planned shrinkage %"].iloc[0].to_dict()
        for w in week_ids:
            if w not in req_w_actual_adj:
                continue
            s_act = _to_frac(act_row.get(w)) if isinstance(act_row, dict) and (w in act_row) else None
            if s_act is None:
                continue
            denom_new = max(0.01, 1.0 - float(s_act or 0.0))
            # Use the unadjusted actual requirement as baseline and scale only by Actual shrink
            req_w_actual_adj[w] = float(req_w_actual.get(w, 0.0)) / denom_new
    except Exception:
        pass

    # ---- BvA ----
    # Align Weekly Budgeted FTE with the same calculation used for
    # 'FTE Required @ Forecast Volume' to avoid discrepancies.
    for w in week_ids:
        if w not in bva.columns:
            bva[w] = pd.Series(np.nan, index=bva.index, dtype="float64")
        elif not pd.api.types.is_float_dtype(bva[w].dtype):
            bva[w] = pd.to_numeric(bva[w], errors="coerce").astype("float64")
        # Use weekly forecast requirement for Budgeted FTE to match
        # 'FTE Required @ Forecast Volume'.
        bud = float(req_w_forecast.get(w, 0.0))
        act = float(req_w_actual.get(w,   0.0))
        bva.loc[bva["metric"] == "Budgeted FTE (#)", w] = bud
        bva.loc[bva["metric"] == "Actual FTE (#)",   w] = act
        bva.loc[bva["metric"] == "Variance (#)",     w] = act - bud

    # Precompute seat variance (pp) from Seats Required vs Available, if present
    try:
        for w in week_ids:
            req = float(pd.to_numeric(seat.loc[seat["metric"] == "Seats Required (#)", w], errors="coerce").fillna(0).iloc[0]) if w in seat.columns else 0.0
            avail = float(pd.to_numeric(seat.loc[seat["metric"] == "Seats Available (#)", w], errors="coerce").fillna(0).iloc[0]) if w in seat.columns else 0.0
            seat_util = ((avail / req) * 100) if avail > 0 else 0.0
            var_val = (avail - req)
            if "Seat Utilization %" in seat["metric"].values:
                seat.loc[seat["metric"] == "Seat Utilization %", w] = seat_util
            if "Variance vs Available" in seat["metric"].values:
                seat.loc[seat["metric"] == "Variance vs Available", w] = var_val
    except Exception:
        pass

    # ---- TL/Agent ratios ----
    planned_ratio = _parse_ratio_setting(settings.get("planned_tl_agent_ratio") or settings.get("tl_agent_ratio") or settings.get("tl_per_agent"))
    actual_ratio = 0.0
    try:
        if isinstance(roster_df, pd.DataFrame) and not roster_df.empty and "role" in roster_df.columns:
            r = roster_df.copy()
            r["role"] = r["role"].astype(str).str.strip().str.lower()
            tl = (r["role"] == "team leader").sum()
            ag = (r["role"] == "agent").sum()
            actual_ratio = (float(tl) / float(ag)) if ag > 0 else 0.0
    except Exception:
        pass
    # Preserve user-edited ratios; recompute Variance only
    for w in week_ids:
        try:
            cur_planned = float(pd.to_numeric(rat.loc[rat["metric"] == "Planned TL/Agent Ratio", w], errors="coerce").fillna(0.0).iloc[0])
        except Exception:
            cur_planned = 0.0
        try:
            cur_actual = float(pd.to_numeric(rat.loc[rat["metric"] == "Actual TL/Agent Ratio",  w], errors="coerce").fillna(0.0).iloc[0])
        except Exception:
            cur_actual = 0.0
        # If both are zero (fresh grid), seed from settings/roster once
        if (cur_planned == 0.0 and cur_actual == 0.0):
            rat.loc[rat["metric"] == "Planned TL/Agent Ratio", w] = planned_ratio
            rat.loc[rat["metric"] == "Actual TL/Agent Ratio",  w] = actual_ratio
            cur_planned, cur_actual = planned_ratio, actual_ratio
        rat.loc[rat["metric"] == "Variance", w] = float(cur_actual) - float(cur_planned)

    # ---- Projected supply (actuals to date; planned + attr/nH forward) ----
    def _row_as_dict(df, metric_name):
        if not isinstance(df, pd.DataFrame) or df.empty or "metric" not in df.columns:
            return {}
        m = df["metric"].astype(str).str.strip()
        if metric_name not in m.values:
            return {}
        row = df.loc[m == metric_name].iloc[0]
        return {w: float(pd.to_numeric(row.get(w), errors="coerce")) for w in week_ids}

    hc_plan_row   = _row_as_dict(hc,  "Planned/Tactical HC (#)")
    hc_actual_row = {w: float(hc_actual_w.get(w, 0)) for w in week_ids}
    att_plan_row  = _row_as_dict(att, "Planned Attrition HC (#)")
    att_act_row   = _row_as_dict(att, "Actual Attrition HC (#)")

    today_w = _monday(dt.date.today()).isoformat()

    # attrition to use (add attr_delta within active window)
    # attrition to use (prefer Actual if present; otherwise Planned) and apply What-If delta within active window
    att_use_row = {}
    for w in week_ids:
        try:
            act_v = float(att_act_row.get(w, 0) or 0.0)
        except Exception:
            act_v = 0.0
        try:
            plan_v = float(att_plan_row.get(w, 0) or 0.0)
        except Exception:
            plan_v = 0.0
        base = act_v if act_v > 0 else plan_v
        if _wf_active(w):
            base += attr_delta
        att_use_row[w] = base

        # NH additions to use: prefer Actual if present (>0), else Planned (for all weeks)
    nh_add_row = {}
    for w in week_ids:
        try:
            a = float(actual_nh_w.get(w, 0) or 0.0)
        except Exception:
            a = 0.0
        try:
            p = float(planned_nh_w.get(w, 0) or 0.0)
        except Exception:
            p = 0.0
        nh_add_row[w] = a if a > 0 else p

    projected_supply = {}
    prev = None
    for w in week_ids:
        if w <= today_w and hc_actual_row.get(w, 0) > 0:
            projected_supply[w] = hc_actual_row.get(w, 0)
            prev = projected_supply[w]
        else:
            # Seed the first forecast week using Planned/Tactical HC if available;
            # otherwise fall back to the roster-derived snapshot for that week.
            if prev is None:
                prev = float(hc_plan_row.get(w, 0) or 0.0)
                if prev <= 0:
                    prev = float(hc_actual_row.get(w, 0) or 0.0)
            next_val = max(prev - float(att_use_row.get(w, 0)) + float(nh_add_row.get(w, 0)), 0.0)
            projected_supply[w] = next_val
            prev = next_val

    # ---- Handling capacity & Projected SL ----
    def _erlang_c(A: float, N: int) -> float:
        if N <= 0: return 1.0
        if A <= 0: return 0.0
        if A >= N: return 1.0
        s = 0.0
        for k in range(N):
            s += (A**k) / math.factorial(k)
        last = (A**N) / math.factorial(N) * (N / (N - A))
        p0 = 1.0 / (s + last)
        return last * p0

    def _erlang_sl(calls_per_ivl: float, aht_sec: float, agents: float, asa_sec: int, ivl_sec: int) -> float:
        if aht_sec <= 0 or ivl_sec <= 0 or agents <= 0 or calls_per_ivl <= 0:
            return 0.0
        A = (calls_per_ivl * aht_sec) / ivl_sec
        pw = _erlang_c(A, int(math.floor(agents)))
        return max(0.0, min(1.0, 1.0 - pw * math.exp(-max(0.0, (agents - A)) * (asa_sec / max(1.0, aht_sec)))))

    def _erlang_calls_capacity(agents: float, aht_sec: float, asa_sec: int, ivl_sec: int, target_pct: float) -> float:
        if agents <= 0 or aht_sec <= 0 or ivl_sec <= 0:
            return 0.0
        target = float(target_pct) / 100.0
        hi = max(1, int((agents * ivl_sec) / aht_sec))
        def sl_for(x): return _erlang_sl(x, aht_sec, agents, asa_sec, ivl_sec)
        lo = 0
        while sl_for(hi) >= target and hi < 10_000_000:
            lo = hi; hi *= 2
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if sl_for(mid) >= target: lo = mid
            else: hi = mid - 1
        return float(lo)

    handling_capacity = {}

    def _metric_for_capacity(actual_map, forecast_map, week):
        # Prefer Forecast (weekly plan) like the monthly view; fall back to Actual.
        def _clean(val):
            try:
                return max(0.0, float(val))
            except Exception:
                return 0.0
        act  = _clean(actual_map.get(week, 0.0))
        fore = _clean(forecast_map.get(week, 0.0))
        if fore > 0.0:
            return fore
        if act > 0.0:
            return act
        return 0.0

    bo_model   = (settings.get("bo_capacity_model") or "tat").lower()
    bo_wd      = int(settings.get("bo_workdays_per_week", 5))
    bo_hpd     = float(settings.get("bo_hours_per_day", settings.get("hours_per_fte", 8.0)))
    # base shrink (fraction 0..1)
    bo_shr_base = float(settings.get("bo_shrinkage_pct", settings.get("shrinkage_pct", 0.0)) or 0.0)
    if bo_shr_base > 1.0: bo_shr_base /= 100.0
    # Voice shrink base (fraction 0..1) ? used to reduce effective agents for Voice capacity/SL
    voice_shr_base = float(settings.get("voice_shrinkage_pct", settings.get("shrinkage_pct", 0.0)) or 0.0)
    if voice_shr_base > 1.0: voice_shr_base /= 100.0
    util_bo    = float(settings.get("util_bo", 0.85))

    for w in week_ids:
        ch_low = ch_first.lower()
        if ch_low == "voice":
            agents_prod = float(schedule_supply_avg.get(w, projected_supply.get(w, 0.0)))
            lc = _lc_with_wf(_learning_curve_for_week(settings, lc_ovr_df, w), w)

            def eff_from_buckets(buckets, prod_pct_list, uplift_pct_list):
                total = 0.0
                login_f = _ovr_login_frac(w)
                aht_m   = _ovr_aht_mult(w)
                for age, cnt in (buckets.get(w, {}) or {}).items():
                    p = (prod_pct_list[age-1] / 100.0) if age-1 < len(prod_pct_list) else 0.0
                    u = (uplift_pct_list[age-1] / 100.0) if age-1 < len(uplift_pct_list) else 0.0
                    if login_f is not None: p *= login_f
                    denom = (1.0 + u)
                    if aht_m is not None:   denom *= aht_m
                    total += float(cnt) * (p / max(1.0, denom))
                return total

            nest_eff = eff_from_buckets(nest_buckets, lc["nesting_prod_pct"], lc["nesting_aht_uplift_pct"])
            sda_eff  = eff_from_buckets(sda_buckets,  lc["sda_prod_pct"],     lc["sda_aht_uplift_pct"])
            # Apply shrink to Voice effective agents as well (in addition to occupancy)
            v_shr_add = (shrink_delta / 100.0) if (_wf_active(w) and shrink_delta) else 0.0
            v_eff_shr = min(0.99, max(0.0, voice_shr_base + v_shr_add))
            agents_eff = max(1.0, (agents_prod + nest_eff + sda_eff) * occ_frac_w[w] * (1.0 - v_eff_shr))
            # Overtime: treat as equivalent agents based on hours per FTE and workdays/week
            try:
                ot = float(overtime_w.get(w, 0.0) or 0.0)
            except Exception:
                ot = 0.0
            if ot:
                wd_voice = int(settings.get("workdays_per_week", 7) or 7)
                hpd      = float(settings.get("hours_per_fte", 8.0) or 8.0)
                agents_eff += max(0.0, ot) / max(1.0, wd_voice * hpd)

            base_aht = _metric_for_capacity(wk_aht_sut_actual, wk_aht_sut_forecast, w)
            # What-If AHT applies to future-only when no explicit window set
            aht = float(base_aht)
            if _wf_active(w) and aht_delta:
                aht = max(1.0, aht * (1.0 + aht_delta / 100.0))
            else:
                aht = max(1.0, aht)
            n = weekly_voice_intervals.get(w)
            intervals = int(n) if isinstance(n, (int, np.integer)) and n > 0 else intervals_per_week_default
            calls_per_ivl = _erlang_calls_capacity(agents_eff, aht, sl_seconds, ivl_sec, sl_target_pct)
            handling_capacity[w] = calls_per_ivl * intervals
            # Optional debug for Voice only (does not alter logic)
            if os.environ.get("CAP_DEBUG_VOICE"):
                try:
                    print(
                        f"[VOICE-WEEK][{w}] vol={weekly_demand_voice.get(w,0)} aht={aht:.1f} occ={occ_frac_w.get(w,0):.2f} agents_eff={agents_eff:.2f} "
                        f"intervals={intervals} cap={handling_capacity[w]:.1f} sl={proj_sl.get(w,0.0):.1f}"
                    )
                except Exception:
                    pass
        elif ch_low in ("back office", "bo"):
            base_sut = _metric_for_capacity(wk_aht_sut_actual, wk_aht_sut_forecast, w)
            sut = float(base_sut)
            if _wf_active(w) and aht_delta:
                sut = max(0.0, sut * (1.0 + aht_delta / 100.0))
            else:
                sut = max(0.0, sut)

            lc = _learning_curve_for_week(settings, lc_ovr_df, w)
            def eff(buckets, prod_pct_list, uplift_pct_list):
                total = 0.0
                login_f = _ovr_login_frac(w)
                aht_m   = _ovr_aht_mult(w)
                for age, cnt in (buckets.get(w, {}) or {}).items():
                    p = (prod_pct_list[age-1] / 100.0) if age-1 < len(prod_pct_list) else 0.0
                    u = (uplift_pct_list[age-1] / 100.0) if age-1 < len(uplift_pct_list) else 0.0
                    if login_f is not None: p *= login_f
                    denom = (1.0 + u)
                    if aht_m is not None:   denom *= aht_m
                    total += float(cnt) * (p / max(1.0, denom))
                return total

            agents_eff = max(1.0, float(projected_supply.get(w, 0.0)) + eff(nest_buckets, lc["nesting_prod_pct"], lc["nesting_aht_uplift_pct"]) + eff(sda_buckets, lc["sda_prod_pct"], lc["sda_aht_uplift_pct"]))
            if bo_model == "tat":
                shr_add = (shrink_delta / 100.0) if (_wf_active(w) and shrink_delta) else 0.0
                eff_shr = min(0.99, max(0.0, bo_shr_base + shr_add))
                # Refactor for clarity: total productive hours = (regular+OT hours) * (1 - shrink) * util
                base_hours = bo_wd * bo_hpd
                try:
                    ot = float(overtime_w.get(w, 0.0) or 0.0)
                except Exception:
                    ot = 0.0
                total_hours = (float(agents_eff) * base_hours) + max(0.0, ot)
                total_prod_hours = total_hours * (1.0 - eff_shr) * util_bo
                # Guard against zero/invalid SUT. Fall back to planned/target if needed.
                try:
                    sut_safe = float(sut)
                except Exception:
                    sut_safe = 0.0
                if sut_safe <= 0.0:
                    try:
                        sut_safe = float(planned_sut_w.get(w, s_target_sut))
                    except Exception:
                        sut_safe = float(s_target_sut)
                sut_safe = max(1.0, float(sut_safe))
                handling_capacity[w] = total_prod_hours * (3600.0 / sut_safe)
            else:
                ivl_per_week = int(round(bo_wd * bo_hpd / (ivl_sec / 3600.0)))
                agents_eff_u = max(1.0, float(agents_eff) * util_bo)
                items_per_ivl = _erlang_calls_capacity(agents_eff_u, sut, sl_seconds, ivl_sec, sl_target_pct)
                handling_capacity[w] = items_per_ivl * ivl_per_week
        elif ch_low in ("outbound", "ob", "out bound"):
            # Outbound via Erlang (expected contacts per interval)
            ob_util = float(settings.get("util_ob", settings.get("occupancy_cap_voice", 0.85)) or 0.85)
            ob_shr  = float(settings.get("ob_shrinkage_pct", settings.get("shrinkage_pct", 0.30)) or 0.30)
            if ob_shr > 1.0: ob_shr /= 100.0
            lc = _learning_curve_for_week(settings, lc_ovr_df, w)
            def eff_ob(buckets, prod_pct_list, uplift_pct_list):
                total = 0.0
                login_f = _ovr_login_frac(w)
                aht_m   = _ovr_aht_mult(w)
                for age, cnt in (buckets.get(w, {}) or {}).items():
                    p = (prod_pct_list[age-1] / 100.0) if age-1 < len(prod_pct_list) else 0.0
                    u = (uplift_pct_list[age-1] / 100.0) if age-1 < len(uplift_pct_list) else 0.0
                    if login_f is not None: p *= login_f
                    denom = (1.0 + u)
                    if aht_m is not None:   denom *= aht_m
                    total += float(cnt) * (p / max(1.0, denom))
                return total
            agents_eff = max(1.0, (float(projected_supply.get(w, 0.0)) + eff_ob(nest_buckets, lc["nesting_prod_pct"], lc["nesting_aht_uplift_pct"]) + eff_ob(sda_buckets, lc["sda_prod_pct"], lc["sda_aht_uplift_pct"])) * ob_util * (1.0 - ob_shr))
            ivl_min_ob = int(float(settings.get("ob_interval_minutes", settings.get("interval_minutes", 30)) or 30))
            ivl_sec_ob = max(60, ivl_min_ob * 60)
            base_aht = _metric_for_capacity(wk_aht_sut_actual, wk_aht_sut_forecast, w)
            aht = float(base_aht)
            if _wf_active(w) and aht_delta:
                aht = max(1.0, aht * (1.0 + aht_delta / 100.0))
            else:
                aht = max(1.0, aht)
            items_per_ivl = _erlang_calls_capacity(agents_eff, aht, sl_seconds, ivl_sec_ob, sl_target_pct)
            # Coverage minutes per day -> intervals per week using bo_wd workdays
            hrs = float(settings.get("hours_per_fte", 8.0) or 8.0)
            cov_min = int(float(settings.get("ob_coverage_minutes", hrs * 60.0) or (hrs * 60.0)))
            ivl_per_week = int(max(1, round((cov_min / ivl_min_ob)))) * max(1, bo_wd)
            handling_capacity[w] = items_per_ivl * ivl_per_week
        elif "chat" in ch_low:
            # Chat via Erlang with concurrency
            chat_util = float(settings.get("util_chat", settings.get("util_bo", 0.85)) or 0.85)
            chat_shr  = float(settings.get("chat_shrinkage_pct", settings.get("shrinkage_pct", 0.30)) or 0.30)
            if chat_shr > 1.0: chat_shr /= 100.0
            conc = float(settings.get("chat_concurrency", 1.0) or 1.0)
            eff_conc = max(1e-6, conc)
            lc = _learning_curve_for_week(settings, lc_ovr_df, w)
            def eff_ch(buckets, prod_pct_list, uplift_pct_list):
                total = 0.0
                login_f = _ovr_login_frac(w)
                aht_m   = _ovr_aht_mult(w)
                for age, cnt in (buckets.get(w, {}) or {}).items():
                    p = (prod_pct_list[age-1] / 100.0) if age-1 < len(prod_pct_list) else 0.0
                    u = (uplift_pct_list[age-1] / 100.0) if age-1 < len(uplift_pct_list) else 0.0
                    if login_f is not None: p *= login_f
                    denom = (1.0 + u)
                    if aht_m is not None:   denom *= aht_m
                    total += float(cnt) * (p / max(1.0, denom))
                return total
            agents_eff = max(1.0, (float(projected_supply.get(w, 0.0)) + eff_ch(nest_buckets, lc["nesting_prod_pct"], lc["nesting_aht_uplift_pct"]) + eff_ch(sda_buckets, lc["sda_prod_pct"], lc["sda_aht_uplift_pct"])) * chat_util * (1.0 - chat_shr))
            ivl_min_ch = int(float(settings.get("chat_interval_minutes", settings.get("interval_minutes", 30)) or 30))
            ivl_sec_ch = max(60, ivl_min_ch * 60)
            base_aht = _metric_for_capacity(wk_aht_sut_actual, wk_aht_sut_forecast, w)
            aht = float(base_aht) / max(1e-6, eff_conc)
            if _wf_active(w) and aht_delta:
                aht = max(1.0, aht * (1.0 + aht_delta / 100.0))
            else:
                aht = max(1.0, aht)
            items_per_ivl = _erlang_calls_capacity(agents_eff, aht, sl_seconds, ivl_sec_ch, sl_target_pct)
            hrs = float(settings.get("hours_per_fte", 8.0) or 8.0)
            cov_min = int(float(settings.get("chat_coverage_minutes", hrs * 60.0) or (hrs * 60.0)))
            ivl_per_week = int(max(1, round((cov_min / ivl_min_ch)))) * max(1, bo_wd)
            handling_capacity[w] = items_per_ivl * ivl_per_week

    # projected service level
    proj_sl = {}
    for w in week_ids:
        ch_low = ch_first.lower()
        if ch_low == "voice":
            weekly_load = float(weekly_demand_voice.get(w, 0.0))
            base_aht_sut = _metric_for_capacity(wk_aht_sut_actual, wk_aht_sut_forecast, w)
            aht_sut = float(base_aht_sut)
            if _wf_active(w) and aht_delta:
                aht_sut = max(1.0, aht_sut * (1.0 + aht_delta / 100.0))
            else:
                aht_sut = max(1.0, aht_sut)
            n = weekly_voice_intervals.get(w)
            intervals = int(n) if isinstance(n, (int, np.integer)) and n > 0 else intervals_per_week_default
            calls_per_ivl = weekly_load / float(max(1, intervals))

            lc = _learning_curve_for_week(settings, lc_ovr_df, w)
            def eff_from_buckets(buckets, prod_pct_list, uplift_pct_list):
                total = 0.0
                login_f = _ovr_login_frac(w)
                aht_m   = _ovr_aht_mult(w)
                for age, cnt in (buckets.get(w, {}) or {}).items():
                    p = (prod_pct_list[age-1] / 100.0) if age-1 < len(prod_pct_list) else 0.0
                    u = (uplift_pct_list[age-1] / 100.0) if age-1 < len(uplift_pct_list) else 0.0
                    if login_f is not None: p *= login_f
                    denom = (1.0 + u)
                    if aht_m is not None:   denom *= aht_m
                    total += float(cnt) * (p / max(1.0, denom))
                return total

            nest_eff = eff_from_buckets(nest_buckets, lc["nesting_prod_pct"], lc["nesting_aht_uplift_pct"])
            sda_eff  = eff_from_buckets(sda_buckets,  lc["sda_prod_pct"],     lc["sda_aht_uplift_pct"])

            # Apply shrink to Voice effective agents in SL too
            v_shr_add = (shrink_delta / 100.0) if (_wf_active(w) and shrink_delta) else 0.0
            v_eff_shr = min(0.99, max(0.0, voice_shr_base + v_shr_add))
            agents_eff = max(1.0, (float(projected_supply.get(w, 0.0)) + nest_eff + sda_eff) * occ_frac_w[w] * (1.0 - v_eff_shr)
)
            sl_frac = _erlang_sl(calls_per_ivl, max(1.0, float(aht_sut)), agents_eff, sl_seconds, ivl_sec)
            proj_sl[w] = 100.0 * sl_frac
            if os.environ.get("CAP_DEBUG_VOICE"):
                try:
                    print(
                        f"[VOICE-WEEK-SL][{w}] load={weekly_load:.1f} aht={aht_sut:.1f} intervals={intervals} agents_eff={agents_eff:.2f} sl={proj_sl[w]:.1f}"
                    )
                except Exception:
                    pass
        elif ch_low in ("back office", "bo"):
            weekly_load = float(weekly_demand_bo.get(w, 0.0))
            if bo_model == "tat":
                cap = float(handling_capacity.get(w, 0.0))
                proj_sl[w] = 0.0 if weekly_load <= 0 else min(100.0, 100.0 * cap / weekly_load)
            else:
                ivl_per_week = int(round(bo_wd * bo_hpd / (ivl_sec / 3600.0)))
                items_per_ivl = weekly_load / float(max(1, ivl_per_week))
                base_sut = _metric_for_capacity(wk_aht_sut_actual, wk_aht_sut_forecast, w)
                sut = float(base_sut)
                if _wf_active(w) and aht_delta:
                    sut = max(0.0, sut * (1.0 + aht_delta / 100.0))
                else:
                    sut = max(0.0, sut)
                lc = _learning_curve_for_week(settings, lc_ovr_df, w)
                def eff(buckets, prod_pct_list, uplift_pct_list):
                    total = 0.0
                    login_f = _ovr_login_frac(w)
                    aht_m   = _ovr_aht_mult(w)
                    for age, cnt in (buckets.get(w, {}) or {}).items():
                        p = (prod_pct_list[age-1] / 100.0) if age-1 < len(prod_pct_list) else 0.0
                        u = (uplift_pct_list[age-1] / 100.0) if age-1 < len(uplift_pct_list) else 0.0
                        if login_f is not None: p *= login_f
                        denom = (1.0 + u)
                        if aht_m is not None:   denom *= aht_m
                        total += float(cnt) * (p / max(1.0, denom))
                    return total
                agents_eff = max(1.0, (float(projected_supply.get(w, 0.0)) + eff(nest_buckets, lc["nesting_prod_pct"], lc["nesting_aht_uplift_pct"]) + eff(sda_buckets, lc["sda_prod_pct"], lc["sda_aht_uplift_pct"])) * util_bo)
                sl_frac = _erlang_sl(items_per_ivl, max(1.0, float(sut)), agents_eff, sl_seconds, ivl_sec)
                proj_sl[w] = 100.0 * sl_frac
        elif ch_low in ("outbound", "ob", "out bound"):
            # Build a weekly expected contacts load from OB DF (forecast/actual/tactical)
            def _expected_contacts_df(df):
                if not isinstance(df, pd.DataFrame) or df.empty:
                    return pd.DataFrame(columns=["date","contacts"])
                d = df.copy()
                cols = {c.lower(): c for c in d.columns}
                c_date = cols.get("date"); c_opc = cols.get("opc"); c_conn = cols.get("connect_rate"); c_rpc = cols.get("rpc"); c_rpc_r = cols.get("rpc_rate")
                if not c_date or not (c_opc or c_rpc or c_conn):
                    return pd.DataFrame(columns=["date","contacts"])
                def _to_frac(x):
                    try:
                        v = float(x); return v/100.0 if v>1.0 else v
                    except Exception:
                        return 0.0
                def _contacts(row):
                    opc = float(row.get(c_opc, 0.0) if c_opc else 0.0)
                    rpc_ct = float(row.get(c_rpc, 0.0) if c_rpc else 0.0)
                    conn = _to_frac(row.get(c_conn, 0.0)) if c_conn else None
                    rpc_rate = _to_frac(row.get(c_rpc_r, 0.0)) if c_rpc_r else None
                    if rpc_ct and rpc_ct>0: return rpc_ct
                    if conn is not None and rpc_rate is not None: return opc * conn * rpc_rate
                    if conn is not None: return opc * conn
                    return opc
                out = pd.DataFrame({
                    "date": pd.to_datetime(d[c_date], errors="coerce").dt.date,
                    "contacts": d.apply(_contacts, axis=1).astype(float)
                }).dropna(subset=["date"]) 
                return out
            weekly_load = 0.0
            for src in (oF, oA, oT):
                tmp = _expected_contacts_df(src)
                if not tmp.empty:
                    tmp["week"] = (pd.to_datetime(tmp["date"]) - pd.to_timedelta(pd.to_datetime(tmp["date"]).dt.weekday, unit="D")).dt.date.astype(str)
                    s = tmp.groupby("week", as_index=False)["contacts"].sum().set_index("week")["contacts"].to_dict()
                    weekly_load = float(s.get(w, 0.0))
                    if weekly_load>0: break
            cap = float(handling_capacity.get(w, 0.0))
            proj_sl[w] = 0.0 if weekly_load <= 0 else min(100.0, 100.0 * cap / weekly_load)
        elif "chat" in ch_low:
            def _items_df(df):
                if not isinstance(df, pd.DataFrame) or df.empty:
                    return pd.DataFrame(columns=["date","items"])
                d = df.copy()
                cols = {c.lower(): c for c in d.columns}
                c_date = cols.get("date"); c_items = cols.get("items") or cols.get("volume")
                if not c_date or not c_items:
                    return pd.DataFrame(columns=["date","items"])
                out = pd.DataFrame({
                    "date": pd.to_datetime(d[c_date], errors="coerce").dt.date,
                    "items": pd.to_numeric(d[c_items], errors="coerce").fillna(0.0)
                }).dropna(subset=["date"]) 
                return out
            weekly_load = 0.0
            for src in (cF, cA, cT):
                tmp = _items_df(src)
                if not tmp.empty:
                    tmp["week"] = (pd.to_datetime(tmp["date"]) - pd.to_timedelta(pd.to_datetime(tmp["date"]).dt.weekday, unit="D")).dt.date.astype(str)
                    s = tmp.groupby("week", as_index=False)["items"].sum().set_index("week")["items"].to_dict()
                    weekly_load = float(s.get(w, 0.0))
                    if weekly_load>0: break
            cap = float(handling_capacity.get(w, 0.0))
            proj_sl[w] = 0.0 if weekly_load <= 0 else min(100.0, 100.0 * cap / weekly_load)

    # ---- Upper summary table ----
    upper_df = _blank_grid(spec["upper"], week_ids)
    if "FTE Required @ Forecast Volume" in spec["upper"]:
        for w in week_ids:
            upper_df.loc[upper_df["metric"] == "FTE Required @ Forecast Volume", w] = float(req_w_forecast.get(w, 0.0))
    # Optional: FTE Required @ Queue (scale forecast requirement by Queue/Forecast load)
    if "FTE Required @ Queue" in spec["upper"]:
        for w in week_ids:
            try:
                fval = float(pd.to_numeric(fw_to_use.loc[fw_to_use["metric"] == "Forecast", w], errors="coerce").fillna(0.0).iloc[0]) if w in fw_to_use.columns else 0.0
            except Exception:
                fval = 0.0
            qval = float(queue_w.get(w, 0.0)) if queue_w else 0.0
            base_req = float(req_w_forecast.get(w, 0.0))
            reqq = (base_req * (qval / fval)) if fval > 0 else 0.0
            upper_df.loc[upper_df["metric"] == "FTE Required @ Queue", w] = reqq
    if "FTE Required @ Actual Volume" in spec["upper"]:
        for w in week_ids:
            upper_df.loc[upper_df["metric"] == "FTE Required @ Actual Volume", w] = float(req_w_actual_adj.get(w, 0.0))
    if "FTE Over/Under MTP Vs Actual" in spec["upper"]:
        for w in week_ids:
            upper_df.loc[upper_df["metric"] == "FTE Over/Under MTP Vs Actual", w] = float(req_w_forecast.get(w, 0.0)) - float(req_w_actual_adj.get(w, 0.0))
    if "FTE Over/Under Tactical Vs Actual" in spec["upper"]:
        for w in week_ids:
            upper_df.loc[upper_df["metric"] == "FTE Over/Under Tactical Vs Actual", w] = float(req_w_tactical.get(w, 0.0)) - float(req_w_actual_adj.get(w, 0.0))
    if "FTE Over/Under Budgeted Vs Actual" in spec["upper"]:
        for w in week_ids:
            upper_df.loc[upper_df["metric"] == "FTE Over/Under Budgeted Vs Actual", w] = float(req_w_budgeted.get(w, 0.0)) - float(req_w_actual_adj.get(w, 0.0))
    if "Projected Supply HC" in spec["upper"]:
        for w in week_ids:
            upper_df.loc[upper_df["metric"] == "Projected Supply HC", w] = projected_supply.get(w, 0.0)
    if "Projected Handling Capacity (#)" in spec["upper"]:
        for w in week_ids:
            upper_df.loc[upper_df["metric"] == "Projected Handling Capacity (#)", w] = handling_capacity.get(w, 0.0)
    if "Projected Service Level" in spec["upper"]:
        for w in week_ids:
            upper_df.loc[upper_df["metric"] == "Projected Service Level", w] = proj_sl.get(w, 0.0)

    # ---- rounding & display formatting ----
    def _round_mixed(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        out = df.copy()
        for w in week_ids:
            if w not in out.columns:
                continue
            m = out["metric"].astype(str).str.lower()
            is_pct = m.str.contains("%") | m.str.contains("ratio") | m.str.contains("variance")
            out.loc[~is_pct, w] = pd.to_numeric(out.loc[~is_pct, w], errors="coerce").fillna(0.0).round(0).astype(int)
            out.loc[is_pct,  w] = pd.to_numeric(out.loc[is_pct,  w], errors="coerce").fillna(0.0).round(1)
        return out

    fw_to_use = _round_week_cols_int(fw_to_use, week_ids)
    hc        = _round_week_cols_int(hc, week_ids)
    att       = _round_mixed(att)
    trn       = _round_week_cols_int(trn, week_ids)
    rat       = _round_mixed(rat)
    seat      = _round_mixed(seat)
    bva       = _round_week_cols_int(bva, week_ids)
    nh        = _round_week_cols_int(nh, week_ids)

    def _format_shrinkage(df):
        if not isinstance(df, pd.DataFrame) or df.empty: return df
        out = df.copy()
        mser = out["metric"].astype(str)
        pct_rows = mser.str.contains("Shrinkage %", regex=False) | mser.str.contains("Variance vs Planned", regex=False)
        hr_rows  = out["metric"].astype(str).str.contains("Hours (#)",   regex=False)
        for w in week_ids:
            if w in out.columns: out[w] = out[w].astype(object)
        for w in week_ids:
            if w not in out.columns: continue
            out.loc[hr_rows,  w] = pd.to_numeric(out.loc[hr_rows,  w], errors="coerce").fillna(0).round(0).astype(int)
            vals = pd.to_numeric(out.loc[pct_rows, w], errors="coerce").fillna(0.0)
            out.loc[pct_rows, w] = vals.round(1).astype(str) + "%"
        return out
    
    def _format_attrition(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df
        out = df.copy()
        m = out["metric"].astype(str)
        # Percent rows: Planned Attrition %, Attrition %, Variance vs Planned
        pct_rows = m.str.lower().str.contains("attrition %") | m.str.lower().str.contains("variance vs planned")
        # Count rows: any HC (#)
        hc_rows = m.str.contains("HC (#)", regex=False)
        for w in week_ids:
            if w in out.columns:
                out[w] = out[w].astype(object)
        for w in week_ids:
            if w not in out.columns:
                continue
            out.loc[hc_rows, w] = pd.to_numeric(out.loc[hc_rows, w], errors="coerce").fillna(0).round(0).astype(int)
            vals = pd.to_numeric(out.loc[pct_rows, w], errors="coerce").fillna(0.0)
            out.loc[pct_rows, w] = vals.round(1).astype(str) + "%"
        return out

    shr_display = _format_shrinkage(shr)    
    att_display = _format_attrition(att)

    # Seat display: append % sign for Seat Utilization %, keep others numeric
    def _format_seat(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df
        out = df.copy()
        mser = out["metric"].astype(str)
        util_rows = mser.str.contains("Seat Utilization %", regex=False)
        for w in week_ids:
            if w in out.columns:
                out[w] = out[w].astype(object)
        for w in week_ids:
            if w not in out.columns:
                continue
            vals = pd.to_numeric(out.loc[util_rows, w], errors="coerce").fillna(0.0)
            out.loc[util_rows, w] = vals.round(1).astype(str) + "%"
        return out
    seat_display = _format_seat(seat)

    # format upper: SL one decimal, others int
    if isinstance(upper_df, pd.DataFrame) and not upper_df.empty:
        for w in week_ids:
            if w not in upper_df.columns: continue
            mask_sl = upper_df["metric"].eq("Projected Service Level")
            mask_not_sl = ~mask_sl
            upper_df.loc[mask_sl, w] = pd.to_numeric(upper_df.loc[mask_sl, w], errors="coerce").fillna(0.0).round(1)
            upper_df.loc[mask_not_sl, w] = pd.to_numeric(upper_df.loc[mask_not_sl, w], errors="coerce").fillna(0.0).round(0).astype(int)

    upper = dash_table.DataTable(
        id="tbl-upper",
        data=upper_df.to_dict("records"),
        columns=[{"name": "Metric", "id": "metric", "editable": False}] +
                [{"name": c["name"], "id": c["id"]} for c in fw_cols if c["id"] != "metric"],
        editable=False,
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={"whiteSpace": "pre"},
    )

    # ---- Roster/Bulk/Notes passthroughs ----
    bulk_df  = _load_or_empty_bulk_files(pid)
    notes_df = _load_or_empty_notes(pid)

    return (
        upper,
        fw_to_use.to_dict("records"),
        hc.to_dict("records"),
        att_display.to_dict("records"),
        shr_display.to_dict("records"),
        trn.to_dict("records"),
        rat.to_dict("records"),
        seat_display.to_dict("records"),
        bva.to_dict("records"),
        nh.to_dict("records"),
        (roster_df.to_dict("records") if isinstance(roster_df, pd.DataFrame) else []),
        bulk_df.to_dict("records")  if isinstance(bulk_df,  pd.DataFrame) else [],
        notes_df.to_dict("records") if isinstance(notes_df, pd.DataFrame) else [],
    )
