import ast
import calendar
import json
import math
import re
import dash
import numpy as np
import pandas as pd
import datetime as dt
from cap_db import load_df
from cap_store import load_roster_long, resolve_holidays, resolve_settings
from capacity_core import required_fte_daily
from plan_detail._calc import _load_roster_normalized, _nh_effective_count 
from plan_detail._common import _assemble_bo, _assemble_chat, _assemble_ob, _assemble_voice, _blank_grid, _canon_scope, _first_non_empty_ts, _learning_curve_for_week, _load_or_blank, _load_or_empty_bulk_files, _load_or_empty_notes, _load_ts_with_fallback, _parse_ratio_setting, get_plan_meta
from plan_store import get_plan
from dash import dash_table

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

    # Derive explicit buckets
    d["bucket"] = d.get("activity", "").map(_bo_bucket)

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

def _get_fw_value(fw_df, metric, col_id, default=0.0):
    """
    Safely get a single numeric value from the FW grid:
    returns float or `default` if missing/NaN.
    """
    try:
        if not isinstance(fw_df, pd.DataFrame) or fw_df.empty or col_id not in fw_df.columns:
            return default
        mask = fw_df["metric"].astype(str).str.strip().eq(metric)
        if not mask.any():
            return default
        ser = pd.to_numeric(fw_df.loc[mask, col_id], errors="coerce").dropna()
        return float(ser.iloc[0]) if not ser.empty else default
    except Exception:
        return default
    
def build_idx(*series_list, base_dates=None):
    """
    Build a unified datetime index from multiple grouped series,
    trimmed to the min/max of the actual uploaded dataset.
    
    Parameters
    ----------
    *series_list : pd.Series
        Any number of grouped series with datetime-like indices.
    base_dates : pd.Series or pd.Index, optional
        The authoritative date column from the uploaded dataset.
        If provided, min/max will be taken from here.
    
    Returns
    -------
    pd.DatetimeIndex
        Sorted index restricted to the dataset's true date range.
    """
    # Union of all indices
    idx = pd.to_datetime(
        pd.Index(set().union(*[s.index for s in series_list])),
        errors="coerce"
    ).dropna().sort_values()

    # Establish min/max bounds
    if base_dates is not None and len(base_dates) > 0:
        dmin, dmax = base_dates.min(), base_dates.max()
    else:
        # fallback: use min/max from the union itself
        dmin, dmax = idx.min(), idx.max()

    # Trim to dataset window
    return idx[(idx >= dmin) & (idx <= dmax)]

def clip_to_dataset_range(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Trim any rows outside the true min/max of the uploaded dataset."""
    dmin, dmax = df[date_col].min(), df[date_col].max()
    return df[(df[date_col] >= dmin) & (df[date_col] <= dmax)]

def _fill_tables_fixed_monthly(ptype, pid, fw_cols, _tick, whatif=None):
    # ---- guards ----
    if not (pid and fw_cols):
        raise dash.exceptions.PreventUpdate
    # calendar columns (YYYY-MM-01 month ids coming from UI)
    month_ids = [c["id"] for c in fw_cols if c.get("id") != "metric"]
    # ---- read persisted What-If ----
    wf_start = ""
    wf_end   = ""
    wf_ovr   = {}
    try:
        wf_df = load_df(f"plan_{pid}_whatif")
        if isinstance(wf_df, pd.DataFrame) and not wf_df.empty:
            last = wf_df.tail(1).iloc[0]
            wf_start = str(last.get("start_week") or "").strip()  # keep window semantics
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
    # per-week Nest/SDA dials (windowed) ? we apply to months by date window as well
    _nest_login_w = dict((whatif.get("nesting_login_pct") or {}))
    _nest_ahtm_w  = dict((whatif.get("nesting_aht_multiplier") or {}))
    # helper: use the persisted window as an ?active future? flag using month ids too

    def _wf_active_month(m):
        # Default: if no explicit window set, apply what-if only to future months
        today_m = pd.to_datetime(dt.date.today()).to_period("M").to_timestamp().date().isoformat()
        if not wf_start and not wf_end:
            return str(m) > today_m
        # If a custom window is provided (as weeks), keep permissive behavior for now
        return True

    # helpers for nest overrides (applied when month is in active window)
    def _ovr_login_frac_m(m):
        # allow monthly override via the same mapping if keys match, else None
        v = _nest_login_w.get(m)
        if v in (None, "") or not _wf_active_month(m): return None
        try:
            x = float(v);  x = x/100.0 if x > 1.0 else x
            return max(0.0, min(1.0, x))
        except Exception:
            return None

    def _ovr_aht_mult_m(m):
        v = _nest_ahtm_w.get(m)
        if v in (None, "") or not _wf_active_month(m): return None
        try:
            mlt = float(v)
            return max(0.1, mlt)
        except Exception:
            return None

    # ---- scope, plan, settings ----
    p = get_plan(pid) or {}
    # Prefer explicit Channel; fallback to LOB if missing, keep first of CSV
    ch_first = (p.get("channel") or p.get("lob") or "").split(",")[0].strip()
    sk = _canon_scope(
        p.get("vertical"),
        p.get("sub_ba"),
        ch_first,
        (p.get("site") or p.get("location") or p.get("country") or "").strip(),
    )
    loc_first = (p.get("location") or p.get("country") or p.get("site") or "").strip()
    settings = resolve_settings(ba=p.get("vertical"), subba=p.get("sub_ba"), lob=ch_first)
    holidays_df = resolve_holidays(
        ba=p.get("vertical"),
        subba=p.get("sub_ba"),
        lob=ch_first,
        site=(p.get("site") or p.get("location") or p.get("country")),
        location=loc_first
    )
    holiday_days_month = {}
    if isinstance(holidays_df, pd.DataFrame) and not holidays_df.empty and "date" in holidays_df.columns:
        h = holidays_df.copy()
        h["date"] = pd.to_datetime(h["date"], errors="coerce")
        h = h.dropna(subset=["date"])
        if not h.empty:
            h = h.drop_duplicates(subset=["date"])
            h["month"] = h["date"].dt.to_period("M").dt.to_timestamp().dt.normalize().dt.date.astype(str)
            holiday_days_month = h.groupby("month").size().to_dict()
        else:
            holiday_days_month = {}
    else:
        holiday_days_month = {}
    try:
        lc_ovr_df = load_df(f"plan_{pid}_lc_overrides")
    except Exception:
        lc_ovr_df = None

    def _lc_with_wf_m(lc_dict, m_id):
        out = dict(lc_dict or {})
        p_ = _ovr_login_frac_m(m_id)
        m_ = _ovr_aht_mult_m(m_id)
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
    planned_sut_df = _load_ts_with_fallback("bo_planned_sut",   sk)
    # Fallbacks: if channel-specific planned series missing, try budget tables like weekly view
    if (not isinstance(planned_aht_df, pd.DataFrame)) or planned_aht_df.empty:
        tmp = _load_ts_with_fallback("voice_budget", sk)
        if isinstance(tmp, pd.DataFrame) and not tmp.empty:
            x = tmp.copy()
            if "week" in x.columns or "date" in x.columns or "month" in x.columns:
                if "budget_aht_sec" in x.columns and "aht_sec" not in x.columns:
                    x = x.rename(columns={"budget_aht_sec": "aht_sec"})
                planned_aht_df = x[[c for c in x.columns if c in ("date","week","month","aht_sec")]]
    if (not isinstance(planned_sut_df, pd.DataFrame)) or planned_sut_df.empty:
        tmp = _load_ts_with_fallback("bo_budget", sk)
        if isinstance(tmp, pd.DataFrame) and not tmp.empty:
            x = tmp.copy()
            if "week" in x.columns or "date" in x.columns or "month" in x.columns:
                if "budget_sut_sec" in x.columns and "sut_sec" not in x.columns:
                    x = x.rename(columns={"budget_sut_sec": "sut_sec"})
                planned_sut_df = x[[c for c in x.columns if c in ("date","week","month","sut_sec")]]

    # util for month id
    def _mid(s):
        return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp().dt.normalize().dt.date.astype(str)

    def _normalize_month_id(label):
        if label in month_ids:
            return label
        try:
            dt = pd.to_datetime(label, errors="coerce")
            if pd.isna(dt):
                return None
            # If the label looks like a weekly column (YYYY-MM-DD Monday),
            # map it to the month that contains the majority of that ISO week
            # by shifting to mid-week (Thursday).
            try:
                lbl = str(label)
                if len(lbl) >= 10 and lbl[4] == '-' and lbl[7] == '-':
                    if pd.Timestamp(dt).weekday() == 0:  # Monday-based week labels in grid
                        dt = pd.Timestamp(dt) + pd.Timedelta(days=3)
            except Exception:
                pass
            return pd.Timestamp(dt).to_period("M").to_timestamp().date().isoformat()
        except Exception:
            return None

    def _collect_saved_metric(df, metric_name, reducer="sum"):
        if not isinstance(df, pd.DataFrame) or df.empty or "metric" not in df.columns:
            return {}
        metrics = df["metric"].astype(str).str.strip()
        if metric_name not in metrics.values:
            return {}
        row = df.loc[metrics == metric_name].iloc[0]
        values = {}
        counts = {}
        for col, raw in row.items():
            if col == "metric":
                continue
            mid = _normalize_month_id(col)
            if not mid or mid not in month_ids:
                continue
            val = pd.to_numeric(raw, errors="coerce")
            if pd.isna(val):
                continue
            if reducer == "sum":
                values[mid] = values.get(mid, 0.0) + float(val)
            elif reducer == "mean":
                values[mid] = values.get(mid, 0.0) + float(val)
                counts[mid] = counts.get(mid, 0) + 1
        if reducer == "mean":
            result = {}
            for m, total in values.items():
                count = counts.get(m, 0)
                if count:
                    result[m] = total / count
            return result
        return values

    # monthly dict from ts (accepts either week or date columns)
    def _ts_month_dict(df: pd.DataFrame, val_candidates: list[str]) -> dict:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}
        d = df.copy()
        if "week" in d.columns:
            d["month"] = _mid(d["week"])
        elif "date" in d.columns:
            d["month"] = _mid(d["date"])
        elif "month" in d.columns:
            d["month"] = _mid(d["month"])
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
        g = d.dropna(subset=["month", vcol]).groupby("month", as_index=True)[vcol].mean()
        return g.astype(float).to_dict()
    # Channel-aware planned AHT/SUT
    planned_aht_m_voice = _ts_month_dict(planned_aht_df, ["aht_sec","sut_sec","aht","avg_aht"])
    planned_aht_df_chat = _load_ts_with_fallback("chat_planned_aht", sk)
    planned_aht_df_ob   = _load_ts_with_fallback("ob_planned_aht",   sk)
    planned_aht_m_chat  = _ts_month_dict(planned_aht_df_chat, ["aht_sec","aht","avg_aht"]) if isinstance(planned_aht_df_chat, pd.DataFrame) else {}
    planned_aht_m_ob    = _ts_month_dict(planned_aht_df_ob,   ["aht_sec","aht","avg_aht"]) if isinstance(planned_aht_df_ob,   pd.DataFrame) else {}
    planned_aht_m = planned_aht_m_voice
    planned_sut_m = _ts_month_dict(planned_sut_df, ["sut_sec","aht_sec","sut","avg_sut"])
    sl_target_pct = None
    for k in ("sl_target_pct","service_level_target","sl_target","sla_target_pct","sla_target","target_sl"):
        v = settings.get(k)
        if v not in (None, ""):
            try:
                x = float(str(v).replace("%",""))
                sl_target_pct = x * 100.0 if x <= 1.0 else x
            except Exception:
                pass
            break
    if sl_target_pct is None:
        sl_target_pct = 80.0

    # ---- assemble time series ----
    vF = _assemble_voice(sk, "forecast"); vA = _assemble_voice(sk, "actual");  vT = _assemble_voice(sk, "tactical")
    bF = _assemble_bo(sk,   "forecast");  bA = _assemble_bo(sk,   "actual");   bT = _assemble_bo(sk,   "tactical")
    oF = _assemble_ob(sk,   "forecast");  oA = _assemble_ob(sk,   "actual");   oT = _assemble_ob(sk,   "tactical")
    cF = _assemble_chat(sk, "forecast");  cA = _assemble_chat(sk, "actual");   cT = _assemble_chat(sk, "tactical")
    use_voice_for_req = vA if isinstance(vA, pd.DataFrame) and not vA.empty else vF
    use_bo_for_req    = bA if isinstance(bA, pd.DataFrame) and not bA.empty else bF

    # ---- FW grid shell (same spec as weekly) ----
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
    # Apply per-plan FW/Upper preferences from plan meta
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
    for m in month_ids:
        fw[m] = 0.0

    # ---- monthly demand + AHT/SUT actual/forecast (voice+bo) ----
    def _norm_voice(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(columns=["date","interval","volume","aht_sec","interval_start"])
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
        if "interval" in d.columns:
            d["interval"] = d["interval"].astype(str)
        if "interval_start" in d.columns:
            d["interval_start"] = d["interval_start"].astype(str)
        d["volume"]  = pd.to_numeric(d.get("volume"),  errors="coerce").fillna(0.0)
        if "aht_sec" not in d.columns and "aht" in d.columns:
            d["aht_sec"] = pd.to_numeric(d["aht"], errors="coerce")
        d["aht_sec"] = pd.to_numeric(d.get("aht_sec"), errors="coerce").fillna(0.0)
        return d.dropna(subset=["date"])

    def _norm_bo(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(columns=["date","items","aht_sec"])
        d = df.copy()
        d["date"]   = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
        d["items"]  = pd.to_numeric(d.get("items", d.get("volume")), errors="coerce").fillna(0.0)
        d["aht_sec"]= pd.to_numeric(d.get("aht_sec", d.get("sut_sec", d.get("sut"))), errors="coerce").fillna(0.0)
        return d.dropna(subset=["date"])

    vF = _norm_voice(vF); vA = _norm_voice(vA); vT = _norm_voice(vT)
    bF = _norm_bo(bF);   bA = _norm_bo(bA);   bT = _norm_bo(bT)

    # Normalize Outbound and Chat for monthly aggregation
    def _norm_ob(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(columns=["date","opc","connect_rate","rpc","rpc_rate","aht_sec"]) 
        d = df.copy()
        d["date"] = pd.to_datetime(d.get("date") if "date" in d.columns else d.get("week"), errors="coerce").dt.normalize()
        # Tolerant picks
        cols = {c.lower(): c for c in d.columns}
        c_opc = cols.get("opc") or cols.get("dials") or cols.get("calls") or cols.get("attempts")
        c_conn = cols.get("connect_rate") or cols.get("connect%") or cols.get("connect pct") or cols.get("connect")
        c_rpc  = cols.get("rpc")
        c_rpcr = cols.get("rpc_rate") or cols.get("rpc%") or cols.get("rpc pct")
        c_aht  = cols.get("aht_sec") or cols.get("talk_sec") or cols.get("avg_talk_sec") or cols.get("aht")
        if c_opc: d[c_opc] = pd.to_numeric(d[c_opc], errors="coerce").fillna(0.0)
        if c_rpc: d[c_rpc] = pd.to_numeric(d[c_rpc], errors="coerce").fillna(0.0)
        if c_conn: d[c_conn] = pd.to_numeric(d[c_conn], errors="coerce").fillna(0.0)
        if c_rpcr: d[c_rpcr] = pd.to_numeric(d[c_rpcr], errors="coerce").fillna(0.0)
        if c_aht: d[c_aht] = pd.to_numeric(d[c_aht], errors="coerce").fillna(float(s_target_aht))
        else:
            d["aht_sec"] = float(s_target_aht)
        return d.dropna(subset=["date"]) 

    def _norm_chat(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(columns=["date","items","aht_sec"]) 
        d = df.copy()
        d["date"] = pd.to_datetime(d.get("date") if "date" in d.columns else d.get("week"), errors="coerce").dt.normalize()
        cols = {c.lower(): c for c in d.columns}
        c_items = cols.get("items") or cols.get("chats") or cols.get("volume") or cols.get("txns") or cols.get("transactions")
        c_aht   = cols.get("aht_sec") or cols.get("sut_sec") or cols.get("aht")
        if c_items: d[c_items] = pd.to_numeric(d[c_items], errors="coerce").fillna(0.0)
        else: d["items"] = 0.0
        if c_aht: d[c_aht] = pd.to_numeric(d[c_aht], errors="coerce").fillna(float(s_target_aht))
        else: d["aht_sec"] = float(s_target_aht)
        return d.dropna(subset=["date"]) 

    oF = _norm_ob(oF); oA = _norm_ob(oA); oT = _norm_ob(oT)
    cF = _norm_chat(cF); cA = _norm_chat(cA); cT = _norm_chat(cT)

    # monthly weighted figures
    def _voice_monthly(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}, {}
        d = df.copy()
        d["_m"] = _mid(d["date"])
        d["_num"] = d["volume"] * d["aht_sec"]
        vol = d.groupby("_m", as_index=True)["volume"].sum()
        num = d.groupby("_m", as_index=True)["_num"].sum()
        aht = (num / vol.replace({0: np.nan})).fillna(np.nan)
        return vol.to_dict(), aht.to_dict()

    def _bo_monthly(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}, {}
        d = df.copy()
        d["_m"] = _mid(d["date"])
        d["_num"] = d["items"] * d["aht_sec"]
        itm = d.groupby("_m", as_index=True)["items"].sum()
        num = d.groupby("_m", as_index=True)["_num"].sum()
        sut = (num / itm.replace({0: np.nan})).fillna(np.nan)
        return itm.to_dict(), sut.to_dict()
    vF_vol, vF_aht = _voice_monthly(vF); vA_vol, vA_aht = _voice_monthly(vA); vT_vol, vT_aht = _voice_monthly(vT)
    bF_itm, bF_sut = _bo_monthly(bF);   bA_itm, bA_sut = _bo_monthly(bA);   bT_itm, bT_sut = _bo_monthly(bT)

    # Outbound monthly expected contacts and AHT
    def _to_frac(x):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return 0.0
            v = float(x)
            if v > 1.0:
                v = v / 100.0
            return max(0.0, min(1.0, v))
        except Exception:
            try:
                s = str(x).strip().rstrip('%')
                v = float(s)
                if v > 1.0:
                    v = v / 100.0
                return max(0.0, min(1.0, v))
            except Exception:
                return 0.0

    def _ob_monthly(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}, {}
        d = df.copy()
        d["_m"] = _mid(d["date"])
        cols = {c.lower(): c for c in d.columns}
        c_opc = cols.get("opc") or cols.get("dials") or cols.get("calls") or cols.get("attempts")
        c_conn = cols.get("connect_rate") or cols.get("connect%") or cols.get("connect pct") or cols.get("connect")
        c_rpc  = cols.get("rpc")
        c_rpcr = cols.get("rpc_rate") or cols.get("rpc%") or cols.get("rpc pct")
        c_aht  = cols.get("aht_sec") or cols.get("talk_sec") or cols.get("avg_talk_sec") or cols.get("aht")
        opc = pd.to_numeric(d.get(c_opc), errors='coerce').fillna(0.0) if c_opc else 0.0
        rpc = pd.to_numeric(d.get(c_rpc), errors='coerce').fillna(0.0) if c_rpc else 0.0
        conn = d.get(c_conn).map(_to_frac) if c_conn else 0.0
        rpcr = d.get(c_rpcr).map(_to_frac) if c_rpcr else 0.0
        expected = rpc if c_rpc else (opc * (conn if c_conn else 1.0) * (rpcr if c_rpcr else 1.0))
        d["_exp"] = pd.to_numeric(expected, errors='coerce').fillna(0.0)
        aht = pd.to_numeric(d.get(c_aht), errors='coerce').fillna(float(s_target_aht)) if c_aht else float(s_target_aht)
        d["_num"] = d["_exp"] * aht
        vol = d.groupby("_m", as_index=True)["_exp"].sum()
        num = d.groupby("_m", as_index=True)["_num"].sum()
        ahtm = (num / vol.replace({0: np.nan})).fillna(np.nan)
        return vol.to_dict(), ahtm.to_dict()

    oF_vol, oF_aht = _ob_monthly(oF); oA_vol, oA_aht = _ob_monthly(oA); oT_vol, oT_aht = _ob_monthly(oT)

    # Chat monthly items and AHT
    def _chat_monthly(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}, {}
        d = df.copy()
        d["_m"] = _mid(d["date"])
        items = pd.to_numeric(d.get("items"), errors='coerce').fillna(0.0)
        aht   = pd.to_numeric(d.get("aht_sec"), errors='coerce').fillna(float(s_target_aht))
        d["_num"] = items * aht
        vol = d.groupby("_m", as_index=True)["items"].sum()
        num = d.groupby("_m", as_index=True)["_num"].sum()
        ahtm = (num / vol.replace({0: np.nan})).fillna(np.nan)
        return vol.to_dict(), ahtm.to_dict()

    cF_vol, cF_aht = _chat_monthly(cF); cA_vol, cA_aht = _chat_monthly(cA); cT_vol, cT_aht = _chat_monthly(cT)

    # Write rows in FW table
    for m in month_ids:
        # Choose channel aggregation based on plan channel
        chk = str(ch_first).strip().lower()
        if chk in ("voice",):
            # Voice plans should reflect Voice only
            f_total = float(vF_vol.get(m, 0.0))
            a_total = float(vA_vol.get(m, 0.0))
            t_total = float(vT_vol.get(m, 0.0))
            aht_actual_numden = [
                (vA_aht.get(m), vA_vol.get(m)),
            ]
            aht_fore_numden = [
                (planned_aht_m_voice.get(m, vF_aht.get(m, s_target_aht)), vF_vol.get(m, 0.0)),
            ]
        elif chk in ("back office", "bo"):
            f_total = float(bF_itm.get(m, 0.0))
            a_total = float(bA_itm.get(m, 0.0))
            t_total = float(bT_itm.get(m, 0.0))
            aht_actual_numden = [(bA_sut.get(m), bA_itm.get(m))]
            # Forecast AHT/SUT should come from forecast SUT when present (fallback to target)
            aht_fore_numden   = [(bF_sut.get(m, s_target_sut), bF_itm.get(m, 0.0))]
        elif chk in ("outbound", "ob", "out bound"):
            f_total = float(oF_vol.get(m, 0.0))
            a_total = float(oA_vol.get(m, 0.0))
            t_total = float(oT_vol.get(m, 0.0))
            aht_actual_numden = [(oA_aht.get(m), oA_vol.get(m))]
            # Forecast from outbound forecast AHT (fallback to target)
            aht_fore_numden   = [(oF_aht.get(m, s_target_aht), oF_vol.get(m, 0.0))]
        elif chk in ("chat",):
            f_total = float(cF_vol.get(m, 0.0))
            a_total = float(cA_vol.get(m, 0.0))
            t_total = float(cT_vol.get(m, 0.0))
            aht_actual_numden = [(cA_aht.get(m), cA_vol.get(m))]
            # Forecast from chat forecast AHT (fallback to target)
            aht_fore_numden   = [(cF_aht.get(m, s_target_aht), cF_vol.get(m, 0.0))]
        else:
            # default to voice + bo
            f_total = float(vF_vol.get(m, 0.0)) + float(bF_itm.get(m, 0.0))
            a_total = float(vA_vol.get(m, 0.0)) + float(bA_itm.get(m, 0.0))
            t_total = float(vT_vol.get(m, 0.0)) + float(bT_itm.get(m, 0.0))
            aht_actual_numden = [
                (vA_aht.get(m), vA_vol.get(m)),
                (bA_sut.get(m), bA_itm.get(m)),
            ]
            aht_fore_numden = [
                (vF_aht.get(m, s_target_aht), vF_vol.get(m, 0.0)),
                (bF_sut.get(m, s_target_sut), bF_itm.get(m, 0.0)),
            ]

        if _wf_active_month(m) and vol_delta:
            f_total *= (1.0 + vol_delta / 100.0)

        if "Forecast" in fw_rows:          fw.loc[fw["metric"] == "Forecast",          m] = f_total
        if "Tactical Forecast" in fw_rows: fw.loc[fw["metric"] == "Tactical Forecast", m] = t_total
        if "Actual Volume" in fw_rows:     fw.loc[fw["metric"] == "Actual Volume",     m] = a_total

        # Actual AHT/SUT (weighted across selected channels)
        a_num = 0.0; a_den = 0.0
        for aht_v, vol_v in aht_actual_numden:
            try:
                vv = float(vol_v or 0.0)
                if vv > 0:
                    aa = float(aht_v if aht_v not in (None, np.nan) else 0.0)
                    if aa <= 0: continue
                    a_num += aa * vv; a_den += vv
            except Exception:
                pass
        actual_aht_sut = (a_num / a_den) if a_den > 0 else 0.0
        actual_aht_sut = (actual_aht_sut if actual_aht_sut > 0 else s_target_aht)
        if "Actual AHT/SUT" in fw_rows:
            fw.loc[fw["metric"] == "Actual AHT/SUT", m] = actual_aht_sut

        # Forecast AHT/SUT (planned overrides aware)
        f_num = 0.0; f_den = 0.0
        for aht_v, vol_v in aht_fore_numden:
            try:
                vv = float(vol_v or 0.0)
                if vv > 0:
                    aa = float(aht_v if aht_v not in (None, np.nan) else 0.0)
                    if aa <= 0: continue
                    f_num += aa * vv; f_den += vv
            except Exception:
                pass
        forecast_aht_sut = (f_num / f_den) if f_den > 0 else s_target_aht
        # What-If: reflect AHT delta in FW row so user sees impact
        try:
            if _wf_active_month(m) and aht_delta:
                forecast_aht_sut = max(1.0, float(forecast_aht_sut) * (1.0 + aht_delta / 100.0))
        except Exception:
            pass
        if "Forecast AHT/SUT" in fw_rows:
            fw.loc[fw["metric"] == "Forecast AHT/SUT", m] = forecast_aht_sut

        # Budgeted AHT/SUT (use planned AHT/SUT where provided, else fallback to budget defaults)
        if "Budgeted AHT/SUT" in fw_rows:
            b_num = b_den = 0.0
            if chk in ("voice",):
                vv = float(vF_vol.get(m, 0.0))
                if vv > 0:
                    aa = float(planned_aht_m_voice.get(m, s_budget_aht))
                    b_num += aa * vv; b_den += vv
            elif chk in ("back office", "bo"):
                vv = float(bF_itm.get(m, 0.0))
                if vv > 0:
                    aa = float(planned_sut_m.get(m, s_budget_sut))
                    b_num += aa * vv; b_den += vv
            elif chk in ("outbound", "ob", "out bound"):
                vv = float(oF_vol.get(m, 0.0))
                if vv > 0:
                    aa = float(planned_aht_m_ob.get(m, s_budget_aht))
                    b_num += aa * vv; b_den += vv
            elif chk in ("chat",):
                vv = float(cF_vol.get(m, 0.0))
                if vv > 0:
                    aa = float(planned_aht_m_chat.get(m, s_budget_aht))
                    b_num += aa * vv; b_den += vv
            else:
                # default combined (Voice + BO)
                vv = float(vF_vol.get(m, 0.0))
                if vv > 0:
                    aa = float(planned_aht_m_voice.get(m, s_budget_aht))
                    b_num += aa * vv; b_den += vv
                vb = float(bF_itm.get(m, 0.0))
                if vb > 0:
                    aa = float(planned_sut_m.get(m, s_budget_sut))
                    b_num += aa * vb; b_den += vb
            budget_aht_sut = (b_num / b_den) if b_den > 0 else float(s_budget_aht)
            fw.loc[fw["metric"] == "Budgeted AHT/SUT", m] = budget_aht_sut

    # Compute Backlog/Queue based on current FW values (honor selection)
    backlog_m_local = {}
    if "Backlog (Items)" in fw_rows:
        for m in month_ids:
            try:
                fval = float(pd.to_numeric(fw.loc[fw["metric"] == "Forecast", m], errors="coerce").fillna(0.0).iloc[0]) if m in fw.columns else 0.0
            except Exception:
                fval = 0.0
            try:
                aval = float(pd.to_numeric(fw.loc[fw["metric"] == "Actual Volume", m], errors="coerce").fillna(0.0).iloc[0]) if m in fw.columns else 0.0
            except Exception:
                aval = 0.0
            bl = max(0.0, aval - fval)
            backlog_m_local[m] = bl
            fw.loc[fw["metric"] == "Backlog (Items)", m] = bl
    queue_m = {}
    if "Queue (Items)" in fw_rows:
        for i, m in enumerate(month_ids):
            prev_bl = float(backlog_m_local.get(month_ids[i-1], 0.0)) if i > 0 else 0.0
            try:
                fval = float(pd.to_numeric(fw.loc[fw["metric"] == "Forecast", m], errors="coerce").fillna(0.0).iloc[0]) if m in fw.columns else 0.0
            except Exception:
                fval = 0.0
            qv = max(0.0, prev_bl + fval)
            queue_m[m] = qv
            fw.loc[fw["metric"] == "Queue (Items)", m] = qv

    # Save current FW to support Backlog/Overtime/Shrinkage readback
    fw_saved = load_df(f"plan_{pid}_fw")

    def _row_to_month_dict(df: pd.DataFrame, metric_name: str) -> dict:
        try:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return {}
            d = df.copy()
            # harmonize column ids to month ids
            for c in list(d.columns):
                if c == "metric": continue
                try:
                    mid = _mid(pd.Series([c]))[0]
                    if mid != c:
                        d.rename(columns={c: mid}, inplace=True)
                except Exception:
                    pass
            m = d["metric"].astype(str).str.strip()
            if metric_name not in m.values:
                return {}
            row = d.loc[m == metric_name].iloc[0]
            out = {}
            for mid in month_ids:
                try:
                    out[mid] = float(pd.to_numeric(row.get(mid), errors="coerce"))
                except Exception:
                    out[mid] = 0.0
            return out
        except Exception:
            return {}
    overtime_m = _row_to_month_dict(fw_saved, "Overtime Hours (#)")
    # Collector for monthly overtime hours derived from raw shrinkage feeds
    # Initialized to avoid NameError where later aggregation appends to it
    overtime_hours_m = {}

    # Independent: compute Back Office Overtime Hours from shrinkage_raw_backoffice (do not touch shrinkage logic)
    def _compute_monthly_ot_from_raw() -> dict:
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
        # Normalize and filter overtime activity
        df[c_date] = pd.to_datetime(df[c_date], errors="coerce")
        df = df.dropna(subset=[c_date])
        act = df[c_act].astype(str).str.strip().str.lower()
        m_ot = act.str.contains(r"\bover\s*time\b|\bovertime\b|\bot\b|\bot\s*hours\b|\bot\s*hrs\b", regex=True, na=False) | act.eq("overtime")
        df = df.loc[m_ot]
        if df.empty:
            return {}
        if c_sec:
            hours = pd.to_numeric(df[c_sec], errors="coerce").fillna(0.0) / 3600.0
        elif c_hr:
            hours = pd.to_numeric(df[c_hr], errors="coerce").fillna(0.0)
        else:
            return {}
        tmp = pd.DataFrame({"date": df[c_date], "hours": hours})
        tmp["month"] = _mid(tmp["date"]) 
        agg = tmp.groupby("month", as_index=False)["hours"].sum()
        return {str(r["month"]): float(r["hours"]) for _, r in agg.iterrows()}

    _ot_m_raw = _compute_monthly_ot_from_raw()
    backlog_m  = _row_to_month_dict(fw_saved, "Backlog (Items)")
    if backlog_m_local and ("backlog" in lower_opts):
        backlog_m = backlog_m_local
 
    # Voice overtime from SC_OVERTIME_DELIVERED
    def _compute_monthly_ot_voice_from_raw() -> dict:
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
        hours = pd.to_numeric(df[c_hours], errors="coerce").fillna(0.0)
        tmp = pd.DataFrame({"date": df[c_date], "hours": hours})
        tmp["month"] = _mid(tmp["date"]) 
        agg = tmp.groupby("month", as_index=False)["hours"].sum()
        return {str(r["month"]): float(r["hours"]) for _, r in agg.iterrows()}
    _ot_voice_m_raw = _compute_monthly_ot_voice_from_raw()
                

    # ---- Apply Backlog carryover (Back Office only): add previous month's backlog to next month's BO forecast ----
    if backlog_carryover and str(ch_first).strip().lower() in ("back office", "bo") and backlog_m and ("backlog" in lower_opts):
        for i in range(len(month_ids) - 1):
            cur_m = month_ids[i]; nxt_m = month_ids[i+1]
            add = float(backlog_m.get(cur_m, 0.0) or 0.0)
            if add:
                fw.loc[fw["metric"] == "Forecast", nxt_m] = float(fw.loc[fw["metric"] == "Forecast", nxt_m]) + add

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
        occ_base_raw = (settings.get("occupancy") or settings.get("occupancy_pct") or settings.get("target_occupancy") or
                        settings.get("budgeted_occupancy") or settings.get("occ") or settings.get("occupancy_cap_voice") or 0.85)
    try:
        if isinstance(occ_base_raw, str) and occ_base_raw.strip().endswith("%"):
            occ_base = float(occ_base_raw.strip()[:-1])
        else:
            occ_base = float(occ_base_raw)
            if occ_base <= 1.0:
                occ_base *= 100.0
    except Exception:
        occ_base = 85.0
    occ_m = {m: int(round(occ_base)) for m in month_ids}
    if occ_override not in (None, ""):
        try:
            ov = float(occ_override); ov = ov*100.0 if ov <= 1.0 else ov
            ov = int(round(ov))
            for m in month_ids:
                if _wf_active_month(m):
                    occ_m[m] = ov
        except Exception:
            pass
    if "Occupancy" in fw_rows:
        for m in month_ids:
            fw.loc[fw["metric"] == "Occupancy", m] = occ_m[m]
    occ_frac_m = {m: min(0.99, max(0.01, float(occ_m[m]) / 100.0)) for m in month_ids}
    # ---- requirements: Interval/Daily ? Monthly (true monthly, no weekly roll-up) ----
    req_daily_actual   = required_fte_daily(use_voice_for_req, use_bo_for_req, oA, settings)
    req_daily_forecast = required_fte_daily(vF, bF, oF, settings)
    req_daily_tactical = required_fte_daily(vT, bT, oT, settings) if (isinstance(vT, pd.DataFrame) and not vT.empty) or (isinstance(bT, pd.DataFrame) and not bT.empty) or (isinstance(oT, pd.DataFrame) and not oT.empty) else pd.DataFrame()
    # add Chat to daily totals
    from capacity_core import chat_fte_daily as _chat_fd
    for _df, chat_df in ((req_daily_actual, cA), (req_daily_forecast, cF), (req_daily_tactical, cT)):
        if isinstance(_df, pd.DataFrame) and not _df.empty and isinstance(chat_df, pd.DataFrame) and not chat_df.empty:
            try:
                ch = _chat_fd(chat_df, settings)
                m = _df.merge(ch, on=["date","program"], how="left")
                m["chat_fte"] = pd.to_numeric(m.get("chat_fte"), errors="coerce").fillna(0.0)
                m["total_req_fte"] = pd.to_numeric(m.get("total_req_fte"), errors="coerce").fillna(0.0) + m["chat_fte"]
                _df.drop(_df.index, inplace=True)
                _df[list(m.columns)] = m
            except Exception:
                pass

    # For budgeted, inject budgeted AHT/SUT (month plans) into copies then run daily engine
    vB = vF.copy(); bB = bF.copy(); oB = oF.copy(); cB = cF.copy()
    if isinstance(vB, pd.DataFrame) and not vB.empty:
        vB["month_id"] = _mid(vB["date"])
        vB["aht_sec"]  = vB["month_id"].map(planned_aht_m).fillna(float(s_budget_aht))
        vB.drop(columns=["month_id"], inplace=True)
    if isinstance(bB, pd.DataFrame) and not bB.empty:
        bB["month_id"] = _mid(bB["date"])
        bB["aht_sec"]  = bB["month_id"].map(planned_sut_m).fillna(float(s_budget_sut))
        bB.drop(columns=["month_id"], inplace=True)
    if isinstance(oB, pd.DataFrame) and not oB.empty:
        oB["month_id"] = _mid(oB["date"])
        oB["aht_sec"]  = oB["month_id"].map(lambda m: planned_aht_m.get(m, None)).fillna(float(s_budget_aht))
        oB.drop(columns=["month_id"], inplace=True)
    req_daily_budgeted = required_fte_daily(vB, bB, oB, settings)
    if isinstance(req_daily_budgeted, pd.DataFrame) and not req_daily_budgeted.empty and isinstance(cB, pd.DataFrame) and not cB.empty:
        try:
            chb = _chat_fd(cB, settings)
            m = req_daily_budgeted.merge(chb, on=["date","program"], how="left")
            m["chat_fte"] = pd.to_numeric(m.get("chat_fte"), errors="coerce").fillna(0.0)
            m["total_req_fte"] = pd.to_numeric(m.get("total_req_fte"), errors="coerce").fillna(0.0) + m["chat_fte"]
            req_daily_budgeted.drop(req_daily_budgeted.index, inplace=True)
            req_daily_budgeted[list(m.columns)] = m
        except Exception:
            pass

    def _workdays_in_month(mid: str, is_bo: bool) -> float:
        t = pd.to_datetime(mid, errors="coerce")
        if pd.isna(t):
            base = 22.0 if is_bo else 30.0
        else:
            y, m = int(t.year), int(t.month)
            days = calendar.monthrange(y, m)[1]
            if not is_bo:
                base = float(days)
            else:
                base = float(sum(1 for d in range(1, days + 1) if pd.Timestamp(year=y, month=m, day=d).weekday() < 5))
        holidays = float(holiday_days_month.get(mid, 0) or 0.0)
        if holidays > base:
            holidays = base
        return max(1.0, base - holidays)
    
    def _daily_to_monthly(df, is_bo: bool):
        if not isinstance(df, pd.DataFrame) or df.empty or "date" not in df.columns or "total_req_fte" not in df.columns:
            return {}
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"])
        d["month"] = _mid(d["date"])
        g = d.groupby("month", as_index=False)["total_req_fte"].sum()
        req_m = {}
        for _, r in g.iterrows():
            mid = str(r["month"])
            wd = _workdays_in_month(mid, is_bo=is_bo)
            req_m[mid] = float(r["total_req_fte"]) / max(1, wd)  # average daily FTE for month
        return req_m
    is_bo_ch = str(ch_first).strip().lower() in ("back office", "bo")
    # Adjust BO daily FTE by actual vs planned shrink before rolling up
    def _adjust_bo_fte_daily(df: pd.DataFrame, use_actual: bool) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df
        if "bo_fte" not in df.columns or "date" not in df.columns:
            return df
        s_plan = planned_shrink_fraction
        try:
            s_plan = float(s_plan); s_plan = s_plan/100.0 if s_plan > 1.0 else s_plan
        except Exception:
            s_plan = 0.0
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date.astype(str)
        old_bo = pd.to_numeric(d["bo_fte"], errors="coerce").fillna(0.0)
        s_target = d["date"].map(lambda x: bo_shrink_frac_daily_m.get(x, s_plan) if use_actual else s_plan)
        try:
            s_target = pd.to_numeric(s_target, errors="coerce").fillna(s_plan)
        except Exception:
            s_target = pd.Series([s_plan]*len(d))
        denom_old = max(0.01, 1.0 - float(s_plan))
        denom_new = np.maximum(0.01, 1.0 - s_target.astype(float))
        factor = denom_old / denom_new
        new_bo = old_bo * factor
        d["bo_fte"] = new_bo
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

    req_m_actual   = _daily_to_monthly(req_daily_actual,   is_bo=is_bo_ch)
    req_m_forecast = _daily_to_monthly(req_daily_forecast, is_bo=is_bo_ch)
    req_m_tactical = _daily_to_monthly(req_daily_tactical, is_bo=is_bo_ch)
    req_m_budgeted = _daily_to_monthly(req_daily_budgeted, is_bo=is_bo_ch)
    # What-If: adjust forecast requirements by volume, AHT and shrink deltas
    # Apply to future months only when no explicit window is set
    if vol_delta or shrink_delta or aht_delta:
        for mid in list(req_m_forecast.keys()):
            if not _wf_active_month(mid):
                continue
            v = float(req_m_forecast[mid])
            if vol_delta:
                v *= (1.0 + vol_delta / 100.0)
            if aht_delta:
                # Approximate: FTE requirement scales ~linearly with AHT/SUT
                v *= (1.0 + aht_delta / 100.0)
            if shrink_delta:
                denom = max(0.1, 1.0 - (shrink_delta / 100.0))
                v /= denom
            req_m_forecast[mid] = v
    # ---- Interval supply from global roster_long (monthly avg per interval) ----
    ivl_min = int(float(settings.get("interval_minutes", 30)) or 30)
    ivl_sec = 60 * ivl_min
    # Estimate per-month interval coverage. Prefer actual interval coverage from voice
    # forecast/actual (counts of interval rows), fall back to 24x7 if unavailable.
    monthly_voice_intervals = {m: 0 for m in month_ids}
    def _cov_counts(df: pd.DataFrame) -> dict:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}
        cols = set(df.columns)
        if "date" not in cols:
            return {}
        ivl_col = None
        for c in ("interval_start", "interval", "time"):
            if c in cols:
                ivl_col = c
                break
        if ivl_col is None:
            return {}
        t = df.copy()
        t["date"] = pd.to_datetime(t["date"], errors="coerce").dt.normalize()
        t = t.dropna(subset=["date"])  
        if t.empty:
            return {}
        t["month"] = t["date"].dt.to_period("M").dt.to_timestamp().dt.normalize().dt.date.astype(str)
        g = t.groupby("month", as_index=False)[ivl_col].count()
        return dict(zip(g["month"], g[ivl_col]))
    try:
        cov_f = _cov_counts(vF)
        cov_a = _cov_counts(vA)
        # Prefer forecast coverage when present, otherwise fall back to actual
        cov_map = cov_f if (isinstance(cov_f, dict) and sum(cov_f.values()) > 0) else cov_a
        for m in month_ids:
            monthly_voice_intervals[m] = int((cov_map or {}).get(m, 0))
    except Exception:
        pass
    # Fallback to 24x7 effective-day coverage when no interval rows found
    for m in month_ids:
        if int(monthly_voice_intervals.get(m, 0) or 0) > 0:
            continue
        try:
            y = int(m[:4]); mo = int(m[5:7])
            days = calendar.monthrange(y, mo)[1]
        except Exception:
            days = 30
        holidays = float(holiday_days_month.get(m, 0) or 0.0)
        if holidays > days:
            holidays = float(days)
        effective_days = max(1.0, float(days) - holidays)
        monthly_voice_intervals[m] = int(effective_days * (24 * 3600 // ivl_sec))
    schedule_supply_avg_m = {}
    try:
        rl = load_roster_long()
    except Exception:
        rl = None
    if isinstance(rl, pd.DataFrame) and not rl.empty:
        df = rl.copy()
        def _col(d, opts):
            for c in opts:
                if c in d.columns:
                    return c
            return None

        c_ba  = _col(df, ["Business Area","business area","vertical"])
        c_sba = _col(df, ["Sub Business Area","sub business area","sub_ba"])
        c_lob = _col(df, ["LOB","lob","Channel","channel"])
        c_site= _col(df, ["Site","site","Location","location","Country","country"])
        BA  = p.get("vertical"); SBA = p.get("sub_ba"); LOB = ch_first
        SITE= p.get("site") or p.get("location") or p.get("country")

        def _match(series, val):
            if not val or not isinstance(series, pd.Series):
                return pd.Series([True]*len(series))
            s = series.astype(str).str.strip().str.lower()
            return s.eq(str(val).strip().lower())
        m = pd.Series([True]*len(df))
        if c_ba:  m &= _match(df[c_ba], BA)
        if c_sba and (SBA not in (None, "")): m &= _match(df[c_sba], SBA)
        if c_lob: m &= _match(df[c_lob], LOB)
        if c_site and (SITE not in (None, "")): m &= _match(df[c_site], SITE)
        df = df[m]
        if "is_leave" in df.columns:
            df = df[~df["is_leave"].astype(bool)]
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
        import re as _re
        def _shift_len_ivl(s: str) -> int:
            try:
                s = str(s or "").strip()
                mm = _re.match(r"^(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})$", s)
                if not mm:
                    return 0
                sh, sm, eh, em = map(int, mm.groups())
                sh = min(23, max(0, sh)); eh = min(24, max(0, eh))
                start = sh*60 + sm; end = eh*60 + em
                if end < start:
                    end += 24*60
                return max(0, int((end - start + (ivl_min-1)) // ivl_min))
            except Exception:
                return 0
        if "entry" in df.columns and "date" in df.columns:
            df["ivl_count"] = df["entry"].apply(_shift_len_ivl)
            df["month"] = _mid(df["date"])
            agg = df.groupby("month", as_index=False)["ivl_count"].sum()
            monthly_agent_ivls = dict(zip(agg["month"], agg["ivl_count"]))
            for m_id in month_ids:
                denom = monthly_voice_intervals.get(m_id, 0)
                if denom <= 0:
                    denom = monthly_voice_intervals.get(m_id, 1)
                schedule_supply_avg_m[m_id] = float(monthly_agent_ivls.get(m_id, 0.0)) / float(denom or 1)

    # ---- lower shells (same metrics) ----
    hc   = _load_or_blank(f"plan_{pid}_hc",   ["Budgeted HC (#)","Planned/Tactical HC (#)","Actual Agent HC (#)","SME Billable HC (#)"], month_ids)
    att  = _load_or_blank(
        f"plan_{pid}_attr",
        [
            "Planned Attrition HC (#)",
            "Actual Attrition HC (#)",
            "Planned Attrition %",
            "Actual Attrition %",
            "Variance vs Planned",
        ],
        month_ids,
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
        month_ids,
    )
    trn  = _load_or_blank(f"plan_{pid}_train",["Training Start (#)","Training End (#)","Nesting Start (#)","Nesting End (#)"], month_ids)
    rat  = _load_or_blank(f"plan_{pid}_ratio",["Planned TL/Agent Ratio","Actual TL/Agent Ratio","Variance"], month_ids)
    seat = _load_or_blank(
        f"plan_{pid}_seat",
        [
            "Seats Required (#)",
            "Seats Available (#)",
            "Seat Utilization %",
            "Variance vs Available",
        ],
        month_ids,
    )
    # Ensure month columns are float-friendly before writing percentage/utilization values
    try:
        for _c in month_ids:
            if _c in seat.columns and not pd.api.types.is_float_dtype(seat[_c].dtype):
                seat[_c] = pd.to_numeric(seat[_c], errors="coerce").astype("float64")
    except Exception:
        pass
    bva  = _load_or_blank(f"plan_{pid}_bva",  ["Budgeted FTE (#)","Actual FTE (#)","Variance (#)"], month_ids)
    nh   = _load_or_blank(f"plan_{pid}_nh",   ["Planned New Hire HC (#)","Actual New Hire HC (#)","Recruitment Achievement"], month_ids)
    # -- New Hire overlay (classes + roster) --
    today_m = pd.to_datetime(dt.date.today()).to_period("M").to_timestamp().date().isoformat()
    # Planned NH from classes: count production_start in month (throughput via LC)
    def _monthly_planned_nh_from_classes(pid, mids):
        classes_df = load_df(f"plan_{pid}_nh_classes")
        out = {m: 0 for m in mids}
        if not isinstance(classes_df, pd.DataFrame) or classes_df.empty:
            # ALWAYS return a tuple
            return out, pd.DataFrame()
        c = classes_df.copy()
        # production_start column normalization
        for cand in ("production_start", "prod_start", "to_production", "go_live"):
            if cand in c.columns:
                c["production_start"] = pd.to_datetime(c[cand], errors="coerce")
                break
        if "production_start" not in c.columns:
            return out, c  # still return tuple
        c = c.dropna(subset=["production_start"]).copy()
        c["month"] = pd.to_datetime(c["production_start"]).dt.to_period("M").dt.to_timestamp().dt.normalize().dt.date.astype(str)
        # Count planned joiners by production month
        g = c.groupby("month")["production_start"].count().to_dict()
        for m in mids:
            out[m] = int(g.get(m, 0))
        return out, c
    planned_nh_m_raw, classes_df = _monthly_planned_nh_from_classes(pid, month_ids)
    # Apply learning-curve throughput to planned NH (month-level)
    def _learning_curve_for_month(settings, lc_ovr_df, mid):
        # reuse weekly curve but just return dict of lists
        return _lc_with_wf_m(_learning_curve_for_week(settings, lc_ovr_df, mid), mid)
    planned_nh_m = {}
    for m in month_ids:
        lc = _learning_curve_for_month(settings, lc_ovr_df, m)
        tp = (lc.get("throughput_train_pct", 100.0) / 100.0) * (lc.get("throughput_nest_pct", 100.0) / 100.0)
        planned_nh_m[m] = int(round(planned_nh_m_raw.get(m, 0) * max(0.0, min(1.0, tp))))

    # Buckets from classes for month (in phase counts)
    def _monthly_buckets_from_classes(df, mids):
        from collections import defaultdict
        nest = {m: defaultdict(int) for m in mids}
        sda  = {m: defaultdict(int) for m in mids}
        if not isinstance(df, pd.DataFrame) or df.empty: return nest, sda
        def _m(d):
            t = pd.to_datetime(d, errors="coerce")
            if pd.isna(t): return None
            return pd.Timestamp(t).to_period("M").to_timestamp().date().isoformat()
        for _, r in df.iterrows():
            n = _nh_effective_count(r)
            if n <= 0: continue
            ns = _m(r.get("nesting_start")); ne = _m(r.get("nesting_end")); ps = _m(r.get("production_start"))
            if ns and ne:
                mlist = [mm for mm in mids if (mm >= ns and mm <= ne)]
                for i, mm in enumerate(mlist, start=1): nest[mm][i] += n
            if ps:
                sda_weeks = int(float(settings.get("sda_weeks", settings.get("default_sda_weeks", 0)) or 0))
                if sda_weeks > 0:
                    mlist = [mm for mm in mids if mm >= ps][:max(1, (sda_weeks+3)//4)]  # approx weeks?months
                    for i, mm in enumerate(mlist, start=1): sda[mm][i] += n
        return nest, sda
    nest_buckets, sda_buckets = _monthly_buckets_from_classes(classes_df, month_ids)
    # in-phase counters (peak within month)
    m_train_in_phase = {m: sum(nest_buckets[m].values()) for m in month_ids}  # training ~= nesting buckets prior to prod
    m_nest_in_phase  = {m: sum(nest_buckets[m].values()) for m in month_ids}
    for m in month_ids:
        trn.loc[trn["metric"] == "Training Start (#)", m] = m_train_in_phase.get(m, 0)
        trn.loc[trn["metric"] == "Training End (#)",   m] = m_train_in_phase.get(m, 0)
        trn.loc[trn["metric"] == "Nesting Start (#)",  m] = m_nest_in_phase.get(m, 0)
        trn.loc[trn["metric"] == "Nesting End (#)",    m] = m_nest_in_phase.get(m, 0)
    # Actual joiners by production month (from roster)
    roster_df = _load_roster_normalized(pid)
    # Actual NH per month from roster
    actual_nh_m = {m: 0 for m in month_ids}
    if isinstance(roster_df, pd.DataFrame) and not roster_df.empty:
        r = roster_df.copy()
        c = None
        for k in ("production_start", "prod_start", "to_production", "go_live"):
            if k in r.columns:
                c = k
                break
        if c:
            try:
                r[c] = pd.to_datetime(r[c], errors="coerce", format="mixed")
            except TypeError:
                r[c] = pd.to_datetime(r[c], errors="coerce")
            r = r.dropna(subset=[c])
            r["month"] = _mid(r[c])
            g = r.groupby("month")[c].count().to_dict()
            actual_nh_m = {m: int(g.get(m, 0)) for m in month_ids}
    for m in month_ids:
        nh.loc[nh["metric"] == "Planned New Hire HC (#)", m] = int(planned_nh_m.get(m, 0))
        nh.loc[nh["metric"] == "Actual New Hire HC (#)",  m] = int(actual_nh_m.get(m, 0) if m <= today_m else 0)
        plan = float(planned_nh_m.get(m, 0))
        act  = float(actual_nh_m.get(m, 0) if m <= today_m else 0)
        nh.loc[nh["metric"] == "Recruitment Achievement", m] = (0.0 if plan <= 0 else 100.0 * act / plan)

    # ---- Actual HC snapshots from roster ? monthly average ----
    def _monthly_hc_average_from_roster(roster_df, mids, role_regex=r"\bagent\b"):
        if not isinstance(roster_df, pd.DataFrame) or roster_df.empty:
            return {m:0 for m in mids}
        R = roster_df.copy()
        # normalize timestamps
        for c in ("production_start","prod_start","terminate_date","term_date"):
            if c in R.columns:
                try:
                    R[c] = pd.to_datetime(R[c], errors="coerce", format="mixed").dt.normalize()
                except TypeError:
                    R[c] = pd.to_datetime(R[c], errors="coerce").dt.normalize()
        if "production_start" not in R.columns and "prod_start" in R.columns:
            R["production_start"] = R["prod_start"]
        if "terminate_date" not in R.columns and "term_date" in R.columns:
            R["terminate_date"] = R["term_date"]
        # role filter
        try:
            R["role"] = R["role"].astype(str)
            mask_role = R["role"].str.contains(role_regex, flags=re.IGNORECASE, regex=True)
            R = R[mask_role]
        except Exception:
            pass
        out = {}
        for mid in mids:
            try:
                base = pd.to_datetime(mid, errors="coerce", format="%Y-%m-%d").normalize()
            except Exception:
                base = pd.to_datetime(mid, errors="coerce").normalize()
            if pd.isna(base):
                out[mid] = 0; continue
            y, m = int(base.year), int(base.month)
            days = calendar.monthrange(y, m)[1]
            dates = pd.date_range(base, periods=days, freq="D")
            counts = []
            for d in dates:
                active = ((R["production_start"].isna()) | (R["production_start"] <= d)) & ((R["terminate_date"].isna()) | (R["terminate_date"] > d))
                counts.append(int(active.sum()))
            out[mid] = int(round(float(np.mean(counts)))) if counts else 0
        return out
    
    def _monthly_hc_step_from_roster(roster_df, mids, role_regex=r"\bagent\b"):
        """Monthly snapshot step function similar to weekly _weekly_hc_step_from_roster.
        Computes headcount as of the first day of each month label in mids.
        """
        if not isinstance(roster_df, pd.DataFrame) or roster_df.empty:
            return {m: 0 for m in mids}
        R = roster_df.copy()
        # column maps
        L = {str(c).strip().lower(): c for c in R.columns}
        c_role = L.get("role") or L.get("position group") or L.get("position description")
        c_cur  = L.get("current status") or L.get("current_status") or L.get("status")
        c_work = L.get("work status")    or L.get("work_status")
        c_ps   = L.get("production start") or L.get("production_start") or L.get("prod start") or L.get("prod_start") or L.get("to_production") or L.get("go_live")
        c_td   = L.get("terminate date")   or L.get("terminate_date")   or L.get("termination date") or L.get("term_date")
        if not c_role:
            return {m: 0 for m in mids}
        # status mask similar to weekly helper
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
        # role mask
        role = R[c_role].astype(str).str.strip().str.lower()
        role_mask = role.str.contains(role_regex, na=False, regex=True)
        X = R[role_mask & status_mask].copy()
        if X.empty:
            return {m: 0 for m in mids}
        # convert dates to month ids
        def _to_month_id(s):
            t = pd.to_datetime(s, errors="coerce")
            if pd.isna(t):
                return None
            return pd.Timestamp(t).to_period("M").to_timestamp().date().isoformat()
        X["_psm"] = X[c_ps].apply(_to_month_id) if c_ps in X else None
        X["_tm"]  = X[c_td].apply(_to_month_id) if c_td in X else None
        # step build
        diffs = {m: 0 for m in mids}
        # Ensure chronological order
        mids_sorted = sorted(mids, key=lambda x: pd.to_datetime(x, errors="coerce"))
        first_m = mids_sorted[0]
        base = 0
        for _, r in X.iterrows():
            psm = r.get("_psm"); tm = r.get("_tm")
            started_before = (psm is None) or (psm < first_m)
            terminated_before_or_on = (tm is not None) and (tm <= first_m)
            if started_before and not terminated_before_or_on:
                base += 1
            if psm is not None and psm in diffs and psm >= first_m:
                diffs[psm] += 1
            if tm is not None and tm in diffs and tm >= first_m:
                diffs[tm] -= 1
        out = {}
        running = base
        for m in mids_sorted:
            running += diffs.get(m, 0)
            out[m] = int(max(0, running))
        # keep original order keys too
        return {m: out.get(m, 0) for m in mids}
    hc_actual_m = _monthly_hc_average_from_roster(roster_df, month_ids, r"\bagent\b")
    hc_baseline_m = {m:0 for m in month_ids}
    if isinstance(roster_df, pd.DataFrame) and not roster_df.empty:
        R = roster_df.copy()
        L = {str(c).strip().lower(): c for c in R.columns}
        c_ps = L.get("production start") or L.get("production_start") or L.get("prod_start") or L.get("to_production") or L.get("go_live")
        c_td = L.get("terminate date") or L.get("terminate_date") or L.get("termination date") or L.get("term_date")
        c_role = L.get("role") or L.get("position group") or L.get("position description")
        roles = R[c_role].astype(str).str.strip().str.lower() if c_role else None
        if c_ps:
            for mm in month_ids:
                try:
                    try:
                        dt0 = pd.to_datetime(mm, errors="coerce", format="%Y-%m-%d")
                    except Exception:
                        dt0 = pd.to_datetime(mm, errors="coerce")
                    if pd.isna(dt0):
                        continue
                    # Month window [begin, end]
                    dt_begin = pd.Timestamp(dt0).to_period('M').to_timestamp().date()
                    y = int(pd.Timestamp(dt0).year); mo = int(pd.Timestamp(dt0).month)
                    dt_end = (pd.Timestamp(dt_begin) + pd.Timedelta(days=(calendar.monthrange(y, mo)[1]-1))).date()
                    try:
                        ps = pd.to_datetime(R[c_ps], errors="coerce", format="mixed").dt.date
                    except TypeError:
                        ps = pd.to_datetime(R[c_ps], errors="coerce").dt.date
                    if c_td:
                        try:
                            td = pd.to_datetime(R[c_td], errors="coerce", format="mixed").dt.date
                        except TypeError:
                            td = pd.to_datetime(R[c_td], errors="coerce").dt.date
                    else:
                        td = None
                    # Active if started by month-end and not terminated before month-begin
                    mask = ps.isna() | (ps <= dt_end)
                    if td is not None:
                        mask &= ((td.isna()) | (td >= dt_begin))
                    if roles is not None:
                        # For Voice channel rosters, broaden the agent pattern to common synonyms
                        ch_low_ = str(ch_first or '').strip().lower()
                        if ch_low_ == 'voice':
                            role_pat = r"\b(agent|csr|advisor|associate|representative|executive|specialist)\b"
                        else:
                            role_pat = r"\bagent\b"
                        mask &= roles.str.contains(role_pat, regex=True, na=False)
                    hc_baseline_m[mm] = int(mask.sum())
                except Exception:
                    pass
    # hc_baseline_m = {m:0 for m in month_ids}
    # if isinstance(roster_df, pd.DataFrame) and not roster_df.empty:
    #     R = roster_df.copy()
    #     L = {str(c).strip().lower(): c for c in R.columns}
    #     c_ps = L.get("production start") or L.get("production_start") or L.get("prod_start") or L.get("to_production") or L.get("go_live")
    #     c_td = L.get("terminate date") or L.get("terminate_date") or L.get("termination date") or L.get("term_date")
    #     c_role = L.get("role") or L.get("position group") or L.get("position description")
    #     roles = R[c_role].astype(str).str.strip().str.lower() if c_role else None
    #     if c_ps:
    #         for mm in month_ids:
    #             try:
    #                 dt0 = pd.to_datetime(mm, errors="coerce")
    #                 if pd.isna(dt0):
    #                     continue
    #                 dt0 = pd.Timestamp(dt0).to_period('M').to_timestamp().date()
    #                 ps = pd.to_datetime(R[c_ps], errors="coerce").dt.date
    #                 td = pd.to_datetime(R[c_td], errors="coerce").dt.date if c_td else None
    #                 mask = ps <= dt0
    #                 if td is not None:
    #                     mask &= ((td.isna()) | (td >= dt0))
    #                 if roles is not None:
    #                     mask &= roles.str.contains(r"\bagent\b", regex=True, na=False)
    #                 hc_baseline_m[mm] = int(mask.sum())
    #             except Exception:
    #                 pass
    # Monthly step snapshot to mimic weekly behavior
    hc_step_m = _monthly_hc_step_from_roster(roster_df, month_ids, r"\bagent\b")
    try:
        if sum(hc_step_m.values()) == 0:
            hc_step_m = _monthly_hc_step_from_roster(roster_df, month_ids, r".*")
    except Exception:
        pass
    sme_billable_m = _monthly_hc_average_from_roster(roster_df, month_ids, r"\bsme\b")
    for m in month_ids:
        hc.loc[hc["metric"] == "Actual Agent HC (#)", m] = int(hc_step_m.get(m, 0) or 0)
        hc.loc[hc["metric"] == "SME Billable HC (#)", m] = sme_billable_m.get(m, 0)

    # ---- Budget vs simple Planned HC (monthly reduce) ----
    def _monthly_reduce(df: pd.DataFrame, value_candidates=("hc","headcount","value","count"), how="sum"):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}
        d = df.copy()
        c_date = None
        for c in ("date","week","month","start_date"):
            if c in d.columns: c_date = c; break
        if not c_date:
            return {}
        d[c_date] = pd.to_datetime(d[c_date], errors="coerce")
        d["month"] = _mid(d[c_date])
        low = {c.lower(): c for c in d.columns}
        vcol = None
        for c in value_candidates:
            vcol = low.get(c.lower()); 
            if vcol: break
        if not vcol: return {}
        d[vcol] = pd.to_numeric(d[vcol], errors="coerce").fillna(0.0)
        g = d.groupby("month", as_index=True)[vcol].agg(how)
        return g.astype(float).to_dict()
    budget_df = _first_non_empty_ts(sk, ["budget_headcount","budget_hc","headcount_budget","hc_budget"])
    # For headcount, monthly should reflect average level across the month, not sum of weeks
    budget_m  = _monthly_reduce(budget_df, value_candidates=("hc","headcount","value","count"), how="mean")
    for m in month_ids:
        hc.loc[hc["metric"] == "Budgeted HC (#)",         m] = float(budget_m.get(m, 0.0))
        hc.loc[hc["metric"] == "Planned/Tactical HC (#)", m] = float(budget_m.get(m, 0.0))

    # ---- Attrition (planned/actual/pct) monthly ----
    # For attrition headcount, weekly inputs should ROLL UP by SUM to a monthly total
    # (not MEAN). Using mean dilutes monthly counts by ~4x when source is weekly.
    att_plan_m = _monthly_reduce(
        _first_non_empty_ts(sk, ["attrition_planned_hc", "attrition_plan_hc", "planned_attrition_hc"]),
        value_candidates=("hc", "headcount", "value", "count"),
        how="sum",
    )
    att_act_m = _monthly_reduce(
        _first_non_empty_ts(sk, ["attrition_actual_hc", "attrition_actual", "actual_attrition_hc"]),
        value_candidates=("hc", "headcount", "value", "count"),
        how="sum",
    )
    att_pct_m  = _monthly_reduce(_first_non_empty_ts(sk, ["attrition_pct","attrition_percent","attrition%","attrition_rate"]),
                                 value_candidates=("pct","percent","value"), how="mean")
    try:
        att_saved_df = load_df(f"plan_{pid}_attr")
    except Exception:
        att_saved_df = None
    att_plan_saved = _collect_saved_metric(att_saved_df, "Planned Attrition HC (#)", "sum")
    att_actual_saved = _collect_saved_metric(att_saved_df, "Actual Attrition HC (#)", "sum")
    att_plan_pct_saved = _collect_saved_metric(att_saved_df, "Planned Attrition %", "mean")
    attr_roster_m = {m: 0 for m in month_ids}
    if isinstance(roster_df, pd.DataFrame) and not roster_df.empty:
        R = roster_df.copy()
        L = {str(c).strip().lower(): c for c in R.columns}
        c_role = L.get("role") or L.get("position group") or L.get("position description")
        c_term = L.get("terminate date") or L.get("terminate_date") or L.get("termination date")
        if c_role and c_term:
            role = R[c_role].astype(str).str.strip().str.lower()
            mask = role.str.contains(r"agent", na=False, regex=True)
            term_series = pd.to_datetime(R.loc[mask, c_term], errors="coerce").dropna()
            if not term_series.empty:
                months = _mid(pd.Series(term_series))
                for tm in months:
                    tm = str(tm)
                    if tm in attr_roster_m:
                        attr_roster_m[tm] += 1
    try:
        _hc_plan_row = hc.loc[hc["metric"].astype(str).str.strip() == "Planned/Tactical HC (#)"].iloc[0]
    except Exception:
        _hc_plan_row = None
    variance_label = None
    try:
        _att_metrics = att["metric"].astype(str).str.strip()
        if "Variance vs Planned" in _att_metrics.values:
            variance_label = "Variance vs Planned"
        elif "Variance vs Planned" in _att_metrics.values:
            variance_label = "Variance vs Planned"
    except Exception:
        variance_label = None
    for m in month_ids:
        plan_ts = float(att_plan_m.get(m, 0.0))
        plan_manual = (att_plan_saved.get(m) if isinstance(att_plan_saved, dict) else None)
        roster_term = float(attr_roster_m.get(m, 0.0))
        if plan_manual is not None:
            plan_hc = plan_manual
        else:
            plan_hc = plan_ts
        if _wf_active_month(m) and attr_delta and m > today_m:
            plan_hc += float(attr_delta)
        plan_hc = max(0.0, plan_hc)
        planned_pct_val = None
        # Always compute monthly planned % from monthly planned HC and a monthly HC baseline.
        # Do not average weekly percentages, which dilutes the monthly rate by ~4x.
        if planned_pct_val is None:
            # Baseline for planned %: previous month's Actual Agent HC snapshot; fallback to current month's start HC,
            # then Planned/Tactical HC, then Budgeted HC.
            try:
                idx = month_ids.index(m)
            except Exception:
                idx = -1
            prev_m = month_ids[idx - 1] if idx > 0 else None
            denom_p = 0.0
            if prev_m:
                try:
                    denom_p = float(hc_baseline_m.get(prev_m, 0.0) or 0.0)
                except Exception:
                    denom_p = 0.0
            if denom_p <= 0:
                try:
                    denom_p = float(hc_baseline_m.get(m, 0.0) or 0.0)
                except Exception:
                    denom_p = 0.0
            # fallback to monthly actual snapshot if baseline is zero
            if denom_p <= 0:
                try:
                    denom_p = float(hc_actual_m.get(prev_m, 0.0) or 0.0) if prev_m else 0.0
                except Exception:
                    denom_p = 0.0
            if denom_p <= 0:
                try:
                    denom_p = float(hc_actual_m.get(m, 0.0) or 0.0)
                except Exception:
                    denom_p = 0.0
            if denom_p <= 0:
                try:
                    denom_p = float(pd.to_numeric((_hc_plan_row or {}).get(m), errors="coerce")) if hasattr(_hc_plan_row, 'get') else float(pd.to_numeric(_hc_plan_row[m], errors="coerce"))
                except Exception:
                    denom_p = 0.0
            if denom_p <= 0:
                try:
                    denom_p = float(budget_m.get(m, 0.0) or 0.0)
                except Exception:
                    denom_p = 0.0
            planned_pct_val = (100.0 * plan_hc / denom_p) if denom_p > 0 else 0.0
        else:
            planned_pct_val = float(planned_pct_val)
        act_manual = (att_actual_saved.get(m) if isinstance(att_actual_saved, dict) else None)
        act_ts = float(att_act_m.get(m, 0.0))
        _cand = [roster_term, act_ts]
        if act_manual is not None:
            try:
                _cand.append(float(act_manual))
            except Exception:
                pass
        act_hc = max([float(c) for c in _cand] + [0.0])
        act_hc = max(0.0, act_hc)
        try:
            denom = float(hc_baseline_m.get(m, 0) or 0.0)
        except Exception:
            denom = 0.0
        if denom <= 0:
            # Fallback to planned/tactical HC for the month
            try:
                denom = float(pd.to_numeric((_hc_plan_row or {}).get(m), errors="coerce")) if hasattr(_hc_plan_row, 'get') else float(pd.to_numeric(_hc_plan_row[m], errors="coerce"))
            except Exception:
                denom = 0.0
        if denom <= 0:
            # As a last resort, use Budgeted HC for the month if present
            try:
                denom = float(budget_m.get(m, 0.0) or 0.0)
            except Exception:
                denom = 0.0
        pct = 100.0 * (act_hc / denom) if denom > 0 else 0.0
        if _wf_active_month(m) and attr_delta and m > today_m:
            try:
                denom_future = float(pd.to_numeric((_hc_plan_row or {}).get(m), errors="coerce")) if hasattr(_hc_plan_row, 'get') else float(pd.to_numeric(_hc_plan_row[m], errors="coerce"))
            except Exception:
                denom_future = denom
            if denom_future > 0:
                pct = 100.0 * (plan_hc / denom_future)
        att.loc[att["metric"] == "Planned Attrition HC (#)", m] = plan_hc
        att.loc[att["metric"] == "Actual Attrition HC (#)",  m] = act_hc
        if "Planned Attrition %" in att["metric"].values:
            att.loc[att["metric"] == "Planned Attrition %", m] = planned_pct_val
        att.loc[att["metric"] == "Attrition %", m] = pct
        if "Actual Attrition %" in att["metric"].values:
            att.loc[att["metric"] == "Actual Attrition %", m] = pct
        if variance_label:
            att.loc[att["metric"] == variance_label, m] = float(pct) - float(planned_pct_val or 0.0)

    # ---- Shrinkage raw - monthly ---
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
    ooo_hours_m, io_hours_m, base_hours_m = {}, {}, {}
    # Extra denominators for BO-style shrinkage
    sc_hours_m, tt_worked_hours_m = {}, {}
    bo_shrink_frac_daily_m = {}

    # overtime_hours_m already initialized above before use

    def _month_key(s):
        ds = pd.to_datetime(s, errors="coerce")
        if isinstance(ds, pd.Series):
            return _mid(ds)
        else:
            return _mid(pd.Series(ds))

    def _agg_monthly(date_idx, ooo_series, ino_series, base_series):
        mk = _month_key(date_idx)
        g = pd.DataFrame({"month": mk, "ooo": ooo_series, "ino": ino_series, "base": base_series}).groupby("month", as_index=False).sum()
        for _, r in g.iterrows():
            k = str(r["month"])
            ooo_hours_m[k]  = ooo_hours_m.get(k, 0.0)  + float(r["ooo"])
            io_hours_m[k]   = io_hours_m.get(k, 0.0)   + float(r["ino"])
            base_hours_m[k] = base_hours_m.get(k, 0.0) + float(r["base"])

    # Tolerant: process both raw sources so BO BAs using voice raw still compute shrinkage
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
                # Align monthly Voice groupings with weekly summarize_shrinkage_voice
                ooo  = _sum_cols_series([
                    r"\bsc[_\s-]*absence",
                    r"\bsc[_\s-]*holiday",
                    r"\bsc[_\s-]*a[_\s-]*sick",
                    r"\bsc[_\s-]*sick",
                    r"\bsc[_\s-]*vacation",
                    r"\bsc[_\s-]*leave",
                    r"\bsc[_\s-]*unpaid",
                ]) 
                ino  = _sum_cols_series([
                    r"\bsc[_\s-]*training",
                    r"\bsc[_\s-]*break",
                    r"\bsc[_\s-]*system[_\s-]*exception",
                    r"\bsc[_\s-]*meeting",
                    r"\bsc[_\s-]*coaching",
                ]) 
                idx_dates = pd.to_datetime(pv.index, errors="coerce")
                _agg_monthly(idx_dates, ooo, ino, base)
                # Do not populate SC denominators from Voice raw; keep Voice on base-hours formula
                # (weekly behavior avoids SC/TTW for Voice so monthly should match)
                try:
                    base_s = pd.to_numeric(base, errors='coerce')
                    ov = pd.to_numeric(ooo, errors='coerce') + pd.to_numeric(ino, errors='coerce')
                    frac = (ov / base_s.replace({0.0: np.nan})).fillna(0.0)
                    for t, f in zip(idx_dates, frac):
                        k = str(pd.to_datetime(t).date())
                        if pd.notna(f):
                            bo_shrink_frac_daily_m[k] = float(max(0.0, min(0.99, f)))
                except Exception:
                    pass
    
    if True:
        try:
            braw = load_df("shrinkage_raw_backoffice")
        except Exception:
            braw = None
        if isinstance(braw, pd.DataFrame) and not braw.empty:
            b = braw.copy()
            L = {str(c).strip().lower(): c for c in b.columns}
            c_date = L.get("date")
            c_act = L.get("activity")
            c_sec  = L.get("duration_seconds") or L.get("seconds") or L.get("duration")
            c_ba   = L.get("journey") or L.get("business area") or L.get("ba") or L.get("vertical")
            c_sba  = L.get("sub_business_area") or L.get("sub business area") or L.get("sub_ba")  or L.get("subba")
            c_ch   = L.get("channel") or L.get("lob")
            mask = pd.Series(True, index=b.index)
            if c_ba and p.get("vertical"): mask &= b[c_ba].astype(str).str.strip().str.lower().eq(p["vertical"].strip().lower())
            if c_sba and p.get("sub_ba"): mask &= b[c_sba].astype(str).str.strip().str.lower().eq(p["sub_ba"].strip().lower())
            if c_ch:
                mask &= b[c_ch].astype(str).str.strip().str.lower().isin(["back office","bo","backoffice"]) 
            # Voice-like BO payload (Superstate/Hours) saved in backoffice store
            c_state = L.get("superstate") or L.get("state")
            c_hours = L.get("hours") or L.get("duration_hours")
            if c_date and c_state and c_hours and not b.empty and (not c_act or b[c_act].isna().all()):
                d2 = b[[c_date, c_state, c_hours]].copy()
                d2[c_hours] = pd.to_numeric(d2[c_hours], errors="coerce").fillna(0.0)
                d2[c_date]  = pd.to_datetime(d2[c_date], errors="coerce").dt.date
                pv = d2.pivot_table(index=c_date, columns=c_state, values=c_hours, aggfunc="sum", fill_value=0.0)
                def col(name): return pv[name] if name in pv.columns else 0.0
                base = col("SC_INCLUDED_TIME")
                # Match weekly buckets exactly (extend OOO and In-Office)
                ooo  = (
                    col("SC_ABSENCE_TOTAL")
                    + col("SC_HOLIDAY")
                    + col("SC_A_Sick_Long_Term")
                    + (pv["SC_VACATION"] if "SC_VACATION" in pv.columns else 0.0)
                    + (pv["SC_LEAVE"] if "SC_LEAVE" in pv.columns else 0.0)
                    + (pv["SC_UNPAID"] if "SC_UNPAID" in pv.columns else 0.0)
                )
                ino  = (
                    col("SC_TRAINING_TOTAL")
                    + col("SC_BREAKS")
                    + col("SC_SYSTEM_EXCEPTION")
                    + (pv["SC_MEETING"] if "SC_MEETING" in pv.columns else 0.0)
                    + (pv["SC_COACHING"] if "SC_COACHING" in pv.columns else 0.0)
                )
                idx_dates = pd.to_datetime(pv.index, errors="coerce")
                _agg_monthly(idx_dates, ooo, ino, base)
            elif c_date and c_act and (c_sec or (L.get("hours") or L.get("duration_hours"))) and not b.empty:
                d = b[[c_date, c_act, c_sec]].copy()
                d[c_act] = d[c_act].astype(str).str.strip().str.lower()
                d[c_sec] = pd.to_numeric(d[c_sec], errors="coerce").fillna(0.0)
                d[c_date] = pd.to_datetime(d[c_date], errors="coerce")
                act = d[c_act] 
                # Broaden classifiers similar to weekly
                m_div  = act.str.contains(r"\bdivert(?:ed)?\b", regex=True, na=False) | act.eq("diverted")
                m_dow  = act.str.contains(r"\bdowntime\b|\bdown\b", regex=True, na=False) | act.eq("downtime")
                m_sc   = act.str.contains(r"\bstaff\s*complement\b|\bsc[_\s-]*included[_\s-]*time\b|\bincluded\s*time\b|\bstaffed\s*hours\b", regex=True, na=False) | act.eq("staff complement")
                m_fx   = act.str.contains(r"\bflexi?time\b|\bflex\b", regex=True, na=False) | act.eq("flexitime")
                m_ot   = act.str.contains(r"\bover\s*time\b|\bovertime\b|\bot\b|\bot\s*hours\b|\bot\s*hrs\b", regex=True, na=False) | act.eq("overtime")
                m_lend = act.str.contains(r"\blend(?:ed)?\b|\blend\s*staff\b", regex=True, na=False) | act.eq("lend staff")
                m_borr = act.str.contains(r"\bborrow(?:ed)?\b|\bborrow\s*staff\b", regex=True, na=False) | act.eq("borrowed staff")
                sec_div  = d.loc[m_div,  c_sec].groupby(d[c_date]).sum()
                sec_dow  = d.loc[m_dow,  c_sec].groupby(d[c_date]).sum()
                sec_sc   = d.loc[m_sc,   c_sec].groupby(d[c_date]).sum()
                sec_fx   = d.loc[m_fx,   c_sec].groupby(d[c_date]).sum()
                sec_ot   = d.loc[m_ot,   c_sec].groupby(d[c_date]).sum()
                sec_lend = d.loc[m_lend, c_sec].groupby(d[c_date]).sum()
                sec_borr = d.loc[m_borr, c_sec].groupby(d[c_date]).sum()
                idx = build_idx(sec_div, sec_dow, sec_sc, sec_fx, sec_ot, sec_lend, sec_borr,
                base_dates=d[c_date])
                def get(s): return s.reindex(idx, fill_value=0.0)
                # Seconds -> Hours
                s_div = get(sec_div).astype(float); h_div = s_div / 3600.0
                s_dow = get(sec_dow).astype(float); h_dow = s_dow / 3600.0
                s_sc  = get(sec_sc).astype(float);  h_sc  = s_sc  / 3600.0
                s_fx  = get(sec_fx).astype(float);  h_fx  = s_fx  / 3600.0
                s_ot  = get(sec_ot).astype(float);  h_ot  = s_ot  / 3600.0
                s_len = get(sec_lend).astype(float);h_len = s_len / 3600.0
                s_bor = get(sec_borr).astype(float);h_bor = s_bor / 3600.0

                # Denominators per business rule
                total_time_worked_h = (h_sc - h_dow + h_fx + h_ot + h_bor - h_len).clip(lower=0)
                non_dow_base_h      = (h_sc + h_fx + h_ot + h_bor - h_len).clip(lower=0)
                # Aggregate OOO (Downtime) and In-Office (Divert) hours; base as non-dow base
                _agg_monthly(idx, h_dow, h_div, non_dow_base_h)
                # Daily overall fraction (activity): (Downtime/SC) + (Diverted/TTW)
                try:
                    for t, dhrs, scv, ttwv in zip(idx, h_div, h_sc, total_time_worked_h):
                        k = str(pd.to_datetime(t).date())
                        ooo_f = (float(h_dow.loc[t]) / float(scv)) if (t in h_dow.index and scv > 0) else 0.0
                        ino_f = (float(dhrs) / float(ttwv)) if ttwv > 0 else 0.0
                        bo_shrink_frac_daily_m[k] = max(0.0, min(0.99, ooo_f + ino_f))
                except Exception:
                    pass
                # Track denominators for BO shrink formulas (month key)
            # mk = _month_key(idx)
            # for t, scv, ttwh in zip(mk, h_sc, total_time_worked_h):
            #     k = str(pd.to_datetime(t).to_period('M').to_timestamp().date())
            #     sc_hours_m[k] = sc_hours_m.get(k, 0.0) + float(scv)
            #     tt_worked_hours_m[k] = tt_worked_hours_m.get(k, 0.0) + float(ttwh)
            # for t, scv, ttwh in zip(idx, h_sc, total_time_worked_h):
            #     k = _month_key([t])[0]  # collapse to month midpoint
            #     sc_hours_m[k] = sc_hours_m.get(k, 0.0) + float(scv)
            #     tt_worked_hours_m[k] = tt_worked_hours_m.get(k, 0.0) + float(ttwh)
            # Capture overtime hours (monthly) from shrinkage raw (Back Office)
            try:
                mk = _month_key(sec_ot.index)
                g_ot = pd.DataFrame({"month": mk, "ot": (sec_ot.astype(float) / 3600)}).groupby("month", as_index=False).sum()
                for _, r2 in g_ot.iterrows():
                    k2 = str(r2["month"])
                    overtime_hours_m[k2] = overtime_hours_m.get(k2, 0.0) + float(r2["ot"])
            except Exception:
                pass
        # As a robustness fallback (and to match weekly behavior), also derive monthly OOO/SC/TTW via common.summarize_shrinkage_bo
        try:
            dsum = summarize_shrinkage_bo(b)
            if not dsum.empty:
                dsum = clip_to_dataset_range(dsum, "date")
            if isinstance(dsum, pd.DataFrame) and not dsum.empty:
                d2 = dsum.copy()
                # Filter to current plan scope
                def _filt_eq(series, val):
                    try:
                        if not val: return pd.Series([True]*len(series))
                        s = series.astype(str).str.strip().str.lower()
                        return s.eq(str(val).strip().lower())
                    except Exception:
                        return pd.Series([True]*len(series))
                m = pd.Series([True]*len(d2))
                if 'Business Area' in d2.columns and p.get('vertical'):
                    m &= _filt_eq(d2['Business Area'], p.get('vertical'))
                if 'Sub Business Area' in d2.columns and p.get('sub_ba'):
                    m &= _filt_eq(d2['Sub Business Area'], p.get('sub_ba'))
                if 'Channel' in d2.columns:
                    m &= d2['Channel'].astype(str).str.strip().str.lower().isin(['back office','bo','backoffice'])
                if loc_first:
                    # if data has a specific country/site and it matches, filter; otherwise don't restrict
                    for col in ('Country','Site'):
                        if col in d2.columns:
                            s = d2[col].astype(str).str.strip()
                            sl = s.str.lower()
                            tgt = loc_first.strip().lower()
                            if sl.ne("").any() and sl.ne("all").any() and sl.eq(tgt).any():
                                m &= sl.eq(tgt)
                d2 = d2.loc[m]
                if not d2.empty:
                    d2['month'] = _mid(d2['date'])
                    agg = d2.groupby('month', as_index=False)[['OOO Hours','In Office Hours','Base Hours','TTW Hours']].sum()
                    for _, r3 in agg.iterrows():
                        k = str(r3['month'])
                        ooo_hours_m[k] = ooo_hours_m.get(k, 0.0) + float(r3['OOO Hours'])
                        io_hours_m[k]  = io_hours_m.get(k, 0.0)  + float(r3['In Office Hours'])
                        sc_hours_m[k]  = sc_hours_m.get(k, 0.0)  + float(r3['Base Hours'])
                        tt_worked_hours_m[k] = tt_worked_hours_m.get(k, 0.0) + float(r3['TTW Hours'])
        except Exception:
            pass
    # After computing shrinkage, write independent overtime (from raw) into FW grid (does not alter shrinkage)
    if "Overtime Hours (#)" in fw_rows:
        for m in month_ids:
            val = None
            ch_key_local = str(ch_first or '').strip().lower()
            if ch_key_local == 'voice':
                val = _ot_voice_m_raw.get(m, None)
            elif ch_key_local in ("back office","bo"):
                val = _ot_m_raw.get(m, None)
            if val is None:
                val = overtime_m.get(m, None)
            if val is not None:
                fw.loc[fw["metric"] == "Overtime Hours (#)", m] = float(val)
    try:
        shr_saved_df = load_df(f"plan_{pid}_shr")
    except Exception:
        shr_saved_df = None
    saved_ooo_hours = _collect_saved_metric(shr_saved_df, "OOO Shrink Hours (#)", "sum")
    saved_ino_hours = _collect_saved_metric(shr_saved_df, "In-Office Shrink Hours (#)", "sum")
    saved_ooo_pct = _collect_saved_metric(shr_saved_df, "OOO Shrinkage %", "mean")
    saved_ino_pct = _collect_saved_metric(shr_saved_df, "In-Office Shrinkage %", "mean")
    saved_ov_pct  = _collect_saved_metric(shr_saved_df, "Overall Shrinkage %", "mean")
    saved_plan_pct = {}
    try:
        if isinstance(shr_saved_df, pd.DataFrame) and not shr_saved_df.empty:
            dfp = shr_saved_df.copy()
            mser = dfp['metric'].astype(str).str.strip()
            if 'Planned Shrinkage %' in mser.values:
                row = dfp.loc[mser == 'Planned Shrinkage %'].iloc[0]
                for m in month_ids:
                    if m in row:
                        v = row.get(m)
                        try:
                            # handle '33.3%' or numeric
                            if isinstance(v, str) and v.endswith('%'):
                                v = float(v.strip().rstrip('%'))
                            else:
                                v = float(pd.to_numeric(v, errors='coerce'))
                        except Exception:
                            v = None
                        if v is not None and not pd.isna(v):
                            saved_plan_pct[m] = float(v)
    except Exception:
        saved_plan_pct = {}
    for m, val in saved_ooo_hours.items():
        if ooo_hours_m.get(m, 0.0) == 0.0:
            ooo_hours_m[m] = float(val)
    for m, val in saved_ino_hours.items():
        if io_hours_m.get(m, 0.0) == 0.0:
            io_hours_m[m] = float(val)

    # Build shrink table (+ What-If ? onto Overall % display)
    import os as _os
    _debug_shr_m = bool(_os.environ.get("CAP_DEBUG_SHRINK_M"))
    for m in month_ids:
        if m not in shr.columns:
            shr[m] = np.nan
        shr[m] = pd.to_numeric(shr[m], errors="coerce").astype("float64")
        base = float(base_hours_m.get(m, 0.0))
        ooo = float(ooo_hours_m.get(m, 0.0))
        ino = float(io_hours_m.get(m, 0.0))
        scm  = float(sc_hours_m.get(m, 0.0))
        ttwm = float(tt_worked_hours_m.get(m, 0.0))
        saved_overall_pct = saved_ov_pct.get(m) if isinstance(saved_ov_pct, dict) else None
        saved_ooo_pct_val = saved_ooo_pct.get(m) if isinstance(saved_ooo_pct, dict) else None
        saved_ino_pct_val = saved_ino_pct.get(m) if isinstance(saved_ino_pct, dict) else None
        if base <= 0 and (ooo or ino):
            if saved_overall_pct not in (None, 0):
                base = (ooo + ino) * 100.0 / saved_overall_pct if (ooo + ino) > 0 else 0.0
            elif planned_shrink_fraction > 0:
                base = (ooo + ino) / planned_shrink_fraction if planned_shrink_fraction > 0 else 0.0
            else:
                base = ooo + ino
            base_hours_m[m] = base
        # Pick formula by channel/denominators EXACTLY like weekly
        ch_lower = str(ch_first or "").strip().lower()
        is_bo_ch = (ch_lower in ("back office","bo")) or ("back office" in ch_lower)
        # Match weekly behavior: prefer BO denominators when available; otherwise fall back to base
        # This mirrors weekly logic where Voice can leverage BO SC/TTW if present
        use_bo_denoms = (scm > 0.0 or ttwm > 0.0) or is_bo_ch
        if use_bo_denoms:
            # Back Office rule: OOO% = Downtime/SC, In-Office% = Divert/TTW
            ooo_pct = (100.0 * ooo / scm) if scm > 0 else 0.0
            ino_pct = (100.0 * ino / ttwm) if ttwm > 0 else 0.0
            ov_pct  = (ooo_pct + ino_pct)
        else:
            # Voice and other non-BO channels: use Base hours as denominator
            ooo_pct = (100.0 * ooo / base) if base > 0 else 0.0
            ino_pct = (100.0 * ino / base) if base > 0 else 0.0
            ov_pct  = (ooo_pct + ino_pct)
        if _wf_active_month(m) and shrink_delta:
            ov_pct = min(100.0, max(0.0, ov_pct + shrink_delta))
        planned_pct = float(saved_plan_pct.get(m, 100.0 * planned_shrink_fraction))
        # What-If: reflect shrink delta onto Planned Shrinkage % for display
        try:
            if _wf_active_month(m) and shrink_delta:
                planned_pct = min(100.0, max(0.0, planned_pct + shrink_delta))
        except Exception:
            pass
        variance_pp = ov_pct - planned_pct
        shr.loc[shr["metric"] == "OOO Shrink Hours (#)",       m] = ooo
        shr.loc[shr["metric"] == "In-Office Shrink Hours (#)", m] = ino
        shr.loc[shr["metric"] == "OOO Shrinkage %",            m] = ooo_pct
        shr.loc[shr["metric"] == "In-Office Shrinkage %",       m] = ino_pct
        shr.loc[shr["metric"] == "Overall Shrinkage %",       m] = ov_pct
        shr.loc[shr["metric"] == "Planned Shrinkage %",       m] = planned_pct
        shr.loc[shr["metric"] == "Variance vs Planned",  m] = variance_pp
        if _debug_shr_m:
            try:
                branch = 'bo' if ((scm>0 or ttwm>0) or (str(ch_first or '').strip().lower() in ('back office','bo') or ('back office' in str(ch_first or '').strip().lower()))) else 'non-bo'
                print(f"[SHR-M][month={m}] ch={str(ch_first).strip().lower()} OOO={ooo:.2f} SC={scm:.2f} TTW={ttwm:.2f} base={base:.2f} branch={branch} OOO%={ooo_pct:.2f} INO%={ino_pct:.2f} OV%={ov_pct:.2f}")
            except Exception:
                pass

    # ---- BvA ----
    for m in month_ids:
        if m not in bva.columns:
            bva[m] = pd.Series(np.nan, index=bva.index, dtype="float64")
        elif not pd.api.types.is_float_dtype(bva[m].dtype):
            bva[m] = pd.to_numeric(bva[m], errors="coerce").astype("float64")
        bud = float(req_m_budgeted.get(m, 0.0))
        act = float(req_m_actual.get(m,   0.0))
        bva.loc[bva["metric"] == "Budgeted FTE (#)", m] = bud
        bva.loc[bva["metric"] == "Actual FTE (#)",   m] = act
        bva.loc[bva["metric"] == "Variance (#)",     m] = act - bud

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
    for m in month_ids:
        rat.loc[rat["metric"] == "Planned TL/Agent Ratio", m] = planned_ratio
        rat.loc[rat["metric"] == "Actual TL/Agent Ratio",  m] = actual_ratio
        rat.loc[rat["metric"] == "Variance",               m] = actual_ratio - planned_ratio

    # ---- Monthly aggregation from weekly for Ratio and Seat ----
    # Only Ratios and Seat tables should roll up as monthly averages of their weekly values
    def _weekly_to_month_avg(df: pd.DataFrame, metrics: list[str]) -> dict[str, dict[str, float]]:
        out = {mn: {mm: np.nan for mm in month_ids} for mn in metrics}
        if not isinstance(df, pd.DataFrame) or df.empty or "metric" not in df.columns:
            return out
        # Build month bucket map: month_id -> list of weekly column names contributing
        month_bucket = {mm: [] for mm in month_ids}
        for c in df.columns:
            if c == "metric":
                continue
            try:
                mid = _mid(pd.Series([c]))[0]
            except Exception:
                continue
            if mid in month_bucket:
                month_bucket[mid].append(c)
        mser = df["metric"].astype(str).str.strip()
        for name in metrics:
            if name not in mser.values:
                continue
            row = df.loc[mser == name].iloc[0]
            for mm, cols in month_bucket.items():
                if not cols:
                    continue
                try:
                    vals = pd.to_numeric(row[cols], errors="coerce")
                    # Treat zeros as missing for these roll-ups so blank weeks don't dilute the average
                    vals = vals.replace(0, np.nan)
                    v = float(vals.dropna().mean()) if not vals.dropna().empty else np.nan
                except Exception:
                    v = np.nan
                out[name][mm] = v
        return out

    try:
        ratio_w = load_df(f"plan_{pid}_ratio")
    except Exception:
        ratio_w = None
    try:
        seat_w = load_df(f"plan_{pid}_seat")
    except Exception:
        seat_w = None

    # Aggregate weekly Ratios -> monthly average
    if isinstance(ratio_w, pd.DataFrame) and not ratio_w.empty:
        agg_ratio = _weekly_to_month_avg(ratio_w, ["Planned TL/Agent Ratio","Actual TL/Agent Ratio","Variance"])
        for m in month_ids:
            pv = agg_ratio.get("Planned TL/Agent Ratio", {}).get(m, np.nan)
            av = agg_ratio.get("Actual TL/Agent Ratio", {}).get(m, np.nan)
            vv = agg_ratio.get("Variance", {}).get(m, np.nan)
            if not pd.isna(pv): rat.loc[rat["metric"] == "Planned TL/Agent Ratio", m] = pv
            if not pd.isna(av): rat.loc[rat["metric"] == "Actual TL/Agent Ratio",  m] = av
            # If weekly lacked Variance, compute from monthly avgs
            if pd.isna(vv):
                try:
                    vv = float(av) - float(pv)
                except Exception:
                    vv = np.nan
            if not pd.isna(vv): rat.loc[rat["metric"] == "Variance", m] = vv

    # Aggregate weekly Seat -> monthly average
    # Weekly variance label may differ (without (pp)); support both
    if isinstance(seat_w, pd.DataFrame) and not seat_w.empty:
        # Normalize weekly variance label to one of the expected names
        w_metrics = seat_w["metric"].astype(str).str.strip().tolist()
        var_name_w = "Variance vs Available"
        metrics = ["Seats Required (#)", "Seats Available (#)", "Seat Utilization %"] + ([var_name_w] if var_name_w else [])
        agg_seat = _weekly_to_month_avg(seat_w, metrics)
        for m in month_ids:
            for nm in ("Seats Required (#)", "Seats Available (#)", "Seat Utilization %"):
                v = agg_seat.get(nm, {}).get(m, np.nan)
                if not pd.isna(v):
                    seat.loc[seat["metric"] == nm, m] = v
            # Do not override monthly Variance from weekly; keep computed (req - avail)

    # ---- Projected supply (actuals to date; prefer actual attrition/NH if present; else planned) ----
    def _row_as_dict(df, metric_name):
        if not isinstance(df, pd.DataFrame) or df.empty or "metric" not in df.columns:
            return {}
        d = df.copy()
        for c in list(d.columns):
            if c == "metric": continue
            try:
                mid = _mid(pd.Series([c]))[0]
                if mid != c: d.rename(columns={c:mid}, inplace=True)
            except Exception:
                pass
        m = d["metric"].astype(str).str.strip()
        if metric_name not in m.values:
            return {}
        row = d.loc[m == metric_name].iloc[0]
        return {mm: float(pd.to_numeric(row.get(mm), errors="coerce")) for mm in month_ids}

    hc_plan_row   = _row_as_dict(hc,  "Planned/Tactical HC (#)")
    # Use monthly step snapshot (weekly-like) for actuals to align carry-forward with weekly view
    hc_actual_row = {m: float((_v if _v is not None else 0)) for m, _v in (hc_step_m.items() if 'hc_step_m' in locals() else {mm:0 for mm in month_ids}.items())}
    att_plan_row  = _row_as_dict(att, "Planned Attrition HC (#)")
    att_act_row   = _row_as_dict(att, "Actual Attrition HC (#)")
    att_plan_pct_row  = _row_as_dict(att, "Planned Attrition %")

    # Attrition to use per month: prefer Actual if present (>0), else Planned; apply What-If delta within window
    att_use_row = {}
    for m in month_ids:
        try:
            a = float(att_act_row.get(m, 0) or 0.0)
        except Exception:
            a = 0.0
        try:
            p = float(att_plan_row.get(m, 0) or 0.0)
        except Exception:
            p = 0.0
        base = a if a > 0 else p

        # If no HC is available, derive planned attrition HC from planned % and monthly HC baseline
        if (base == 0) and isinstance(att_plan_pct_row, dict):
            try:
                pct = float(att_plan_pct_row.get(m, 0) or 0.0)
            except Exception:
                pct = 0.0
            if pct > 0:
                if pct > 1.0:
                    pct = pct / 100.0
                # Denominator preference: actual HC -> planned HC snapshot -> budget
                try:
                    denom_f = float(hc_actual_row.get(m, 0.0) or 0.0)
                except Exception:
                    denom_f = 0.0
                if denom_f <= 0:
                    try:
                        denom_f = float(hc_plan_row.get(m, 0.0) or 0.0)
                    except Exception:
                        denom_f = 0.0
                if denom_f <= 0:
                    try:
                        denom_f = float(budget_m.get(m, 0.0) or 0.0)
                    except Exception:
                        denom_f = 0.0
                if denom_f > 0:
                    base = float(pct) * float(denom_f)
        # If no HC is available, derive planned attrition HC from planned % and monthly HC baseline
        if (base == 0) and isinstance(att_plan_pct_row, dict):
            try:
                pct = float(att_plan_pct_row.get(m, 0) or 0.0)
            except Exception:
                pct = 0.0
            if pct > 0:
                if pct > 1.0: pct = pct / 100.0
                # Denominator preference: planned HC -> actual HC snapshot -> budget
                try:
                    denom_f = float(hc_plan_row.get(m, 0.0) or 0.0)
                except Exception:
                    denom_f = 0.0
                if denom_f <= 0:
                    try:
                        denom_f = float(hc_actual_row.get(m, 0.0) or 0.0)
                    except Exception:
                        denom_f = 0.0
                if denom_f <= 0:
                    try:
                        denom_f = float(budget_m.get(m, 0.0) or 0.0)
                    except Exception:
                        denom_f = 0.0
                if denom_f > 0:
                    base = float(pct) * float(denom_f)
        if _wf_active_month(m):
            base += attr_delta
        att_use_row[m] = base

    # NH additions to use: actual for past/current, planned for future
    # NH additions: prefer Actual if present (>0), else Planned
    # Compute first-day joiners and joins-after-first for actual months
    first_day_joins_m = {m: 0 for m in month_ids}
    try:
        if isinstance(roster_df, pd.DataFrame) and not roster_df.empty:
            r2 = roster_df.copy()
            c = None
            for k in ("production_start", "prod_start", "to_production", "go_live"):
                if k in r2.columns:
                    c = k
                    break
            if c:
                try:
                    r2[c] = pd.to_datetime(r2[c], errors="coerce", format="mixed")
                except TypeError:
                    r2[c] = pd.to_datetime(r2[c], errors="coerce")
                r2 = r2.dropna(subset=[c])
                r2["month"] = _mid(r2[c])
                try:
                    first = pd.to_datetime(r2["month"], errors="coerce", format="%Y-%m-%d").dt.normalize()
                except Exception:
                    first = pd.to_datetime(r2["month"], errors="coerce").dt.normalize()
                on_first = (r2[c].dt.normalize() == first)
                g2 = r2.loc[on_first].groupby("month")[c].count().to_dict()
                first_day_joins_m = {m: int(g2.get(m, 0)) for m in month_ids}
    except Exception:
        first_day_joins_m = {m: 0 for m in month_ids}
    actual_nh_after_first_m = {m: max(0, int(actual_nh_m.get(m, 0)) - int(first_day_joins_m.get(m, 0))) for m in month_ids}

    nh_add_row = {}
    for m in month_ids:
        if m <= today_m:
            # Actual months: only add joiners after the 1st (snapshot already includes day-1)
            nh_add_row[m] = int(actual_nh_after_first_m.get(m, 0))
        else:
            # Future months: use planned joiners
            nh_add_row[m] = int(planned_nh_m.get(m, 0))

    projected_supply = {}
    prev = None
    _month_ids_cf = sorted(month_ids, key=lambda x: pd.to_datetime(x, errors="coerce"))
    for m in _month_ids_cf:
        if m <= today_m and float(hc_actual_row.get(m, 0) or 0.0) > 0.0:
            # Weekly-consistent behavior: use the actual snapshot for current/past months
            projected_supply[m] = float(hc_actual_row.get(m, 0) or 0.0)
            prev = projected_supply[m]
        else:
            # First future month: seed from Plan/Tactical HC; fallback to actual snapshot (no budget fallback in weekly)
            if prev is None:
                prev = float(hc_plan_row.get(m, 0) or 0.0)
                if prev <= 0:
                    prev = float(hc_actual_row.get(m, 0) or 0.0)
            next_val = max(prev - float(att_use_row.get(m, 0)) + float(nh_add_row.get(m, 0)), 0.0)
            projected_supply[m] = next_val
            prev = next_val

    # ---- Handling capacity & Projected SL (monthly) ----
    def _erlang_c(A: float, N: int) -> float:
        # Numerically stable Erlang C using recursive terms
        if N <= 0:
            return 1.0
        if A <= 0:
            return 0.0
        if A >= N:
            return 1.0
        term = 1.0
        s = term
        for k in range(1, N):
            term *= A / k
            s += term
        term *= A / N  # A^N / N!
        last = term * (N / (N - A))
        denom = s + last
        if denom <= 0:
            return 1.0
        p0 = 1.0 / denom
        return last * p0

    def _erlang_sl(calls_per_ivl: float, aht_sec: float, agents: float, asa_sec: int, ivl_sec: int) -> float:
        if aht_sec <= 0 or ivl_sec <= 0 or agents <= 0:
            return 0.0
        if calls_per_ivl <= 0:
            return 1.0
        A = (calls_per_ivl * aht_sec) / ivl_sec
        pw = _erlang_c(A, int(math.floor(agents)))
        return max(0.0, min(1.0, 1.0 - pw * math.exp(-max(0.0, (agents - A)) * (asa_sec / max(1.0, aht_sec)))))

    def _erlang_calls_capacity(agents: float, aht_sec: float, asa_sec: int, ivl_sec: int, target_pct: float) -> float:
        # If target SL is unattainable at max throughput, return throughput cap instead of zero
        if agents <= 0 or aht_sec <= 0 or ivl_sec <= 0:
            return 0.0
        target = float(target_pct) / 100.0
        def sl_for(x: int) -> float:
            return _erlang_sl(x, aht_sec, agents, asa_sec, ivl_sec)
        hi = max(1, int((agents * ivl_sec) / aht_sec))
        if sl_for(hi) < target:
            return float(hi)
        lo = 0
        while sl_for(hi) >= target and hi < 10_000_000:
            lo = hi
            hi *= 2
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if sl_for(mid) >= target:
                lo = mid
            else:
                hi = mid - 1
        return float(lo)

    handling_capacity = {}
    bo_model   = (settings.get("bo_capacity_model") or "tat").lower()
    bo_hpd     = float(settings.get("bo_hours_per_day", settings.get("hours_per_fte", 8.0)))
    util_bo    = float(settings.get("util_bo", 0.85))
    # base shrink (fraction)
    bo_shr_base = float(settings.get("bo_shrinkage_pct", settings.get("shrinkage_pct", 0.0)) or 0.0)
    if bo_shr_base > 1.0: bo_shr_base /= 100.0
    voice_shr_base = float(settings.get("voice_shrinkage_pct", settings.get("shrinkage_pct", 0.0)) or 0.0)
    if voice_shr_base > 1.0: voice_shr_base /= 100.0
    for m in month_ids:
        if ch_first.lower() == "voice":
            agents_prod = float(schedule_supply_avg_m.get(m, projected_supply.get(m, 0.0)))
            lc = _learning_curve_for_month(settings, lc_ovr_df, m)

            def eff_from_buckets(buckets, prod_pct_list, uplift_pct_list):
                total = 0.0
                login_f = _ovr_login_frac_m(m)
                aht_m   = _ovr_aht_mult_m(m)
                for age, cnt in (buckets.get(m, {}) or {}).items():
                    p = (prod_pct_list[age-1] / 100.0) if age-1 < len(prod_pct_list) else 0.0
                    u = (uplift_pct_list[age-1] / 100.0) if age-1 < len(uplift_pct_list) else 0.0
                    if login_f is not None: p *= login_f
                    denom = (1.0 + u)
                    if aht_m is not None:   denom *= aht_m
                    total += float(cnt) * (p / max(1.0, denom))
                return total
            
            nest_eff = eff_from_buckets(nest_buckets, lc["nesting_prod_pct"], lc["nesting_aht_uplift_pct"])
            sda_eff  = eff_from_buckets(sda_buckets,  lc["sda_prod_pct"],     lc["sda_aht_uplift_pct"])
            v_shr_add = (shrink_delta / 100.0) if (_wf_active_month(m) and shrink_delta) else 0.0
            v_eff_shr = min(0.99, max(0.0, voice_shr_base + v_shr_add))
            eff_agents = max(1.0, (agents_prod + nest_eff + sda_eff) * (1.0 - v_eff_shr))
            # Pick AHT: prefer Actual > Forecast; fallback to Planned/Target to avoid zero capacity
            _a = vA_aht.get(m, np.nan)
            _f = vF_aht.get(m, np.nan)
            try:
                av = float(_a) if (not pd.isna(_a) and _a > 0) else None
            except Exception:
                av = None
            try:
                fv = float(_f) if (not pd.isna(_f) and _f > 0) else None
            except Exception:
                fv = None
            aht = av if (av is not None) else (fv if (fv is not None) else float(planned_aht_m_voice.get(m, s_target_aht)))
            try:
                if _wf_active_month(m) and aht_delta:
                    aht = max(1.0, float(aht) * (1.0 + aht_delta / 100.0))
                else:
                    aht = max(1.0, float(aht))
            except Exception:
                aht = max(1.0, float(aht))
            intervals = monthly_voice_intervals.get(m, 0)
            if eff_agents <= 0.0 or intervals <= 0:
                handling_capacity[m] = 0.0
            else:
                calls_per_ivl = _erlang_calls_capacity(eff_agents, aht, sl_seconds, ivl_sec, sl_target_pct)
                # Occupancy cap as a limit on erlangs (not on agents)
                occ_cap_erlangs = float(occ_m.get(m, 85))
                occ_cap_erlangs = (occ_cap_erlangs/100.0) * eff_agents
                occ_calls_cap = (occ_cap_erlangs * ivl_sec) / max(1.0, aht)
                calls_per_ivl = min(calls_per_ivl, occ_calls_cap)
                handling_capacity[m] = calls_per_ivl * intervals
            # Optional debug for Voice monthly
            import os as _os
            if _os.environ.get("CAP_DEBUG_VOICE"):
                try:
                    print(
                        f"[VOICE-MONTH][{m}] vol(F/A/T)={vF_vol.get(m,0)}/{vA_vol.get(m,0)}/{vT_vol.get(m,0)} aht={aht:.1f} occ={occ_frac_m.get(m,0):.2f} "
                        f"agents_eff={agents_eff:.2f} intervals={intervals} cap={handling_capacity[m]:.1f}"
                    )
                except Exception:
                    pass
        else:
            # Backoffice
            # Only use SUT if an Actual or Forecast value exists; otherwise 0 (no capacity)
            _sa = bA_sut.get(m, np.nan)
            _sf = bF_sut.get(m, np.nan)
            if (pd.isna(_sa) or _sa <= 0) and (pd.isna(_sf) or _sf <= 0):   
                sut = 0.0
            else:
                sut = float(_sa) if (not pd.isna(_sa) and _sa > 0) else float(_sf)
                if _wf_active_month(m) and aht_delta:
                    sut = max(1.0, sut * (1.0 + aht_delta / 100.0))
                else:
                    sut = max(1.0, sut)
            lc = _learning_curve_for_month(settings, lc_ovr_df, m)

            def eff(buckets, prod_pct_list, uplift_pct_list):
                total = 0.0
                login_f = _ovr_login_frac_m(m)
                aht_m   = _ovr_aht_mult_m(m)
                for age, cnt in (buckets.get(m, {}) or {}).items():
                    p = (prod_pct_list[age-1] / 100.0) if age-1 < len(prod_pct_list) else 0.0
                    u = (uplift_pct_list[age-1] / 100.0) if age-1 < len(uplift_pct_list) else 0.0
                    if login_f is not None: p *= login_f
                    denom = (1.0 + u)
                    if aht_m is not None:   denom *= aht_m
                    total += float(cnt) * (p / max(1.0, denom))
                return total
            agents_eff = max(1.0, float(projected_supply.get(m, 0.0)) + eff(nest_buckets, lc["nesting_prod_pct"], lc["nesting_aht_uplift_pct"]) + eff(sda_buckets, lc["sda_prod_pct"], lc["sda_aht_uplift_pct"]))
            wd = _workdays_in_month(m, is_bo=True)
            if bo_model == "tat":
                shr_add = (shrink_delta / 100.0) if (_wf_active_month(m) and shrink_delta) else 0.0
                eff_shr = min(0.99, max(0.0, bo_shr_base + shr_add))
                base_prod_hours = wd * bo_hpd * (1.0 - eff_shr) * util_bo
                try:
                    # Prefer saved FW overtime; fallback to raw-derived monthly OT when not present
                    ot = float(overtime_m.get(m, _ot_m_raw.get(m, 0.0)) or 0.0)
                except Exception:
                    try:
                        ot = float(_ot_m_raw.get(m, 0.0) or 0.0)
                    except Exception:
                        ot = 0.0
                if sut <= 0.0 or agents_eff <= 0.0:
                    handling_capacity[m] = 0.0
                else:
                    total_prod_hours = (float(agents_eff) * base_prod_hours) + (max(0.0, ot) * (1.0 - eff_shr) * util_bo)
                    handling_capacity[m] = total_prod_hours * (3600.0 / sut)
            else:
                if sut <= 0.0 or agents_eff <= 0.0:
                    handling_capacity[m] = 0.0
                else:
                    ivl_per_month = int(round(wd * bo_hpd / (ivl_sec / 3600.0)))
                    items_per_ivl = _erlang_calls_capacity(max(1.0, float(agents_eff) * util_bo), sut, sl_seconds, ivl_sec, sl_target_pct)
                    handling_capacity[m] = items_per_ivl * max(1, ivl_per_month)

    # projected service level
    proj_sl = {}
    for m in month_ids:
        if ch_first.lower() == "voice":
            # Prefer Actual > Forecast > Tactical for demand
            try:
                av = float(vA_vol.get(m, 0.0) or 0.0)
            except Exception:
                av = 0.0
            if av > 0:
                monthly_load = av
            else:
                fv = float(vF_vol.get(m, 0.0) or 0.0)
                if fv > 0:
                    monthly_load = fv
                else:
                    monthly_load = float(vT_vol.get(m, 0.0) or 0.0)
            # Robust AHT pick: prefer Actual > Forecast > Target, handling None/NaN safely
            _aa = vA_aht.get(m, None)
            _fa = vF_aht.get(m, None)
            try:
                av = float(_aa) if _aa is not None and not pd.isna(_aa) else None
            except Exception:
                av = None
            try:
                fv = float(_fa) if _fa is not None and not pd.isna(_fa) else None
            except Exception:
                fv = None
            aht_sut = av if (av is not None and av > 0) else (fv if (fv is not None and fv > 0) else float(s_target_aht))
            try:
                if _wf_active_month(m) and aht_delta:
                    aht_sut = max(1.0, float(aht_sut) * (1.0 + aht_delta / 100.0))
                else:
                    aht_sut = max(1.0, float(aht_sut))
            except Exception:
                aht_sut = max(1.0, float(aht_sut))
            intervals = monthly_voice_intervals.get(m, 1)
            calls_per_ivl = monthly_load / float(max(1, intervals))
            lc = _learning_curve_for_month(settings, lc_ovr_df, m)

            def eff_from_buckets(buckets, prod_pct_list, uplift_pct_list):
                total = 0.0
                login_f = _ovr_login_frac_m(m)
                aht_m   = _ovr_aht_mult_m(m)
                for age, cnt in (buckets.get(m, {}) or {}).items():
                    p = (prod_pct_list[age-1] / 100.0) if age-1 < len(prod_pct_list) else 0.0
                    u = (uplift_pct_list[age-1] / 100.0) if age-1 < len(uplift_pct_list) else 0.0
                    if login_f is not None: p *= login_f
                    denom = (1.0 + u)
                    if aht_m is not None:   denom *= aht_m
                    total += float(cnt) * (p / max(1.0, denom))
                return total
            nest_eff = eff_from_buckets(nest_buckets, lc["nesting_prod_pct"], lc["nesting_aht_uplift_pct"])
            sda_eff  = eff_from_buckets(sda_buckets,  lc["sda_prod_pct"],     lc["sda_aht_uplift_pct"])
            v_shr_add = (shrink_delta / 100.0) if (_wf_active_month(m) and shrink_delta) else 0.0
            v_eff_shr = min(0.99, max(0.0, voice_shr_base + v_shr_add))
            # Use schedule-based average supply when available (align with capacity calc); fallback to projected HC
            agents_prod = schedule_supply_avg_m.get(m, None)
            if agents_prod is None or agents_prod <= 0:
                agents_prod = float(projected_supply.get(m, 0.0))
            agents_eff = max(1.0, (float(agents_prod) + nest_eff + sda_eff) * (1.0 - v_eff_shr))
            sl_frac = _erlang_sl(calls_per_ivl, max(1.0, float(aht_sut)), agents_eff, sl_seconds, ivl_sec)
            proj_sl[m] = 100.0 * sl_frac
            import os as _os2
            if _os2.environ.get("CAP_DEBUG_VOICE"):
                try:
                    print(
                        f"[VOICE-MONTH-SL][{m}] load={monthly_load:.1f} aht={aht_sut:.1f} intervals={intervals} agents_eff={agents_eff:.2f} sl={proj_sl[m]:.1f}"
                    )
                except Exception:
                    pass
        else:
            # Back Office and others: Actual > Forecast > Tactical items
            try:
                ab = float(bA_itm.get(m, 0.0) or 0.0)
            except Exception:
                ab = 0.0
            if ab > 0:
                monthly_load = ab
            else:
                fb = float(bF_itm.get(m, 0.0) or 0.0)
                if fb > 0:
                    monthly_load = fb
                else:
                    monthly_load = float(bT_itm.get(m, 0.0) or 0.0)
            cap = float(handling_capacity.get(m, 0.0))
            proj_sl[m] = 0.0 if monthly_load <= 0 else min(100.0, 100.0 * cap / monthly_load)

    # ---- Upper summary table (same rows) ----
    upper_df = _blank_grid(spec["upper"], month_ids)
    # For Back Office in monthly view, align FTE to linear monthly formula:
    # FTE = ((Monthly Items * SUT_sec) / 3600) / MonthlyHoursPerFTE / (1 - Shrink)
    # Use planned shrink for consistency with examples; can be toggled later if needed.
    is_bo_monthly = str(ch_first or '').strip().lower() in ("back office", "bo")
    weekly_hours = float(settings.get("weekly_hours", settings.get("weekly_hours_per_fte", 40.0)) or 40.0)
    monthly_hours = weekly_hours * (52.0/12.0)
    # Read monthly shrink rows (numeric, pre-formatting)
    def _row_as_dict_safe(df, metric_name):
        try:
            return _row_as_dict(df, metric_name)
        except Exception:
            return {m: np.nan for m in month_ids}
    shr_planned_pct_m = _row_as_dict_safe(shr, "Planned Shrinkage %")
    shr_actual_pct_m  = _row_as_dict_safe(shr, "Overall Shrinkage %")

    if ("FTE Required @ Forecast Volume" in spec["upper"]) and is_bo_monthly:
        for mm in month_ids:
            vol = float(bF_itm.get(mm, 0.0))
            sut = float(bF_sut.get(mm, s_target_sut))
            # What-If: apply AHT/SUT delta to requirement calc as well
            try:
                if _wf_active_month(mm) and aht_delta:
                    sut = max(1.0, sut * (1.0 + aht_delta / 100.0))
            except Exception:
                pass
            sh_pct = float(shr_planned_pct_m.get(mm, np.nan))
            sh_frac = (sh_pct/100.0) if (not pd.isna(sh_pct)) else float(planned_shrink_fraction)
            denom = max(1e-6, monthly_hours * float(util_bo) * max(0.01, 1.0 - sh_frac))
            fte = ((vol * sut) / 3600.0) / denom
            upper_df.loc[upper_df["metric"] == "FTE Required @ Forecast Volume", mm] = fte
    elif "FTE Required @ Forecast Volume" in spec["upper"]:
        for mm in month_ids:
            upper_df.loc[upper_df["metric"] == "FTE Required @ Forecast Volume", mm] = float(req_m_forecast.get(mm, 0.0))
    # Optional: FTE Required @ Queue (scale forecast requirement by Queue/Forecast load)
    if "FTE Required @ Queue" in spec["upper"]:
        for m in month_ids:
            try:
                fval = float(pd.to_numeric(fw.loc[fw["metric"] == "Forecast", m], errors="coerce").fillna(0.0).iloc[0]) if m in fw.columns else 0.0
            except Exception:
                fval = 0.0
            qval = float(queue_m.get(m, 0.0)) if isinstance(queue_m, dict) else 0.0
            base_req = float(req_m_forecast.get(m, 0.0))
            reqq = (base_req * (qval / fval)) if fval > 0 else 0.0
            upper_df.loc[upper_df["metric"] == "FTE Required @ Queue", m] = reqq
    if ("FTE Required @ Actual Volume" in spec["upper"]) and is_bo_monthly:
        for mm in month_ids:
            vol = float(bA_itm.get(mm, 0.0))
            # Prefer actual SUT; fallback to forecast/target
            sut = float(bA_sut.get(mm, bF_sut.get(mm, s_target_sut)))
            try:
                if _wf_active_month(mm) and aht_delta:
                    sut = max(1.0, sut * (1.0 + aht_delta / 100.0))
            except Exception:
                pass
            # By default use planned shrink to match provided examples; switch to 'shr_actual_pct_m' if desired
            sh_pct = float(shr_actual_pct_m.get(mm, np.nan))
            sh_frac = (sh_pct/100.0) if (not pd.isna(sh_pct)) else float(planned_shrink_fraction)
            denom = max(1e-6, monthly_hours * float(util_bo) * max(0.01, 1.0 - sh_frac))
            fte = ((vol * sut) / 3600.0) / denom
            upper_df.loc[upper_df["metric"] == "FTE Required @ Actual Volume", mm] = fte
    elif "FTE Required @ Actual Volume" in spec["upper"]:
        for mm in month_ids:
            upper_df.loc[upper_df["metric"] == "FTE Required @ Actual Volume", mm] = float(req_m_actual.get(mm, 0.0))
    if "FTE Over/Under MTP Vs Actual" in spec["upper"]:
        for mm in month_ids:
            if is_bo_monthly:
                try:
                    mtp = float(pd.to_numeric(upper_df.loc[upper_df["metric"] == "FTE Required @ Forecast Volume", mm], errors="coerce").fillna(0).iloc[0])
                    act = float(pd.to_numeric(upper_df.loc[upper_df["metric"] == "FTE Required @ Actual Volume", mm], errors="coerce").fillna(0).iloc[0])
                except Exception:
                    mtp, act = float(req_m_forecast.get(mm, 0.0)), float(req_m_actual.get(mm, 0.0))
                upper_df.loc[upper_df["metric"] == "FTE Over/Under MTP Vs Actual", mm] = mtp - act
            else:
                upper_df.loc[upper_df["metric"] == "FTE Over/Under MTP Vs Actual", mm] = float(req_m_forecast.get(mm, 0.0)) - float(req_m_actual.get(mm, 0.0))
    if "FTE Over/Under Tactical Vs Actual" in spec["upper"]:
        for mm in month_ids:
            upper_df.loc[upper_df["metric"] == "FTE Over/Under Tactical Vs Actual", mm] = float(req_m_tactical.get(mm, 0.0)) - float(req_m_actual.get(mm, 0.0))
    if "FTE Over/Under Budgeted Vs Actual" in spec["upper"]:
        for mm in month_ids:
            upper_df.loc[upper_df["metric"] == "FTE Over/Under Budgeted Vs Actual", mm] = float(req_m_budgeted.get(mm, 0.0)) - float(req_m_actual.get(mm, 0.0))
    if "Projected Supply HC" in spec["upper"]:
        for mm in month_ids:
            upper_df.loc[upper_df["metric"] == "Projected Supply HC", mm] = projected_supply.get(mm, 0.0)
    if "Projected Handling Capacity (#)" in spec["upper"]:
        for mm in month_ids:
            upper_df.loc[upper_df["metric"] == "Projected Handling Capacity (#)", mm] = handling_capacity.get(mm, 0.0)
    if "Projected Service Level" in spec["upper"]:
        for mm in month_ids:
            upper_df.loc[upper_df["metric"] == "Projected Service Level", mm] = proj_sl.get(mm, 0.0)
    # Precompute seat variance (pp) for monthly
    try:
        for m in month_ids:
            req = float(pd.to_numeric(seat.loc[seat["metric"] == "Seats Required (#)", m], errors="coerce").fillna(0).iloc[0]) if m in seat.columns else 0.0
            avail = float(pd.to_numeric(seat.loc[seat["metric"] == "Seats Available (#)", m], errors="coerce").fillna(0).iloc[0]) if m in seat.columns else 0.0
            seat_util = ((avail / req) * 100) if avail > 0 else 0.0
            var_val = (avail - req)
            if "Seat Utilization %" in seat["metric"].values:
                seat.loc[seat["metric"] == "Seat Utilization %", m] = seat_util
            if "Variance vs Available" in seat["metric"].values:
                seat.loc[seat["metric"] == "Variance vs Available", m] = var_val
    except Exception:
        pass

    # Re-override Seat from weekly averages right before formatting (ensure no earlier logic overwrites)
    try:
        seat_w2 = load_df(f"plan_{pid}_seat")
    except Exception:
        seat_w2 = None
    if isinstance(seat_w2, pd.DataFrame) and not seat_w2.empty and "metric" in seat_w2.columns:
        w_metrics = seat_w2["metric"].astype(str).str.strip().tolist()
        var_name_w = "Variance vs Available"
        metrics = ["Seats Required (#)", "Seats Available (#)"] + ([var_name_w] if var_name_w else [])
        def _weekly_to_month_avg_local(df: pd.DataFrame, metrics: list[str]) -> dict[str, dict[str, float]]:
            out = {mn: {mm: np.nan for mm in month_ids} for mn in metrics}
            if not isinstance(df, pd.DataFrame) or df.empty or "metric" not in df.columns:
                return out
            month_bucket = {mm: [] for mm in month_ids}
            for c in df.columns:
                if c == "metric": continue
                try:
                    mid = _mid(pd.Series([c]))[0]
                except Exception:
                    continue
                if mid in month_bucket:
                    month_bucket[mid].append(c)
            mser = df["metric"].astype(str).str.strip()
            for name in metrics:
                if name not in mser.values:
                    continue
                row = df.loc[mser == name].iloc[0]
                for mm, cols in month_bucket.items():
                    if not cols:
                        continue
                    try:
                        vals = pd.to_numeric(row[cols], errors="coerce")
                        # For Seat Utilization %, ignore zeros to avoid diluting averages when weeks are blank
                        if name in ("Seats Required (#)", "Seats Available (#)"):
                            vals = vals.replace(0, np.nan)
                        v = float(vals.dropna().mean()) if not vals.dropna().empty else np.nan
                    except Exception:
                        v = np.nan
                    out[name][mm] = v
            return out
        agg_seat2 = _weekly_to_month_avg_local(seat_w2, metrics)
        for m in month_ids:
            for nm in ("Seats Required (#)", "Seats Available (#)"):
                v = agg_seat2.get(nm, {}).get(m, np.nan)
                if not pd.isna(v):
                    seat.loc[seat["metric"] == nm, m] = v
            # Do not override monthly Variance from weekly; keep computed (req - avail)

    # ---- Align BvA (BO) with Upper linear formula ----
    try:
        is_bo_monthly = str(ch_first or '').strip().lower() in ("back office", "bo")
        if is_bo_monthly:
            # Helper to read a month->value dict for a metric from 'shr'
            def _metric_row_map(df: pd.DataFrame, metric_name: str) -> dict:
                if not isinstance(df, pd.DataFrame) or df.empty or "metric" not in df.columns:
                    return {}
                mser = df["metric"].astype(str).str.strip()
                if metric_name not in mser.values:
                    return {}
                row = df.loc[mser == metric_name].iloc[0]
                out = {}
                for mm in month_ids:
                    try:
                        out[mm] = float(pd.to_numeric(row.get(mm), errors="coerce"))
                    except Exception:
                        out[mm] = np.nan
                return out

            weekly_hours = float(settings.get("weekly_hours", settings.get("weekly_hours_per_fte", 40.0)) or 40.0)
            monthly_hours = weekly_hours * (52.0/12.0)
            shr_plan_pct = _metric_row_map(shr, "Planned Shrinkage %")
            shr_act_pct  = _metric_row_map(shr, "Overall Shrinkage %")

            for mm in month_ids:
                # Budgeted FTE (#): use forecast items and planned SUT
                vol_b = float(bF_itm.get(mm, 0.0))
                sut_b = float(bF_sut.get(mm, s_target_sut))
                sp = shr_plan_pct.get(mm, np.nan)
                sp = float(sp) if not pd.isna(sp) else float(planned_shrink_fraction * 100.0)
                sp_frac = (sp/100.0) if sp > 1.0 else max(0.0, float(sp))
                denom_b = max(1e-6, monthly_hours * float(util_bo) * max(0.01, 1.0 - sp_frac))
                bud_linear = ((vol_b * sut_b) / 3600.0) / denom_b

                # Actual FTE (#): use actual items and actual SUT; shrink = Overall Shrinkage %
                vol_a = float(bA_itm.get(mm, 0.0))
                sut_a = float(bA_sut.get(mm, bF_sut.get(mm, s_target_sut)))
                sa = shr_act_pct.get(mm, np.nan)
                sa = float(sa) if not pd.isna(sa) else float(planned_shrink_fraction * 100.0)
                sa_frac = (sa/100.0) if sa > 1.0 else max(0.0, float(sa))
                denom_a = max(1e-6, monthly_hours * float(util_bo) * max(0.01, 1.0 - sa_frac))
                act_linear = ((vol_a * sut_a) / 3600.0) / denom_a

                bva.loc[bva["metric"] == "Budgeted FTE (#)", mm] = bud_linear
                bva.loc[bva["metric"] == "Actual FTE (#)",   mm] = act_linear
                bva.loc[bva["metric"] == "Variance (#)",     mm] = act_linear - bud_linear
    except Exception:
        pass

    # ---- rounding & display formatting ----
    def _round_cols_int(df, col_ids):  # month friendly
        if not isinstance(df, pd.DataFrame) or df.empty: return df
        out = df.copy()
        for c in col_ids:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).round(0).astype(int)
        return out

    fw_to_use = _round_cols_int(fw, month_ids)
    hc        = _round_cols_int(hc, month_ids)

    # Mixed rounding for attr/ratio/seat: keep %/Ratio/Variance at 1-dec
    def _round_mixed(df):
        if not isinstance(df, pd.DataFrame) or df.empty: return df
        out = df.copy()
        for c in month_ids:
            if c not in out.columns: continue
            m = out["metric"].astype(str).str.lower()
            is_pct = m.str.contains("%") | m.str.contains("ratio") | m.str.contains("variance")
            out.loc[~is_pct, c] = pd.to_numeric(out.loc[~is_pct, c], errors="coerce").fillna(0.0).round(0).astype(int)
            out.loc[is_pct,  c] = pd.to_numeric(out.loc[is_pct,  c], errors="coerce").fillna(0.0).round(1)
        return out

    att       = _round_mixed(att)
    trn       = _round_cols_int(trn, month_ids)
    rat       = _round_mixed(rat)
    seat      = _round_mixed(seat)
    bva       = _round_cols_int(bva, month_ids)
    nh        = _round_cols_int(nh, month_ids)

    def _format_shrinkage_month(df):
        if not isinstance(df, pd.DataFrame) or df.empty: return df
        out = df.copy()
        mser = out["metric"].astype(str)
        pct_rows = mser.str.contains("Shrinkage %", regex=False) | mser.str.contains("Variance vs Planned", regex=False)
        hr_rows  = out["metric"].astype(str).str.contains("Hours (#)",   regex=False)

        for c in month_ids:
            if c in out.columns: out[c] = out[c].astype(object)
        for c in month_ids:
            if c not in out.columns: continue
            out.loc[hr_rows,  c] = pd.to_numeric(out.loc[hr_rows,  c], errors="coerce").fillna(0).round(0).astype(int)
            vals = pd.to_numeric(out.loc[pct_rows, c], errors="coerce").fillna(0.0)
            out.loc[pct_rows, c] = vals.round(1).astype(str) + "%"
        return out

    def _format_attrition_month(df):
        if not isinstance(df, pd.DataFrame) or df.empty: return df
        out = df.copy()
        m = out["metric"].astype(str)
        pct_rows = m.str.lower().str.contains("attrition %") | m.str.lower().str.contains("variance vs planned")
        hc_rows  = m.str.contains("HC (#)", regex=False)

        for c in month_ids:
            if c in out.columns: out[c] = out[c].astype(object)
        for c in month_ids:
            if c not in out.columns: continue
            out.loc[hc_rows, c] = pd.to_numeric(out.loc[hc_rows, c], errors='coerce').fillna(0).round(0).astype(int)
            vals = pd.to_numeric(out.loc[pct_rows, c], errors='coerce').fillna(0.0)
            out.loc[pct_rows, c] = vals.round(1).astype(str) + '%'
        return out

    shr_display = _format_shrinkage_month(shr)
    att_display = _format_attrition_month(att)

    # Seat display: add % sign to Seat Utilization %
    def _format_seat_month(df):
        if not isinstance(df, pd.DataFrame) or df.empty: return df
        out = df.copy()
        mser = out["metric"].astype(str)
        util_rows = mser.str.contains("Seat Utilization %", regex=False)
        for c in month_ids:
            if c in out.columns: out[c] = out[c].astype(object)
        for c in month_ids:
            if c not in out.columns: continue
            vals = pd.to_numeric(out.loc[util_rows, c], errors="coerce").fillna(0.0)
            out.loc[util_rows, c] = vals.round(1).astype(str) + "%"
        return out
    seat_display = _format_seat_month(seat)

    # format upper: SL one decimal, others int
    if isinstance(upper_df, pd.DataFrame) and not upper_df.empty:
        for c in month_ids:
            if c not in upper_df.columns: continue
            mask_sl = upper_df["metric"].eq("Projected Service Level")
            mask_not_sl = ~mask_sl
            upper_df.loc[mask_sl, c] = pd.to_numeric(upper_df.loc[mask_sl, c], errors="coerce").fillna(0.0).round(1)
            upper_df.loc[mask_not_sl, c] = pd.to_numeric(upper_df.loc[mask_not_sl, c], errors="coerce").fillna(0.0).round(0).astype(int)

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
