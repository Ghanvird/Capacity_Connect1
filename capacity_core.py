# capacity_core.py — core math + samples + KPIs
from __future__ import annotations
import math, datetime as dt
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from itertools import cycle
from cap_store import load_timeseries
# --- ADD: helpers + precise rollups ------------------------------------------
import re

def _to_frac(x) -> float:
    """Convert percent-like values to fraction.
    Accepts numbers (e.g., 0.3, 30) and strings (e.g., "30%", "30").
    Returns a float in 0..1 range; falls back to 0.0 on parse failure.
    """
    try:
        v = float(x)
        return v/100.0 if v > 1.0 else v
    except Exception:
        try:
            s = str(x).strip()
            if s.endswith('%'):
                s = s[:-1]
            v = float(s)
            return v/100.0 if v > 1.0 else v
        except Exception:
            return 0.0

def week_floor(d: pd.Timestamp | str | dt.date, week_start: str = "Monday") -> dt.date:
    d = pd.to_datetime(d).date()
    wd = d.weekday()  # Mon=0..Sun=6
    if (week_start or "Monday").lower().startswith("sun"):
        # week starts on Sunday
        return d - dt.timedelta(days=(wd + 1) % 7)
    return d - dt.timedelta(days=wd)

def add_week_month_keys(df: pd.DataFrame, date_col: str, week_start: str = "Monday") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.date
    out["week"] = out[date_col].apply(lambda x: week_floor(x, week_start))
    out["month"] = pd.to_datetime(out[date_col]).dt.to_period("M").dt.to_timestamp().dt.date
    return out

def _ivl_minutes_from_str(s: str | int | float, default: int = 30) -> int:
    """Infer interval minutes from '09:00-09:30' etc.; else return default."""
    try:
        if isinstance(s, (int, float)) and not pd.isna(s):
            v = int(s)
            return v if v > 0 else default
        st = str(s)
        m = re.search(r"(\d{1,2}):(\d{2})\s*(?:-\s*(\d{1,2}):(\d{2}))?", st)
        if m:
            h1, m1 = int(m.group(1)), int(m.group(2))
            if m.group(3):
                h2, m2 = int(m.group(3)), int(m.group(4))
                t1 = h1*60 + m1; t2 = h2*60 + m2
                diff = (t2 - t1) % (24*60)
                return diff if diff>0 else default
    except Exception:
        pass
    return int(default)

def voice_requirements_interval(voice_df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """
    Input: interval rows with columns like ['date','interval','volume/calls','aht_sec', optional 'program'].
    Output per interval:
      ['date','interval','program','calls','aht_sec','A_erlangs','agents_req','service_level','occupancy','asa_sec','staff_seconds']
    """
    if not isinstance(voice_df, pd.DataFrame) or voice_df.empty:
        return pd.DataFrame(columns=["date","interval","program","calls","aht_sec","A_erlangs","agents_req","service_level","occupancy","asa_sec","staff_seconds"])

    df = voice_df.copy()
    L = {c.lower(): c for c in df.columns}
    date_c = L.get("date")
    ivl_c  = L.get("interval") or L.get("interval_start") or L.get("time")
    calls_c= L.get("calls") or L.get("volume")
    aht_c  = L.get("aht_sec") or L.get("aht (sec)") or L.get("aht")
    prog_c = L.get("program") or L.get("business area")

    if not all([date_c, ivl_c, calls_c, aht_c]):
        return pd.DataFrame(columns=["date","interval","program","calls","aht_sec","A_erlangs","agents_req","service_level","occupancy","asa_sec","staff_seconds"])

    ivl_min  = int(settings.get("interval_minutes", 30) or 30)
    target_sl= float(settings.get("target_sl", 0.8) or 0.8)
    T_sec    = float(settings.get("sl_seconds", 20) or 20)
    occ_cap  = float(settings.get("occupancy_cap_voice", 0.85) or 0.85)

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_c], errors="coerce").dt.date,
        "interval": df[ivl_c].astype(str),
        "calls": pd.to_numeric(df[calls_c], errors="coerce"),
        "aht_sec": pd.to_numeric(df[aht_c], errors="coerce"),
        "program": (df[prog_c].astype(str) if prog_c else "All"),
    }).dropna(subset=["date","interval","calls","aht_sec"])

    # infer interval length per-row (supports mixed 15/30/60)
    ivl_sec = out["interval"].map(lambda s: _ivl_minutes_from_str(s, ivl_min)).fillna(ivl_min).astype(int) * 60

    rows = []
    for i, r in out.iterrows():
        calls = float(r["calls"] or 0.0)
        aht   = float(r["aht_sec"] or 0.0)
        ivalsec = int(ivl_sec.iloc[i])
        ivlmin_row = max(5, round(ivalsec/60))
        A = offered_load_erlangs(calls, aht, ivlmin_row)
        N, sl, occ, asa_val = min_agents(calls, aht, ivlmin_row, target_sl, T_sec, occ_cap)
        rows.append({
            "date": r["date"], "interval": r["interval"], "program": r["program"],
            "calls": calls, "aht_sec": aht,
            "A_erlangs": A, "agents_req": N, "service_level": sl, "occupancy": occ, "asa_sec": asa_val,
            "staff_seconds": N * ivalsec
        })
    res = pd.DataFrame(rows)
    return res[["date","interval","program","calls","aht_sec","A_erlangs","agents_req","service_level","occupancy","asa_sec","staff_seconds"]]

def voice_rollups(voice_ivl: pd.DataFrame, settings: dict, week_start: str = "Monday"):
    """Interval → day/week/month FTE for Voice."""
    if voice_ivl is None or voice_ivl.empty:
        empty = pd.DataFrame(columns=["date","program","fte_req"])
        return {"interval": pd.DataFrame(), "daily": empty, "weekly": empty, "monthly": empty}

    shrink = float(settings.get("shrinkage_pct", 0.30) or 0.30)
    hrs    = float(settings.get("hours_per_fte", 8.0) or 8.0)
    denom  = hrs * 3600.0 * (1.0 - shrink)

    base = voice_ivl.copy()
    base["date"] = pd.to_datetime(base["date"]).dt.date
    daily = (base.groupby(["date","program"], as_index=False)["staff_seconds"].sum())
    daily["fte_req"] = daily["staff_seconds"] / max(1e-6, denom)
    daily = daily[["date","program","fte_req"]].sort_values(["date","program"])

    wk = add_week_month_keys(daily, "date", week_start)
    weekly  = wk.groupby(["week","program"], as_index=False)["fte_req"].sum().rename(columns={"week":"start_week"})
    monthly = wk.groupby(["month","program"], as_index=False)["fte_req"].sum().rename(columns={"month":"month_start"})
    return {"interval": base, "daily": daily, "weekly": weekly, "monthly": monthly}

def bo_rollups(bo_df: pd.DataFrame, settings: dict, week_start: str = "Monday"):
    """Day → week/month FTE for Back Office."""
    if bo_df is None or bo_df.empty:
        empty = pd.DataFrame(columns=["date","program","fte_req"])
        return {"daily": empty, "weekly": empty, "monthly": empty}

    hrs    = float(settings.get("hours_per_fte", 8.0) or 8.0)
    shrink = float(settings.get("shrinkage_pct", 0.30) or 0.30)
    util   = float(settings.get("util_bo", 0.85) or 0.85)
    denom  = hrs * 3600.0 * (1.0 - shrink) * util

    df = bo_df.copy()
    L = {c.lower(): c for c in df.columns}
    date_c = L.get("date")
    items_c= L.get("items") or L.get("volume")
    aht_c  = L.get("aht_sec") or L.get("sut_sec") or L.get("sut")
    prog_c = L.get("program") or L.get("business area")

    if not all([date_c, items_c, aht_c]):
        empty = pd.DataFrame(columns=["date","program","fte_req"])
        return {"daily": empty, "weekly": empty, "monthly": empty}

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_c], errors="coerce").dt.date,
        "items": pd.to_numeric(df[items_c], errors="coerce"),
        "aht_sec": pd.to_numeric(df[aht_c], errors="coerce"),
        "program": (df[prog_c].astype(str) if prog_c else "All"),
    }).dropna(subset=["date","items","aht_sec"])

    out["work_seconds"] = out["items"] * out["aht_sec"]
    daily = (out.groupby(["date","program"], as_index=False)["work_seconds"].sum())
    daily["fte_req"] = daily["work_seconds"] / max(1e-6, denom)
    daily = daily[["date","program","fte_req"]].sort_values(["date","program"])

    wk = add_week_month_keys(daily, "date", week_start)
    weekly  = wk.groupby(["week","program"], as_index=False)["fte_req"].sum().rename(columns={"week":"start_week"})
    monthly = wk.groupby(["month","program"], as_index=False)["fte_req"].sum().rename(columns={"month":"month_start"})
    return {"daily": daily, "weekly": weekly, "monthly": monthly}


def chat_fte_daily(chat_df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Daily FTE for Chat using Erlang C with concurrency adjustments."""
    if not isinstance(chat_df, pd.DataFrame) or chat_df.empty:
        return pd.DataFrame(columns=["date","program","chat_fte"])

    hrs = float(settings.get("hours_per_fte", 8.0) or 8.0)
    shrink = float(settings.get("chat_shrinkage_pct", settings.get("shrinkage_pct", 0.30)) or 0.30)
    target_sl = float(settings.get("chat_target_sl", settings.get("target_sl", 0.8)) or 0.8)
    T_sec = float(settings.get("chat_sl_seconds", settings.get("sl_seconds", 20)) or 20.0)
    occ_cap = settings.get("occupancy_cap_chat")
    if occ_cap is None:
        occ_cap = settings.get("util_chat", settings.get("occupancy_cap_voice", settings.get("occupancy_cap", 0.85)))
    occ_cap = float(occ_cap or 0.85)
    conc = float(settings.get("chat_concurrency", 1.5) or 1.0)
    default_interval = float(settings.get("chat_interval_minutes", settings.get("interval_minutes", 30)) or 30.0)
    coverage_min = float(settings.get("chat_coverage_minutes", hrs * 60.0) or (hrs * 60.0))

    d = chat_df.copy()
    L = {str(c).strip().lower(): c for c in d.columns}
    date_c = L.get("date")
    items_c = L.get("items") or L.get("volume")
    aht_c = L.get("aht_sec") or L.get("aht")
    prog_c = L.get("program") or L.get("business area") or L.get("journey")
    interval_c = L.get("interval") or L.get("time")
    ivl_minutes_c = L.get("interval_minutes") or L.get("interval_mins")

    if not date_c or not items_c:
        return pd.DataFrame(columns=["date","program","chat_fte"])

    out = pd.DataFrame({
        "date": pd.to_datetime(d[date_c], errors="coerce").dt.date,
        "items": pd.to_numeric(d[items_c], errors="coerce"),
    })

    default_aht = float(settings.get("chat_aht_sec", settings.get("target_aht", 240)) or 240.0)
    if aht_c:
        out["aht_sec"] = pd.to_numeric(d[aht_c], errors="coerce").fillna(default_aht)
    else:
        out["aht_sec"] = default_aht

    if prog_c:
        out["program"] = d[prog_c].astype(str).replace("", "Chat")
    else:
        out["program"] = "Chat"

    out["interval_minutes"] = coverage_min
    if interval_c:
        out["interval_minutes"] = d[interval_c].map(lambda s: _ivl_minutes_from_str(s, default_interval)).fillna(coverage_min)
    if ivl_minutes_c:
        out["interval_minutes"] = pd.to_numeric(d[ivl_minutes_c], errors="coerce").fillna(out["interval_minutes"])

    out = out.dropna(subset=["date"]).fillna({"items": 0.0, "aht_sec": default_aht})
    if out.empty:
        return pd.DataFrame(columns=["date","program","chat_fte"])

    eff_conc = max(conc, 1e-6)

    def _staff_seconds(row: pd.Series) -> float:
        calls = float(row.get("items", 0.0) or 0.0)
        if calls <= 0:
            return 0.0
        aht = float(row.get("aht_sec", default_aht) or 0.0) / eff_conc
        if aht <= 0:
            return 0.0
        ivl_min = float(row.get("interval_minutes", coverage_min) or coverage_min)
        if not ivl_min or ivl_min <= 0:
            ivl_min = coverage_min
        ivl_min = max(1.0, ivl_min)
        agents, _, _, _ = min_agents(calls, aht, int(round(ivl_min)), target_sl, T_sec, occ_cap)
        return agents * ivl_min * 60.0

    out["_staff_seconds"] = out.apply(_staff_seconds, axis=1)
    agg = out.groupby(["date","program"], as_index=False)["_staff_seconds"].sum()
    denom = hrs * 3600.0 * max(1e-6, (1.0 - shrink))
    agg["chat_fte"] = agg["_staff_seconds"] / denom if denom > 0 else 0.0
    return agg[["date","program","chat_fte"]]

def required_fte_daily(voice_df: pd.DataFrame, bo_df: pd.DataFrame, ob_df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """
    Backward-compatible daily totals across Voice, Back Office and Outbound.
    Voice: interval→day using agent-seconds→FTE conversion (time-weighted).
    BO:    TAT model (items×SUT / productive-seconds) if bo_capacity_model='tat', else bo_rollups fallback.
    OB:    Erlang staffing on expected connects.
    Returns: ['date','program','voice_fte','bo_fte','ob_fte','total_req_fte']
    """
    frames = []

    # Voice (unchanged)
    try:
        vi   = voice_requirements_interval(voice_df, settings)
        vday = voice_rollups(vi, settings)["daily"].rename(columns={"fte_req": "voice_fte"})
        frames.append(vday)
    except Exception:
        pass

    # --- Back Office (TAT) ----------------------------------------------------
    def _bo_daily_tat(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(columns=["date","program","bo_fte"])
        d = df.copy()

        # tolerant column picks
        cols = {c.lower(): c for c in d.columns}
        c_date = cols.get("date")
        c_prog = cols.get("program") or cols.get("journey") or cols.get("ba") or cols.get("business area")
        c_items = (cols.get("items") or cols.get("txns") or cols.get("transactions") or cols.get("volume"))
        c_sut   = (cols.get("sut") or cols.get("sut_sec") or cols.get("aht_sec") or cols.get("avg_sut"))

        # defaults if missing
        if not c_sut:
            # fall back to settings target/budget
            sut = float(settings.get("target_sut", settings.get("budgeted_sut", 600)) or 600.0)
            d["_sut_sec_"] = float(sut)
            c_sut = "_sut_sec_"
        if not c_items:
            # nothing to compute
            return pd.DataFrame(columns=["date","program","bo_fte"])

        d[c_sut]   = pd.to_numeric(d[c_sut], errors="coerce").fillna(0.0)
        d[c_items] = pd.to_numeric(d[c_items], errors="coerce").fillna(0.0)
        d["date"]  = pd.to_datetime(d[c_date]).dt.date if c_date else pd.to_datetime(d.get("date")).dt.date

        # TAT denominator: productive seconds per FTE per day
        bo_hpd  = float(settings.get("bo_hours_per_day", settings.get("hours_per_fte", 8.0)))
        bo_shr  = float(settings.get("bo_shrinkage_pct", settings.get("shrinkage_pct", 0.0)))
        util_bo = float(settings.get("util_bo", 0.85))
        denom_day = max(bo_hpd * 3600.0 * (1.0 - bo_shr) * util_bo, 1e-6)

        d["bo_fte"] = (d[c_items] * d[c_sut]) / denom_day
        if not c_prog:
            d["_program_"] = "Back Office"
            c_prog = "_program_"

        g = d.groupby(["date", c_prog], as_index=False)["bo_fte"].sum()
        g = g.rename(columns={c_prog: "program"})
        return g

    try:
        model = str(settings.get("bo_capacity_model", "tat")).lower()
        if model == "tat":
            bday = _bo_daily_tat(bo_df, settings)
            frames.append(bday)
        else:
            # keep existing behavior if you explicitly choose a non-TAT model
            bday = bo_rollups(bo_df, settings)["daily"].rename(columns={"fte_req": "bo_fte"})
            frames.append(bday)
    except Exception:
        pass



    # Outbound (OPC x Connect Rate x RPC Rate) via Erlang staffing
    if isinstance(ob_df, pd.DataFrame) and not ob_df.empty:
        d = ob_df.copy()
        cols = {c.lower(): c for c in d.columns}
        c_date = cols.get("date")
        c_prog = cols.get("program") or cols.get("business area") or cols.get("journey")
        c_opc = cols.get("opc") or cols.get("dials") or cols.get("calls") or cols.get("attempts")
        c_conn = cols.get("connect_rate") or cols.get("connect%") or cols.get("connect pct") or cols.get("connect")
        c_rpc = cols.get("rpc")
        c_rpc_r = cols.get("rpc_rate") or cols.get("rpc%") or cols.get("rpc pct")
        c_aht = cols.get("aht_sec") or cols.get("talk_sec") or cols.get("avg_talk_sec")
        c_interval = cols.get("interval") or cols.get("time")
        c_ivl_min = cols.get("interval_minutes") or cols.get("interval_mins")

        d["date"] = pd.to_datetime(d[c_date], errors="coerce").dt.date if c_date else pd.to_datetime(d.get("date"), errors="coerce").dt.date
        if c_opc:
            d[c_opc] = pd.to_numeric(d[c_opc], errors="coerce").fillna(0.0)
        if c_rpc:
            d[c_rpc] = pd.to_numeric(d[c_rpc], errors="coerce").fillna(0.0)
        if c_conn:
            d[c_conn] = d[c_conn].map(_to_frac)
        if c_rpc_r:
            d[c_rpc_r] = d[c_rpc_r].map(_to_frac)

        default_aht = float(settings.get("ob_aht_sec", settings.get("target_aht", 240)) or 240.0)
        if c_aht:
            d[c_aht] = pd.to_numeric(d[c_aht], errors="coerce").fillna(default_aht)
        else:
            c_aht = "_aht_ob_"
            d[c_aht] = default_aht

        target_sl_ob = float(settings.get("ob_target_sl", settings.get("target_sl", 0.8)) or 0.8)
        T_sec_ob = float(settings.get("ob_sl_seconds", settings.get("sl_seconds", 20)) or 20.0)
        occ_cap_ob = settings.get("occupancy_cap_ob")
        if occ_cap_ob is None:
            occ_cap_ob = settings.get("util_ob", settings.get("occupancy_cap_voice", settings.get("occupancy_cap", 0.85)))
        occ_cap_ob = float(occ_cap_ob or 0.85)
        shrink_ob = float(settings.get("ob_shrinkage_pct", settings.get("shrinkage_pct", 0.30)) or 0.30)
        hrs = float(settings.get("hours_per_fte", 8.0) or 8.0)
        default_interval = float(settings.get("ob_interval_minutes", settings.get("interval_minutes", 30)) or 30.0)
        coverage_min = float(settings.get("ob_coverage_minutes", hrs * 60.0) or (hrs * 60.0))

        def _expected_contacts(row: pd.Series) -> float:
            opc = float(row.get(c_opc, 0.0) if c_opc else 0.0)
            rpc_ct = float(row.get(c_rpc, 0.0) if c_rpc else 0.0)
            conn = _to_frac(row.get(c_conn, 0.0)) if c_conn else None
            rpc_rate = _to_frac(row.get(c_rpc_r, 0.0)) if c_rpc_r else None
            if rpc_ct and rpc_ct > 0:
                return rpc_ct
            if conn is not None and rpc_rate is not None:
                return opc * conn * rpc_rate
            if conn is not None:
                return opc * conn
            return opc

        d["_contacts"] = d.apply(_expected_contacts, axis=1).astype(float)
        d["_interval_minutes"] = coverage_min
        if c_interval:
            d["_interval_minutes"] = d[c_interval].map(lambda s: _ivl_minutes_from_str(s, default_interval)).fillna(coverage_min)
        if c_ivl_min:
            d["_interval_minutes"] = pd.to_numeric(d[c_ivl_min], errors="coerce").fillna(d["_interval_minutes"])

        if not c_prog:
            d["_program_"] = "Outbound"
            c_prog = "_program_"
        else:
            d[c_prog] = d[c_prog].astype(str).replace("", "Outbound")

        def _staff_seconds(row: pd.Series) -> float:
            calls = float(row.get("_contacts", 0.0) or 0.0)
            if calls <= 0:
                return 0.0
            aht = float(row.get(c_aht, default_aht) or 0.0)
            if aht <= 0:
                return 0.0
            ivl_min = float(row.get("_interval_minutes", coverage_min) or coverage_min)
            if not ivl_min or ivl_min <= 0:
                ivl_min = coverage_min
            ivl_min = max(1.0, ivl_min)
            agents, _, _, _ = min_agents(calls, aht, int(round(ivl_min)), target_sl_ob, T_sec_ob, occ_cap_ob)
            return agents * ivl_min * 60.0

        d["_staff_seconds"] = d.apply(_staff_seconds, axis=1)
        g = d.groupby(["date", c_prog], as_index=False)["_staff_seconds"].sum()
        g["date"] = pd.to_datetime(g["date"], errors="coerce").dt.date
        g = g.dropna(subset=["date"])
        denom = hrs * 3600.0 * max(1e-6, (1.0 - shrink_ob))
        g["ob_fte"] = g["_staff_seconds"] / denom if denom > 0 else 0.0
        g = g.rename(columns={c_prog: "program"})[["date","program","ob_fte"]]
        frames.append(g)
    if not frames:
        return pd.DataFrame(columns=["date","program","voice_fte","bo_fte","ob_fte","total_req_fte"])

    out = frames[0]
    for f in frames[1:]:
        out = pd.merge(out, f, on=["date","program"], how="outer")

    for c in ["voice_fte","bo_fte","ob_fte"]:
        if c not in out: out[c] = 0.0
    out["total_req_fte"] = out[["voice_fte","bo_fte","ob_fte"]].fillna(0).sum(axis=1)
    return out.fillna(0)


# ─── Erlang / Queueing ───────────────────────────────────────
def erlang_b(A: float, N: int) -> float:
    if N <= 0: return 1.0
    B = 1.0
    for n in range(1, N+1):
        B = (A * B) / (n + A * B)
    return B

def erlang_c(A: float, N: int) -> float:
    if N <= 0: return 1.0
    if A <= 0: return 0.0
    if N <= A: return 1.0
    rho = A / N
    B = erlang_b(A, N)
    denom = 1 - rho + rho * B
    if denom <= 0: return 1.0
    return B / denom

def service_level(A: float, N: int, aht_sec: float, T_sec: float) -> float:
    if N <= 0: return 0.0
    if A <= 0: return 1.0
    pw = erlang_c(A, N)
    gap = N - A
    if gap <= 0: return 0.0
    return 1.0 - pw * math.exp(-gap * (T_sec / max(aht_sec, 1e-9)))

def asa(A: float, N: int, aht_sec: float) -> float:
    if N <= 0 or A <= 0 or N <= A: return float("inf")
    pw = erlang_c(A, N)
    return (pw * aht_sec) / (N - A)

def offered_load_erlangs(calls: float, aht_sec: float, interval_minutes: int) -> float:
    interval_minutes = max(5, int(interval_minutes or 30))
    return (calls * aht_sec) / (interval_minutes * 60.0)

def min_agents(calls: float, aht_sec: float, ivl_min: int, target_sl: float, T: float,
               occ_cap: Optional[float] = None, asa_cap: Optional[float] = None, Ncap: int = 2000) -> Tuple[int,float,float,float]:
    A = offered_load_erlangs(calls, aht_sec, ivl_min)
    if A <= 0: return 0, 1.0, 0.0, 0.0
    start = max(1, math.ceil(A))
    for N in range(start, min(start+1000, Ncap)):
        sl = service_level(A, N, aht_sec, T)
        occ = A / N
        _asa = asa(A, N, aht_sec)
        ok = True
        if occ_cap is not None and occ > occ_cap: ok = False
        if target_sl is not None and sl < target_sl: ok = False
        if asa_cap is not None and _asa > asa_cap: ok = False
        if ok: return N, sl, occ, _asa
    N = min(start+1000, Ncap) - 1
    return N, service_level(A, N, aht_sec, T), A/max(N,1), asa(A, N, aht_sec)

# ─── Samples ─────────────────────────────────────────────────
def make_projects_sample() -> pd.DataFrame:
    """
    Return a tiny summary table for the Home page: Business Area vs Active Plans.
    Active = rows where (is_current=1 OR status='current'), excluding soft-deleted rows
    (is_deleted=1 or deleted_at NOT NULL) when those columns exist.
    """
    try:
        from cap_store import _conn  # existing DB connector

        with _conn() as cx:
            # Detect soft-delete columns safely
            col_rows = cx.execute("PRAGMA table_info(capacity_plans)").fetchall()
            cols = {r[1] if isinstance(r, tuple) else r["name"] for r in col_rows}

            where = [
                "(is_current = 1) OR (LOWER(COALESCE(status,'')) = 'current')",
            ]
            if "is_deleted" in cols:
                where.append("COALESCE(is_deleted,0) = 0")
            elif "deleted_at" in cols:
                where.append("deleted_at IS NULL")

            sql = (
                "SELECT COALESCE(vertical, '') AS vertical, COUNT(*) AS cnt "
                "FROM capacity_plans "
                f"WHERE {' AND '.join(f'({w})' for w in where)} "
                "GROUP BY vertical "
                "ORDER BY vertical"
            )
            rows = cx.execute(sql).fetchall()

        data = [
            {"Business Area": (r[0] if isinstance(r, tuple) else r["vertical"]) or "—",
             "Active Plans": int(r[1] if isinstance(r, tuple) else r["cnt"]) }
            for r in rows
        ]
        return pd.DataFrame(data, columns=["Business Area", "Active Plans"])

    except Exception:
        # Fallback to plan_store API (already excludes deleted rows)
        try:
            from plan_store import list_plans
            rows = list_plans(status_filter="current") or []
            if not rows:
                return pd.DataFrame(columns=["Business Area", "Active Plans"])
            df = pd.DataFrame(rows)
            if "is_current" in df.columns:
                df = df[df["is_current"] == 1]
            elif "status" in df.columns:
                df = df[df["status"].astype(str).str.lower() == "current"]
            if df.empty:
                return pd.DataFrame(columns=["Business Area", "Active Plans"])
            ba_col = "vertical" if "vertical" in df.columns else (
                "business_area" if "business_area" in df.columns else None
            )
            if not ba_col:
                return pd.DataFrame(columns=["Business Area", "Active Plans"])
            grp = (
                df.groupby(ba_col)
                  .size()
                  .reset_index(name="Active Plans")
                  .rename(columns={ba_col: "Business Area"})
                  .sort_values("Business Area")
            )
            return grp[["Business Area", "Active Plans"]]
        except Exception:
            return pd.DataFrame(columns=["Business Area", "Active Plans"]) 

def make_voice_sample(interval_minutes: int = 30, days: int = 5) -> pd.DataFrame:
    today = dt.date.today()
    rows: List[dict] = []
    rng = np.random.default_rng(42)
    for d in range(days):
        date = today + dt.timedelta(days=d)
        start = dt.datetime.combine(date, dt.time(9,0))
        end   = dt.datetime.combine(date, dt.time(18,0))
        t = start
        while t < end:
            rows.append({"date": date, "interval": t.time().strftime("%H:%M"),
                         "volume": int(rng.integers(20,80)), "aht_sec": 300, "program":"WFM"})
            t += dt.timedelta(minutes=interval_minutes)
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out

def make_backoffice_sample(days: int = 5) -> pd.DataFrame:
    """Backoffice demo data; all columns length == days."""
    today = dt.date.today()
    dates = [(today + dt.timedelta(days=i)).isoformat() for i in range(days)]
    rng = np.random.default_rng(7)

    sub_tasks_cycle = cycle(["Case Review", "KYB", "Doc Check", "QA", "QC"])

    return pd.DataFrame({
        "date": dates,
        "items": rng.integers(200, 600, size=days),
        "aht_sec": np.full(days, 600),
        "sub_task": [next(sub_tasks_cycle) for _ in range(days)],
        "program": ["WFM"] * days,
    })

def make_outbound_sample(days: int = 5) -> pd.DataFrame:
    """Outbound demo data; all columns length == days."""
    today = dt.date.today()
    dates = [(today + dt.timedelta(days=i)).isoformat() for i in range(days)]
    rng = np.random.default_rng(11)

    campaigns_cycle = cycle(["Retention", "Welcome", "NPS", "Upsell", "Collections"])

    return pd.DataFrame({
        "date": dates,
        "calls": rng.integers(300, 700, size=days),
        "aht_sec": np.full(days, 240),
        "campaign": [next(campaigns_cycle) for _ in range(days)],
        "program": ["WFM"] * days,
    })

def make_roster_sample() -> pd.DataFrame:
    today = dt.date.today()
    return pd.DataFrame({
        "employee_id":["101","102","103","104"],
        "name":["Asha","Bala","Chan","Drew"],
        "status":["Active"]*4,
        "employment_type":["FT","FT","FT","PT"],
        "fte":[1.0,1.0,1.0,0.5],
        "contract_hours_per_week":[40,40,40,20],
        "country":["India","India","India","India"],
        "site":["Chennai"]*4, "timezone":["Asia/Kolkata"]*4,
        "program":["WFM"]*4, "sub_business_area":["Retail"]*4, "lob":["Cards"]*4, "channel":["Voice"]*4,
        "skill_voice":[True,True,False,True], "skill_bo":[True,False,True,True], "skill_ob":[False,True,True,False],
        "start_date":[today.isoformat()]*4, "end_date":[""]*4
    })

def make_hiring_sample() -> pd.DataFrame:
    today = dt.date.today()
    sow = today - dt.timedelta(days=today.weekday())  # Monday
    return pd.DataFrame({
        "start_week":[(sow - dt.timedelta(weeks=1)).isoformat(), sow.isoformat(), (sow + dt.timedelta(weeks=1)).isoformat()],
        "fte":[2,0,5], "program":["WFM","WFM","WFM"], "country":["India"]*3, "site":["Chennai"]*3
    })


def make_shrinkage_sample() -> pd.DataFrame:
    today = dt.date.today()
    sow = today - dt.timedelta(days=today.weekday())
    weeks = [(sow - dt.timedelta(weeks=i)).isoformat() for i in (3, 2, 1, 0)]
    base_hours = np.array([320.0, 318.0, 325.0, 330.0])
    ooo_hours = np.array([36.0, 34.0, 35.0, 36.5])
    ino_hours = np.array([22.0, 23.5, 24.0, 23.0])
    overall_pct = ((ooo_hours + ino_hours) / base_hours) * 100.0
    return pd.DataFrame({
        "week": weeks,
        "program": ["WFM"] * 4,
        "ooo_hours": ooo_hours,
        "ino_hours": ino_hours,
        "base_hours": base_hours,
        "ooo_pct": (ooo_hours / base_hours) * 100.0,
        "ino_pct": (ino_hours / base_hours) * 100.0,
        "overall_pct": overall_pct,
    })

def make_attrition_sample() -> pd.DataFrame:
    today = dt.date.today()
    sow = today - dt.timedelta(days=today.weekday())
    return pd.DataFrame({"week":[(sow - dt.timedelta(weeks=i)).isoformat() for i in [3,2,1,0]],
                         "attrition_pct":[0.7, 0.8, 0.9, 0.85], "program":["WFM"]*4})

def assemble_voice(scope_key, which="forecast"):
    vol = load_timeseries(f"voice_{which}_volume", scope_key)
    aht = load_timeseries(f"voice_{which}_aht",    scope_key)
    if vol.empty or aht.empty:
        return pd.DataFrame(columns=["date","interval","volume","aht_sec","program"])
    df = pd.merge(vol, aht, on=["date","interval"], how="inner")
    df["program"] = "WFM"
    return df[["date","interval","volume","aht_sec","program"]]

def assemble_bo(scope_key, which="forecast"):
    vol = load_timeseries(f"bo_{which}_volume", scope_key)
    sut = load_timeseries(f"bo_{which}_sut",    scope_key)
    if vol.empty or sut.empty:
        return pd.DataFrame(columns=["date","items","aht_sec","program"])
    df = pd.merge(vol, sut, on=["date"], how="inner")
    df = df.rename(columns={"sut_sec":"aht_sec","items":"items"})
    df["program"] = "WFM"
    return df[["date","items","aht_sec","program"]]
# ─── Core daily requirement/supply ───────────────────────────
# def required_fte_daily(voice_df: pd.DataFrame, bo_df: pd.DataFrame, ob_df: pd.DataFrame, settings: dict) -> pd.DataFrame:
#     frames = []
#     # Voice per-interval → daily
#     if isinstance(voice_df, pd.DataFrame) and not voice_df.empty:
#         vrows = []
#         for _, r in voice_df.iterrows():
#             calls = float(r.get("volume", 0) or 0)
#             aht   = float(r.get("aht_sec", 0) or 0)
#             A     = offered_load_erlangs(calls, aht, int(settings["interval_minutes"]))
#             # use min agents; convert to FTE via shrinkage
#             N, sl, occ, asa_val = min_agents(calls, aht, int(settings["interval_minutes"]),
#                                              float(settings["target_sl"]), float(settings["sl_seconds"]),
#                                              float(settings["occupancy_cap_voice"]))
#             fte = N / max(1e-6, (1 - float(settings["shrinkage_pct"])))
#             vrows.append({"date": pd.to_datetime(r["date"]).date(), "program": r.get("program","WFM"),
#                           "fte_req": fte})
#         v = pd.DataFrame(vrows).groupby(["date","program"], as_index=False)["fte_req"].sum().rename(columns={"fte_req":"voice_fte"})
#         frames.append(v)

#     # Backoffice (daily)
#     if isinstance(bo_df, pd.DataFrame) and not bo_df.empty:
#         denom = float(settings["hours_per_fte"]) * 3600.0 * (1 - float(settings["shrinkage_pct"])) * float(settings["util_bo"])
#         b = bo_df.copy()
#         b["date"] = pd.to_datetime(b["date"]).dt.date
#         b["bo_fte"] = b.apply(lambda r: (float(r["items"]) * float(r["aht_sec"])) / max(denom, 1e-6), axis=1)
#         b = b.groupby(["date","program"], as_index=False)["bo_fte"].sum()
#         frames.append(b)

#     # Outbound (daily)
#     if isinstance(ob_df, pd.DataFrame) and not ob_df.empty:
#         denom = float(settings["hours_per_fte"]) * 3600.0 * (1 - float(settings["shrinkage_pct"])) * float(settings["util_ob"])
#         o = ob_df.copy()
#         o["date"] = pd.to_datetime(o["date"]).dt.date
#         o["ob_fte"] = o.apply(lambda r: (float(r["calls"]) * float(r["aht_sec"])) / max(denom, 1e-6), axis=1)
#         o = o.groupby(["date","program"], as_index=False)["ob_fte"].sum()
#         frames.append(o)

#     if not frames:
#         return pd.DataFrame(columns=["date","program","voice_fte","bo_fte","ob_fte","total_req_fte"])
#     out = frames[0]
#     for f in frames[1:]:
#         out = pd.merge(out, f, on=["date","program"], how="outer")
#     for c in ["voice_fte","bo_fte","ob_fte"]:
#         if c not in out: out[c] = 0.0
#     out["total_req_fte"] = out[["voice_fte","bo_fte","ob_fte"]].fillna(0).sum(axis=1)
#     return out.fillna(0)

def supply_fte_daily(roster: pd.DataFrame, hiring: pd.DataFrame) -> pd.DataFrame:
    if roster is None or roster.empty:
        return pd.DataFrame(columns=["date","program","supply_fte"])
    # expand roster by date
    rows: List[dict] = []
    today = dt.date.today()
    horizon = today + dt.timedelta(days=28)
    date_list = [today + dt.timedelta(days=i) for i in range((horizon - today).days + 1)]
    for _, r in roster.iterrows():
        try:
            sd = pd.to_datetime(r.get("start_date")).date()
        except Exception:
            sd = today
        ed_val = r.get("end_date", "")
        ed = pd.to_datetime(ed_val).date() if str(ed_val).strip() else horizon
        fte = float(r.get("fte", 1.0) or 0.0)
        prog = r.get("program", "WFM")
        stat = str(r.get("status","Active")).strip().lower()
        if stat and stat != "active":
            continue
        for d in date_list:
            if sd <= d <= ed:
                rows.append({"date": d, "program": prog, "fte": fte})
    sup = pd.DataFrame(rows)
    if sup.empty:
        return pd.DataFrame(columns=["date","program","supply_fte"])
    sup = sup.groupby(["date","program"], as_index=False)["fte"].sum().rename(columns={"fte":"supply_fte"})

    # add hiring by week
    if isinstance(hiring, pd.DataFrame) and not hiring.empty and "start_week" in hiring.columns:
        add_rows: List[dict] = []
        for _, r in hiring.iterrows():
            try:
                s = str(r.get("start_week", "")).strip()
                if re.match(r"^\d{1,2}-\d{1,2}-\d{4}$", s):
                    ws = pd.to_datetime(s, dayfirst=True).date()
                else:
                    ws = pd.to_datetime(s, errors="coerce").date()
            except Exception:
                continue
            prog = r.get("program","WFM")
            f = float(r.get("fte",0) or 0)
            for i in range(7):
                add_rows.append({"date": ws + dt.timedelta(days=i), "program": prog, "supply_fte": f})
        add = pd.DataFrame(add_rows)
        if not add.empty:
            sup = pd.concat([sup, add]).groupby(["date","program"], as_index=False)["supply_fte"].sum()
    sup["date"] = pd.to_datetime(sup["date"]).dt.date
    return sup

# ─── KPIs ────────────────────────────────────────────────────
def kpi_hiring(hiring_df: pd.DataFrame) -> Tuple[float,float,float]:
    if hiring_df is None or hiring_df.empty: return 0,0,0
    today = dt.date.today()
    sow = today - dt.timedelta(days=today.weekday())
    lw, tw, nw = sow - dt.timedelta(weeks=1), sow, sow + dt.timedelta(weeks=1)
    def wk(x): 
        try:
            s = str(x).strip()
            if re.match(r"^\d{1,2}-\d{1,2}-\d{4}$", s):
                return pd.to_datetime(s, dayfirst=True).date()
            return pd.to_datetime(s, errors="coerce", dayfirst=False).date()
        except Exception:
            return None
    ser = hiring_df["start_week"].map(wk)
    fte = hiring_df["fte"].astype(float)
    last = fte[ser==lw].sum(); this = fte[ser==tw].sum(); nxt = fte[ser==nw].sum()
    return float(last), float(this), float(nxt)

def kpi_shrinkage(shrink_df: pd.DataFrame) -> Tuple[float,float]:
    if shrink_df is None or shrink_df.empty:
        return 0.0, 0.0
    df = shrink_df.copy()
    col = None
    for candidate in ("overall_pct", "shrinkage_pct"):
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        return 0.0, 0.0
    df["week"] = pd.to_datetime(df.get("week"), errors="coerce")
    df = df.dropna(subset=["week"]).sort_values("week")
    if df.empty:
        return 0.0, 0.0
    values = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    last4 = values.tail(4).mean()
    next4 = values.tail(1).mean()
    return float(last4 or 0.0), float(next4 or 0.0)

def understaffed_accounts_next_4w(req: pd.DataFrame, sup: pd.DataFrame) -> int:
    if req is None or req.empty: return 0
    if sup is None or sup.empty: return 0
    df = pd.merge(req.copy(), sup.copy(), on=["date","program"], how="left").fillna({"supply_fte":0})
    df["gap"] = df["supply_fte"] - df["total_req_fte"]
    by_prog = df.groupby("program")["gap"].min()
    return int((by_prog < 0).sum())


def _last_next_4(df: pd.DataFrame, week_col: str, value_col: str):
    """Return (avg of last 4 weeks, avg of next 4 weeks) based on current Monday."""
    if df is None or df.empty:
        return 0.0, 0.0
    tmp = df.copy()
    tmp[week_col] = pd.to_datetime(tmp[week_col]).dt.date
    today = dt.date.today()
    sow = today - dt.timedelta(days=today.weekday())  # Monday
    past   = tmp[tmp[week_col] <= sow].sort_values(week_col).tail(4)
    future = tmp[tmp[week_col] >  sow].sort_values(week_col).head(4)
    last4 = float(past[value_col].mean()) if not past.empty else 0.0
    next4 = float(future[value_col].mean()) if not future.empty else last4
    return last4, next4



