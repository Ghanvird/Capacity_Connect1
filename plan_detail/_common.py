from __future__ import annotations
import datetime as dt
from typing import List, Tuple, Dict
import base64, io, re

import pandas as pd
import numpy as np

import dash
from dash import html, dcc, dash_table, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from plan_store import get_plan, create_plan
from cap_db import save_df, load_df
from cap_store import load_headcount, get_clients_hierarchy  # <-- single source for BA/Level 3/Site
from cap_store import load_timeseries, resolve_settings, load_roster, load_hiring, load_roster_long, load_defaults, load_roster_wide
from capacity_core import voice_requirements_interval, voice_rollups, bo_rollups, required_fte_daily, supply_fte_daily
import os, getpass, re
import pandas as pd
from datetime import datetime, timezone
from flask import request, session
# Optional capacity_core
try:
    from capacity_core import min_agents, offered_load_erlangs  # (placeholder for future calcs)
except Exception:
    def min_agents(*args, **kwargs): return None
    def offered_load_erlangs(*args, **kwargs): return None

# Dash ctx fallback
try:
    from dash import ctx
except Exception:
    from dash import callback_context as ctx  # type: ignore

pd.set_option('future.no_silent_downcasting', True)

CHANNEL_DEFAULTS = ["Voice", "Back Office", "Outbound", "Blended", "Chat", "MessageUs"]

_PLANS_INDEX_KEY = "plans_index"  # where plan metadata lives
# adjust this list if you add more per-plan tables
_PLAN_TABLE_SUFFIXES = [
    "fw","hc","attr","shr","train","ratio","seat","bva","nh","emp","bulk_files","notes","lc_overrides"
]
# ──────────────────────────────────────────────────────────────────────────────
# === Learning-curve + per-week overrides helpers =============================

def _as_pct_list(val, length: int | None = None, default: float = 0.0) -> list[float]:
    """
    Accepts '50,60,70' | [50,60,70] | None, returns list of floats (percent, not fraction).
    Pads/truncates to `length` (if provided).
    """
    import numpy as np
    out: list[float] = []
    if isinstance(val, str):
        parts = [p.strip().replace("%", "") for p in val.split(",")]
        for p in parts:
            try: out.append(float(p))
            except Exception: out.append(default)
    elif isinstance(val, (list, tuple, np.ndarray)):
        for p in val:
            try: out.append(float(str(p).replace("%","")))
            except Exception: out.append(default)
    if length is not None:
        out = (out + [default]*length)[:length]
    return out

def _pick_override_for_week(ovr_list, week_id: str):
    """
    Given a list of dicts like [{"start_week":"2025-09-01","nesting_prod_pct":"50,60"}...]
    return the last row with start_week <= week_id. Returns {} if none.
    """
    import pandas as pd
    if not isinstance(ovr_list, (list, tuple)) or not ovr_list:
        return {}
    df = pd.DataFrame(ovr_list)
    if "start_week" not in df.columns:  # tolerate different casing
        return {}
    df = df.copy()
    df["start_week"] = pd.to_datetime(df["start_week"], errors="coerce").dt.date.astype(str)
    df = df.dropna(subset=["start_week"])
    if df.empty:
        return {}
    df = df.sort_values("start_week")
    pick = df[df["start_week"] <= str(week_id)]
    return (pick.iloc[-1].to_dict() if not pick.empty else {})

def _learning_curve_for_week(settings: dict, lc_overrides_df, week_id: str) -> dict:
    """
    Returns dict with lists (percents, not fractions):
      nesting_prod_pct, nesting_aht_uplift_pct,
      sda_prod_pct, sda_aht_uplift_pct,
      throughput_train_pct, throughput_nest_pct
    Pulls base from `settings` then applies latest override in lc_overrides_df up to week_id.
    """
    import pandas as pd
    base = dict(settings or {})
    # defaults
    n_weeks  = int(float(base.get("nesting_weeks", base.get("default_nesting_weeks", 0)) or 0))
    s_weeks  = int(float(base.get("sda_weeks",      base.get("default_sda_weeks", 0)) or 0))

    out = dict(
        nesting_prod_pct       = _as_pct_list(base.get("nesting_productivity_pct"), n_weeks, default=100.0),
        nesting_aht_uplift_pct = _as_pct_list(base.get("nesting_aht_uplift_pct"), n_weeks, default=0.0),
        sda_prod_pct           = _as_pct_list(base.get("sda_productivity_pct"),     s_weeks, default=100.0),
        sda_aht_uplift_pct     = _as_pct_list(base.get("sda_aht_uplift_pct"),       s_weeks, default=0.0),
        throughput_train_pct   = float(str(base.get("throughput_train_pct", 100)).replace("%","")),
        throughput_nest_pct    = float(str(base.get("throughput_nest_pct",  100)).replace("%","")),
    )

    # optional: per-week overrides stored as a table (we load it in _calc and pass df here)
    if isinstance(lc_overrides_df, pd.DataFrame) and not lc_overrides_df.empty:
        sel = _pick_override_for_week(lc_overrides_df.to_dict("records"), week_id)
        if sel:
            if "nesting_weeks" in sel: n_weeks = int(float(sel.get("nesting_weeks") or n_weeks))
            if "sda_weeks"     in sel: s_weeks = int(float(sel.get("sda_weeks") or s_weeks))
            out["nesting_prod_pct"]       = _as_pct_list(sel.get("nesting_prod_pct", out["nesting_prod_pct"]), n_weeks, 100.0)
            out["nesting_aht_uplift_pct"] = _as_pct_list(sel.get("nesting_aht_uplift_pct", out["nesting_aht_uplift_pct"]), n_weeks, 0.0)
            out["sda_prod_pct"]           = _as_pct_list(sel.get("sda_prod_pct", out["sda_prod_pct"]), s_weeks, 100.0)
            out["sda_aht_uplift_pct"]     = _as_pct_list(sel.get("sda_aht_uplift_pct", out["sda_aht_uplift_pct"]), s_weeks, 0.0)
            if "throughput_train_pct" in sel: out["throughput_train_pct"] = float(str(sel["throughput_train_pct"]).replace("%",""))
            if "throughput_nest_pct"  in sel: out["throughput_nest_pct"]  = float(str(sel["throughput_nest_pct"]).replace("%",""))
    return out

def _user():
    return os.environ.get("HOSTNAME") or os.environ.get("USERNAME") or getpass.getuser() or "system"

def _load_index() -> pd.DataFrame:
    try:
        df = load_df(_PLANS_INDEX_KEY)
        if not isinstance(df, pd.DataFrame): df = pd.DataFrame()
    except Exception:
        df = pd.DataFrame()
    if "plan_id" not in df.columns:
        df["plan_id"] = pd.Series(dtype="int64")
    return df

def get_plan_meta(pid: int | str) -> dict:
    """Return plan metadata row from the plans index as a dict.
    Falls back to empty dict when not found.
    """
    try:
        pid_i = int(pid)
    except Exception:
        return {}
    df = _load_index()
    if not isinstance(df, pd.DataFrame) or df.empty or "plan_id" not in df.columns:
        return {}
    try:
        mask = df["plan_id"].astype("Int64").fillna(-1).astype(int) == pid_i
        if getattr(mask, "any", lambda: False)():
            row = df.loc[mask].tail(1).iloc[0].to_dict()
            # Clean NaNs to friendly values
            out = {}
            for k, v in row.items():
                if k == "plan_id":
                    out[k] = pid_i
                else:
                    try:
                        out[k] = ("" if pd.isna(v) else v)
                    except Exception:
                        out[k] = v
            return out
    except Exception:
        pass
    return {}

def save_plan_meta(pid: int, meta: dict) -> None:
    """Upsert row in the plans index."""
    df = _load_index()
    pid = int(pid)
    meta = dict(meta or {})
    meta["plan_id"] = pid
    now = pd.Timestamp.utcnow().isoformat(timespec="seconds")
    meta.setdefault("last_updated_on", now)
    meta.setdefault("last_updated_by", _user())

    if not df.empty and "plan_id" in df.columns:
        mask = df["plan_id"].astype(int) == pid
    else:
        mask = pd.Series(False, index=df.index)

    # ensure all columns exist
    for k in meta.keys():
        if k not in df.columns:
            df[k] = None

    if getattr(mask, "any", lambda: False)():
        for k, v in meta.items():
            df.loc[mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([meta])], ignore_index=True)

    save_df(_PLANS_INDEX_KEY, df)

def clone_plan(pid: int, new_name: str) -> int:
    """Clone an existing plan into a history entry and copy all datasets."""
    pid = int(pid)
    base = get_plan(pid) or {}

    payload = {
        "org": base.get("org"),
        "business_entity": base.get("business_entity"),
        "vertical": base.get("vertical"),
        "sub_ba": base.get("sub_ba"),
        "channel": base.get("channel"),
        "location": base.get("location"),
        "site": base.get("site"),
        "plan_type": base.get("plan_type"),
        "start_week": base.get("start_week"),
        "end_week": base.get("end_week"),
        "ft_weekly_hours": base.get("ft_weekly_hours"),
        "pt_weekly_hours": base.get("pt_weekly_hours"),
        "tags": base.get("tags"),
        "plan_name": new_name,
        "status": "history",
        "is_current": 0,
    }
    new_pid = create_plan(payload)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new_meta = {
        **base,
        "plan_id": new_pid,
        "plan_name": new_name,
        "status": "history",
        "is_current": 0,
        "created_on": base.get("created_on") or now,
        "created_by": base.get("created_by") or _user(),
        "last_updated_on": now,
        "last_updated_by": _user(),
    }
    save_plan_meta(new_pid, new_meta)

    for sfx in _PLAN_TABLE_SUFFIXES:
        try:
            df = load_df(f"plan_{pid}_{sfx}")
        except Exception:
            df = None
        if isinstance(df, pd.DataFrame):
            save_df(f"plan_{new_pid}_{sfx}", df)

    return new_pid

def extend_plan_weeks(pid: int, add_weeks: int) -> None:
    """Push plan end week forward by N weeks and save meta."""
    meta = get_plan(pid) or {}
    end = pd.to_datetime(meta.get("end_week"), errors="coerce")
    if pd.isna(end):
        end = pd.Timestamp.utcnow().normalize()
    new_end = (end + pd.Timedelta(days=7*int(add_weeks or 0))).date().isoformat()
    meta["end_week"] = new_end
    meta["last_updated_on"] = pd.Timestamp.utcnow().isoformat(timespec="seconds")
    meta["last_updated_by"] = _user()
    save_plan_meta(pid, meta)
# --- New-Hire: centralized options & helpers ---------------------------------


# Storage key for class dataset
CLASS_STORE_FMT = "plan_{pid}_nh_classes"

# Centralized dropdown options (edit here to change everywhere)
CLASS_TYPE_OPTIONS = [
    {"label": "Ramp-Up", "value": "ramp-up"},
    {"label": "Backfill", "value": "backfill"},
]

CLASS_LEVEL_OPTIONS = [
    {"label": "Trainee",        "value": "trainee"},
    {"label": "New Agent",      "value": "new-agent"},
    {"label": "Tenured Agent",  "value": "tenured"},
    {"label": "Senior Agent",   "value": "senior-agent"},
    {"label": "SME",            "value": "sme"},
    {"label": "Cross-Skill",    "value": "cross-skill"},
]

def get_class_type_options():
    return CLASS_TYPE_OPTIONS

def get_class_level_options():
    return CLASS_LEVEL_OPTIONS

def load_nh_classes(pid: str) -> pd.DataFrame:
    df = load_df(CLASS_STORE_FMT.format(pid=pid))
    cols = ["class_reference","source_system_id","emp_type","status","class_type","class_level",
            "grads_needed","billable_hc","training_weeks","nesting_weeks",
            "induction_start","training_start","training_end","nesting_start","nesting_end","production_start",
            "created_by","created_ts"]
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]

def save_nh_classes(pid: str, df: pd.DataFrame):
    save_df(CLASS_STORE_FMT.format(pid=pid), df)

def current_user_fallback() -> str:
    try:
        return (
            session.get("user_email") or session.get("user")
            or request.headers.get("X-Auth-Email") or request.headers.get("X-Forwarded-User")
            or request.headers.get("X-User") or request.headers.get("remote-user")
            or os.getenv("USERNAME") or os.getenv("USER") or getpass.getuser() or "unknown"
        )
    except Exception:
        return os.getenv("USERNAME") or os.getenv("USER") or "unknown"

def next_class_reference(pid: str, df: pd.DataFrame | None = None) -> str:
    """NH-<pid>-YYYYMMDD-## (sequential per day & plan)."""
    df = df if isinstance(df, pd.DataFrame) else load_nh_classes(pid)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    prefix = f"NH-{pid}-{today}-"
    seq = 1
    if "class_reference" in df.columns:
        existing = df["class_reference"].dropna().astype(str)
        pat = re.compile(rf"^{re.escape(prefix)}(\d{{2,}})$")
        nums = []
        for s in existing:
            m = pat.match(s.strip())
            if m:
                try:
                    nums.append(int(m.group(1)))
                except:
                    pass
        if nums:
            seq = max(nums) + 1
    return f"{prefix}{seq:02d}"


def _settings_volume_aht_overrides(sk, which: str):
    """
    Read 'Settings' uploads where AHT/SUT is uploaded together with volume.
    Returns dicts:
      - vol_w: week -> total volume/items
      - aht_w (voice) or sut_w (bo): week -> weighted AHT/SUT
    `which` is 'voice' or 'bo'.
    """
    # Try a handful of tolerant keys; keep/add keys that match your storage
    keys_voice = [
        "settings_voice_volume_aht", "voice_settings_upload",
        "settings_volume_aht_voice", "voice_volume_aht", "voice_forecast_settings"
    ]
    keys_bo = [
        "settings_bo_volume_sut", "settings_backoffice_volume_sut",
        "bo_settings_upload", "backoffice_volume_sut", "bo_forecast_settings"
    ]
    df = _first_non_empty_ts(sk, keys_voice if which.lower()=="voice" else keys_bo)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"vol_w": {}, "aht_or_sut_w": {}}

    d = df.copy()
    # Column detection
    L = {str(c).strip().lower(): c for c in d.columns}
    c_date = L.get("date") or L.get("week") or L.get("start_date")
    c_vol  = L.get("vol") or L.get("volume") or L.get("calls") or L.get("items") or L.get("txns") or L.get("transactions")
    if which.lower()=="voice":
        c_time = L.get("aht_sec") or L.get("aht") or L.get("avg_aht")
    else:
        c_time = L.get("sut_sec") or L.get("sut") or L.get("avg_sut") or L.get("aht_sec")  # tolerate 'aht_sec' for BO

    if not c_date or not c_vol or not c_time:
        return {"vol_w": {}, "aht_or_sut_w": {}}

    # Normalize & weekly group
    d[c_date] = pd.to_datetime(d[c_date], errors="coerce")
    d = d.dropna(subset=[c_date])
    d["week"] = (d[c_date] - pd.to_timedelta(d[c_date].dt.weekday, unit="D")).dt.date.astype(str)

    d[c_vol]  = pd.to_numeric(d[c_vol],  errors="coerce").fillna(0.0)
    d[c_time] = pd.to_numeric(d[c_time], errors="coerce").fillna(0.0)

    g = d.groupby("week", as_index=False)[[c_vol]].sum().rename(columns={c_vol: "_vol_"})
    # weighted time per week
    d["_num_"] = d[c_time] * d[c_vol]
    w = d.groupby("week", as_index=False)[["_num_", c_vol]].sum()
    w["_wt_"] = np.where(w[c_vol] > 0, w["_num_"] / w[c_vol], np.nan)

    vol_w = dict(zip(g["week"], g["_vol_"]))
    aht_or_sut_w = dict(zip(w["week"], w["_wt_"]))  # may be NaN if no volume

    # Clean NaNs
    aht_or_sut_w = {k: float(v) for k, v in aht_or_sut_w.items() if pd.notna(v) and v > 0}
    vol_w        = {k: float(v) for k, v in vol_w.items()        if pd.notna(v) and v > 0}
    return {"vol_w": vol_w, "aht_or_sut_w": aht_or_sut_w}



# _____________________ erlangs ___________________

import math

def _metric_week_dict(df, row_name, week_ids):
    """Read a single 'metric' row into {week -> float}."""
    if not isinstance(df, pd.DataFrame) or df.empty or "metric" not in df.columns:
        return {}
    m = df["metric"].astype(str).str.strip().eq(row_name)
    if not m.any():
        return {}
    ser = pd.to_numeric(df.loc[m, week_ids].iloc[0], errors="coerce").fillna(0.0)
    return {str(k): float(v) for k, v in ser.to_dict().items()}

def _prev_week_id(w, week_ids):
    try:
        i = week_ids.index(w)
        return week_ids[i-1] if i > 0 else None
    except Exception:
        return None

def _erlang_c(traffic_erlangs: float, agents: float) -> float:
    """Classic Erlang-C waiting probability; agents may be float, we floor to int."""
    a = max(0, int(math.floor(agents)))
    A = max(0.0, float(traffic_erlangs))
    if a <= 0:
        return 1.0
    if A >= a:
        return 1.0
    # p0
    s = sum((A**k)/math.factorial(k) for k in range(a))
    last = (A**a)/math.factorial(a) * (a/(a - A))
    p0 = 1.0 / (s + last)
    pw = last * p0
    return min(1.0, max(0.0, pw))

def _erlang_service_level(offered_calls: float, aht_sec: float, agents: float,
                          interval_sec: int, asa_sec: int) -> float:
    """Return service level in [0..1]."""
    if aht_sec <= 0 or interval_sec <= 0 or agents <= 0 or offered_calls <= 0:
        return 0.0

# Normalized roster loader (shadows legacy version)
def _legacy_load_or_empty_roster(pid):
    """Load roster normalized to current _roster_columns() ids (backward compatible)."""
    cols = [c["id"] for c in _roster_columns()]
    empty = pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    try:
        df = load_df(f"plan_{pid}_emp")
    except Exception:
        return empty
    if not isinstance(df, pd.DataFrame) or df.empty:
        return empty
    df = df.copy()
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
    # Traffic in Erlangs in this interval
    A = (offered_calls * aht_sec) / float(interval_sec)
    pw = _erlang_c(A, agents)
    # P(wait <= T) = 1 - Pw * exp(-(a - A) * T / AHT)
    tail = math.exp(-max(0.0, (agents - A)) * (asa_sec / max(1.0, aht_sec)))
    sl = 1.0 - pw * tail
    return min(1.0, max(0.0, sl))
# ──────────────────────────────────────────────────────────────────────────────
def _build_global_hierarchy() -> dict:
    try:
        hmap, sites, _locs = get_clients_hierarchy()   # {BA:{SubBA:[LOBs...]}}
    except Exception:
        hmap, sites = {}, []

    ba_list = sorted(hmap.keys())
    sub_map = {ba: sorted((hmap.get(ba) or {}).keys()) for ba in ba_list}

    lob_map = {}
    for ba, subs in (hmap or {}).items():
        for sba, channels in (subs or {}).items():
            lob_map[f"{ba}|{sba}"] = list(channels or CHANNEL_DEFAULTS)

    return {
        "ba": ba_list,
        "subba": sub_map,
        "lob": lob_map,
        "site": sites or []
    }


def _lower_map(df: pd.DataFrame) -> dict[str, str]:
    return {str(c).strip().lower(): c for c in df.columns}

# ---------- Parsing helpers ----------
def pretty_columns(df_or_cols) -> list[dict]:
    cols = list(df_or_cols.columns) if hasattr(df_or_cols, "columns") else list(df_or_cols)
    return [{"name": c, "id": c} for c in cols]

def lock_variance_cols(cols):
    out = []
    for col in cols:
        c = dict(col)  # copy
        name_txt = c.get("name", "")
        if isinstance(name_txt, list):  # sometimes headers are multi-line arrays
            name_txt = " ".join(map(str, name_txt))
        id_txt = str(c.get("id", ""))
        if "variance" in str(name_txt).lower() or "variance" in id_txt.lower():
            c["editable"] = False
        else:
            # make other columns explicitly editable unless already set
            c.setdefault("editable", True)
        out.append(c)
    return out

def _scope_key(ba, subba, channel):
    return f"{(ba or '').strip()}|{(subba or '').strip()}|{(channel or '').strip()}"

# ──────────────────────────────────────────────────────────────────────────────
# helpers shared

def _week_monday(d):  # returns Monday date
    d = pd.to_datetime(d, errors="coerce").date()
    return d - pd.Timedelta(days=d.weekday())

# Fullscreen overlay style (centered)

def _settings_for_scope_key(sk: str) -> dict:
    try:
        ba, sba, ch = (sk.split("|", 2) + ["",""])[:3]
    except Exception:
        ba, sba, ch = "", "", ""
    return resolve_settings(ba=ba, subba=sba, lob=ch)

def _canon_scope(ba, sba, ch, site=None):
    """
    Build canonical scope key. If `site` present -> 4-part, else legacy 3-part.
    Why: uploads now save per (BA|SubBA|Channel|Site) and plans must read same key.
    """
    canon = lambda x: (x or "").strip()
    if site and str(site).strip():
        return f"{canon(ba)}|{canon(sba)}|{canon(ch)}|{canon(site)}"
    return f"{canon(ba)}|{canon(sba)}|{canon(ch)}|{canon(site)}"  # legacy

def _monday(x):
    s = pd.to_datetime(x, errors="coerce")
    if isinstance(s, (pd.Series, pd.DatetimeIndex)):
        if isinstance(s, pd.DatetimeIndex):
            return (s - pd.to_timedelta(s.weekday, unit="D")).date
        else:
            return (s - pd.to_timedelta(s.dt.weekday, unit="D")).dt.date
    if pd.isna(s):
        s = pd.Timestamp(dt.date.today())
    return (s - pd.Timedelta(days=int(s.weekday()))).date()

def _weekly_voice(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["week", "vol", "aht"])

    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.dropna(subset=["date"])

    # Week bucket (Monday)
    x["week"] = _monday(x["date"])

    # Safe numerics
    x["w"] = pd.to_numeric(x.get("volume"),  errors="coerce").fillna(0.0)
    x["a"] = pd.to_numeric(x.get("aht_sec"), errors="coerce").fillna(0.0)

    # Weighted-average pieces
    x["num"] = x["w"] * x["a"]

    g = (
        x.groupby("week", as_index=False)
         .agg(vol=("w", "sum"), num=("num", "sum"))
    )
    g["aht"] = np.where(g["vol"] > 0, g["num"] / g["vol"], np.nan)
    g = g.drop(columns=["num"])
    g["week"] = g["week"].astype(str)

    return g[["week", "vol", "aht"]]


def _weekly_bo(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["week", "items", "sut"])

    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.dropna(subset=["date"])

    # Week bucket (Monday)
    x["week"] = _monday(x["date"])

    # Safe numerics (for BO, aht_sec column carries SUT)
    x["i"] = pd.to_numeric(x.get("items"),   errors="coerce").fillna(0.0)
    x["s"] = pd.to_numeric(x.get("aht_sec"), errors="coerce").fillna(0.0)

    # Weighted-average pieces
    x["num"] = x["i"] * x["s"]

    g = (
        x.groupby("week", as_index=False)
         .agg(items=("i", "sum"), num=("num", "sum"))
    )
    g["sut"] = np.where(g["items"] > 0, g["num"] / g["items"], np.nan)
    g = g.drop(columns=["num"])
    g["week"] = g["week"].astype(str)

    return g[["week", "items", "sut"]]


# def _assemble_voice(scope_key, which):
#     vol = load_timeseries(f"voice_{which}_volume", scope_key)
#     aht = load_timeseries(f"voice_{which}_aht",    scope_key)
#     if vol is None or vol.empty:
#         return pd.DataFrame(columns=["date","interval","volume","aht_sec","program"])
#     df = vol.copy()
#     if isinstance(aht, pd.DataFrame) and not aht.empty:
#         df = df.merge(aht, on=["date","interval"], how="left")
#     if "aht_sec" not in df.columns or df["aht_sec"].isna().all():
#         s = _settings_for_scope_key(scope_key)
#         df["aht_sec"] = float(s.get("target_aht", s.get("budgeted_aht", 300)) or 300)
#     df["program"] = "WFM"
#     return df[["date","interval","volume","aht_sec","program"]]

# def _assemble_bo(scope_key, which):
#     vol = load_timeseries(f"bo_{which}_volume", scope_key)
#     sut = load_timeseries(f"bo_{which}_sut",    scope_key)
#     if vol is None or vol.empty:
#         return pd.DataFrame(columns=["date","items","aht_sec","program"])
#     df = vol.rename(columns={"volume":"items"}).copy()
#     if isinstance(sut, pd.DataFrame) and not sut.empty:
#         df = df.merge(sut, on=["date"], how="left")
#         if "aht_sec" not in df.columns and "sut_sec" in df.columns:
#             df = df.rename(columns={"sut_sec":"aht_sec"})
#     if "aht_sec" not in df.columns or df["aht_sec"].isna().all():
#         s = _settings_for_scope_key(scope_key)
#         df["aht_sec"] = float(s.get("target_sut", s.get("budgeted_sut", 600)) or 600)
#     df["program"] = "WFM"
#     return df[["date","items","aht_sec","program"]]

# def _assemble_voice(scope_key, which):
#     which = (which or "forecast").strip().lower()
#     vol = load_timeseries(f"voice_{which}_volume", scope_key)
#     aht = load_timeseries(f"voice_{which}_aht",    scope_key)

#     # Empty volume → return canonical empty frame
#     if vol is None or vol.empty:
#         return pd.DataFrame(columns=["date","interval","volume","aht_sec","program"])

#     # Canonicalize volume
#     df = vol.copy()
#     # Allow 'week' instead of 'date'
#     if "date" not in df.columns and "week" in df.columns:
#         df["date"] = pd.to_datetime(df["week"], errors="coerce").dt.normalize()
#     else:
#         df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
#     # Allow alt volume col names
#     if "volume" not in df.columns:
#         for alt in ("vol","calls"):
#             if alt in df.columns:
#                 df = df.rename(columns={alt: "volume"})
#                 break
#     # interval may or may not exist (OK)

#     # Merge AHT if provided
#     if isinstance(aht, pd.DataFrame) and not aht.empty:
#         ah = aht.copy()
#         # Accept week/date and normalize
#         if "date" not in ah.columns and "week" in ah.columns:
#             ah["date"] = pd.to_datetime(ah["week"], errors="coerce").dt.normalize()
#         else:
#             ah["date"] = pd.to_datetime(ah["date"], errors="coerce").dt.normalize()
#         # Canonicalize AHT column name(s)
#         if "aht_sec" not in ah.columns:
#             for alt in ("aht","avg_aht","aht_seconds"):
#                 if alt in ah.columns:
#                     ah = ah.rename(columns={alt: "aht_sec"})
#                     break
#         # Pick best join keys available
#         join_keys = [k for k in ["date","interval"] if k in df.columns and k in ah.columns]
#         if not join_keys:  # fall back to date-only
#             join_keys = ["date"]
#         df = df.merge(ah[[*join_keys, *(["aht_sec"] if "aht_sec" in ah.columns else [])]],
#                       on=join_keys, how="left")

#     # If still missing AHT, fill from settings defaults
#     if "aht_sec" not in df.columns or df["aht_sec"].isna().all():
#         s = _settings_for_scope_key(scope_key)
#         df["aht_sec"] = float(s.get("target_aht", s.get("budgeted_aht", 300)) or 300)

#     df["program"] = "WFM"
#     # Ensure canonical columns exist
#     if "interval" not in df.columns:
#         df["interval"] = pd.NaT  # keeps downstream code happy if it expects the column
#     if "volume" not in df.columns:
#         df["volume"] = 0.0

#     return df[["date","interval","volume","aht_sec","program"]]


# def _assemble_bo(scope_key, which):
#     which = (which or "forecast").strip().lower()
#     vol = load_timeseries(f"bo_{which}_volume", scope_key)
#     sut = load_timeseries(f"bo_{which}_sut",    scope_key)

#     if vol is None or vol.empty:
#         return pd.DataFrame(columns=["date","items","aht_sec","program"])

#     # Canonicalize volume/items
#     df = vol.copy()
#     if "date" not in df.columns and "week" in df.columns:
#         df["date"] = pd.to_datetime(df["week"], errors="coerce").dt.normalize()
#     else:
#         df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
#     if "items" not in df.columns:
#         for alt in ("volume","txns","transactions"):
#             if alt in df.columns:
#                 df = df.rename(columns={alt: "items"})
#                 break

#     # Merge SUT (rename to aht_sec for a common downstream field)
#     if isinstance(sut, pd.DataFrame) and not sut.empty:
#         su = sut.copy()
#         if "date" not in su.columns and "week" in su.columns:
#             su["date"] = pd.to_datetime(su["week"], errors="coerce").dt.normalize()
#         else:
#             su["date"] = pd.to_datetime(su["date"], errors="coerce").dt.normalize()
#         if "aht_sec" not in su.columns:
#             for alt in ("sut_sec","sut","aht","avg_sut","sut_seconds"):
#                 if alt in su.columns:
#                     su = su.rename(columns={alt: "aht_sec"})
#                     break
#         df = df.merge(su[["date","aht_sec"]], on="date", how="left")

#     if "aht_sec" not in df.columns or df["aht_sec"].isna().all():
#         s = _settings_for_scope_key(scope_key)
#         df["aht_sec"] = float(s.get("target_sut", s.get("budgeted_sut", 600)) or 600)

#     df["program"] = "WFM"
#     if "items" not in df.columns:
#         df["items"] = 0.0

#     return df[["date","items","aht_sec","program"]]

def _assemble_voice(scope_key, which):
    which = (which or "forecast").strip().lower()
    vol = _load_ts_with_fallback(f"voice_{which}_volume", scope_key)
    aht = _load_ts_with_fallback(f"voice_{which}_aht",    scope_key)

    # Empty volume → return canonical empty frame
    if vol is None or vol.empty:
        return pd.DataFrame(columns=["date","interval","volume","aht_sec","program"])

    # Canonicalize volume
    df = vol.copy()
    # Allow 'week' instead of 'date'
    if "date" not in df.columns and "week" in df.columns:
        df["date"] = pd.to_datetime(df["week"], errors="coerce").dt.normalize()
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    # Allow alt volume col names
    if "volume" not in df.columns:
        for alt in ("vol","calls"):
            if alt in df.columns:
                df = df.rename(columns={alt: "volume"})
                break

    # Merge AHT if provided
    if isinstance(aht, pd.DataFrame) and not aht.empty:
        ah = aht.copy()
        # Accept week/date and normalize
        if "date" not in ah.columns and "week" in ah.columns:
            ah["date"] = pd.to_datetime(ah["week"], errors="coerce").dt.normalize()
        else:
            ah["date"] = pd.to_datetime(ah["date"], errors="coerce").dt.normalize()
        # Canonicalize AHT column name(s)
        if "aht_sec" not in ah.columns:
            for alt in ("aht","avg_aht","aht_seconds"):
                if alt in ah.columns:
                    ah = ah.rename(columns={alt: "aht_sec"})
                    break
        # Pick best join keys available
        join_keys = [k for k in ["date","interval"] if k in df.columns and k in ah.columns]
        if not join_keys:  # fall back to date-only
            join_keys = ["date"]
        df = df.merge(ah[[*join_keys, *(["aht_sec"] if "aht_sec" in ah.columns else [])]],
                      on=join_keys, how="left")

    # If still missing AHT, fill from settings defaults (BA/SubBA/LOB only)
    if "aht_sec" not in df.columns or df["aht_sec"].isna().all():
        s = _settings_for_scope_key(scope_key)  # split first 3 parts internally
        df["aht_sec"] = float(s.get("target_aht", s.get("budgeted_aht", 300)) or 300)

    df["program"] = "WFM"
    if "interval" not in df.columns:  # required downstream
        df["interval"] = pd.NaT
    if "volume" not in df.columns:
        df["volume"] = 0.0

    return df[["date","interval","volume","aht_sec","program"]]

def _assemble_bo(scope_key, which):
    which = (which or "forecast").strip().lower()
    vol = _load_ts_with_fallback(f"bo_{which}_volume", scope_key)
    sut = _load_ts_with_fallback(f"bo_{which}_sut",    scope_key)

    if vol is None or vol.empty:
        return pd.DataFrame(columns=["date","items","aht_sec","program"])

    # Canonicalize volume/items
    df = vol.copy()
    if "date" not in df.columns and "week" in df.columns:
        df["date"] = pd.to_datetime(df["week"], errors="coerce").dt.normalize()
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if "items" not in df.columns:
        for alt in ("volume","txns","transactions"):
            if alt in df.columns:
                df = df.rename(columns={alt: "items"})
                break

    # Merge SUT (rename to aht_sec for common field)
    if isinstance(sut, pd.DataFrame) and not sut.empty:
        su = sut.copy()
        if "date" not in su.columns and "week" in su.columns:
            su["date"] = pd.to_datetime(su["week"], errors="coerce").dt.normalize()
        else:
            su["date"] = pd.to_datetime(su["date"], errors="coerce").dt.normalize()
        if "aht_sec" not in su.columns:
            for alt in ("sut_sec","sut","aht","avg_sut","sut_seconds"):
                if alt in su.columns:
                    su = su.rename(columns={alt: "aht_sec"})
                    break
        df = df.merge(su[["date","aht_sec"]], on="date", how="left")

    if "aht_sec" not in df.columns or df["aht_sec"].isna().all():
        s = _settings_for_scope_key(scope_key)
        df["aht_sec"] = float(s.get("target_sut", s.get("budgeted_sut", 600)) or 600)

    df["program"] = "WFM"
    if "items" not in df.columns:
        df["items"] = 0.0
    return df[["date","items","aht_sec","program"]]

def _assemble_chat(scope_key, which):
    which = (which or "forecast").strip().lower()
    vol = _load_ts_with_fallback(f"chat_{which}_volume", scope_key)
    aht = _load_ts_with_fallback(f"chat_{which}_aht",    scope_key)

    if vol is None or vol.empty:
        return pd.DataFrame(columns=["date","items","aht_sec","program"])

    df = vol.copy()
    # normalize date
    if "date" not in df.columns and "week" in df.columns:
        df["date"] = pd.to_datetime(df["week"], errors="coerce").dt.normalize()
    else:
        df["date"] = pd.to_datetime(df.get("date"), errors="coerce").dt.normalize()
    # normalize items column
    if "items" not in df.columns:
        for alt in ("chats","volume","txns","transactions","count"):
            if alt in df.columns:
                df = df.rename(columns={alt: "items"}); break

    # merge aht if provided
    if isinstance(aht, pd.DataFrame) and not aht.empty:
        ah = aht.copy()
        if "date" not in ah.columns and "week" in ah.columns:
            ah["date"] = pd.to_datetime(ah["week"], errors="coerce").dt.normalize()
        else:
            ah["date"] = pd.to_datetime(ah.get("date"), errors="coerce").dt.normalize()
        if "aht_sec" not in ah.columns:
            for alt in ("aht","avg_aht","sut","sut_sec","aht_seconds"):
                if alt in ah.columns:
                    ah = ah.rename(columns={alt: "aht_sec"}); break
        df = df.merge(ah[["date","aht_sec"]], on="date", how="left")

    if "aht_sec" not in df.columns or df["aht_sec"].isna().all():
        s = _settings_for_scope_key(scope_key)
        df["aht_sec"] = float(s.get("chat_aht_sec", s.get("target_aht", 240)) or 240)

    df["program"] = "Chat"
    if "items" not in df.columns:
        df["items"] = 0.0
    return df[["date","items","aht_sec","program"]]

def _assemble_ob(scope_key, which):
    which = (which or "forecast").strip().lower()
    # Try tolerant keys for OPC/dials and rates
    opc   = _load_ts_with_fallback(f"ob_{which}_opc",   scope_key)
    if opc is None or opc.empty:
        for k in [f"outbound_{which}_opc", f"ob_{which}_dials", f"outbound_{which}_dials", f"ob_{which}_calls"]:
            tmp = _load_ts_with_fallback(k, scope_key)
            if isinstance(tmp, pd.DataFrame) and not tmp.empty:
                opc = tmp; break
    conn  = _load_ts_with_fallback(f"ob_{which}_connect_rate", scope_key)
    if conn is None or conn.empty:
        for k in [f"outbound_{which}_connect_rate", f"ob_{which}_connect%"]:
            tmp = _load_ts_with_fallback(k, scope_key)
            if isinstance(tmp, pd.DataFrame) and not tmp.empty:
                conn = tmp; break
    rpc   = _load_ts_with_fallback(f"ob_{which}_rpc", scope_key)
    rpc_r = _load_ts_with_fallback(f"ob_{which}_rpc_rate", scope_key)
    aht   = _load_ts_with_fallback(f"ob_{which}_aht", scope_key)

    if opc is None or opc.empty:
        return pd.DataFrame(columns=["date","opc","connect_rate","rpc","rpc_rate","aht_sec","program"])

    # Canonicalize dates
    d = opc.copy()
    d["date"] = pd.to_datetime(d["date"] if "date" in d.columns else d.get("week"), errors="coerce").dt.normalize()
    # Canonicalize OPC column
    if "opc" not in d.columns:
        for alt in ("dials","calls","attempts","volume"):
            if alt in d.columns:
                d = d.rename(columns={alt: "opc"}); break

    def pick_rate(df, name):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        x = df.copy()
        x["date"] = pd.to_datetime(x["date"] if "date" in x.columns else x.get("week"), errors="coerce").dt.normalize()
        # normalize column name
        cols = {c.lower(): c for c in x.columns}
        for c in (name, name.replace("_"," "), f"{name}%"):
            lc = c.lower()
            if lc in cols:
                col = cols[lc]
                x = x.rename(columns={col: name})
                return x[["date", name]]
        # If numeric unnamed column, try first non-date
        for c in x.columns:
            if c.lower() not in ("date","week"):
                return x.rename(columns={c: name})[["date", name]]
        return None

    cr = pick_rate(conn, "connect_rate")
    rpcr = pick_rate(rpc_r, "rpc_rate")

    if isinstance(rpc, pd.DataFrame) and not rpc.empty:
        rr = rpc.copy()
        rr["date"] = pd.to_datetime(rr["date"] if "date" in rr.columns else rr.get("week"), errors="coerce").dt.normalize()
        cols = {c.lower(): c for c in rr.columns}
        c_rpc = cols.get("rpc") or cols.get("right party connects") or cols.get("right_party_connects")
        if c_rpc and c_rpc != "rpc":
            rr = rr.rename(columns={c_rpc: "rpc"})
        d = d.merge(rr[["date","rpc"]], on="date", how="left")

    if isinstance(cr, pd.DataFrame):
        d = d.merge(cr, on="date", how="left")
    if isinstance(rpcr, pd.DataFrame):
        d = d.merge(rpcr, on="date", how="left")

    if isinstance(aht, pd.DataFrame) and not aht.empty:
        ah = aht.copy()
        ah["date"] = pd.to_datetime(ah["date"] if "date" in ah.columns else ah.get("week"), errors="coerce").dt.normalize()
        cols = {c.lower(): c for c in ah.columns}
        c_aht = cols.get("aht_sec") or cols.get("talk_sec") or cols.get("avg_talk_sec") or cols.get("aht")
        if c_aht and c_aht != "aht_sec":
            ah = ah.rename(columns={c_aht: "aht_sec"})
        d = d.merge(ah[["date","aht_sec"]], on="date", how="left")

    d["program"] = "Outbound"
    # Ensure columns present
    for c in ["connect_rate","rpc","rpc_rate","aht_sec"]:
        if c not in d.columns:
            d[c] = None

    return d[["date","opc","connect_rate","rpc","rpc_rate","aht_sec","program"]]

def _snap_to_monday(value: str | dt.date | None) -> str:
    if not value:
        return ""
    return _monday(value).isoformat()

def _week_span(start_week: str | None, end_week: str | None, fallback_weeks: int = 12) -> list[str]:
    start = _monday(start_week) if start_week else _monday(dt.date.today())
    end   = _monday(end_week)   if end_week   else (pd.Timestamp(start) + pd.Timedelta(weeks=fallback_weeks-1)).date()
    if end < start:
        start, end = end, start
    weeks = []
    cur = pd.Timestamp(start)
    stop = pd.Timestamp(end)
    while cur <= stop:
        weeks.append(cur.date().isoformat())
        cur += pd.Timedelta(weeks=1)
    return weeks

def _week_cols(weeks: list[str]):
    today = dt.date.today()
    cols = [{"name": "Metric", "id": "metric", "editable": False}]
    week_ids: list[str] = []
    for w in weeks:
        wd = pd.to_datetime(w, errors="coerce")
        if pd.isna(wd):
            continue
        if not isinstance(wd, pd.Timestamp):
            wd = pd.Timestamp(wd)
        d = wd.date()
        tag = "Actual" if d <= today else "Plan"
        cols.append({"name": f"{tag}\n{d.strftime('%m/%d/%y')}", "id": d.isoformat()})
        week_ids.append(d.isoformat())
    return cols, week_ids

def _month_cols(weeks: list[str]):
    today = dt.date.today()
    # derive month starts covering the span of given week Mondays
    dts = [pd.to_datetime(w, errors="coerce") for w in weeks]
    dts = [pd.Timestamp(d) for d in dts if not pd.isna(d)]
    if not dts:
        return _week_cols(weeks)
    start = min(dts).to_period("M").to_timestamp()
    end   = max(dts).to_period("M").to_timestamp()
    months = []
    cur = start
    while cur <= end:
        months.append(cur.date())
        cur = (cur + pd.offsets.MonthBegin(1))
    cols = [{"name": "Metric", "id": "metric", "editable": False}]
    ids  = []
    for m in months:
        base = dt.date(m.year, m.month, 1)
        tag = "Actual" if base <= dt.date(today.year, today.month, 1) else "Plan"
        cols.append({"name": f"{tag}\n{m.strftime('%b %Y')}", "id": base.isoformat()})
        ids.append(base.isoformat())
    return cols, ids

def _blank_grid(metrics: List[str], week_ids: List[str]) -> pd.DataFrame:
    rows = []
    for m in metrics:
        r = {"metric": m}
        for wid in week_ids:
            r[wid] = 0.0
        rows.append(r)
    return pd.DataFrame(rows)

def _load_or_blank(key: str, metrics: List[str], week_ids: List[str]) -> pd.DataFrame:
    df = load_df(key)
    if isinstance(df, pd.DataFrame) and not df.empty:
        for wid in week_ids:
            if wid not in df.columns:
                df[wid] = 0.0
        if "metric" not in df.columns:
            df.insert(0, "metric", metrics[: len(df)])
        return df[["metric"] + week_ids].copy()
    return _blank_grid(metrics, week_ids)

def _round_week_cols_int(df: pd.DataFrame, week_ids: list[str]) -> pd.DataFrame:
    """Round all weekly numeric columns to integers (no decimals) for display."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out = df.copy()
    for wid in week_ids:
        if wid in out.columns:
            out[wid] = pd.to_numeric(out[wid], errors="coerce").fillna(0).round(0).astype(int)
    return out


# def _save_table(pid: int, tab_key: str, df: pd.DataFrame):
#     save_df(f"plan_{pid}_{tab_key}", df if isinstance(df, pd.DataFrame) else pd.DataFrame())

DEBUG_STORAGE_TRACE = True # flip to False to silence
def _trace(msg: str) -> None:
    if DEBUG_STORAGE_TRACE:
        print(msg)

def _save_table(pid: str | int, suffix: str, df: pd.DataFrame) -> None:
    """Centralized writer for plan tables (adds debug tracing)."""
    key = f"plan_{pid}_{suffix}"
    try:
        n = len(df) if isinstance(df, pd.DataFrame) else -1
    except Exception:
        n = -1
    _trace(f"[SAVE_TABLE] key={key} rows={n}")
    save_df(key, pd.DataFrame(df))

# ──────────────────────────────────────────────────────────────────────────────
# roster / bulk columns

def _roster_columns() -> List[dict]:
    names = [
        ("BRID", "brid"), ("Name", "name"), ("Class Reference", "class_ref"),
        ("Work Status", "work_status"), ("Role", "role"),
        ("FT/PT Status", "ftpt_status"), ("FT/PT Hours", "ftpt_hours"),
        ("Current Status", "current_status"),
        ("Training Start", "training_start"), ("Training End", "training_end"),
        ("Nesting Start", "nesting_start"), ("Nesting End", "nesting_end"),
        ("Production Start", "production_start"), ("Terminate Date", "terminate_date"),
        ("Team Leader", "team_leader"), ("AVP", "avp"),
        ("Business Area", "biz_area"), ("Sub Business Area", "sub_biz_area"),
        ("LOB", "lob"), ("LOA Date", "loa_date"), ("Back from LOA Date", "back_from_loa_date"),
        ("Site", "site"),
    ]
    cols = []
    for n, cid in names:
        cols.append({"name": n, "id": cid, "presentation": "input"})
    return cols

def _bulkfile_columns() -> List[dict]:
    return [
        {"name": "File Name", "id": "file_name"},
        {"name": "Extension", "id": "ext"},
        {"name": "File Size (in KB)", "id": "size_kb", "type": "numeric"},
        {"name": "Is Valid✅", "id": "is_valid"},
        {"name": "File Status", "id": "status"},
    ]

_ROSTER_REQUIRED_IDS = [c["id"] for c in _roster_columns()]

# plan_detail/_common.py



def _load_or_empty_bulk_files(pid: int) -> pd.DataFrame:
    cols = [c["id"] for c in _bulkfile_columns()]

    # sensible defaults for an empty grid
    empty = pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    if "size_kb" in empty.columns:
        empty["size_kb"] = empty["size_kb"].astype("float64")

    try:
        df = load_df(f"plan_{pid}_bulk_files")
    except Exception:
        # covers pandas.errors.EmptyDataError and any bad/corrupt payloads
        return empty

    if isinstance(df, pd.DataFrame) and not df.empty:
        # ensure required columns exist and types align
        for c in cols:
            if c not in df.columns:
                df[c] = "" if c != "size_kb" else 0.0
        # coerce size_kb numeric
        if "size_kb" in df.columns:
            df["size_kb"] = pd.to_numeric(df["size_kb"], errors="coerce").fillna(0.0)
        return df[cols].copy()

    return empty


def _load_or_empty_notes(pid: int) -> pd.DataFrame:
    """Notes table: always return a DF with ['when','user','note'] and handle empty payloads."""
    cols = ["when", "user", "note"]

    # canonical empty frame (preserve dtypes)
    empty = pd.DataFrame({c: pd.Series(dtype="object") for c in cols})

    try:
        df = load_df(f"plan_{pid}_notes")
    except Exception:
        # covers pandas.errors.EmptyDataError and other corrupt/empty payloads
        return empty

    if isinstance(df, pd.DataFrame) and not df.empty:
        # ensure required columns exist
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        # keep only expected columns / order
        return df[cols].copy()

    return empty


def _enrich_roster_from_headcount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing Team Leader and AVP (and optionally BA/SubBA/Site) from Headcount via BRID.
    - team_leader: headcount.line_manager_full_name for the employee BRID
    - avp:         two-hop -> the employee's manager's line_manager_full_name
    - biz_area:    headcount.journey
    - sub_biz_area:headcount.level_3
    - site:        headcount.position_location_building_description

    Only fills when target cells are empty/blank. Safe when headcount is empty.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

    try:
        hc = load_headcount()
    except Exception:
        hc = pd.DataFrame()
    if not isinstance(hc, pd.DataFrame) or hc.empty:
        return df

    H = {str(c).strip().lower(): c for c in hc.columns}
    def pick(name: str, *alts: str) -> str | None:
        for n in (name, *alts):
            c = H.get(str(n).strip().lower())
            if c:
                return c
        return None

    c_brid   = pick("brid")
    c_lm_b   = pick("line_manager_brid", "manager_brid", "tl_brid")
    c_lm_n   = pick("line_manager_full_name", "manager_name", "tl_name", "team_manager")
    c_ba     = pick("journey", "business area", "vertical", "current_org_unit_description")
    c_sba    = pick("level_3", "sub business area", "sub_business_area")
    c_site   = pick("position_location_building_description", "building", "site")

    if not c_brid:
        return df

    # Build maps
    hc2 = hc[[c_brid] + [c for c in [c_lm_b, c_lm_n, c_ba, c_sba, c_site] if c]].copy()
    for c in hc2.columns:
        if hc2[c].dtype == object:
            hc2[c] = hc2[c].astype(str).str.strip()

    lm_brid_by_emp = dict(zip(hc2[c_brid], hc2[c_lm_b] if c_lm_b else [None]*len(hc2)))
    lm_name_by_emp = dict(zip(hc2[c_brid], hc2[c_lm_n] if c_lm_n else [None]*len(hc2)))
    ba_by_emp      = dict(zip(hc2[c_brid], hc2[c_ba] if c_ba else [None]*len(hc2)))
    sba_by_emp     = dict(zip(hc2[c_brid], hc2[c_sba] if c_sba else [None]*len(hc2)))
    site_by_emp    = dict(zip(hc2[c_brid], hc2[c_site] if c_site else [None]*len(hc2)))

    def _avp_for(brid: str) -> str | None:
        tlb = (lm_brid_by_emp.get(brid) or "").strip()
        if not tlb:
            return None
        # manager's manager full name
        return (lm_name_by_emp.get(tlb) or None)

    out = df.copy()
    L = {str(c).strip().lower(): c for c in out.columns}
    col_brid = L.get("brid")
    col_tl   = L.get("team_leader")
    col_avp  = L.get("avp")
    col_ba   = L.get("biz_area")
    col_sba  = L.get("sub_biz_area")
    col_site = L.get("site")

    if not col_brid:
        return out

    # Normalize types
    out[col_brid] = out[col_brid].astype(str).str.strip()
    for c in [col_tl, col_avp, col_ba, col_sba, col_site]:
        if c and out[c].dtype == object:
            out[c] = out[c].astype(str)

    # Fill Team Leader
    if col_tl:
        mask_tl = out[col_tl].astype(str).str.strip().eq("")
        out.loc[mask_tl, col_tl] = out.loc[mask_tl, col_brid].map(lambda b: (lm_name_by_emp.get(b) or ""))

    # Fill AVP (two-hop via manager's manager)
    if col_avp:
        mask_avp = out[col_avp].astype(str).str.strip().eq("")
        out.loc[mask_avp, col_avp] = out.loc[mask_avp, col_brid].map(lambda b: (_avp_for(b) or ""))

    # Fill hierarchy helpers if blank
    if col_ba:
        mask_ba = out[col_ba].astype(str).str.strip().eq("")
        out.loc[mask_ba, col_ba] = out.loc[mask_ba, col_brid].map(lambda b: (ba_by_emp.get(b) or ""))
    if col_sba:
        mask_sba = out[col_sba].astype(str).str.strip().eq("")
        out.loc[mask_sba, col_sba] = out.loc[mask_sba, col_brid].map(lambda b: (sba_by_emp.get(b) or ""))
    if col_site:
        mask_site = out[col_site].astype(str).str.strip().eq("")
        out.loc[mask_site, col_site] = out.loc[mask_site, col_brid].map(lambda b: (site_by_emp.get(b) or ""))

    return out


def _parse_upload(contents: str, filename: str) -> Tuple[pd.DataFrame, dict]:
    if not contents or not filename:
        return pd.DataFrame(), {}
    try:
        header, b64 = contents.split(",", 1)
    except ValueError:
        return pd.DataFrame(), {"file_name": filename, "ext": "", "size_kb": 0, "is_valid": "No", "status": "Invalid format"}

    raw = base64.b64decode(b64)
    ext = filename.split(".")[-1].lower()
    try:
        if ext in ("csv",):
            df = pd.read_csv(io.BytesIO(raw))
        elif ext in ("xlsx","xls"):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            return pd.DataFrame(), {"file_name": filename, "ext": ext, "size_kb": round(len(raw)/1024,1),
                                    "is_valid": "No", "status": "Unsupported"}
    except Exception:
        return pd.DataFrame(), {"file_name": filename, "ext": ext, "size_kb": round(len(raw)/1024,1),
                                "is_valid": "No", "status": "Read Error"}

    rename_map = {c["name"]: c["id"] for c in _roster_columns()}
    lower_map = {k.lower(): v for k,v in rename_map.items()}
    df = df.rename(columns={col: lower_map.get(str(col).lower(), col) for col in df.columns})

    # Treat Team Leader + AVP as optional; we can enrich them from Headcount by BRID
    optional_ids = {"team_leader", "avp"}
    missing_all = [cid for cid in _ROSTER_REQUIRED_IDS if cid not in df.columns]
    missing_req = [cid for cid in missing_all if cid not in optional_ids]

    valid = len(missing_req) == 0
    status_msg = "Loaded" if valid else f"Missing: {', '.join(missing_req[:3])}"
    ledger = {"file_name": filename, "ext": ext, "size_kb": round(len(raw)/1024,1),
              "is_valid": "Yes" if valid else "No", "status": status_msg}
    if not valid:
        return pd.DataFrame(), ledger

    # Ensure optional columns exist even if not provided
    for opt in optional_ids:
        if opt not in df.columns:
            df[opt] = ""

    # Keep known columns, create any other missing ones as empty to satisfy downstream order
    for cid in _ROSTER_REQUIRED_IDS:
        if cid not in df.columns:
            df[cid] = ""
    df = df[_ROSTER_REQUIRED_IDS].copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    # Enrich missing TL/AVP (and a few hierarchy hints) from Headcount via BRID
    df = _enrich_roster_from_headcount(df)
    return df, ledger

def _format_crumb(p: dict) -> str:
    ba  = (p.get("business_area") or p.get("plan_ba") or p.get("ba") or p.get("vertical") or "").strip()
    sba = (p.get("sub_business_area") or p.get("plan_sub_ba") or p.get("sub_ba") or p.get("subba") or "").strip()
    lob = (p.get("lob") or p.get("channel") or "").strip()
    if lob:
        lob = lob.title()
    site = (p.get("site") or "").strip()
    parts = [x for x in [ba, sba, lob, site] if x]
    return " > ".join(parts)

# ──────────────────────────────────────────────────────────────────────────────
# NEW: Build BA -> Sub-BA (Level 3) and Sites from Headcount Update only

def _load_hcu_df() -> pd.DataFrame:
    try:
        df = load_headcount()
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def _build_hierarchy_sites_from_headcount() -> tuple[dict[str, list[str]], list[str]]:
    df = _load_hcu_df()
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}, []

    L = {str(c).strip().lower(): c for c in df.columns}

    ba_col   = L.get("journey") or L.get("business area") or L.get("vertical") \
               or L.get("current_org_unit_description") or L.get("current org unit description") \
               or L.get("current_org_unit") or L.get("current org unit") \
               or L.get("level_0") or L.get("level 0")
    sba_col  = L.get("level_3") or L.get("level 3") or L.get("sub business area") or L.get("sub_business_area")
    site_col = L.get("position_location_building_description") or L.get("building") \
               or L.get("building description") or L.get("site")

    hmap: dict[str, set[str]] = {}
    if ba_col:
        for _, r in df.iterrows():
            ba = str(r.get(ba_col, "")).strip()
            if not ba:
                continue
            sba_val = str(r.get(sba_col, "")).strip() if sba_col else ""
            hmap.setdefault(ba, set())
            if sba_val:
                hmap[ba].add(sba_val)

    out_hmap = {ba: sorted(list(subs)) for ba, subs in hmap.items()}
    sites: list[str] = []
    if site_col:
        s = (
            df[site_col].dropna().astype(str).str.strip().replace({"": np.nan}).dropna().unique().tolist()
        )
        sites = sorted(set(s))
    return out_hmap, sites

def _hier_from_hcu() -> dict:
    hmap, sites = _build_hierarchy_sites_from_headcount()  # {BA: [SubBAs]}, [sites]
    bas = sorted(hmap.keys())
    sub_map = {ba: sorted(list(subs or [])) for ba, subs in (hmap or {}).items()}

    lob_map = {}
    for ba, subs in sub_map.items():
        for sba in subs:
            lob_map[f"{ba}|{sba}"] = list(CHANNEL_DEFAULTS)

    return {"ba": bas, "subba": sub_map, "lob": lob_map, "site": sorted(sites or [])}

# ──────────────────────────────────────────────────────────────────────────────
# NEW: generic weekly loaders & small parsers

# def _first_non_empty_ts(scope_key: str, keys: list[str]) -> pd.DataFrame:
#     """Return the first non-empty timeseries DF for any of the given keys."""
#     for k in keys:
#         try:
#             df = load_timeseries(k, scope_key)
#             if isinstance(df, pd.DataFrame) and not df.empty:
#                 return df.copy()
#         except Exception:
#             pass
#     return pd.DataFrame()

def _first_non_empty_ts(scope_key: str, keys: list[str]) -> pd.DataFrame:
    """Return the first non-empty timeseries DF for any of the given keys (4→3 fallback)."""
    for k in keys:
        df = _load_ts_with_fallback(k, scope_key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy()
    return pd.DataFrame()

def _strip_site(scope_key: str) -> str:
    """Return the 3-part (BA|SubBA|Channel) from a 4-part key; else identity."""
    parts = (scope_key or "").split("|")
    return "|".join(parts[:3]) if len(parts) >= 4 else scope_key

def _load_ts_with_fallback(ts_key: str, scope_key: str):
    """
    Try timeseries with 4-part key; if empty and key has site, retry with 3-part.
    Why: allows reading both new (site) and legacy data seamlessly.
    """
    try:
        df = load_timeseries(ts_key, scope_key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass
    sk3 = _strip_site(scope_key)
    if sk3 != scope_key:
        try:
            df = load_timeseries(ts_key, sk3)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            pass
    return pd.DataFrame()

def _weekly_reduce(df: pd.DataFrame, value_candidates=("value","hc","headcount","hours","items","volume","count","amt","amount"),
                   how: str = "sum") -> dict:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    date_col = None
    for c in ("date","dt","day","when"):
        if c in df.columns:
            date_col = c
            break
    if not date_col:
        # try to cast an index to dates
        if isinstance(df.index, (pd.DatetimeIndex, pd.Index)):
            df = df.reset_index().rename(columns={df.columns[0]:"date"})
            date_col = "date"
        else:
            return {}
    val_col = None
    for c in value_candidates:
        if c in df.columns:
            val_col = c
            break
    if not val_col:
        # single numeric column fallback
        nums = [c for c in df.columns if c != date_col and np.issubdtype(df[c].dtype, np.number)]
        if nums:
            val_col = nums[0]
        else:
            return {}

    d = df[[date_col, val_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    d["week"] = (d[date_col] - pd.to_timedelta(d[date_col].dt.weekday, unit="D")).dt.date.astype(str)
    d[val_col] = pd.to_numeric(d[val_col], errors="coerce").fillna(0.0)

    if how == "mean":
        s = d.groupby("week", as_index=False)[val_col].mean().set_index("week")[val_col]
    else:
        s = d.groupby("week", as_index=False)[val_col].sum().set_index("week")[val_col]
    return s.to_dict()

def _parse_ratio_setting(v) -> float:
    try:
        if isinstance(v, str) and ":" in v:
            a, b = v.split(":", 1)
            a = float(str(a).strip()); b = float(str(b).strip())
            return (a / b) if b else 0.0
        return float(v)
    except Exception:
        return 0.0

# Final override: ensure the normalized roster loader is the active definition
def _load_or_empty_roster(pid):
    """Load roster normalized to current _roster_columns() ids (backward compatible)."""
    cols = [c["id"] for c in _roster_columns()]
    empty = pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    try:
        df = load_df(f"plan_{pid}_emp")
    except Exception:
        return empty

    if not isinstance(df, pd.DataFrame) or df.empty:
        return empty
    df = df.copy()
    # Map legacy short names -> canonical
    if "ftpt" in df.columns and "ftpt_status" not in df.columns:
        df["ftpt_status"] = df["ftpt"]
    if "tl" in df.columns and "team_leader" not in df.columns:
        df["team_leader"] = df["tl"]
    if "status" in df.columns:
        if "work_status" not in df.columns:
            df["work_status"] = df["status"]
        if "current_status" not in df.columns:
            df["current_status"] = df["status"]
    # Legacy date fields
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


# ──────────────────────────────────────────────────────────────────────────────
