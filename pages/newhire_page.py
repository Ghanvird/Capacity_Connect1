# file: pages/newhire_page.py
from __future__ import annotations
from dash import html, dcc, dash_table, Output, Input, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from app_instance import app
from common import *  # noqa

# --- New Hire: canonical schema used by BOTH the Summary page and plan tabs ---

NH_LABELS = {
    "business_area": "Business Area",
    "class_reference": "Class Reference",
    "source_system_id": "Source System ID",
    "emp_type": "Emp Type",
    "status": "Status",
    "class_type": "Class Type",
    "class_level": "Class Level",
    "grads_needed": "Grads Needed",
    "billable_hc": "Billable HC",
    "training_weeks": "Training Weeks",
    "nesting_weeks": "Nesting Weeks",
    "induction_start": "Induction Start",
    "training_start": "Training Start",
    "training_end": "Training End",
    "nesting_start": "Nesting Start",
    "nesting_end": "Nesting End",
    "production_start": "Production Start",
    "created_by": "Created By",
    "created_ts": "Created On",
}

NH_COLS = [
    "business_area",
    "class_reference",
    "source_system_id",
    "emp_type",            # full-time | part-time
    "status",              # tentative | confirmed
    "class_type",          # ramp-up | backfill
    "class_level",
    "grads_needed",
    "billable_hc",
    "training_weeks",
    "nesting_weeks",
    "induction_start",
    "training_start",
    "training_end",
    "nesting_start",
    "nesting_end",
    "production_start",
    "created_by",
    "created_ts",
]

def new_hire_template_df() -> pd.DataFrame:
    """
    One-row template with correct columns. Users can duplicate rows.
    class_reference & source_system_id are left blank on purpose;
    your upload code auto-generates them.
    """
    sample = {
        "business_area": "Example BA",
        "class_reference": "",     # auto
        "source_system_id": "",    # auto
        "emp_type": "full-time",   # full-time | part-time
        "status": "tentative",     # tentative | confirmed
        "class_type": "ramp-up",   # ramp-up | backfill
        "class_level": "new-agent",
        "grads_needed": 10,
        "billable_hc": 0,
        "training_weeks": 2,
        "nesting_weeks": 1,
        "induction_start": "",
        "training_start": "",      # YYYY-MM-DD
        "training_end": "",
        "nesting_start": "",
        "nesting_end": "",
        "production_start": "",
        "created_by": "",
        "created_ts": "",
    }
    return pd.DataFrame([sample])[NH_COLS]

def get_class_type_options():
    return [{"label": "Ramp-Up", "value": "ramp-up"},
            {"label": "Backfill", "value": "backfill"}]

def get_class_level_options():
    return [{"label": "New Agent", "value": "new-agent"},
            {"label": "Cross-Skill", "value": "cross-skill"},
            {"label": "Up-Skill", "value": "up-skill"}]

def current_user_fallback():
    import os, getpass
    return os.environ.get("USERNAME") or os.environ.get("USER") or getpass.getuser() or "system"

def _iso_date(v):
    import pandas as pd
    if v in (None, "", "nan"): return None
    t = pd.to_datetime(v, errors="coerce")
    if pd.isna(t): return None
    return t.date().isoformat()

def _auto_dates(training_start, training_weeks, nesting_start, nesting_weeks, production_start):
    from datetime import timedelta, date
    import pandas as pd
    def to_date(d):
        if not d: return None
        if isinstance(d, date): return d
        return pd.to_datetime(d, errors="coerce").date()
    ts = to_date(training_start)
    ns = to_date(nesting_start)
    ps = to_date(production_start)
    tw = int(training_weeks or 0)
    nw = int(nesting_weeks or 0)
    te = ts + timedelta(days=7*tw) - timedelta(days=1) if (ts and tw>0) else None
    if ns is None and te is not None:
        ns = te + timedelta(days=1)
    ne = ns + timedelta(days=7*nw) - timedelta(days=1) if (ns and nw>0) else None
    if ps is None and ne is not None:
        ps = ne + timedelta(days=1)
    return _iso_date(ts), _iso_date(te), _iso_date(ns), _iso_date(ne), _iso_date(ps)

def _ensure_nh_cols(df):
    import pandas as pd
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=NH_COLS)
    out = df.copy()
    for c in NH_COLS:
        if c not in out.columns:
            if c in ("grads_needed","billable_hc","training_weeks","nesting_weeks"):
                out[c] = 0
            elif c == "created_by":
                out[c] = current_user_fallback()
            elif c == "created_ts":
                out[c] = pd.Timestamp.utcnow().isoformat(timespec="seconds")
            else:
                out[c] = None
    return out[NH_COLS + [c for c in out.columns if c not in NH_COLS]]

def load_hiring():
    """Global master list used by the New Hire Summary page."""
    import pandas as pd
    try:
        df = load_df("nh_classes_master")
    except Exception:
        df = pd.DataFrame(columns=NH_COLS)
    return _ensure_nh_cols(df)

def save_hiring(df):
    save_df("nh_classes_master", _ensure_nh_cols(df))

def next_class_reference(pid: str | None = None, existing_df=None, ba: str | None = None) -> str:
    """Readable, unique class ref. If BA supplied, include a short BA tag. pid is optional."""
    import re, datetime as dt, pandas as pd
    df = existing_df if hasattr(existing_df, "columns") else load_hiring()
    today = dt.date.today().strftime("%Y%m%d")
    ba_tag = ""
    if ba:
        s = re.sub(r"[^A-Za-z0-9]", "", str(ba))
        ba_tag = f"-{s[:6].upper()}" if s else ""
    head = f"NH-{today}{ba_tag}" if not pid else f"NH-{pid}-{today}{ba_tag}"
    seq = 1
    if "class_reference" in df.columns and not df.empty:
        m = df["class_reference"].astype(str).str.extract(rf"^{re.escape(head)}-(\d+)$", expand=False).dropna()
        if not m.empty:
            seq = int(m.astype(int).max()) + 1
    return f"{head}-{seq:02d}"

def nh_template_df():
    """Template offered on 'Download Sample' (same fields users can upload)."""
    import pandas as pd
    cols = [
        "business_area","emp_type","status","class_type","class_level",
        "grads_needed","billable_hc","training_weeks","nesting_weeks",
        "induction_start","training_start","nesting_start","production_start",
        "class_reference","source_system_id"
    ]
    return pd.DataFrame(columns=cols)

def _nh_effective_count(row) -> int:
    """FT = grads; PT = ceil(grads/2); billable_hc overrides when > 0."""
    import math, pandas as pd
    bill = pd.to_numeric(row.get("billable_hc"), errors="coerce")
    if pd.notna(bill) and bill > 0:
        return int(bill)
    grads = int(pd.to_numeric(row.get("grads_needed"), errors="coerce") or 0)
    emp   = str(row.get("emp_type","")).strip().lower()
    return int(math.ceil(grads/2.0)) if emp == "part-time" else int(grads)

def normalize_nh_upload_master(raw_df, source_id: str | None = None, default_ba: str | None = None):
    """Map arbitrary columns into the canonical schema and auto-assign IDs."""
    import pandas as pd, numpy as np, datetime as dt
    if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
        return _ensure_nh_cols(raw_df)

    d = raw_df.copy()
    L = {str(c).strip().lower(): c for c in d.columns}
    def pick(*names):
        for n in names:
            if n in L: return L[n]
        return None

    col_map = dict(
        business_area = pick("business area","ba","journey","vertical","business_area"),
        emp_type      = pick("emp type","emptype","ft/pt","ftpt","emp_type"),
        status        = pick("status","class status","class_status"),
        class_type    = pick("class type","classtype","class_type"),
        class_level   = pick("class level","classlevel","class_level"),
        grads_needed  = pick("grads needed","graduates","grads","headcount","grads_needed"),
        billable_hc   = pick("billable hc","billable","billable_headcount","billable_hc"),
        training_weeks= pick("training weeks","train weeks","training_weeks"),
        nesting_weeks = pick("nesting weeks","nest weeks","nesting_weeks"),
        induction_start = pick("induction start","induction_start"),
        training_start  = pick("training start","training_start","train start"),
        training_end    = pick("training end","training_end","train end"),
        nesting_start   = pick("nesting start","nesting_start"),
        nesting_end     = pick("nesting end","nesting_end"),
        production_start= pick("production start","production_start","go live","golive"),
        class_reference = pick("class reference","class_reference","class"),
        source_system_id= pick("source system id","source_system_id","source id","sourceid"),
    )

    out = pd.DataFrame()
    for k in [
        "business_area","emp_type","status","class_type","class_level",
        "grads_needed","billable_hc","training_weeks","nesting_weeks",
        "induction_start","training_start","training_end","nesting_start","nesting_end","production_start",
        "class_reference","source_system_id"
    ]:
        src = col_map.get(k)
        out[k] = d[src] if src in d else None

    # normalize text
    out["business_area"] = out["business_area"].fillna(default_ba).astype(object)
    out["emp_type"] = out["emp_type"].astype(str).str.lower().replace({
        "full time":"full-time","fulltime":"full-time","ft":"full-time","pt":"part-time","nan":None,"none":None,"":None
    })
    out["status"] = out["status"].astype(str).str.lower().replace({"nan":None,"none":None,"":None})
    out["class_type"] = out["class_type"].astype(str).str.lower().replace({
        "ramp up":"ramp-up","rampup":"ramp-up","nan":None,"none":None,"":None
    })
    out["class_level"] = out["class_level"].astype(str).str.lower().replace({
        "newagent":"new-agent","new agent":"new-agent","cross skill":"cross-skill","up skill":"up-skill",
        "nan":None,"none":None,"":None
    })
    for n in ("grads_needed","billable_hc","training_weeks","nesting_weeks"):
        out[n] = pd.to_numeric(out[n], errors="coerce").fillna(0).astype(int)

    # dates + auto-derivations
    for i, r in out.iterrows():
        ts, te, ns, ne, ps = _auto_dates(
            r.get("training_start"), r.get("training_weeks"),
            r.get("nesting_start"),  r.get("nesting_weeks"),
            r.get("production_start")
        )
        out.at[i,"induction_start"] = _iso_date(r.get("induction_start"))
        out.at[i,"training_start"]  = ts
        out.at[i,"training_end"]    = te
        out.at[i,"nesting_start"]   = ns
        out.at[i,"nesting_end"]     = ne
        out.at[i,"production_start"]= ps

    # one Source System Id for the whole upload (if missing)
    batch_id = source_id or f"upload-{current_user_fallback()}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d%H%M%S')}"
    out["source_system_id"] = out["source_system_id"].replace({None: batch_id, "": batch_id})

    # unique Class Reference per Business Area (fill only if missing)
    existing = load_hiring()
    for ba_val, idx in out.groupby(out["business_area"].fillna("").astype(str)).groups.items():
        for i in idx:
            if not str(out.at[i,"class_reference"] or "").strip():
                out.at[i,"class_reference"] = next_class_reference(existing_df=existing, ba=ba_val)
                # extend existing so subsequent rows get incremented sequence
                existing = pd.concat([existing, out.loc[[i], existing.columns.intersection(["class_reference"])]], ignore_index=True)

    out["created_by"] = current_user_fallback()
    out["created_ts"] = pd.Timestamp.utcnow().isoformat(timespec="seconds")
    return _ensure_nh_cols(out)


def page_new_hire():
    df = load_hiring()
    cols = pretty_columns(NH_COLS)  # canonical schema
    return html.Div(dbc.Container([
        header_bar(),
        html.H4("New Hire Summary", className="ghanii"),
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(dcc.Upload(
                    id="up-hire",
                    children=html.Div(["⬆️ Upload CSV/XLSX"], className="upload-box"),
                    multiple=False,
                    className="upload-box"
                ), md=4),
                dbc.Col(dbc.Button("Save", id="btn-save-hire", color="primary"), md=2),
                dbc.Col(html.Div(html.Span(id="hire-save-msg", className="text-success"), className="loading-block"), md=3),
                dbc.Col(dbc.Button("Download Sample", id="btn-dl-hire", outline=True, color="secondary"), md=3),
            ], className="my-2"),

            dcc.Download(id="dl-hire-sample"),

            html.Div(dash_table.DataTable(
                id="tbl-hire",
                columns=[{"name": NH_LABELS[c], "id": c} for c in NH_COLS],
                data=df.to_dict("records"),
                editable=True, row_deletable=True, page_size=10,
                style_table={"overflowX":"auto"},
                style_as_list_view=True,
                style_header={"textTransform":"none"},
            ), className="loading-block"),
            html.Hr(),
            html.Div(dcc.Graph(id="fig-hire", style={"height":"280px"}, config={"displayModeBar": False}), className="loading-block"),
        ])),
    ], fluid=True), className="loading-page")



