# file: plan_detail/_ui.py
from __future__ import annotations
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from ._common import *  # noqa
from datetime import date, timedelta
from ._common import _load_or_empty_roster
from dash import html, dcc, dash_table, Input, Output, State, callback, ctx, no_update
from plan_detail._expand_week_bar import expand_week_bar_component
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import date, timedelta
from plan_detail._common import (
    get_class_type_options, get_class_level_options,
    load_nh_classes, save_nh_classes,
    current_user_fallback, next_class_reference,
    _load_or_empty_roster, save_df, load_df
)
from datetime import date, timedelta
import pandas as pd

# ---- Active Plan ID helper ---------------------------------------------------
from flask import request, session, has_request_context, current_app
from urllib.parse import urlparse, parse_qs

def _safe_session_get(*keys):
    try:
        if has_request_context() and getattr(current_app, "secret_key", None):
            for k in keys:
                v = session.get(k)
                if v not in (None, ""):
                    return str(v)
    except Exception:
        pass
    return None

def _safe_session_set(key, value):
    try:
        if has_request_context() and getattr(current_app, "secret_key", None):
            session[key] = str(value)
    except Exception:
        # swallow: session not available (e.g., no SECRET_KEY or no context)
        pass

def set_active_plan_id(pid: str | None) -> None:
    _safe_session_set("active_plan_id", pid or "")

def _pid_from_layout(default: str = "") -> str:
    # 1) Try session
    pid = _safe_session_get("active_plan_id", "plan_pid")
    if pid:
        return pid

    # 2) Parse current request URL (or referrer)
    def _parse_pid_from_url(url: str | None) -> str | None:
        if not url:
            return None
        try:
            parsed = urlparse(url)
            q = parse_qs(parsed.query or "")
            cand = q.get("pid", [None])[0] or q.get("plan_id", [None])[0]
            if cand:
                return str(cand).strip()
            parts = [p for p in (parsed.path or "").split("/") if p]
            for i, seg in enumerate(parts):
                if seg.lower() in ("plan", "plans", "capacity", "cap-plan", "cap", "detail"):
                    if i + 1 < len(parts) and parts[i + 1]:
                        return str(parts[i + 1]).strip()
            if parts and parts[-1] and parts[-1].strip():
                return str(parts[-1].strip())
        except Exception:
            return None
        return None

    pid = _parse_pid_from_url(getattr(request, "url", None)) or _parse_pid_from_url(getattr(request, "referrer", None))
    if pid:
        _safe_session_set("active_plan_id", pid)  # best-effort
        return pid

    return str(default or "")


# ---- NH helpers --------------------------------------------------------------
def _auto_dates(training_start, training_weeks, nesting_start, nesting_weeks, production_start):
    def to_date(d):
        if not d: return None
        if isinstance(d, date): return d
        return pd.to_datetime(d, errors="coerce").date()
    ts  = to_date(training_start)
    ns  = to_date(nesting_start)
    ps  = to_date(production_start)
    tw  = int(training_weeks or 0)
    nw  = int(nesting_weeks or 0)
    te = ts + timedelta(days=7*tw) - timedelta(days=1) if (ts and tw>0) else None
    if ns is None and te is not None:
        ns = te + timedelta(days=1)
    ne = ns + timedelta(days=7*nw) - timedelta(days=1) if (ns and nw>0) else None
    if ps is None and ne is not None:
        ps = ne + timedelta(days=1)
    iso = lambda d: (d.date().isoformat() if isinstance(d, pd.Timestamp) else (d.isoformat()) if d is not None else None)
    return iso(ts), iso(te), iso(ns), iso(ne), iso(ps)

def _update_roster_by_class(pid: str, class_ref: str, ts, te, ns, ne, ps):
    # uses _load_or_empty_roster, save_df already imported from _common
    roster = _load_or_empty_roster(pid)
    if not isinstance(roster, pd.DataFrame) or roster.empty or not class_ref:
        return 0
    L = {str(c).strip().lower(): c for c in roster.columns}
    c_ref = L.get("class reference") or L.get("class_reference") or L.get("class") or L.get("classref")
    if not c_ref:
        return 0

    def pick_or_create(*names):
        for n in names:
            if n in roster.columns:
                return n
        friendly = names[0]
        roster[friendly] = None
        return friendly

    c_ts = pick_or_create("training start", "training_start", "training start date", "training_starts_on")
    c_te = pick_or_create("training end", "training_end", "training end date")
    c_ns = pick_or_create("nesting start", "nesting_start", "nesting start date", "nesting_starts_on")
    c_ne = pick_or_create("nesting end", "nesting_end", "nesting end date")
    c_ps = pick_or_create("production start", "production_start", "production start date", "production_starts_on")

    m = roster[c_ref].astype(str).str.strip().str.lower() == str(class_ref).strip().lower()
    n_before = int(m.sum())
    if n_before:
        if ts: roster.loc[m, c_ts] = ts
        if te: roster.loc[m, c_te] = te
        if ns: roster.loc[m, c_ns] = ns
        if ne: roster.loc[m, c_ne] = ne
        if ps: roster.loc[m, c_ps] = ps
        # ? correct key
        save_df(f"plan_{pid}_emp", roster)
    return n_before



# --- New Hire helpers (robust to older saved payloads) -----------------------
# --- New Hire: column/IO helpers (keep these with your other helpers) ----------
NH_STORE_KEY = "plan_{pid}_nh_classes"
NH_COLS = [
    "class_reference","source_system_id","emp_type","status","class_type","class_level",
    "grads_needed","billable_hc","training_weeks","nesting_weeks",
    "induction_start","training_start","training_end","nesting_start","nesting_end","production_start",
    "created_ts",
]

def _ensure_nh_cols(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=NH_COLS)
    for c in NH_COLS:
        if c not in df.columns:
            if c in ("grads_needed","billable_hc","training_weeks","nesting_weeks"):
                df[c] = 0
            elif c == "created_ts":
                df[c] = pd.Timestamp.utcnow().isoformat(timespec="seconds")
            else:
                df[c] = None
    # tidy order
    keep = [c for c in NH_COLS if c in df.columns]
    keep += [c for c in df.columns if c not in keep]
    return df[keep]

def _load_nh_classes(pid: int) -> pd.DataFrame:
    try:
        df = load_df(NH_STORE_KEY.format(pid=pid))
    except Exception:
        df = pd.DataFrame(columns=NH_COLS)
    return _ensure_nh_cols(df if isinstance(df, pd.DataFrame) else pd.DataFrame())

def _save_nh_classes(pid: int, df: pd.DataFrame):
    save_df(NH_STORE_KEY.format(pid=pid), _ensure_nh_cols(df))

_OVERLAY_STYLE = {
    "position": "fixed",
    "inset": "0",
    "background": "rgba(0,0,0,0.6)",
    "display": "none",                   # toggled by callback
    "alignItems": "center",
    "justifyContent": "center",
    "flexDirection": "column",
    "zIndex": 9999
}
def _loading_overlay() -> html.Div:
    return html.Div(
        id="plan-loading-overlay",
        children=[
            html.Img(src="/assets/Infinity.svg", style={"width": "96px", "height": "96px"}, className="avy"),
            html.Div("Preparing your plan...", style={"color": "white", "marginLeft": "10px"})
        ],
        style=_OVERLAY_STYLE
    )
# layout builders
toggle_switch = dbc.Switch(id="toggle-switch", value=False, className="me-2 gha", label="")

def _upper_summary_header_card() -> html.Div:
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Div([
                    dcc.Link(dbc.Button("🢀", id="plan-hdr-back", color="light", title="Back"),
                             href="/planning", className="me-2"),
                    html.Span(id="plan-hdr-name", className="fw-bold")
                ], className="d-flex align-items-center"),
                html.Div([
                    dbc.Button("💾", id="btn-plan-save", color="light", title="Save", className="me-1"),
                    dbc.Button("🔄", id="btn-plan-refresh", color="light", title="Refresh", className="me-1"),
                    toggle_switch,
                    html.Div(id="plan-msg", className="text-success mt-2"),
                ], style={"display":"flex"}),
                html.Div([
                    dbc.Button("?", id="plan-hdr-collapse", color="light", title="Collapse/Expand")
                ]),
            ], className="d-flex justify-content-between align-items-center mb-2 hhh"),
        ], style={"padding": "3px"}),
        className="mb-3"
    )

def _upper_summary_body_card() -> html.Div:
    return dbc.Card(
        dbc.CardBody([
            html.Div(id="plan-upper", className="cp-grid")
        ], class_name="gaurav"),
        className="mb-3"
    )

def _lower_tabs() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            dbc.Tabs(id="plan-tabs", active_tab="tab-fw", children=[
                # Auto-computed tabs ? editable=False
                dbc.Tab(label="Forecast & Workload", tab_id="tab-fw",
                        children=[
                            # expand_week_bar_component(),
                            dash_table.DataTable(id="tbl-fw", editable=True,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)
                        ], class_name="ddd"),
                dbc.Tab(label="Headcount", tab_id="tab-hc",
                        children=[dash_table.DataTable(id="tbl-hc", editable=False,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)], class_name="ddd"),
                dbc.Tab(label="Attrition", tab_id="tab-attr",
                        children=[dash_table.DataTable(id="tbl-attr", editable=True,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)], class_name="ddd"),
                dbc.Tab(label="Shrinkage", tab_id="tab-shr",
                        children=[dash_table.DataTable(id="tbl-shr", editable=False,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"WhiteSpace":"pre"},
                                                       page_size=12)], class_name="ddd"),

                # User-editable tabs ? editable=True (keep as-is)
                dbc.Tab(label="Training Lifecycle", tab_id="tab-train",
                        children=[dash_table.DataTable(id="tbl-train", editable=True,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)], class_name="ddd"),
                dbc.Tab(label="Ratios", tab_id="tab-ratio",
                        children=[dash_table.DataTable(id="tbl-ratio", editable=True,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)], class_name="ddd"),
                dbc.Tab(label="Seat Utilization", tab_id="tab-seat",
                        children=[dash_table.DataTable(id="tbl-seat", editable=True,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)], class_name="ddd"),
                dbc.Tab(label="Budget vs Actual", tab_id="tab-bva",
                        children=[dash_table.DataTable(id="tbl-bva", editable=False,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)], class_name="ddd"),
                # In _lower_tabs(), replace the current "New Hire" tab's children with both the UI and the old grid:
                dbc.Tab(label="New Hire", tab_id="tab-nh", children=[
                    *_new_hire_ui(),
                    dash_table.DataTable(
                        id="tbl-nh", editable=True,
                        style_as_list_view=True,
                        style_table={"overflowX":"auto"},
                        style_header={"whiteSpace":"pre"},
                        page_size=12
                    ),
                ]),


                # Employee roster & Notes (unchanged)
                dbc.Tab(label="Employee Roster", tab_id="tab-roster", children=[
                    dbc.Tabs([
                        dbc.Tab(label="Roster", tab_id="tab-roster-main", children=[
                            html.Div([
                                dbc.Button("+ Add new", id="btn-emp-add", className="me-1", color="secondary"),
                                dbc.Button("Transfer & Promotion", id="btn-emp-tp", className="me-1", color="secondary", disabled=True),
                                dbc.Button("Move to LOA", id="btn-emp-loa", className="me-1", color="secondary", disabled=True),
                                dbc.Button("Back from LOA", id="btn-emp-back", className="me-1", color="secondary", disabled=True),
                                dbc.Button("Terminate", id="btn-emp-term", className="me-1", color="secondary", disabled=True),
                                dbc.Button("FT/PT Conversion", id="btn-emp-ftp", className="me-1", color="secondary", disabled=True),
                                dbc.Button("Undo", id="btn-emp-undo", className="me-1", color="secondary", disabled=True),
                                dbc.Button("Change Class", id="btn-emp-class", className="me-1", color="secondary", disabled=True),
                                dbc.Button("Remove", id="btn-emp-remove", className="me-2", color="secondary", disabled=True),
                                html.Span("Total: 00 Records", id="lbl-emp-total", className="me-2"),
                                dbc.Button("Workstatus Dataset", id="btn-emp-dl", color="warning", outline=True),
                                dcc.Download(id="dl-workstatus"),
                            ], className="mb-2 ashwini"),
                            dash_table.DataTable(
                                id="tbl-emp-roster",
                                editable=True,
                                row_selectable=False,
                                selected_rows=[],
                                style_as_list_view=True,
                                style_table={"overflowX": "auto"},
                                page_size=10,
                            ),
                        ]),
                        dbc.Tab(label="Bulk Upload", tab_id="tab-roster-bulk", children=[
                            html.Div([
                                dcc.Upload(id="up-roster-bulk", children=html.Div(["⬆️ Upload CSV/XLSX"]),
                                           multiple=False, className="upload-box"),
                                dbc.Button("Download Template", id="btn-template-dl", color="secondary"),
                                dcc.Download(id="dl-template")
                            ], className="mb-2", style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
                            dash_table.DataTable(
                                id="tbl-bulk-files", editable=False, style_as_list_view=True,
                                filter_action="native", sort_action="native",
                                style_table={"overflowX":"auto"}, page_size=10
                            ),
                        ])
                    ], style={"marginBottom": "1rem", "marginTop": "1rem"})
                ]),

                # Notes
                dbc.Tab(label="Notes", tab_id="tab-notes", children=[
                    dbc.Row([
                        dbc.Col(dcc.Textarea(id="notes-input", style={"width":"100%","height":"120px"},
                                             placeholder="Write a note and click Save�"), md=9, class_name="panwar"),
                        dbc.Col(dbc.Button("Save Note", id="btn-note-save", color="primary", className="mt-2"), md=3, class_name="aggarwal"),
                    ], className="mb-2"),
                    dash_table.DataTable(
                        id="tbl-notes",
                        columns=[{"name":"Date","id":"when"},{"name":"User","id":"user"},{"name":"Note","id":"note"}],
                        data=[], editable=False, style_as_list_view=True, page_size=10,
                        style_table={"overflowX":"auto"}
                    )
                ], class_name="akl"),
            ]),
        ], class_name="ankit"),
        className="mb-3"
    )

def _new_hire_ui() -> list:
    return [
        html.Div([
            html.Div([dbc.Button("+ Add New", id="btn-nh-add", color="primary", className="w-100")]),
            html.Div([
                dbc.Button("Training Details", id="btn-nh-details", outline=True, color="secondary", className="w-100"),
                dbc.Button("Download dataset", id="btn-nh-dl", outline=True, color="secondary", className="w-100"),
            ], className="ye"),
            html.Div(id="nh-msg", style={"display": "none"})
        ], className="g-2 bhaiya"),
        html.Hr(className="my-3"),

        dash_table.DataTable(
            id="tbl-nh-recent",
            page_size=5,
            style_table={"overflowX":"auto"},
            style_as_list_view=True,
            style_header={"textTransform":"none"},
        ),

        html.Hr(className="my-4"),
        html.Div(className="mt-2", children=[
            html.Div("Weekly New Hire Plan/actual grid:", className="text-muted mb-1"),
            dash_table.DataTable(
                id="tbl-nh",
                editable=True,
                style_as_list_view=True,
                style_table={"overflowX":"auto"},
                style_header={"whiteSpace":"pre"},
                page_size=12),
        ]),

        dcc.Download(id="dl-nh-dataset"),
        dcc.Store(id="store-nh-classes"),

        dbc.Modal(
            id="nh-modal", is_open=False, size="lg", backdrop="static",
            children=[
                dbc.ModalHeader(dbc.ModalTitle("Add New Hire Class"), style={"background": "#2f3747", "color": "white"}),
                dbc.ModalBody([
                    dbc.Row([
                        dbc.Col(dbc.RadioItems(
                            id="nh-form-emp-type",
                            options=[{"label":"Full-Time","value":"full-time"}, {"label":"Part-Time","value":"part-time"}],
                            value="full-time", inline=True), md=6, class_name="anki"),
                        dbc.Col(dbc.RadioItems(
                            id="nh-form-status",
                            options=[{"label":"Tentative","value":"tentative"}, {"label":"Confirmed","value":"confirmed"}],
                            value="tentative", inline=True), md=6, class_name="anki"),
                    ], class_name="mb-2"),
                    dbc.Row([
                        dbc.Col(dbc.Input(id="nh-form-class-ref", placeholder="Class Reference *", disabled=True), md=6, class_name="anki"),
                        dbc.Col(dbc.Input(id="nh-form-source-id", placeholder="Source System Id (auto)", disabled=True), md=6, class_name="anki"),
                    ], class_name="mb-2"),
                    dbc.Row([
                        dbc.Col(dbc.Input(id="nh-form-grads", type="number", min=0, step=1,
                                          placeholder="How many graduates you need? *"), md=6, class_name="anki"),
                        dbc.Col(dbc.Input(id="nh-form-billable", type="number", min=0, step=1,
                                          placeholder="Billable Headcount"), md=6, class_name="anki"),
                    ], class_name="mb-2 govin"),
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(
                            id="nh-form-class-type", options=get_class_type_options(),
                            placeholder="Class Type", clearable=True), md=6, class_name="anki"),
                        dbc.Col(dcc.Dropdown(
                            id="nh-form-class-level", options=get_class_level_options(),
                            placeholder="Class Level", clearable=True), md=6, class_name="anki"),
                    ], class_name="mb-2"),
                    dbc.Row([
                        dbc.Col(dbc.Input(id="nh-form-train-weeks", type="number", min=0, step=1, value=2,
                                          placeholder="Training Weeks"), md=6, class_name="anki"),
                        dbc.Col(dbc.Input(id="nh-form-nest-weeks",  type="number", min=0, step=1, value=1,
                                          placeholder="Nesting Weeks"), md=6, class_name="anki"),
                    ], class_name="mb-2"),
                    dbc.Row([
                        dbc.Col(dcc.DatePickerSingle(id="nh-form-date-induction", placeholder="Induction Starts On"), md=3, class_name="anki"),
                        dbc.Col(dcc.DatePickerSingle(id="nh-form-date-training",  placeholder="Training Starts On *"), md=3, class_name="anki"),
                        dbc.Col(dcc.DatePickerSingle(id="nh-form-date-nesting",   placeholder="Nesting Starts On"), md=3, class_name="anki"),
                        dbc.Col(dcc.DatePickerSingle(id="nh-form-date-production",placeholder="Production Starts On"), md=3, class_name="anki"),
                    ]),
                    html.Div(id="nh-form-error", className="text-danger mt-2"),
                    ]),
                dbc.ModalFooter([
                    dbc.Button("Save", id="btn-nh-save", color="primary", className="me-2"),
                    dbc.Button("Cancel", id="btn-nh-cancel", outline=True, color="secondary"),
                ]),
            ]),
                dbc.Modal(
                    id="nh-details-modal", is_open=False, size="xl",
                    children=[
                        dbc.ModalBody([
                            html.H4("Training Details", className="mb-3"),
                            dash_table.DataTable(
                                id="tbl-nh-details",
                                page_size=10, style_table={"overflowX":"auto"},
                                style_as_list_view=True, style_header={"textTransform":"none"},
                                row_selectable="multi",
                            ),
                        ]),
                        html.Div(className="d-flex mb-3 yebhilo", children=[
                            # dbc.Button("Close", id="btn-nh-details-close", outline=True, color="secondary", className="ms-auto"),
                            dbc.Button("Confirm Selected", id="btn-nh-confirm", color="success", className="me-2"),
                            dbc.Button("Close", id="btn-nh-details-close", outline=True, color="secondary"),
                        ])
                    ]
                ),
        ]


def _add_employee_modal() -> dbc.Modal:
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Add New Employee"), style={"background": "#2f3747", "color": "white"}),
            dbc.ModalBody([
                html.Div(id="modal-roster-crumb", className="text-muted small mb-2"),
                dbc.Row([
                    dbc.Col(dbc.Input(id="inp-brid", placeholder="BRID"), md=6),
                    dbc.Col(dbc.Input(id="inp-name", placeholder="Employee Name"), md=6),
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col(dbc.RadioItems(
                        id="inp-ftpt", value="Full-time",
                        options=[{"label":" Full-time","value":"Full-time"},
                                 {"label":" Part-time","value":"Part-time"}],
                        inline=True
                    ), md=6),
                    dbc.Col(dcc.Dropdown(
                        id="inp-role", placeholder="Role",
                        options=[{"label":r,"value":r} for r in ["Agent","SME","Trainer","Team Leader","QA","HR","WFM","AVP","VP"]],
                        value="Agent", clearable=False
                    ), md=6),
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col(dcc.DatePickerSingle(id="inp-prod-date", placeholder="Production Date"), md=6),
                    dbc.Col(dbc.Input(id="inp-tl", placeholder="Team Leader"), md=6),
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col(dbc.Input(id="inp-avp", placeholder="AVP"), md=6),
                    dbc.Col(dbc.Input(id="inp-ftpt-hours", placeholder="FT/PT Hours", type="number"), md=6),
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col(dcc.Dropdown(id="inp-work-status", placeholder="Work Status",
                                         options=[{"label":x,"value":x} for x in ["Production","Training","Nesting","LOA"]],
                                         value="Production", clearable=False), md=6),
                    dbc.Col(dcc.Dropdown(id="inp-current-status", placeholder="Current Status",
                                         options=[{"label":x,"value":x} for x in ["Production","LOA","Terminated"]],
                                         value="Production", clearable=False), md=6),
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col(dcc.DatePickerSingle(id="inp-term-date", placeholder="Terminate Date"), md=6),
                ]),
            ]),
            dbc.ModalFooter([
                dbc.Button("Save", id="btn-emp-modal-save", color="primary", className="me-2"),
                dbc.Button("Cancel", id="btn-emp-modal-cancel", color="secondary"),
            ]),
        ],
        id="modal-emp-add", is_open=False, size="lg", backdrop="static"
    )

def _actions_modals() -> list:
    return [
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Please confirm"), style={"background": "#2f3747", "color": "white"}),
                dbc.ModalBody([
                    html.Div("Are you sure?"),
                    html.Div(
                        "Deleting this record will remove employee from database and will impact headcount projections.",
                        className="text-muted small mt-1"
                    ),
                ]),
                dbc.ModalFooter([
                    dbc.Button("Yes", id="btn-remove-ok", color="danger", className="me-2"),
                    dbc.Button("No", id="btn-remove-cancel", color="secondary"),
                ])
            ],
            id="modal-remove", is_open=False, backdrop="static"
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Change Class Reference"), style={"background": "#2f3747", "color": "white"}),
                dbc.ModalBody([
                    dbc.Row([
                        dbc.Col(dbc.Label("Class Reference"), md=12),
                        dbc.Col(dcc.Dropdown(id="inp-class-ref", placeholder="Select class reference�"), md=12),
                    ]),
                    html.Div(id="class-change-hint", className="text-muted small mt-2"),
                ]),
                dbc.ModalFooter([
                    dbc.Button("Save", id="btn-class-save", color="primary", className="me-2"),
                    dbc.Button("Cancel", id="btn-class-cancel", color="secondary"),
                ]),
            ],
            id="modal-class", is_open=False, backdrop="static"
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("FT/PT Conversion"), style={"background": "#2f3747", "color": "white"}),
                dbc.ModalBody([
                    dbc.Row([
                        dbc.Col(dbc.Label("Effective Date"), md=6),
                        dbc.Col(dcc.DatePickerSingle(id="inp-ftp-date"), md=6),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(dbc.Label("Hours (weekly)"), md=6),
                        dbc.Col(dbc.Input(id="inp-ftp-hours", type="number", min=1, step=0.5, placeholder="e.g. 20"), md=6),
                    ], className="mb-2"),
                    html.Div(id="ftp-who", className="text-muted small"),
                ]),
                dbc.ModalFooter([
                    dbc.Button("Save", id="btn-ftp-save", color="primary", className="me-2"),
                    dbc.Button("Cancel", id="btn-ftp-cancel", color="secondary"),
                ])
            ],
            id="modal-ftp", is_open=False, backdrop="static"
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Move to LOA"), style={"background": "#2f3747", "color": "white"}),
                dbc.ModalBody([
                    dbc.Label("Effective Date"),
                    dcc.DatePickerSingle(id="inp-loa-date"),
                    html.Div(id="loa-who", className="text-muted small mt-2"),
                ]),
                dbc.ModalFooter([
                    dbc.Button("Save", id="btn-loa-save", color="primary", className="me-2"),
                    dbc.Button("Cancel", id="btn-loa-cancel", color="secondary"),
                ])
            ],
            id="modal-loa", is_open=False, backdrop="static"
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Back from LOA"), style={"background": "#2f3747", "color": "white"}),
                dbc.ModalBody([
                    dbc.Label("Effective Date"),
                    dcc.DatePickerSingle(id="inp-back-date"),
                    html.Div(id="back-who", className="text-muted small mt-2"),
                ]),
                dbc.ModalFooter([
                    dbc.Button("Save", id="btn-back-save", color="primary", className="me-2"),
                    dbc.Button("Cancel", id="btn-back-cancel", color="secondary"),
                ])
            ],
            id="modal-back", is_open=False, backdrop="static"
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Terminate"), style={"background": "#2f3747", "color": "white"}),
                dbc.ModalBody([
                    dbc.Label("Termination Date"),
                    dcc.DatePickerSingle(id="inp-term-date"),
                    html.Div(id="term-who", className="text-muted small mt-2"),
                ]),
                dbc.ModalFooter([
                    dbc.Button("Save", id="btn-term-save", color="danger", className="me-2"),
                    dbc.Button("Cancel", id="btn-term-cancel", color="secondary"),
                ])
            ],
            id="modal-term", is_open=False, backdrop="static"
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Transfer & Promotion"), style={"background": "#2f3747", "color": "white"}),
                dbc.ModalBody([
                    dcc.Tabs(id="tp-active-tab", value="tp-transfer", children=[
                        dcc.Tab(label="Transfer", value="tp-transfer", children=[
                            dbc.Row([
                                dbc.Col(dcc.Dropdown(id="tp-ba", placeholder="Business Area"), md=3),
                                dbc.Col(dcc.Dropdown(id="tp-subba", placeholder="Sub Business Area"), md=3),
                                dbc.Col(dcc.Dropdown(id="tp-lob", placeholder="Channel"), md=3),
                                dbc.Col(dcc.Dropdown(id="tp-site", placeholder="Site"), md=3),
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col(dbc.RadioItems(
                                    id="tp-transfer-type",
                                    options=[{"label":" Permanent","value":"perm"},
                                             {"label":" Interim","value":"interim"}],
                                    value="perm", inline=True
                                ), md=6),
                                dbc.Col(dbc.Checklist(
                                    id="tp-new-class",
                                    options=[{"label":" Transfer with new class","value":"yes"}],
                                    value=[]
                                ), md=6),
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col(dbc.Label("Effective Date"), md=6),
                                dbc.Col(dcc.DatePickerSingle(id="tp-date-from"), md=6),
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col(dbc.Label("Return Date (Interim only)"), md=6),
                                dbc.Col(dcc.DatePickerSingle(id="tp-date-to"), md=6),
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col(dcc.Dropdown(id="tp-class-ref", placeholder="Class Reference (if new class)"), md=6),
                            ]),
                        ]),
                        dcc.Tab(label="Promotion", value="tp-promo", children=[
                            dbc.Row([
                                dbc.Col(dbc.RadioItems(
                                    id="promo-type",
                                    options=[{"label":" Permanent","value":"perm"},
                                             {"label":" Temporary","value":"interim"}],
                                    value="perm", inline=True
                                ), md=6),
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col(dbc.Label("Effective Date"), md=6),
                                dbc.Col(dcc.DatePickerSingle(id="promo-date-from"), md=6),
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col(dbc.Label("Stop Date (Temporary only)"), md=6),
                                dbc.Col(dcc.DatePickerSingle(id="promo-date-to"), md=6),
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col(dcc.Dropdown(
                                    id="promo-role",
                                    placeholder="Role (e.g., Team Leader, Trainer, SME, QA �)",
                                    options=[{"label":r,"value":r} for r in ["Agent","SME","Trainer","Team Leader","QA","HR","WFM","AVP","VP"]],
                                ), md=6),
                            ]),
                        ]),
                        dcc.Tab(label="Transfer with Promotion", value="tp-both", children=[
                            dbc.Row([
                                dbc.Col(dcc.Dropdown(id="twp-ba", placeholder="Business Area"), md=3),
                                dbc.Col(dcc.Dropdown(id="twp-subba", placeholder="Sub Business Area"), md=3),
                                dbc.Col(dcc.Dropdown(id="twp-lob", placeholder="Channel"), md=3),
                                dbc.Col(dcc.Dropdown(id="twp-site", placeholder="Site"), md=3),
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col(dbc.RadioItems(
                                    id="twp-type",
                                    options=[{"label":" Permanent","value":"perm"},
                                             {"label":" Temporary","value":"interim"}],
                                    value="perm", inline=True
                                ), md=6),
                                dbc.Col(dbc.Checklist(
                                    id="twp-new-class",
                                    options=[{"label":" Transfer with new class","value":"yes"}],
                                    value=[]
                                ), md=6),
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col(dbc.Label("Effective Date"), md=6),
                                dbc.Col(dcc.DatePickerSingle(id="twp-date-from"), md=6),
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col(dbc.Label("Stop Date (Temporary only)"), md=6),
                                dbc.Col(dcc.DatePickerSingle(id="twp-date-to"), md=6),
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col(dcc.Dropdown(id="twp-class-ref", placeholder="Class Reference (if new class)"), md=6),
                                dbc.Col(dcc.Dropdown(
                                    id="twp-role", placeholder="Role",
                                    options=[{"label":r,"value":r} for r in ["Agent","SME","Trainer","Team Leader","QA","HR","WFM","AVP","VP"]]
                                ), md=6),
                            ])
                        ]),
                    ]),
                    html.Div(id="tp-who", className="text-muted small mt-2"),
                ]),
                dbc.ModalFooter([
                    dbc.Button("Save", id="btn-tp-save", color="primary", className="me-2"),
                    dbc.Button("Cancel", id="btn-tp-cancel", color="secondary"),
                ])
            ],
            id="modal-tp", is_open=False, size="xl", backdrop="static"
        ),
    ]

# === Right-side Options bar (floating) =======================================
options_toggle = html.Div(
    id="plan-opt-toggle",
    children="⚙ Options",
    style={
        "position":"fixed","right":"0","top":"45%","transform":"translateY(-50%)",
        "zIndex": 1030, "background":"#f8f9fa","border":"1px solid #ddd","borderRight":"none",
        "padding":"8px 10px","cursor":"pointer","writingMode":"vertical-rl","textOrientation":"mixed",
        "borderTopLeftRadius":"8px","borderBottomLeftRadius":"8px", "boxShadow":"0 0 6px rgba(0,0,0,.1)"
    }
)

options_panel = dbc.Offcanvas(
    id="plan-opt-canvas",
    placement="end",
    backdrop=True,
    title="Options",
    children=[
        html.Div([
            html.Div([html.Small("Plan Type: "),  html.B(id="opt-plan-type")]),
            html.Div([html.Small("Start Week: "), html.B(id="opt-plan-start")]),
            html.Div([html.Small("End Week: "),   html.B(id="opt-plan-end")]),
            html.Hr(),
            html.Div([
                html.Small("Created by: "),html.Span(id="opt-created-by"), html.Span(""),
            ]),
            html.Div([
                html.Small("Created on: "),html.Span(id="opt-created-on"),
            ]),
            html.Div([
                html.Small("Last Updated by: "),html.Span(id="opt-updated-by"), html.Span(""),
            ]),
            html.Div([
                html.Small("Last Updated on: "),html.Span(id="opt-updated-on"),
            ]),
            html.Hr(),
        ]),    
    # --- Icon-only actions (text shown on hover via tooltips) ---
    dbc.Row([
            dbc.Col(dbc.Button(html.I(className="bi bi-arrow-left-right"), id="opt-switch", color="secondary", outline=True,
                               className="w-100 mb-2", title="Switch Plan")),
            dbc.Col(dbc.Button(html.I(className="bi bi-columns-gap"), id="opt-compare", color="secondary", outline=True,
                               className="w-100 mb-2", title="Compare")),
    ]),
    dbc.Row([
            dbc.Col(dbc.Button(html.I(className="bi bi-calendar-range"), id="opt-view", color="secondary", outline=True,
                               className="w-100 mb-2", title="View Between Weeks")),
            dbc.Col(dbc.Button(html.I(className="bi bi-save"), id="opt-saveas", color="secondary", outline=True,
                               className="w-100 mb-2", title="Save As")),
    ]),
    dbc.Row([
            dbc.Col(dbc.Button(html.I(className="bi bi-download"), id="opt-export", color="secondary", outline=True,
                               className="w-100 mb-2", title="Export")),
            dbc.Col(dbc.Button(html.I(className="bi bi-arrows-expand"), id="opt-extend", color="secondary", outline=True,
                               className="w-100 mb-2", title="Extend Plan")),
    ]),
    html.Small("View Grain"),
    dbc.Row([
            dbc.Col(dbc.Button("Day", id="opt-view-day", color="secondary", outline=True,
                               className="w-100 mb-2", title="Daily View")),
            dbc.Col(dbc.Button("Interval", id="opt-view-interval", color="secondary", outline=True,
                               className="w-100 mb-2", title="Interval View")),
    ]),
    dbc.Row([
            dbc.Col(dbc.Button("Delete Plan", id="opt-delete", color="danger", outline=True,
                               className="w-100 mb-2", title="Delete Plan"), md=12),
    ]),
    # Tooltips that show labels on hover
    dbc.Tooltip("Switch Plan", target="opt-switch", placement="left"),
    dbc.Tooltip("Compare", target="opt-compare", placement="left"),
    dbc.Tooltip("View Between Weeks", target="opt-view", placement="left"),
    dbc.Tooltip("Save As", target="opt-saveas", placement="left"),
    dbc.Tooltip("Export", target="opt-export", placement="left"),
    dbc.Tooltip("Extend Plan", target="opt-extend", placement="left"),
    dbc.Tooltip("Daily View", target="opt-view-day", placement="left"),
    dbc.Tooltip("Interval View", target="opt-view-interval", placement="left"),


    html.Hr(className="my-3"),
    html.H5("Live What-If"),
    dbc.Row([
        dbc.Col(dbc.InputGroup([
        dbc.InputGroupText("AHT/SUT Δ %"),
        dbc.Input(id="whatif-aht-delta", type="number", value=0, step=5)
    ]), md=6),
        dbc.Col(dbc.InputGroup([
        dbc.InputGroupText("Shrink Δ %"),
        dbc.Input(id="whatif-shr-delta", type="number", value=0, step=1)
    ]), md=6),
    ], className="mb-2"),
    dbc.Row([
        dbc.Col(dbc.InputGroup([
        dbc.InputGroupText("Attrition Δ HC"),
        dbc.Input(id="whatif-attr-delta", type="number", value=0, step=1)
    ]), md=6),
        dbc.Col(dbc.InputGroup([
        dbc.InputGroupText("Forecast Δ %"),
        dbc.Input(id="whatif-vol-delta", type="number", value=0, step=1)
        ]), md=6),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(dbc.Checkbox(id="fw-backlog-carryover", value=True,
                              label="Apply Backlog to next week (Back Office)"), md=12)
    ], className="mb-2"),
    dbc.Button("Apply What-If", id="whatif-apply", color="primary", className="me-2"),
    dbc.Button("Clear", id="whatif-clear", outline=True, color="secondary"),
    dcc.Store(id="plan-whatif", data={}),
    dcc.Download(id="dl-plan-export"),
    ],
)

# View-between-weeks modal
view_modal = dbc.Modal(
    id="opt-view-modal", is_open=False, size="md",
    children=dbc.ModalBody([
        html.H4("View Plan Between Weeks", className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.DatePickerSingle(id="opt-view-from"), md=6),
            dbc.Col(dcc.DatePickerSingle(id="opt-view-to"), md=6),
        ], className="mb-3"),
        dbc.Alert("Tip: keep to = 24 weeks for best performance.", color="light"),
        dbc.Button("Save", id="opt-view-save", color="primary", className="me-2"),
        dbc.Button("Cancel", id="opt-view-cancel", outline=True),
    ])
)

# Save-As modal
saveas_modal = dbc.Modal(
    id="opt-saveas-modal", is_open=False,
    children=dbc.ModalBody([
        html.H4("Save Plan As", className="mb-3"),
        dbc.Input(id="opt-saveas-name", placeholder="New plan name"),
        html.Div(id="opt-saveas-msg", className="text-success mt-2"),
        dbc.Button("Save", id="opt-saveas-save", color="primary", className="me-2 mt-3"),
        dbc.Button("Cancel", id="opt-saveas-cancel", outline=True, className="mt-3"),
    ])
)

# Extend modal
extend_modal = dbc.Modal(
    id="opt-extend-modal", is_open=False,
    children=dbc.ModalBody([
        html.H4("Extend Plan", className="mb-3"),
        dbc.InputGroup([
            dbc.InputGroupText("Add weeks"),
            dbc.Input(id="opt-extend-weeks", type="number", value=4, min=1, step=1),
        ]),
        html.Div(id="opt-extend-msg", className="text-success mt-2"),
        dbc.Button("Extend", id="opt-extend-save", color="primary", className="me-2 mt-3"),
        dbc.Button("Cancel", id="opt-extend-cancel", outline=True, className="mt-3"),
    ])
)

# Compare & Switch shells (you can flesh out listing later)
switch_modal = dbc.Modal(
    id="opt-switch-modal",
    is_open=False,
    children=dbc.ModalBody([
        html.H4("Switch Plan"),
        html.Div(
            id="opt-switch-body",
            children=[
                dbc.Label("Select a plan to switch to", className="mt-3"),
                dcc.Dropdown(id="opt-switch-select", placeholder="Select plan", className="mb-3"),
                html.Div(id="opt-switch-warning", className="text-danger mb-2"),
                dbc.Button("Switch", id="opt-switch-apply", color="primary", className="mt-2", disabled=True),
            ],
        ),
        dbc.Button("Close", id="opt-switch-close", className="mt-2"),
    ]),
)

compare_modal = dbc.Modal(id="opt-compare-modal", is_open=False,
                          children=dbc.ModalBody([html.H4("Compare Plans"), html.Div(id="opt-compare-body"),
                                                  dbc.Button("Close", id="opt-compare-close", className="mt-2")]))

compare_result_modal = dbc.Modal(id="opt-compare-result-modal", is_open=False, size="xl",
                          children=[
                              dbc.ModalHeader(html.H4("Comparison Result")),
                              dbc.ModalBody(html.Div(id="opt-compare-result")),
                              dbc.ModalFooter(dbc.Button("Close", id="opt-compare-result-close", color="secondary"))
                          ])

# Add to the main plan-detail layout container:
# ...
# add after tabs/card:
html.Div([options_toggle, options_panel, view_modal, saveas_modal, extend_modal, switch_modal, compare_modal, compare_result_modal])


# ------------------------------------------------------------------------------

def layout_for_plan(pid: int) -> html.Div:
    """Main page UI; data comes from callbacks."""
    return dbc.Container([
        dcc.Store(id="plan-detail-id", data=pid),
        dcc.Store(id="plan-upper-collapsed", data=False),
        dcc.Store(id="plan-type"),
        dcc.Store(id="plan-weeks"),
        dcc.Store(id="tp-hier-map"),
        dcc.Store(id="tp-sites-map"),
        dcc.Store(id="plan-loading", data=True), 
        dcc.Store(id="plan-refresh-tick", data=0),  # NEW: refresh trigger store
        dcc.Store(id="plan-grain", data="week"),   # week or month view toggle
        dcc.Store(id="emp-undo", data={}),          # NEW: undo snapshot for roster
        dcc.Store(id="tp-current", data={}),
        dcc.Interval(id="plan-msg-timer", interval=5000, n_intervals=0, disabled=True),
        dcc.Store(id="plan-debug", storage_type="memory"),
        dcc.Store(id="plan-hydrated", storage_type="memory", data=False),
        # Interval-day selector (populated when Interval grain is active)
        html.Div([
            dbc.Label("Interval Date", html_for="interval-date", className="me-2"),
            dcc.Dropdown(id="interval-date", placeholder="Select date", clearable=False, style={"maxWidth": "240px"}),
        ], id="interval-date-wrap", style={"display": "none"}),
        html.Div([
        # Debug store to surface verify_storage logs (optional; safe no-op)
        dcc.Store(id="plan-debug", storage_type="memory"),
         # rest of layout follows
        ]),

        _upper_summary_header_card(),
        _upper_summary_body_card(),
        # _loading_overlay(),
        _lower_tabs(),
        _add_employee_modal(),
        *_actions_modals(),
        html.Div([options_toggle, options_panel, view_modal, saveas_modal, extend_modal, switch_modal, compare_modal, compare_result_modal])
    ], fluid=True)

def plan_detail_validation_layout() -> html.Div:
    dummy_cols = [{"name": "Metric", "id": "metric"}] + [{"name": "Plan\\n01/01/70", "id": "1970-01-01"}]
    return html.Div(
        [
            dcc.Store(id="plan-detail-id"),
            dcc.Store(id="plan-upper-collapsed"),
            dcc.Store(id="plan-type"),
            dcc.Store(id="plan-loading"),
            dcc.Store(id="plan-weeks"),
            dcc.Store(id="plan-refresh-tick"),  # ensure present in validation layout too
            dcc.Store(id="plan-grain"),         # ensure present in validation layout too
            dcc.Interval(id="plan-msg-timer"),
            html.Div(id="interval-date-wrap"),
            dcc.Dropdown(id="interval-date"),

            html.Div(id="plan-hdr-name"),
            html.Div(id="plan-upper"),
            html.Div(id="plan-msg"),

            dash_table.DataTable(id="tbl-fw", columns=dummy_cols, data=[]),
            dash_table.DataTable(id="tbl-hc", columns=dummy_cols, data=[]),
            dash_table.DataTable(id="tbl-attr", columns=dummy_cols, data=[]),
            dash_table.DataTable(id="tbl-shr", columns=dummy_cols, data=[]),
            dash_table.DataTable(id="tbl-train", columns=dummy_cols, data=[]),
            dash_table.DataTable(id="tbl-ratio", columns=dummy_cols, data=[]),
            dash_table.DataTable(id="tbl-seat", columns=dummy_cols, data=[]),
            dash_table.DataTable(id="tbl-bva", columns=dummy_cols, data=[]),
            dash_table.DataTable(id="tbl-nh", columns=dummy_cols, data=[]),

            dash_table.DataTable(id="tbl-emp-roster", columns=[{"name":"BRID","id":"brid"}], data=[]),
            dash_table.DataTable(
                id="tbl-bulk-files",
                columns=[{"name":"File Name","id":"file_name"},{"name":"Extension","id":"ext"},
                         {"name":"File Size (in KB)","id":"size_kb"},{"name":"Is Valid?","id":"is_valid"},
                         {"name":"File Status","id":"status"}],
                data=[]
            ),
            dash_table.DataTable(id="tbl-notes", columns=[{"name":"Date","id":"when"},
                                                          {"name":"User","id":"user"},
                                                          {"name":"Note","id":"note"}], data=[]),

            dcc.Upload(id="up-roster-bulk"),
            dcc.Download(id="dl-template"),
            dcc.Download(id="dl-workstatus"),

            dbc.Button(id="btn-plan-save"),
            dbc.Button(id="btn-plan-refresh"),
            dbc.Switch(id="toggle-switch", value=False),
            dbc.Button(id="btn-plan-grain"),
            dbc.Button(id="plan-hdr-collapse"),
            dbc.Button(id="btn-template-dl"),
            dbc.Button(id="btn-emp-modal-save"),
            dbc.Button(id="btn-emp-modal-cancel"),
            dbc.Modal(id="modal-emp-add"),
            dcc.DatePickerSingle(id="inp-prod-date"),
            dbc.RadioItems(id="inp-ftpt"),
            dbc.Input(id="inp-brid"), dbc.Input(id="inp-name"),
            dbc.Input(id="inp-role"), dbc.Input(id="inp-tl"), dbc.Input(id="inp-avp"),
            html.Div(id="modal-roster-crumb"),
            dbc.Tabs(id="plan-tabs"),
            dcc.Store(id="tp-hier-map"), dcc.Store(id="tp-current"), dcc.Tabs(id="tp-active-tab"),
            dcc.Dropdown(id="tp-ba"), dcc.Dropdown(id="tp-subba"), dcc.Dropdown(id="tp-lob"), dcc.Dropdown(id="tp-site"),
            dcc.Dropdown(id="tp-class-ref"), dcc.RadioItems(id="tp-transfer-type"),
            dcc.Checklist(id="tp-new-class"), dcc.DatePickerSingle(id="tp-date-from"), dcc.DatePickerSingle(id="tp-date-to"),
            dcc.Dropdown(id="promo-role"), dcc.RadioItems(id="promo-type"), dcc.DatePickerSingle(id="promo-date-from"), dcc.DatePickerSingle(id="promo-date-to"),
            dcc.Dropdown(id="twp-ba"), dcc.Dropdown(id="twp-subba"), dcc.Dropdown(id="twp-lob"), dcc.Dropdown(id="twp-site"),
            dcc.Dropdown(id="twp-class-ref"), dcc.Dropdown(id="twp-role"), dcc.RadioItems(id="twp-type"),
            dcc.DatePickerSingle(id="twp-date-from"), dcc.DatePickerSingle(id="twp-date-to"),
            html.Div(id="tp-who"),
            html.Div(id="lbl-emp-total"),
            dbc.Tabs(id="plan-tabs"),
            # ?? Add ALL the action buttons used as Inputs in callbacks
            dbc.Button(id="btn-emp-tp"),       dbc.Button(id="btn-tp-save"),       dbc.Button(id="btn-tp-cancel"),
            dbc.Button(id="btn-emp-loa"),      dbc.Button(id="btn-loa-save"),      dbc.Button(id="btn-loa-cancel"),
            dbc.Button(id="btn-emp-back"),     dbc.Button(id="btn-back-save"),     dbc.Button(id="btn-back-cancel"),
            dbc.Button(id="btn-emp-term"),     dbc.Button(id="btn-term-save"),     dbc.Button(id="btn-term-cancel"),
            dbc.Button(id="btn-emp-ftp"),      dbc.Button(id="btn-ftp-save"),      dbc.Button(id="btn-ftp-cancel"),
            dbc.Button(id="btn-emp-class"),    dbc.Button(id="btn-class-save"),    dbc.Button(id="btn-class-cancel"),
            dbc.Button(id="btn-emp-remove"),   dbc.Button(id="btn-remove-ok"),     dbc.Button(id="btn-remove-cancel"),
            # (optional) placeholder for the "Undo" button even though it has no callback yet
            dbc.Button(id="btn-emp-undo"),

            # ?? Add Inputs/State targets referenced by callbacks (if any were missing)
            dcc.DatePickerSingle(id="inp-loa-date"),
            dcc.DatePickerSingle(id="inp-back-date"),   # you already had this � keep it
            dcc.DatePickerSingle(id="inp-term-date"),
            dcc.DatePickerSingle(id="inp-ftp-date"),
            dbc.Input(id="inp-ftp-hours", type="number"),
            dcc.Dropdown(id="inp-class-ref"),           # you already had this � keep it

            # ?? Add **all** the modal shells used as Outputs
            dbc.Modal(id="modal-remove"),
            dbc.Modal(id="modal-class"),
            dbc.Modal(id="modal-ftp"),
            dbc.Modal(id="modal-loa"),
            dbc.Modal(id="modal-back"),
            dbc.Modal(id="modal-term"),
            dbc.Modal(id="modal-tp"),
            html.Div(id="nh-msg"),
            dcc.Store(id="store-nh-classes"),
            dbc.Modal(id="nh-modal"),
            dbc.Modal(id="nh-details-modal"),
            dash_table.DataTable(id="tbl-nh-recent"),
            dash_table.DataTable(id="tbl-nh-details"),
            dbc.Button(id="btn-nh-add"),
            dbc.Button(id="btn-nh-details"),
            dbc.Button(id="btn-nh-details-close"),
            dbc.Button(id="btn-nh-dl"),
            dcc.Download(id="dl-nh-dataset"),
            dbc.RadioItems(id="nh-form-emp-type"),
            dbc.RadioItems(id="nh-form-status"),
            dbc.Input(id="nh-form-class-ref"),
            dbc.Input(id="nh-form-source-id"),
            dbc.Input(id="nh-form-grads"),
            dbc.Input(id="nh-form-billable"),
            dbc.Input(id="nh-form-class-type"),
            dbc.Input(id="nh-form-class-level"),
            dbc.Input(id="nh-form-train-weeks"),
            dbc.Input(id="nh-form-nest-weeks"),
            dcc.DatePickerSingle(id="nh-form-date-induction"),
            dcc.DatePickerSingle(id="nh-form-date-training"),
            dcc.DatePickerSingle(id="nh-form-date-nesting"),
            dcc.DatePickerSingle(id="nh-form-date-production"),
            html.Div(id="nh-form-error"),
            dbc.Button(id="btn-nh-save"),
            dbc.Button(id="btn-nh-cancel"),
            html.Div([options_toggle, options_panel, view_modal, saveas_modal, extend_modal, switch_modal, compare_modal, compare_result_modal])

        ],
        style={"display": "none"}
    )
