from __future__ import annotations
import json
import os
import string
from collections import defaultdict
from typing import Iterable, List, Tuple
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update
from urllib.parse import quote
from dash.dependencies import ALL
from dash.exceptions import PreventUpdate
import pandas as pd

from plan_detail._common import current_user_fallback, save_plan_meta

try:
    from dash import ctx
except Exception:
    from dash import callback_context as ctx

# ---------- Data access ----------
from cap_store import _conn, get_roster_locations
from plan_store import create_plan, delete_plan, list_business_areas, list_plans, get_plan

ADMIN_MODE = os.getenv("ADMIN_MODE", "1") in ("1", "true", "yes", "on")
ADMIN_DELETE_ENABLED = False  # keep off by default

# ---------- Constants ----------
ALPHABET = ["All"] + list(string.ascii_uppercase)
CHANNEL_OPTIONS = ["Voice", "Back Office", "Chat", "MessageUs", "Outbound", "Blended"]
PLAN_TYPE_OPTIONS = [
    "Volume Based",
    "Billable Hours Based",
    "FTE Based",
    "FTE Based Billable Transaction",
]
WEEK_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DEFAULT_LOCATIONS = ["India", "UK"]

# ---------- Channel → Icon mapping + normalization ----------
CHANNEL_ICON = {
    "Backoffice": "💼",
    "Voice": "📞",
    "Chat": "💬",
    "MessageUs": "📩",
    "Outbound": "📣",
    "Blended": "🔀",
    "Email": "✉️",
    "Omni": "🌐",
}
CHAN_ALIASES = {
    "back office": "Backoffice",
    "back-office": "Backoffice",
    "backoffice": "Backoffice",
    "voice": "Voice",
    "phone": "Voice",
    "telephony": "Voice",
    "call": "Voice",
    "chat": "Chat",
    "messageus": "MessageUs",
    "message us": "MessageUs",
    "outbound": "Outbound",
    "blended": "Blended",
    "email": "Email",
    "mail": "Email",
    "omni": "Omni",
}

# ========= Headcount-backed helpers =========

def _headcount_sites_for_ba_loc(ba: str, location: str | None) -> List[str]:
    """
    Sites for a BA filtered by selected location.
    Location matches either Country or City (case-insensitive).
    """
    if not ba:
        return []
    try:
        with _conn() as cx:
            params = [ba]
            where = "journey = ?"
            if location:
                where += """
                    AND (
                        LOWER(COALESCE(position_location_country,'')) = LOWER(?)
                        OR LOWER(COALESCE(position_location_city,'')) = LOWER(?)
                    )
                """
                params += [location, location]
            rows = cx.execute(f"""
                SELECT DISTINCT position_location_building_description AS site
                FROM headcount
                WHERE {where}
                  AND position_location_building_description IS NOT NULL
                  AND TRIM(position_location_building_description) <> ''
                ORDER BY LOWER(position_location_building_description)
            """, params).fetchall()
        return [r["site"] for r in rows]
    except Exception:
        return []


def _headcount_bas() -> List[str]:
    """Business Areas from Headcount Update: Journey (distinct, non-empty)."""
    try:
        with _conn() as cx:
            rows = cx.execute("""
                SELECT DISTINCT journey
                FROM headcount
                WHERE journey IS NOT NULL AND TRIM(journey) <> ''
                ORDER BY LOWER(journey)
            """).fetchall()
        return [r["journey"] for r in rows]
    except Exception:
        return []

def _headcount_sbas_for_ba(ba: str) -> List[str]:
    """Sub Business Area for a BA = Level 3 in Headcount (filtered by Journey)."""
    if not ba:
        return []
    try:
        with _conn() as cx:
            rows = cx.execute("""
                SELECT DISTINCT level_3 AS sba
                FROM headcount
                WHERE journey = ?
                  AND level_3 IS NOT NULL
                  AND TRIM(level_3) <> ''
                ORDER BY LOWER(level_3)
            """, (ba,)).fetchall()
        return [r["sba"] for r in rows]
    except Exception:
        return []

def _headcount_sites_for_ba(ba: str) -> List[str]:
    """Sites for BA from Headcount: Position Location Building Description (distinct, non-empty)."""
    if not ba:
        return []
    try:
        with _conn() as cx:
            rows = cx.execute("""
                SELECT DISTINCT position_location_building_description AS site
                FROM headcount
                WHERE journey = ?
                  AND position_location_building_description IS NOT NULL
                  AND TRIM(position_location_building_description) <> ''
                ORDER BY LOWER(position_location_building_description)
            """, (ba,)).fetchall()
        return [r["site"] for r in rows]
    except Exception:
        return []

def _headcount_locations_for_ba(ba: str) -> List[str]:
    """Locations for BA from Headcount: prefer Country; else fallback to City."""
    if not ba:
        return []
    try:
        with _conn() as cx:
            countries = cx.execute("""
                SELECT DISTINCT position_location_country AS loc
                FROM headcount
                WHERE journey = ?
                  AND position_location_country IS NOT NULL
                  AND TRIM(position_location_country) <> ''
                ORDER BY LOWER(position_location_country)
            """, (ba,)).fetchall()
            if countries:
                return [r["loc"] for r in countries]

            cities = cx.execute("""
                SELECT DISTINCT position_location_city AS loc
                FROM headcount
                WHERE journey = ?
                  AND position_location_city IS NOT NULL
                  AND TRIM(position_location_city) <> ''
                ORDER BY LOWER(position_location_city)
            """, (ba,)).fetchall()
            return [r["loc"] for r in cities]
    except Exception:
        return []

# ========= Existing helpers (kept) =========

def _collect_site_options() -> list[dict]:
    """Union of sites from roster + headcount (building description)."""
    sites: set[str] = set(get_roster_locations() or [])
    try:
        with _conn() as cx:
            rows = cx.execute(
                """
                SELECT DISTINCT position_location_building_description AS site
                FROM headcount
                WHERE position_location_building_description IS NOT NULL
                  AND TRIM(position_location_building_description) <> ''
                """
            ).fetchall()
        for r in rows:
            s = (r["site"] if isinstance(r, dict) else r[0]) or ""
            s = str(s).strip()
            if s:
                sites.add(s)
    except Exception:
        pass
    return [{"label": s, "value": s} for s in sorted(sites)]

def _canonical_channel(label: str | None) -> str:
    if not label:
        return "Backoffice"
    s = label.strip().lower()
    return CHAN_ALIASES.get(s, label.strip().title())

def _chan_icon(label: str | None) -> str:
    c = _canonical_channel(label or "")
    return CHANNEL_ICON.get(c, "👥")

# ---------- UI bits ----------
def _ba_chip_card(ba: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            html.Div(
                [html.Span("💼", className="me-2"), html.Span(ba or "—", className="fw-semibold")],
                className="d-flex align-items-center",
            )
        ),
        className="ws-ba-card",
    )

def _ba_union_for_dropdown(status_filter: str) -> List[dict]:
    """
    Business Area options = union of:
    - BAs that already have plans (Current/History)
    - Headcount Update 'Journey' values
    """
    from_plans = set(list_business_areas(status_filter) or [])
    from_headcount = set(_headcount_bas() or [])
    union = sorted(from_plans | from_headcount)
    return [{"label": b, "value": b} for b in union]

def _sbas_for_ba(ba: str, plans: list[dict]) -> list[str]:
    """
    Column order for Kanban = Headcount SBAs (Level 3) + any SBAs seen in plans.
    """
    hc_sbas = _headcount_sbas_for_ba(ba) or []
    seen_in_plans: list[str] = []
    for r in plans or []:
        s = (r.get("sub_ba") or "").strip()
        if s and s not in seen_in_plans:
            seen_in_plans.append(s)
    out = hc_sbas + [s for s in seen_in_plans if s not in hc_sbas]
    return out or ["Overall"]

def _group_plans_by_sba_and_channel(plans: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for r in plans or []:
        sba = (r.get("sub_ba") or "Overall").strip()
        ch_field = (r.get("channel") or "").strip()
        raw_channels = [c.strip() for c in ch_field.split(",") if c.strip()] or ["Unspecified"]
        for ch in raw_channels:
            cch = _canonical_channel(ch)
            node = out.setdefault(sba, {}).setdefault(cch, {"site_pairs": set(), "plans": []})
            node["plans"].append(r)
            site = (r.get("site") or "").strip()
            loc = (r.get("location") or "").strip()
            node["site_pairs"].add((site, loc))
    return out

def _kanban_column(sba: str, data_for_sba: dict) -> html.Div:
    ch_keys = sorted(data_for_sba.keys()) if data_for_sba else []
    cards: list = []
    if ch_keys:
        for ch in ch_keys:
            node = data_for_sba.get(ch) or {}
            plans = node.get("plans") or []
            # group by site
            site_groups: dict[str, list[dict]] = defaultdict(list)
            for p in plans:
                site_label = (p.get("site") or "").strip() or "Sites not specified"
                site_groups[site_label].append(p)

            rows: list = []
            chan_icon = _chan_icon(ch)
            chan_label = _canonical_channel(ch)

            if site_groups:
                for site_label in sorted(site_groups.keys(), key=lambda s: s.lower()):
                    rows.append(
                        html.Div([html.Span(chan_icon, className="me-2"), html.Span(chan_label)], className="ws-card-row ws-l1")
                    )
                    rows.append(
                        html.Div([html.Span("📍", className="me-2"), html.Span(site_label)], className="ws-card-row ws-l2")
                    )
                    # dedupe within site/channel
                    seen = set()
                    for p in site_groups[site_label]:
                        pname = (p.get("plan_name") or "").strip()
                        if not pname:
                            continue
                        pid = p.get("id")
                        key = (pname.lower(), chan_label.lower(), site_label.lower())
                        if key in seen:
                            continue
                        seen.add(key)
                        row_children = [html.Span("📝", className="me-2"), html.Span(pname)]
                        if ADMIN_DELETE_ENABLED and pid is not None:
                            row_children.append(
                                html.Button(
                                    "🗑",
                                    id={"type": "del-plan", "pid": int(pid)},
                                    n_clicks=0,
                                    className="ws-del-btn ms-2",
                                    title="Delete plan",
                                )
                            )
                        rows.append(
                            dcc.Link(
                                html.Div(row_children, className="ws-card-row ws-l3"),
                                href=f"/plan/{int(pid)}" if pid is not None else "/planning",
                                style={"textDecoration":"none","color":"inherit"}
                            )
                        )
            else:
                rows.append(html.Div("No plans yet", className="text-muted small ws-card-empty"))

            cards.append(
                html.Div(
                    [
                        html.Div(
                            [html.Span(chan_icon, className="me-2"), html.Span(chan_label, className="fw-semibold")],
                            className="ws-card-title",
                        ),
                        html.Div(rows, className="ws-card-body"),
                    ],
                    className="ws-kanban-card",
                )
            )
    else:
        cards.append(html.Div("No plans yet", className="text-muted small ws-card-empty"))
    return html.Div([html.Div(sba or "Overall", className="ws-col-head"), html.Div(cards, className="ws-col-body")], className="ws-kanban-col")

def _render_ba_detail(ba: str, status_filter: str) -> dbc.Card:
    plans = list_plans(vertical=ba, status_filter=status_filter) or []
    order = _sbas_for_ba(ba, plans)
    grouped = _group_plans_by_sba_and_channel(plans)
    cols = [_kanban_column(sba, grouped.get(sba, {})) for sba in (order or [])]
    if not cols:
        cols = [_kanban_column("Overall", grouped.get("Overall", {}))]
    return dbc.Card(
        dbc.CardBody(html.Div([html.Div(cols, className="ws-kanban")], className="d-flex flex-column gap-2")),
        className="ws-right-card",
    )

# ---------- Layout ----------
def planning_layout():

    return dbc.Container(
        [
            # Top bar: tabs / search / actions
            dcc.Store(id="planning-ready", data={"ready": False, "ver": 0, "path": None}),
            dcc.Interval(id="planning-ready-poller", interval=250, n_intervals=0, disabled=False),
            dbc.Card(
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button("Current", id="btn-tab-current", className="ws-tab ws-tab--active me-2", color="primary", n_clicks=0),
                                width="auto",
                            ),
                            dbc.Col(dbc.Button("History", id="btn-tab-history", className="ws-tab", color="light", n_clicks=0), width="auto"),
                            dbc.Col(dbc.Button("+ New Cap Plan", id="btn-new-plan", color="primary"), width="auto"),
                            dbc.Col(dbc.Input(id="search-ba", placeholder="Search Business Area", type="text"), md=5),
                            dbc.Col(html.Div(id="ws-message", className="text-success small text-end"), width=True),
                        ],
                        className="g-2",
                    )
                ),
                className="mb-3",
            ),
            # Workspace: left list + right kanban
            html.Div(
                [
                    dcc.Store(id="kanban-scroll-sync"),
                    dbc.Row(
                        [
                            # LEFT
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.Div(id="ws-caption", children="Current", className="ws-caption mb-2"),
                                            html.Div(
                                                className="ws-left-grid",
                                                children=[
                                                    html.Div(
                                                        id="alpha-rail",
                                                        children=[
                                                            html.Div(
                                                                dcc.RadioItems(
                                                                    id="alpha-filter",
                                                                    options=[{"label": a, "value": a} for a in ALPHABET],
                                                                    value="All",
                                                                    labelClassName="ws-alpha-label",
                                                                    inputClassName="ws-alpha-input",
                                                                    style={
                                                                        "display": "flex",
                                                                        "flexDirection": "column",
                                                                        "fontSize": "small",
                                                                    },
                                                                ),
                                                                className="ws-alpha-wrapper",
                                                            )
                                                        ],
                                                        className="ws-alpha-col",
                                                    ),
                                                    html.Div(id="ba-list", className="ws-ba-col"),
                                                ],
                                            ),
                                        ]
                                    ),
                                    className="h-100",
                                ),
                                xs=12,
                                md=4,
                                lg=3,
                            ),
                            # RIGHT
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Button("◀", id="kanban-left", size="sm", className="me-2"),
                                                    dbc.Button("▶", id="kanban-right", size="sm"),
                                                ],
                                                className="text-end mb-2",
                                            ),
                                            html.Div(id="ba-detail-col"),
                                        ]
                                    ),
                                    className="h-100",
                                ),
                                xs=12,
                                md=8,
                                lg=9,
                                style={"height": "-webkit-fill-available"},
                            ),
                        ],
                        className="g-2 align-items-stretch",
                    ),
                ]
            ),
            # Modal: New Cap Plan (no Setup New Site link anymore)
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Add New Plan"), className="ws-modal-header"),
                    dbc.ModalBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col([dbc.Label("Organizations"), dcc.Dropdown(id="org", value="Barclays", options=[{"label": "Barclays", "value": "Barclays"}])], md=3),
                                    dbc.Col([dbc.Label("Business Entity"), dcc.Dropdown(id="entity", value="Barclays", options=[{"label": "Barclays", "value": "Barclays"}])], md=3),
                                    dbc.Col(
                                        [
                                            dbc.Label("Verticals (Business Area)"),
                                            dcc.Dropdown(id="vertical"),
                                        ],
                                        md=3,
                                    ),
                                    dbc.Col([dbc.Label("Sub Business Area"), dcc.Dropdown(id="subba", placeholder="Select Sub Business Area", clearable=True)], md=3),
                                ],
                                className="mb-2 g-2 teri",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col([dbc.Label("Plan Name"), dbc.Input(id="plan-name")], md=220),
                                    dbc.Col([dbc.Label("Plan Type"), dcc.Dropdown(id="plan-type", options=[{"label": x, "value": x} for x in PLAN_TYPE_OPTIONS])], md=220),
                                    dbc.Col([dbc.Label("Channels"), dcc.Dropdown(id="channels", multi=True, options=[{"label": x, "value": x} for x in CHANNEL_OPTIONS])], md=220),
                                    dbc.Col([dbc.Label("Location"), dcc.Dropdown(id="location", placeholder="Select Location")], md=220),
                                    dbc.Col([dbc.Label("Site"), dcc.Dropdown(id="site", placeholder="Select Site", clearable=True)], md=220),
                                ],
                                className="mb-2 g-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col([dbc.Label("Start Week"), dcc.DatePickerSingle(id="start-week", className="lelo")], md=3),
                                    dbc.Col([dbc.Label("End Week"), dcc.DatePickerSingle(id="end-week", className="lelo")], md=3),
                                    dbc.Col([dbc.Label("Full-time Weekly Hours"), dbc.Input(id="ft-hrs", type="number", value=40)], md=3),
                                    dbc.Col([dbc.Label("Part-time Weekly Hours"), dbc.Input(id="pt-hrs", type="number", value=20)], md=3),
                                ],
                                className="mb-2 g-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Upper Grid: Include"),
                                            dcc.Dropdown(
                                                id="upper-options",
                                                multi=True,
                                                options=[
                                                    {"label": "FTE Required @ Queue", "value": "req_queue"},
                                                ],
                                                placeholder="Select additional upper metrics",
                                            ),
                                        ],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Lower Grid: Include"),
                                            dcc.Dropdown(
                                                id="fw-lower-options",
                                                multi=True,
                                                options=[
                                                    {"label": "Backlog (Items)", "value": "backlog"},
                                                    {"label": "Queue (Items)",   "value": "queue"},
                                                ],
                                                placeholder="Select FW rows to include",
                                            ),
                                        ],
                                        md=6,
                                    ),
                                ],
                                className="mb-2 g-2",
                            ),
                            dbc.Row([dbc.Col(dbc.Checklist(options=[{"label": " Is Current Plan?", "value": "yes"}], id="is-current", value=["yes"]), md=4)]),
                            html.Div(id="newplan-msg", className="text-danger mt-2"),
                        ]
                    ),
                    dbc.ModalFooter([dbc.Button("Create Plan", id="btn-create-plan", color="primary"), dbc.Button("Cancel", id="btn-cancel", className="ms-2")]),
                ],
                id="modal-newplan",
                is_open=False,
                size="xl",
            ),
            # (Setup New Business Area modal removed)
            # Modal: Confirm delete (DEV only)
            dcc.Store(id="ws-del-pid"),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Delete capacity plan?")),
                    dbc.ModalBody([html.Div("This will remove the selected capacity plan (soft delete).", className="text-muted"), html.Div(id="ws-del-plan-label", className="mt-2 fw-semibold")]),
                    dbc.ModalFooter([dbc.Button("Delete", id="btn-del-confirm", color="danger"), dbc.Button("Cancel", id="btn-del-cancel", className="ms-2")]),
                ],
                id="modal-del",
                is_open=False,
                size="sm",
            ),
        ],
        fluid=True,
    )

# ---------- Shared rendering (DRY) ----------
def _render_left_and_right(
    bas_left: list[str], status_filter: str, selected_ba: str | None
) -> Tuple[html.Component, html.Component, str | None]:
    if bas_left:
        items = [
            dbc.ListGroupItem(
                [html.Span("💼", className="ba-ico me-1"), html.Span(b)],
                id={"type": "ba-item", "name": b},
                action=True,
                className="py-2",
            )
            for b in bas_left
        ]
        left_list = dbc.ListGroup(items, flush=True, id="ba-list-group")
        chosen = selected_ba if (selected_ba in bas_left) else bas_left[0]
        right = html.Div([
                dcc.Link(
                    _ba_chip_card(chosen),
                    href=f"/plan/ba/{quote(chosen)}",
                    style={"textDecoration": "none", "color": "inherit"},
                    title="Open roll-up capacity plan",
                ),
                _render_ba_detail(chosen, status_filter)
            ], className="ws-right-stack")
        return left_list, right, chosen
    left_list = html.Div("No Business Areas found.", className="text-muted small")
    return left_list, html.Div(), None

# ---------- Callbacks ----------
def register_planning_ws(app):
    app.clientside_callback(
        """
        function(n, store) {
        // This callback only exists when /planning is rendered.
        const hasBAList = !!document.querySelector('#ba-list');
        if (hasBAList) {
            const prevReady = !!(store && store.ready);
            if (prevReady) { return [dash_clientside.no_update, true]; }  // debounce
            const ver = (store && store.ver ? store.ver + 1 : 1);
            return [{ ready: true, ver: ver, path: "/planning", t: Date.now() }, true];
        }
        return [dash_clientside.no_update, false];  // keep polling until DOM mounts
        }
        """,
        Output("planning-ready", "data"),
        Output("planning-ready-poller", "disabled"),
        Input("planning-ready-poller", "n_intervals"),
        State("planning-ready", "data"),
    )

    # Bump ws-refresh AFTER ready
    @app.callback(
        Output("ws-refresh", "data"),
        Input("planning-ready", "data"),
        State("ws-refresh", "data"),
        prevent_initial_call=True
    )
    def _bump_refresh_on_ready(ready_store, cur):
        if not ready_store or not ready_store.get("ready"):
            raise PreventUpdate
        return int(cur or 0) + 1

    # Breadcrumb
    @app.callback(
        Output("ws-breadcrumb", "items"),
        Input("url-router", "pathname"),
        Input("ws-status", "data"),
        Input("ws-selected-ba", "data"),
        prevent_initial_call=False,
    )
    def _crumb(pathname, status, selected_ba):
        path = (pathname or "").rstrip("/")

        # BA roll-up plan detail
        if path.startswith("/plan/ba/"):
            try:
                from urllib.parse import unquote as _unq
                ba = _unq(path.split("/plan/ba/", 1)[-1])
            except Exception:
                ba = path.split("/plan/ba/", 1)[-1]
            return [
                {"label": "CAP-CONNECT", "href": "/", "active": False},
                {"label": "Planning Workspace", "href": "/planning", "active": False},
                {"label": f"{ba} (Roll-up)", "active": True},
            ]

        # Plan detail
        if path.startswith("/plan/"):
            try:
                pid = int(path.split("/")[-1])
                p = get_plan(pid) or {}
            except Exception:
                p = {}

            items = [
                {"label": "CAP-CONNECT", "href": "/", "active": False},
                {"label": "Planning Workspace", "href": "/planning", "active": False},
            ]
            if p.get("vertical"):
                try:
                    from urllib.parse import quote as _quote
                    href = f"/plan/ba/{_quote(p['vertical'])}"
                except Exception:
                    href = "/planning"
                items.append({"label": p["vertical"], "href": href, "active": False})
            items.append({"label": (p.get("plan_name") or f"Plan {pid}"), "active": True})
            return items

        # Planning workspace
        if path == "/planning":
            items = [
                {"label": "CAP-CONNECT", "href": "/", "active": False},
                {"label": "Planning Workspace", "href": "/planning", "active": selected_ba is None},
            ]
            if selected_ba:
                items.append({
                    "label": f"{selected_ba} ({(status or 'current').title()})",
                    "href": f"/plan/ba/{quote(selected_ba)}",
                    "active": False,
                })
            return items

        # Home / others (show home emoji for non-planning pages)
        return [{"label": "🏠 Home", "href": "/", "active": False}]
    
    # Open/close the New Plan modal (no pathname gating)
    @app.callback(
        Output("modal-newplan", "is_open", allow_duplicate=True),
        Input("btn-new-plan", "n_clicks"),
        Input("btn-cancel", "n_clicks"),
        State("modal-newplan", "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_newplan(n_open, n_cancel, is_open):
        trig = getattr(ctx, "triggered_id", None)
        if trig == "btn-new-plan":
            return True
        if trig == "btn-cancel":
            return False
        return is_open


    # Tabs
    @app.callback(
        Output("ws-status", "data"),
        Output("btn-tab-current", "className"),
        Output("btn-tab-history", "className"),
        Input("btn-tab-current", "n_clicks"),
        Input("btn-tab-history", "n_clicks"),
        State("url-router", "pathname"),
        prevent_initial_call=False,
    )
    def _tabs(nc, nh, pathname):
        if (pathname or "").rstrip("/") != "/planning":
            raise PreventUpdate
        trig = ctx.triggered[0]["prop_id"] if getattr(ctx, "triggered", None) else "btn-tab-current.n_clicks"
        status = "history" if "btn-tab-history" in trig else "current"
        curr_cls = "ws-tab ws-tab--active me-2" if status == "current" else "ws-tab me-2"
        hist_cls = "ws-tab ws-tab--active" if status == "history" else "ws-tab"
        return status, curr_cls, hist_cls

    # Caption
    @app.callback(
        Output("ws-caption", "children"),
        Input("ws-status", "data"),
        State("url-router", "pathname"),
        prevent_initial_call=False,
    )
    def _caption(status, pathname):
        if (pathname or "").rstrip("/") != "/planning":
            raise PreventUpdate
        return "Current" if (status or "current") == "current" else "History"

    # ===== A) Alpha/search → render list & panel + BA dropdown options
    @app.callback(
        Output("ba-list", "children"),
        Output("ba-detail-col", "children"),
        Output("vertical", "options"),
        Output("ws-selected-ba", "data"),
        Input("alpha-filter", "value"),
        Input("search-ba", "value"),
        State("ws-status", "data"),
        State("ws-selected-ba", "data"),
        State("url-router", "pathname"),
        prevent_initial_call=True,  # user interaction only
    )
    def _fill_alpha_search(alpha, q, status_filter, selected_ba, pathname):
        if (pathname or "").rstrip("/") != "/planning":
            raise PreventUpdate

        status_filter = (status_filter or "current")
        bas_left = sorted(set(list_business_areas(status_filter) or []))
        if alpha and alpha != "All":
            bas_left = [b for b in bas_left if b.upper().startswith(str(alpha).upper())]
        if q:
            bas_left = [b for b in bas_left if q.lower() in b.lower()]

        left_list, right, chosen = _render_left_and_right(bas_left, status_filter, selected_ba)
        opts_union = _ba_union_for_dropdown(status_filter)
        return left_list, right, opts_union, chosen

    # Global renderer duplicate-safe (refresh)
    @app.callback(
        Output("ba-list", "children", allow_duplicate=True),
        Output("ba-detail-col", "children", allow_duplicate=True),
        Output("vertical", "options", allow_duplicate=True),
        Output("ws-selected-ba", "data", allow_duplicate=True),
        Input("ws-status", "data"),
        Input("ws-refresh", "data"),
        State("ws-selected-ba", "data"),
        prevent_initial_call=True,
    )
    def _fill_on_refresh(status_filter, _refresh, selected_ba):
        status_filter = (status_filter or "current")
        bas_left = sorted(set(list_business_areas(status_filter) or []))
        left_list, right, chosen = _render_left_and_right(bas_left, status_filter, selected_ba)
        opts_union = _ba_union_for_dropdown(status_filter)
        return left_list, right, opts_union, chosen

    # BA click → set selected + update right pane
    @app.callback(
        Output("ws-selected-ba", "data", allow_duplicate=True),
        Output("ba-detail-col", "children", allow_duplicate=True),
        Input({"type": "ba-item", "name": ALL}, "n_clicks"),
        State("ws-status", "data"),
        State("url-router", "pathname"),
        prevent_initial_call=True,
    )
    def _pick_ba(_n, status_filter, pathname):
        if (pathname or "").rstrip("/") != "/planning":
            raise PreventUpdate

        trig_id = getattr(ctx, "triggered_id", None)
        if isinstance(trig_id, dict):
            ba = trig_id.get("name")
        else:
            if not getattr(ctx, "triggered", None):
                raise PreventUpdate
            key = ctx.triggered[0]["prop_id"].split(".")[0]
            ba = json.loads(key).get("name")

        if not ba:
            raise PreventUpdate

        return ba, html.Div([
                dcc.Link(
                    _ba_chip_card(ba),
                    href=f"/plan/ba/{quote(ba)}",
                    style={"textDecoration": "none", "color": "inherit"},
                    title="Open roll-up capacity plan",
                ),
                _render_ba_detail(ba, status_filter)
            ], className="ws-right-stack")

    # Prefill BA in New Plan modal
    @app.callback(
        Output("vertical", "value"),
        Input("modal-newplan", "is_open"),
        State("ws-selected-ba", "data"),
        State("url-router", "pathname"),
        prevent_initial_call=True,
    )
    def _prefill_vertical(on, selected_ba, pathname):
        if (pathname or "").rstrip("/") != "/planning":
            raise PreventUpdate
        if on and selected_ba:
            return selected_ba
        raise PreventUpdate

    # Sub BA options (from Headcount Level 3)
    @app.callback(
        Output("subba", "options"),
        Output("subba", "value"),
        Input("vertical", "value"),
        State("url-router", "pathname"),
        prevent_initial_call=False,
    )
    def _fill_subba(ba, pathname):
        if (pathname or "").rstrip("/") != "/planning":
            raise PreventUpdate
        if not ba:
            return [], None
        sbas = _headcount_sbas_for_ba(ba)
        if not sbas:
            return [], None
        opts = [{"label": s, "value": s} for s in sbas]
        return opts, (sbas[0] if sbas else None)

    # Location & Site in Add New Plan (from Headcount; Site depends on BA)
# (A) When BA changes: set Location options/value and prime Site by the first Location
    @app.callback(
        Output("location", "options"),
        Output("location", "value"),
        Output("site", "options"),
        Output("site", "value"),
        Input("vertical", "value"),
        State("url-router", "pathname"),
        prevent_initial_call=False,
    )
    def _prime_loc_site_on_ba(ba, pathname):
        if (pathname or "").rstrip("/") != "/planning":
            raise PreventUpdate
        if not ba:
            return [{"label": v, "value": v} for v in DEFAULT_LOCATIONS], None, [], None

        locs = _headcount_locations_for_ba(ba) or DEFAULT_LOCATIONS
        loc_val = (locs[0] if locs else None)

        sites = _headcount_sites_for_ba_loc(ba, loc_val) or []
        site_val = (sites[0] if sites else None)

        loc_opts = [{"label": v, "value": v} for v in locs]
        site_opts = [{"label": s, "value": s} for s in sites]
        return loc_opts, loc_val, site_opts, site_val


    # (B) When Location changes: update Site options/value for the selected BA
    @app.callback(
        Output("site", "options", allow_duplicate=True),
        Output("site", "value", allow_duplicate=True),
        Input("location", "value"),
        State("vertical", "value"),
        State("site", "value"),
        State("url-router", "pathname"),
        prevent_initial_call=True,
    )
    def _update_site_on_location(location, ba, current_site, pathname):
        if (pathname or "").rstrip("/") != "/planning":
            raise PreventUpdate
        if not ba or not location:
            return [], None

        sites = _headcount_sites_for_ba_loc(ba, location) or []
        site_opts = [{"label": s, "value": s} for s in sites]

        # keep current site if still valid; else pick first
        site_val = current_site if (current_site in sites) else (sites[0] if sites else None)
        return site_opts, site_val


    # Create plan
    @app.callback(
        Output("newplan-msg", "children", allow_duplicate=True),
        Output("ws-message", "children", allow_duplicate=True),
        Output("modal-newplan", "is_open", allow_duplicate=True),
        Output("ws-refresh", "data", allow_duplicate=True),
        Input("btn-create-plan", "n_clicks"),
        State("org", "value"),
        State("entity", "value"),
        State("vertical", "value"),
        State("subba", "value"),
        State("plan-name", "value"),
        State("plan-type", "value"),
        State("channels", "value"),
        State("location", "value"),
        State("site", "value"),
        State("start-week", "date"),
        State("end-week", "date"),
        State("ft-hrs", "value"),
        State("pt-hrs", "value"),
        State("is-current", "value"),
        State("fw-lower-options", "value"),
        State("upper-options", "value"),
        State("ws-refresh", "data"),
        State("url-router", "pathname"),
        prevent_initial_call=True,
    )
    def _create_plan(
        n,
        org,
        ent,
        vertical,
        subba,
        name,
        ptype,
        channels,
        location,
        site,
        sw,
        ew,
        ft,
        pt,
        iscur,
        fw_lower_opts,
        upper_opts,
        refresh_counter,
        pathname,
    ):
        if (pathname or "").rstrip("/") != "/planning":
            raise PreventUpdate
        if not (vertical and name and sw):
            return "Business Area, Plan Name and Start Week are required.", no_update, no_update, no_update

        payload = dict(
            org=org or "Barclays",
            business_entity=ent or "Barclays",
            vertical=vertical,
            sub_ba=subba,
            channel=", ".join(channels or []),
            location=location,
            site=site,
            plan_name=name,
            plan_type=ptype,
            start_week=sw,
            end_week=ew,
            ft_weekly_hours=ft,
            pt_weekly_hours=pt,
            tags=json.dumps([]),
            is_current=("yes" in (iscur or [])),
            status=("current" if ("yes" in (iscur or [])) else "draft"),
            hierarchy_json=json.dumps(dict(vertical=vertical, sub_ba=subba, channels=channels, location=location, site=site)),
            owner=current_user_fallback(),
        )

        try:
            pid = create_plan(payload)
            now = pd.Timestamp.utcnow().isoformat(timespec="seconds")
            save_plan_meta(pid, {
                "plan_id": pid,
                "plan_name": name,
                "vertical": vertical, "sub_ba": subba,
                "channel": ", ".join(channels or []),
                "location": location, "site": site,
                "start_week": sw, "end_week": ew,
                "created_by": current_user_fallback(),
                "created_on": now,
                "last_updated_by": current_user_fallback(),
                "last_updated_on": now,
                # FW/Upper selections
                "fw_lower_options": json.dumps(list(fw_lower_opts or [])),
                "upper_options": json.dumps(list(upper_opts or [])),
            })

        except Exception as e:
            return f"Error: {e}", no_update, no_update, no_update

        return "", f"Created plan '{name}' (ID {pid})", False, int(refresh_counter or 0) + 1

    # Admin delete (confirm modal)
    @app.callback(
        Output("modal-del", "is_open"),
        Output("ws-del-pid", "data"),
        Output("ws-del-plan-label", "children"),
        Input({"type": "del-plan", "pid": ALL}, "n_clicks"),
        State("url-router", "pathname"),
        prevent_initial_call=True,
    )
    def _ask_delete(ns, pathname):
        if (pathname or "").rstrip("/") != "/planning":
            raise PreventUpdate
        trig_id = getattr(ctx, "triggered_id", None)
        if isinstance(trig_id, dict) and trig_id.get("type") == "del-plan" and any(ns):
            pid = trig_id.get("pid")
            return True, pid, f"Plan ID: {pid}"
        raise PreventUpdate

    @app.callback(
        Output("modal-del", "is_open", allow_duplicate=True),
        Output("ws-refresh", "data", allow_duplicate=True),
        Output("ws-message", "children", allow_duplicate=True),
        Input("btn-del-confirm", "n_clicks"),
        Input("btn-del-cancel", "n_clicks"),
        State("ws-del-pid", "data"),
        State("ws-refresh", "data"),
        State("url-router", "pathname"),
        prevent_initial_call=True,
    )
    def _do_delete(n_yes, n_no, pid, refresh_counter, pathname):
        if (pathname or "").rstrip("/") != "/planning":
            raise PreventUpdate
        trig = (ctx.triggered[0]["prop_id"].split(".")[0] if getattr(ctx, "triggered", None) else "")
        if trig == "btn-del-confirm" and pid:
            try:
                delete_plan(int(pid), hard_if_missing=True)
                return False, int(refresh_counter or 0) + 1, "Plan deleted."
            except Exception as e:
                return False, no_update, f"Delete failed: {e}"
        if trig == "btn-del-cancel":
            return False, no_update, no_update
        raise PreventUpdate

    # Kanban horizontal scroll (arrows)
    app.clientside_callback(
        """
        function(nL, nR, pathname) {
        const path = (pathname || "").replace(/\\/+$/,'');
        if (path !== "/planning") { return null; }
        const el = document.querySelector('.ws-right-card .ws-kanban');
        if (!el) { return null; }

        const trigArr = dash_clientside.callback_context.triggered;
        const trig = (trigArr && trigArr.length ? trigArr[0].prop_id : "");
        let dir = 0;
        if (trig.indexOf("kanban-right") === 0) dir = 1;
        if (trig.indexOf("kanban-left")  === 0) dir = -1;
        if (!dir) { return null; }

        const tile = el.querySelector('.ws-kanban-col');
        const step = tile ? (tile.getBoundingClientRect().width + 16) : 420;
        el.scrollBy({ left: dir * step, behavior: 'smooth' });
        return Date.now();
        }
        """,
        Output("kanban-scroll-sync", "data"),
        Input("kanban-left", "n_clicks"),
        Input("kanban-right", "n_clicks"),
        Input("url-router", "pathname"),
    )

    # (legacy helper kept; not wired)
    def _refresh_site_options(path, open_clicks, map1_saved):
        path = (path or "").rstrip("/")
        if path and (path != "/planning"):
            raise PreventUpdate
        return _collect_site_options()
