from __future__ import annotations
from dash import html, dcc, dash_table, Output, Input, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import dash_svg as svg
from app_instance import app
from urllib.parse import unquote
from common import *  # noqa
import common  # for internal helpers with leading underscore
from pages.forecast_page import page_forecast
from plan_detail._common import current_user_fallback
try:
    from auth import get_user_role
except Exception:
    def get_user_role(_): return 'viewer'
from pages import (page_default_settings, page_roster, page_new_hire, page_shrink_attr,
    page_budget, page_dataset, page_planning, page_ops, page_help)

# ---------------------- Router + Home ----------------------
@app.callback(
    Output("tbl-projects", "data"),
    Input("cap-plans-refresh", "n_intervals"),
    State("url-router", "pathname"),
    prevent_initial_call=False
)
def _refresh_projects_table(_n, pathname):
    path = (pathname or "").rstrip("/")
    if path not in ("", "/"):
        raise PreventUpdate
    df = make_projects_sample()
    return df.to_dict("records")

def home_layout():
    return dbc.Container([
        header_bar(),
        dbc.Row([
            dbc.Col(left_capability_panel(), width=5),
            dbc.Col(html.Div([center_projects_table(), timeline_card()]), width=4),
            dbc.Col(right_kpi_cards(), width=3, className="col-kpis"),
        ], className="g-3"),
    ], fluid=True)

def not_found_layout():
    return dbc.Container([header_bar(), dbc.Alert("Page not found.", color="warning"), dcc.Link("← Home", href="/")], fluid=True)

@app.callback(Output("root","children"), Input("url-router","pathname"))
def route(pathname: str):
    path = (pathname or "").rstrip("/")

    if path.startswith("/plan/ba/"):
        # BA roll-up capacity plan (pid is a synthetic key "ba::<BA>")
        ba_slug = path.split("/plan/ba/", 1)[-1]
        ba = unquote(ba_slug)
        pid = f"ba::{ba}"
        return dbc.Container([header_bar(), layout_for_plan(pid)], fluid=True)

    if path.startswith("/plan/"):
        try:
            pid = int(path.rsplit("/", 1)[-1])
        except Exception:
            return not_found_layout()
        return dbc.Container([header_bar(), layout_for_plan(pid)], fluid=True)

    if path in ("/", None, ""):
        return home_layout()

    # Forecasting Workspace route
    if path == "/forecast":
        return dbc.Container([header_bar(), dbc.Alert("Forecasting Workspace — coming soon", color="info")], fluid=True)

    # Override placeholder: route /ops to the full Operational Dashboard page
    if path == "/ops":
        return page_ops()

    pages = {
        "/settings": page_default_settings,
        "/roster":   page_roster,
        "/newhire":  page_new_hire,
        "/shrink":   page_shrink_attr,
        "/dataset":  page_dataset,
        "/planning": lambda: dbc.Container([header_bar(), planning_layout()], fluid=True),
        "/ops":      lambda: dbc.Container([header_bar(), dbc.Alert("Operational Dashboard — todo", color="info")], fluid=True),
        "/budget":   page_budget,
        "/help":     page_help,
    }
    fn = pages.get(path)
    return fn() if fn else not_found_layout()

# Log navigation globally
@callback(Output("nav-log-dummy","data"), Input("url-router","pathname"), prevent_initial_call=False)
def _nav_log(pathname: str):
    user = current_user_fallback()
    try:
        log_activity(user, "nav", (pathname or ""))
    except Exception:
        pass
    return (pd.Timestamp.utcnow().isoformat(timespec="seconds"))

# ---------------------- Home KPIs dynamic update ----------------------
@callback(
    Output("right-kpis","children"),
    Input("tbl-projects","active_cell"),
    Input("tbl-projects","data"),
    Input("url-router","pathname"),
    prevent_initial_call=False
)
def _update_home_kpis(active_cell, rows, path):
    path = (path or "").rstrip("/")
    if path not in ("", "/"):
        raise PreventUpdate
    ba = None
    if active_cell and isinstance(rows, list):
        try:
            idx = int(active_cell.get("row", -1))
            if 0 <= idx < len(rows):
                r = rows[idx]
                for key in ("vertical","Vertical","Business Area","business_area","program","Program","Journey"):
                    if key in r and str(r.get(key, "")).strip():
                        ba = str(r.get(key)).strip()
                        break
        except Exception:
            ba = None
    try:
        k = common._home_kpis_for_ba(ba)
        return common._kpi_cards_children(k)
    except Exception:
        return common._kpi_cards_children(common._home_kpis_for_ba(None))

# Fill timeline on home (vertical timeline)
@callback(Output("timeline-body","children"), Input("cap-plans-refresh","n_intervals"), prevent_initial_call=False)
def _fill_timeline(_n):
    try:
        df = load_df("activity_log")
    except Exception:
        return []
    user = current_user_fallback(); role = get_user_role(user)
    if role != 'admin':
        try:
            df = df[df["user"].astype(str) == str(user)]
        except Exception:
            pass
    try:
        df = df.sort_values("ts", ascending=False).head(50)
        out = []
        for _, r in df.iterrows():
            ts = r.get("ts")
            try:
                ts_disp = pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M")
            except Exception:
                ts_disp = str(ts)
            path = r.get("path") or "/"
            out.append(
                html.Div([
                    html.Span(ts_disp, className="timeline-time"),
                    html.Span(f"Visited {path}", className="timeline-label"),
                ], className="timeline-item")
            )
        return out
    except Exception:
        return []

# Toggle collapse
@callback(
    Output("timeline-collapse","is_open"),
    Output("timeline-toggle","children"),
    Input("timeline-toggle","n_clicks"),
    State("timeline-collapse","is_open"),
    prevent_initial_call=False
)
def _toggle_timeline(n, is_open):
    # SVGs: down when open, up when collapsed
    def svg_down():
        return svg.Svg([
            svg.Path(
                d="m19 9-7 7-7-7",
                stroke="currentColor",
                strokeLinecap="round",
                strokeLinejoin="round",
                strokeWidth="2"
            )
        ], xmlns="http://www.w3.org/2000/svg", width=24, height=24, fill="none", viewBox="0 0 24 24")

    def svg_up():
        return svg.Svg([
            svg.Path(
                d="m5 15 7-7 7 7",
                stroke="currentColor",
                strokeLinecap="round",
                strokeLinejoin="round",
                strokeWidth="2"
            )
        ], xmlns="http://www.w3.org/2000/svg", width=24, height=24, fill="none", viewBox="0 0 24 24")


    if not n:
        return (is_open, svg_down())
    try:
        new_state = not bool(is_open)
    except Exception:
        new_state = True
    return (new_state, svg_up() if new_state is False else svg_down())
