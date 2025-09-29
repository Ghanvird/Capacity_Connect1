from __future__ import annotations
from dash import html, dcc, dash_table, Output, Input
import dash_bootstrap_components as dbc
from app_instance import app, server
from router import home_layout, not_found_layout
from common import _planning_ids_skeleton
from common import header_bar, sidebar_component
from planning_workspace import planning_layout, register_planning_ws
from plan_detail import plan_detail_validation_layout, register_plan_detail
from plan_store import auto_lock_previous_month_plans
from callbacks_pkg import *  # registers callbacks


# ---- Main Layout (verbatim) ----
app.layout = html.Div([
    dcc.Location(id="url-router"),
    dcc.Store(id="sidebar_collapsed", data=True, storage_type="session"),
    dcc.Store(id="nav-log-dummy"),
    dcc.Store(id="global-loading", data=False),
    _planning_ids_skeleton(),
    html.Div(id="app-wrapper", className="sidebar-collapsed", children=[
        html.Div(id="sidebar", children=sidebar_component(False).children),
        html.Div(id="root")
    ]),
    # Global loading overlay
    html.Div(
        id="global-loading-overlay",
        children=[
            html.Img(
                src="/assets/Infinity.svg",
                className="avy",
                style={"width": "96px", "height": "96px"},
            ),
            html.Div("Working...", style={"color": "white", "marginLeft": "10px"}),
        ],
        style={
            "position": "fixed",
            "inset": "0",
            "background": "rgba(0,0,0,0.6)",
            "display": "none",
            "alignItems": "center",
            "justifyContent": "center",
            "flexDirection": "column",
            "zIndex": 9999,
        },
    ),
    dcc.Interval(id="cap-plans-refresh", interval=5000, n_intervals=0)
])


# ---- Validation Layout (verbatim) ----
# ✅ VALIDATION LAYOUT — include plan-detail skeleton
app.validation_layout = html.Div([
    dcc.Location(id="url-router"),
    dcc.Store(id="sidebar_collapsed"),
    dcc.Store(id="ws-status"),
    dcc.Store(id="ws-selected-ba"),
    dcc.Store(id="ws-refresh"),
    dcc.Store(id="global-loading"),
    header_bar(),
    planning_layout(),
    plan_detail_validation_layout(),
    dash_table.DataTable(id="tbl-projects"),
    # Placeholders for Home timeline callback targets so Dash can validate callbacks globally
    html.Div(id="timeline-body"),
    dbc.Collapse(id="timeline-collapse"),
    html.Button(id="timeline-toggle"),
    # Global loading overlay in validation context
    html.Div(id="global-loading-overlay")
])

# Global loading overlay visibility toggler
@app.callback(Output("global-loading-overlay", "style"), Input("global-loading", "data"))
def _toggle_global_loading(is_on):
    base = {
        "position": "fixed",
        "inset": "0",
        "background": "rgba(0,0,0,0.6)",
        "alignItems": "center",
        "justifyContent": "center",
        "flexDirection": "column",
        "zIndex": 9999,
    }
    try:
        if bool(is_on):
            base["display"] = "flex"
        else:
            base["display"] = "none"
    except Exception:
        base["display"] = "none"
    return base

# Show global overlay on route change (start of navigation)
@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("url-router", "pathname"),
    prevent_initial_call=True,
)
def _nav_start(_path):
    return True

# Hide global overlay after new page content renders
@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("root", "children"),
    prevent_initial_call=True,
)
def _nav_end(_children):
    return False

# Register callbacks (planning + plan-detail)
register_planning_ws(app)
register_plan_detail(app)

# Auto-lock previous months' plans at startup (idempotent)
try:
    auto_lock_previous_month_plans()
except Exception:
    pass

# ---- Entrypoint (verbatim) ----
# ---------------------- Main ----------------------
if __name__ == "__main__":
    app.run(debug=True)
