from __future__ import annotations
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from app_instance import app
from common import header_bar


SECTIONS = [
    dict(id="nav", title="Navigation Overview",
         search="home planning plan detail settings shrinkage attrition dataset budget ops navigation",
         body=html.Div([
             html.P("Home: KPIs (Staffing, Hiring, Shrinkage, Attrition). Click a BA row in the center table to filter the right KPIs; no selection shows All BAs."),
             html.P("Planning Workspace: Create/duplicate plans, then open Plan Detail."),
             html.P("Plan Detail: Roster, Bulk Upload, New Hire classes, weekly tables, and notes."),
             html.P("Default Settings: Global, Location, or BA → Sub BA → Channel → Site (effective-dated)."),
             html.P("Uploads: Shrinkage & Attrition raw → normalized → weekly; Planner Dataset, Budget, Ops."),
             html.Img(src="/assets/help/screen-1.png", style={"maxWidth":"100%","marginTop":"8px"}),
         ])),
    dict(id="roles", title="Roles & Permissions",
         search="roles admin planner viewer permissions",
         body=html.Ul([
             html.Li("Admin/Planner can save settings; Admin can delete plans."),
             html.Li("Viewer is read-only."),
         ])),
    dict(id="settings", title="Data Scope & Effective-Dated Settings",
         search="settings scope effective dated monday week",
         body=html.Div([
             html.P("Settings store an Effective Week (Monday). Computations use the latest settings where effective_week ≤ the target date."),
             html.Ul([
                 html.Li("Save today → applies to this and future weeks."),
                 html.Li("Change next week → applies from next week onward; past weeks keep older settings when recomputed."),
             ]),
             html.Img(src="/assets/help/screen-2.png", style={"maxWidth":"100%","marginTop":"8px"}),
         ])),
    dict(id="uploads", title="Uploads & Storage",
         search="headcount roster voice back office forecast actual timeseries storage keys",
         body=html.Ul([
             html.Li("Headcount: upsert by BRID; provides hierarchy + manager mapping."),
             html.Li("Roster: WIDE/LONG; stored as roster_wide + normalized roster_long (dedupe by BRID,date)."),
             html.Li("Forecasts/Actuals: Voice (volume + AHT by date/interval), BO (items + SUT by date) per scope."),
             html.Li("Shrinkage & Attrition: raw → weekly series for dashboards and KPIs."),
         ])),
    dict(id="kpis", title="Home KPIs (Calculations)",
         search="kpi staffing hiring shrinkage attrition requirements supply",
         body=html.Ul([
             html.Li("Requirements: prefer saved Voice/BO series; convert staff_seconds → FTE using settings."),
             html.Li("Supply: prefer saved roster + hiring; roster FTE + hiring injection on start week."),
             html.Li("Understaffed: supply < required for next week."),
         ])),
    dict(id="plan", title="Plan Detail Highlights",
         search="roster bulk upload new hire notes",
         body=html.Ul([
             html.Li("Roster: add/remove/edit, FT/PT toggle, LOA, change class ref, undo snapshot."),
             html.Li("Bulk Upload: standardized roster columns; merge by BRID (update/insert)."),
             html.Li("New Hire Classes: class types/levels/dates; additions by Production Start week."),
         ])),
    dict(id="lock", title="Automatic Plan Locking",
         search="auto lock previous month history",
         body=html.P("On the 1st monthly, any plan ending before the 1st of the previous month is auto-locked (status=history).")),
    dict(id="calcs", title="Calculation Details",
         search="voice erlangs aht bo items sut fte formula",
         body=html.Ul([
             html.Li("Voice: Erlangs + min agents for SL target & occupancy cap → staff_seconds per interval."),
             html.Li("Back Office: staff_seconds = items × SUT_sec per day."),
             html.Li("FTE/day = staff_seconds / (hours_per_day × 3600 × (1 - shrinkage))."),
         ])),
    dict(id="templates", title="Templates & Normalizers",
         search="templates normalizers",
         body=html.Ul([
             html.Li("Headcount: BRID, Full Name, Line Manager BRID/Name, Journey, Level 3, Position Location, Position Group."),
             html.Li("Voice/BO: Date (+ Interval for Voice), Volume/Items, AHT/SUT."),
             html.Li("Shrinkage/Attrition: Raw → normalized + weekly series."),
         ])),
    dict(id="dupes", title="Duplicate Uploads",
         search="duplicate overwrite upsert append",
         body=html.Ul([
             html.Li("Headcount: upsert by BRID (last wins)."),
             html.Li("Timeseries: append by date/week; overlapping dates are replaced."),
             html.Li("Roster snapshots: overwrite; long dedupes by (BRID,date)."),
             html.Li("Plan bulk roster: upsert per BRID within a plan."),
             html.Li("Shrinkage weekly: merged; Attrition weekly: overwrite."),
         ])),
    dict(id="media", title="Quickstart Video & Screenshots",
         search="video screenshots quickstart",
         body=html.Div([
             html.Video(src="/assets/help/quickstart.mp4", controls=True, style={"maxWidth":"100%"}),
             html.P("Place /assets/help/quickstart.mp4 and /assets/help/screen-*.png to enable media."),
         ])),
    dict(id="troubleshoot", title="Troubleshooting",
         search="kpi not updating role settings",
         body=html.Ul([
             html.Li("If KPIs don’t reflect uploads, ensure saves are complete and BA labels match."),
             html.Li("Effective settings: refresh/re-run for target dates to pick the correct version."),
             html.Li("Role errors on save: verify Admin/Planner."),
         ])),
]


def _toc():
    return html.Ul([html.Li(html.A(s["title"], href=f"#{s['id']}")) for s in SECTIONS])


def _section_card(sec):
    return dbc.Card(dbc.CardBody([
        html.H4(sec["title"], id=sec["id"]),
        sec["body"],
        html.Div(html.A("Back to top", href="#help-top"), className="mt-2")
    ]), className="mb-3")


def page_help():
    return html.Div(dbc.Container([
        header_bar(),
        # Anchor for "Back to top" links (kept out of flex row so it doesn't affect alignment)
        html.Div(id="help-top"),
        html.Div([
            html.H3("Help & Documentation", style={"margin": "12px", "color":"#3a5166"}),
            html.Div([
                html.Button(
                    html.I(className="bi bi-search"),
                    id="help-search-toggle",
                    className="btn btn-link p-0 search-btn",
                    title="Search",
                    style={"fontSize": "1.25rem"},
                ),
                dbc.Input(id="help-search", placeholder="Search help...", type="text", debounce=True, className="search-input")
            ], id="help-search-wrap", className="help-search d-flex align-items-center ms-auto")
        ], className="d-flex align-items-center mb-2", style={"border":"rgba(0,0,0,0.175) 1px solid","borderRadius": "5px", "background": "#fff", "boxShadow": "rgba(0, 0, 0, 0.06) 0px 2px 8px", "marginLeft":"12px", "marginRight":"12px"}),
        dcc.Store(id="help-search-open", data=False),
        dbc.Card(dbc.CardBody([html.H5("Table of Contents"), _toc()]), className="mb-3"),
        html.Div(id="help-sections", children=[_section_card(s) for s in SECTIONS])
    ], fluid=True, className="help-page"), className="loading-page")


@app.callback(Output("help-sections", "children"), Input("help-search", "value"))
def _filter_sections(q):
    q = (q or "").strip().lower()
    if not q:
        return [_section_card(s) for s in SECTIONS]
    out = []
    for s in SECTIONS:
        hay = f"{s['title']} {s.get('search','')}".lower()
        if q in hay:
            out.append(s)
    if not out:
        return [dbc.Alert(f"No sections match '{q}'.", color="warning")]
    return [_section_card(s) for s in out]


@app.callback(
    Output("help-search-open", "data"),
    Input("help-search-toggle", "n_clicks"),
    State("help-search-open", "data"),
    prevent_initial_call=True,
)
def _toggle_search(n, is_open):
    try:
        return not bool(is_open)
    except Exception:
        return True


@app.callback(
    Output("help-search-wrap", "className"),
    Input("help-search-open", "data"),
)
def _set_search_class(open_now):
    base = "help-search d-flex align-items-center ms-auto"
    return f"{base} open" if open_now else base
