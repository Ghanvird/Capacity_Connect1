from __future__ import annotations
from dash import html, dcc, dash_table, Output, Input, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from app_instance import app
try:
    from dash import ctx
except ImportError:
    from dash import callback_context as ctx
from common import parse_upload, pretty_columns, enrich_with_manager, normalize_roster_wide, build_roster_template_wide, save_roster_wide, save_roster_long
from dash.exceptions import PreventUpdate 



@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("up-roster-wide", "contents"),
    Input("btn-save-roster-wide", "n_clicks"),
    Input("btn-apply-clear", "n_clicks"),
    prevent_initial_call=True,
)
def _roster_show_loader(*_):
    trig = getattr(ctx, "triggered_id", None)
    if not trig:
        raise PreventUpdate
    triggered_value = ctx.triggered[0].get("value") if ctx.triggered else None
    if trig.startswith("up-") and not triggered_value:
        raise PreventUpdate
    if trig in {"btn-save-roster-wide", "btn-apply-clear"} and not triggered_value:
        raise PreventUpdate
    return True


@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("tbl-roster-wide", "data"),
    Input("tbl-roster-long", "data"),
    Input("roster-wide-msg", "children"),
    Input("roster-save-msg", "children"),
    Input("bulk-clear-msg", "children"),
    prevent_initial_call=True,
)
def _roster_hide_loader(*_):
    return False

@callback(
    Output("dl-roster-template","data"),
    Input("btn-dl-roster-template","n_clicks"),
    State("roster-template-dates","start_date"),
    State("roster-template-dates","end_date"),
    prevent_initial_call=True
)
def dl_roster_tmpl(n, sd, ed):
    df = build_roster_template_wide(sd, ed, include_sample=False)
    s, e = pd.to_datetime(sd).date(), pd.to_datetime(ed).date()
    return dcc.send_data_frame(df.to_csv, f"roster_template_{s}_{e}.csv", index=False)

@callback(
    Output("dl-roster-sample","data"),
    Input("btn-dl-roster-sample","n_clicks"),
    State("roster-template-dates","start_date"),
    State("roster-template-dates","end_date"),
    prevent_initial_call=True
)
def dl_roster_sample(n, sd, ed):
    df = build_roster_template_wide(sd, ed, include_sample=True)
    s, e = pd.to_datetime(sd).date(), pd.to_datetime(ed).date()
    return dcc.send_data_frame(df.to_csv, f"roster_sample_{s}_{e}.csv", index=False)


@callback(
    Output("tbl-roster-wide","data"),
    Output("tbl-roster-wide","columns"),
    Output("roster_wide_store","data"),
    Output("tbl-roster-long", "data", allow_duplicate=True),
    Output("tbl-roster-long","columns"),
    Output("roster_long_store", "data", allow_duplicate=True),
    Output("roster-wide-msg","children"),
    Input("up-roster-wide","contents"),
    State("up-roster-wide","filename"),
    prevent_initial_call=True
)
def on_upload_roster(contents, filename):
    df = parse_upload(contents, filename)
    if df.empty:
        empty_cols = pretty_columns(pd.DataFrame())
        msg = "Upload produced no rows."
        return [], empty_cols, None, [], empty_cols, None, msg
    df = enrich_with_manager(df)
    long = normalize_roster_wide(df)
    msg = f"Loaded {len(df)} rows. Normalized rows: {len(long)}."
    return (
        df.to_dict("records"), pretty_columns(df), df.to_dict("records"),
        long.to_dict("records"), pretty_columns(long), long.to_dict("records"),
        msg
    )

    if df.empty:
        empty_cols = pretty_columns(pd.DataFrame())
        msg = "Upload produced no rows."
        return [], empty_cols, None, [], empty_cols, None, msg
    df = enrich_with_manager(df)
    long = normalize_roster_wide(df)
    msg = f"Loaded {len(df)} rows. Normalized rows: {len(long)}."
    return (
        df.to_dict("records"), pretty_columns(df), df.to_dict("records"),
        long.to_dict("records"), pretty_columns(long), long.to_dict("records"),
        msg
    )


@callback(
    Output("tbl-roster-long", "data", allow_duplicate=True),
    Input("roster-preview-dates","start_date"),
    Input("roster-preview-dates","end_date"),
    State("roster_long_store","data"),
    prevent_initial_call=True
)
def filter_long_for_preview(sd, ed, store):
    base = pd.DataFrame(store or [])
    if base.empty or not sd or not ed:
        raise PreventUpdate
    sd = pd.to_datetime(sd).date()
    ed = pd.to_datetime(ed).date()
    df = base.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df[(df["date"] >= sd) & (df["date"] <= ed)]
    df["is_leave"] = df.get("entry","").astype(str).str.lower().isin({"leave","l","off","pto"})
    return df.to_dict("records")

@callback(
    Output("roster-save-msg","children"),
    Input("btn-save-roster-wide","n_clicks"),
    State("tbl-roster-wide","data"),
    State("tbl-roster-long","data"),
    prevent_initial_call=True
)
def save_roster_wide_and_long(n, rows_wide, rows_long):
    dfw = pd.DataFrame(rows_wide or [])
    dfl = pd.DataFrame(rows_long or [])
    dfw = enrich_with_manager(dfw)
    dfl = enrich_with_manager(dfl)
    save_roster_wide(dfw)
    save_roster_long(dfl)
    return "Saved ✓ (wide + normalized)"

@callback(
    Output("clear-brids", "options"),
    Input("tbl-roster-long", "data"),
    prevent_initial_call=False
)
def _fill_brid_options(rows):
    df = pd.DataFrame(rows or [])
    if df.empty:
        return []
    brid_col = "BRID" if "BRID" in df.columns else ("brid" if "brid" in df.columns else None)
    if not brid_col:
        return []
    vals = sorted(df[brid_col].dropna().astype(str).unique().tolist())
    return [{"label": v, "value": v} for v in vals]

@callback(
    Output("tbl-roster-long", "data", allow_duplicate=True),
    Output("roster_long_store", "data", allow_duplicate=True),
    Output("bulk-clear-msg", "children"),
    Input("btn-apply-clear", "n_clicks"),
    State("tbl-roster-long", "data"),
    State("clear-range", "start_date"),
    State("clear-range", "end_date"),
    State("clear-brids", "value"),
    State("clear-action", "value"),
    prevent_initial_call=True
)
def apply_bulk_clear(n, rows, start, end, brids, action):
    if not n or not start or not end:
        raise PreventUpdate
    df = pd.DataFrame(rows or [])
    if df.empty or "date" not in df.columns:
        raise PreventUpdate

    date_ser = pd.to_datetime(df["date"], errors="coerce").dt.date
    target_col = "entry" if "entry" in df.columns else ("value" if "value" in df.columns else None)
    if target_col is None:
        raise PreventUpdate

    mask = (date_ser >= pd.to_datetime(start).date()) & (date_ser <= pd.to_datetime(end).date())
    brid_col = "BRID" if "BRID" in df.columns else ("brid" if "brid" in df.columns else None)
    if brids and brid_col:
        mask &= df[brid_col].astype(str).isin([str(b) for b in brids])

    edits = int(mask.sum())
    if edits == 0:
        return rows, rows, "No matching rows in range."

    if action == "blank":
        df.loc[mask, target_col] = ""
        msg = f"Cleared {edits} cells."
    else:
        df.loc[mask, target_col] = action
        msg = f"Set '{action}' on {edits} cells."

    df["is_leave"] = df[target_col].astype(str).str.lower().isin({"leave","l","off","pto"})
    updated = df.to_dict("records")
    return updated, updated, msg

# ---------------------- New Hire / Shrinkage / Attrition (unchanged) ----------------------


