# file: callbacks_pkg/newhire.py
from __future__ import annotations
from dash import Output, Input, State, callback, no_update, dcc
import dash
import pandas as pd
import numpy as np
import base64, io, plotly.express as px
from app_instance import app
try:
    from dash import ctx
except ImportError:
    from dash import callback_context as ctx
from dash.exceptions import PreventUpdate
from common import *
from pages.newhire_page import _ensure_nh_cols, _nh_effective_count, new_hire_template_df, normalize_nh_upload_master  # noqa

# Upload ingest (auto Source ID + per-BA distinct Class Ref)


@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("up-hire", "contents"),
    Input("btn-save-hire", "n_clicks"),
    prevent_initial_call=True,
)
def _newhire_show_loader(*_):
    trig = getattr(ctx, "triggered_id", None)
    if not trig:
        raise PreventUpdate
    triggered_value = ctx.triggered[0].get("value") if ctx.triggered else None
    if trig.startswith("up-") and not triggered_value:
        raise PreventUpdate
    if trig.startswith("btn-") and triggered_value in (None, 0):
        raise PreventUpdate
    return True


@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("tbl-hire", "data"),
    Input("hire-save-msg", "children"),
    prevent_initial_call=True,
)
def _newhire_hide_loader(*_):
    return False

@app.callback(
    Output("tbl-hire", "data", allow_duplicate=True),
    Output("hire-save-msg", "children", allow_duplicate=True),
    Input("up-hire", "contents"),
    State("up-hire", "filename"),
    prevent_initial_call=True
)
def nh_upload(contents, filename):
    if not contents:
        raise dash.exceptions.PreventUpdate

    content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)

    # read CSV/XLSX
    try:
        if (filename or "").lower().endswith(".csv"):
            raw = pd.read_csv(io.StringIO(decoded.decode("utf-8")), dtype=str)
        else:
            raw = pd.read_excel(io.BytesIO(decoded))
    except Exception:
        return no_update, "Upload failed ✗"

    norm = normalize_nh_upload_master(raw)

    cur = load_hiring()
    key = "class_reference"
    if not cur.empty and key in cur and key in norm:
        # upsert by class_reference (uploaded wins on overlaps)
        keep_old = cur[~cur[key].astype(str).isin(norm[key].astype(str))]
        merged = pd.concat([keep_old, norm], ignore_index=True)
    else:
        merged = norm

    save_hiring(merged)

    msg = f"Uploaded {len(norm)} row(s) ✓"
    return merged.to_dict("records"), msg

# Save (manual edits inside the grid)
@app.callback(
    Output("hire-save-msg", "children", allow_duplicate=True),
    Input("btn-save-hire", "n_clicks"),
    State("tbl-hire", "data"),
    prevent_initial_call=True
)
def nh_save(_n, data):
    df = _ensure_nh_cols(pd.DataFrame(data or []))
    save_hiring(df)
    return f"Saved {len(df)} row(s) ✓"

# Summary figure — weekly planned (effective) by production start week
@app.callback(
    Output("fig-hire", "figure"),
    Input("tbl-hire", "data"),
    prevent_initial_call=False
)
def nh_fig(data):
    df = _ensure_nh_cols(pd.DataFrame(data or []))
    if df.empty:
        return px.bar(pd.DataFrame({"week": [], "count": []}), x="week", y="count", title="Planned New Hires by Production Week")

    def week_label(d):
        t = pd.to_datetime(d, errors="coerce")
        if pd.isna(t): return None
        monday = t - pd.to_timedelta(t.dt.weekday if hasattr(t, "dt") else t.weekday(), unit="D")
        return pd.to_datetime(monday).dt.date.astype(str)

    df["_week"] = week_label(df["production_start"])
    df["_eff"]  = df.apply(_nh_effective_count, axis=1)
    g = df.groupby("_week", as_index=False)["_eff"].sum().rename(columns={"_eff":"count"}).sort_values("_week")
    return px.bar(g, x="_week", y="count", title="Planned New Hires by Production Week", labels={"_week":"Week","count":"Planned HC"})

@app.callback(
    Output("dl-hire-sample", "data"),
    Input("btn-dl-hire", "n_clicks"),
    prevent_initial_call=True
)
def download_new_hire_sample(_n):
    df = new_hire_template_df()   # keep only ONE template function; prefer this one
    return dcc.send_data_frame(df.to_csv, "new_hire_template.csv", index=False)
