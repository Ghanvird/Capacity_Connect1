from __future__ import annotations
from dash import html, dcc, dash_table, Output, Input, State, callback
import dash_bootstrap_components as dbc
try:
    from dash import ctx
except ImportError:
    from dash import callback_context as ctx
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from app_instance import app
from common import _sbas_from_headcount, _bas_from_headcount, parse_upload, pretty_columns, lock_variance_cols, _all_sites, _sites_for_ba_location, _canon_scope, CHANNEL_LIST, load_timeseries, save_timeseries, _save_budget_hc_timeseries, _budget_voice_template, _budget_bo_template, _budget_chat_template, _budget_ob_template, _budget_normalize_voice, _budget_normalize_bo, _budget_normalize_chat, _budget_normalize_ob



@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("up-bud-voice", "contents"),
    Input("up-bud-bo", "contents"),
    Input("up-bud-chat", "contents"),
    Input("up-bud-ob", "contents"),
    Input("btn-save-bud-voice", "n_clicks"),
    Input("btn-save-bud-bo", "n_clicks"),
    Input("btn-save-bud-chat", "n_clicks"),
    Input("btn-save-bud-ob", "n_clicks"),
    prevent_initial_call=True,
)
def _budget_show_loader(*_):
    trig = getattr(ctx, "triggered_id", None)
    if not trig:
        raise PreventUpdate
    triggered_value = ctx.triggered[0].get("value") if ctx.triggered else None
    if trig.startswith("up-") and not triggered_value:
        raise PreventUpdate
    return True


@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("tbl-bud-voice", "data"),
    Input("tbl-bud-bo", "data"),
    Input("tbl-bud-chat", "data"),
    Input("tbl-bud-ob", "data"),
    Input("msg-save-bud-voice", "children"),
    Input("msg-save-bud-bo", "children"),
    Input("msg-save-bud-chat", "children"),
    Input("msg-save-bud-ob", "children"),
    prevent_initial_call=True,
)
def _budget_hide_loader(*_):
    return False

@app.callback(
    Output("bud-site", "options"),
    Output("bud-site", "value"),
    Input("bud-ba", "value"),
    prevent_initial_call=False,
    )
def _bud_fill_site(ba):
    sites = _sites_for_ba_location(ba, None) if ba else _all_sites()
    opts = [{"label": s, "value": s} for s in (sites or [])]
    return opts, (sites[0] if sites else None)
#============= Tactical Upload =============#
# ====================== TEMPLATE DOWNLOADS ======================

@app.callback(
    Output("bud-ba", "options"),
    Output("bud-ba", "value"),
    Input("url-router", "pathname"),
    prevent_initial_call=False
)
def _bud_fill_ba(path):
    bas = _bas_from_headcount()  # from Headcount Update
    opts = [{"label": b, "value": b} for b in bas]
    return opts, (bas[0] if bas else None)

@app.callback(
    Output("bud-subba", "options"),
    Output("bud-subba", "value"),
    Input("bud-ba", "value"),
    prevent_initial_call=False
)
def _bud_fill_subba(ba):
    if not ba:
        return [], None
    subs = _sbas_from_headcount(ba)
    opts = [{"label": s, "value": s} for s in subs]
    return opts, (subs[0] if subs else None)

@app.callback(
    Output("bud-channel", "options"),
    Output("bud-channel", "value"),
    Input("bud-subba", "value"),
    State("bud-channel", "value"),
    prevent_initial_call=False
)
def _bud_fill_channels(_subba, current):
    opts = [{"label": c, "value": c} for c in CHANNEL_LIST]
    val = current if current in CHANNEL_LIST else "Voice"
    return opts, val


# ---- Load existing budgets when scope changes ----

@app.callback(
    Output("tbl-bud-voice", "data", allow_duplicate=True),
    Output("tbl-bud-voice", "columns", allow_duplicate=True),
    Output("store-bud-voice", "data", allow_duplicate=True),
    Input("bud-ba", "value"), Input("bud-subba", "value"), Input("bud-channel", "value"), Input("bud-site", "value"),
    prevent_initial_call=True,
)
def load_voice_budget(ba, subba, channel, site):
    if not (ba and subba and channel and site):
        return [], [], None
    if (channel or "").strip().lower() != "voice":
        return [], [], None
    key4 = _canon_scope(ba, subba, "Voice", site)
    df = load_timeseries("voice_budget", key4)
    if df is None or df.empty:
        key3 = _canon_scope(ba, subba, "Voice") # legacy
        df = load_timeseries("voice_budget", key3)
    if df is None or df.empty:
        return [], [], None
    return df.to_dict("records"), pretty_columns(df), df.to_dict("records")

@app.callback(
    Output("tbl-bud-bo", "data", allow_duplicate=True),
    Output("tbl-bud-bo", "columns", allow_duplicate=True),
    Output("store-bud-bo", "data", allow_duplicate=True),
    Input("bud-ba", "value"), Input("bud-subba", "value"), Input("bud-channel", "value"), Input("bud-site", "value"),
    prevent_initial_call=True,
)
def load_bo_budget(ba, subba, channel, site):
    if not (ba and subba and channel and site):
        return [], [], None
    ch = (channel or "").strip().lower()
    if ch not in ("back office", "bo"):
        return [], [], None
    key4 = _canon_scope(ba, subba, "Back Office", site)
    df = load_timeseries("bo_budget", key4)
    if df is None or df.empty:
        key3 = _canon_scope(ba, subba, "Back Office") # legacy
        df = load_timeseries("bo_budget", key3)
    if df is None or df.empty:
        return [], [], None
    return df.to_dict("records"), pretty_columns(df), df.to_dict("records")

@app.callback(
    Output("tbl-bud-chat", "data", allow_duplicate=True),
    Output("tbl-bud-chat", "columns", allow_duplicate=True),
    Output("store-bud-chat", "data", allow_duplicate=True),
    Input("bud-ba", "value"), Input("bud-subba", "value"), Input("bud-channel", "value"), Input("bud-site", "value"),
    prevent_initial_call=True,
)
def load_chat_budget(ba, subba, channel, site):
    if not (ba and subba and channel and site):
        return [], [], None
    if (channel or "").strip().lower() != "chat":
        return [], [], None
    key4 = _canon_scope(ba, subba, "Chat", site)
    df = load_timeseries("chat_budget", key4)
    if df is None or df.empty:
        key3 = _canon_scope(ba, subba, "Chat")
        df = load_timeseries("chat_budget", key3)
    if df is None or df.empty:
        return [], [], None
    return df.to_dict("records"), pretty_columns(df), df.to_dict("records")

@app.callback(
    Output("tbl-bud-ob", "data", allow_duplicate=True),
    Output("tbl-bud-ob", "columns", allow_duplicate=True),
    Output("store-bud-ob", "data", allow_duplicate=True),
    Input("bud-ba", "value"), Input("bud-subba", "value"), Input("bud-channel", "value"), Input("bud-site", "value"),
    prevent_initial_call=True,
)
def load_ob_budget(ba, subba, channel, site):
    if not (ba and subba and channel and site):
        return [], [], None
    if (channel or "").strip().lower() not in ("outbound", "ob", "out bound"):
        return [], [], None
    key4 = _canon_scope(ba, subba, "Outbound", site)
    df = load_timeseries("ob_budget", key4)
    if df is None or df.empty:
        key3 = _canon_scope(ba, subba, "Outbound")
        df = load_timeseries("ob_budget", key3)
    if df is None or df.empty:
        return [], [], None
    return df.to_dict("records"), pretty_columns(df), df.to_dict("records")

# ---- Download templates ----

@app.callback(Output("dl-bud-voice","data"),
          Input("btn-bud-voice-tmpl","n_clicks"),
          State("bud-voice-start","date"), State("bud-voice-weeks","value"),
          prevent_initial_call=True)
def dl_voice_tmpl(_n, start_date, weeks):
    df = _budget_voice_template(start_date, int(weeks or 8))
    return dcc.send_data_frame(df.to_csv, "voice_budget_template.csv", index=False)

@app.callback(Output("dl-bud-bo","data"),
          Input("btn-bud-bo-tmpl","n_clicks"),
          State("bud-bo-start","date"), State("bud-bo-weeks","value"),
          prevent_initial_call=True)
def dl_bo_tmpl(_n, start_date, weeks):
    df = _budget_bo_template(start_date, int(weeks or 8))
    return dcc.send_data_frame(df.to_csv, "bo_budget_template.csv", index=False)

@app.callback(Output("dl-bud-chat","data"),
          Input("btn-bud-chat-tmpl","n_clicks"),
          State("bud-chat-start","date"), State("bud-chat-weeks","value"),
          prevent_initial_call=True)
def dl_chat_tmpl(_n, start_date, weeks):
    df = _budget_chat_template(start_date, int(weeks or 8))
    return dcc.send_data_frame(df.to_csv, "chat_budget_template.csv", index=False)

@app.callback(Output("dl-bud-ob","data"),
          Input("btn-bud-ob-tmpl","n_clicks"),
          State("bud-ob-start","date"), State("bud-ob-weeks","value"),
          prevent_initial_call=True)
def dl_ob_tmpl(_n, start_date, weeks):
    df = _budget_ob_template(start_date, int(weeks or 8))
    return dcc.send_data_frame(df.to_csv, "ob_budget_template.csv", index=False)

# ---- Upload / normalize ----

@app.callback(
    Output("tbl-bud-voice", "data", allow_duplicate=True),
    Output("tbl-bud-voice", "columns", allow_duplicate=True),
    Output("store-bud-voice", "data", allow_duplicate=True),
    Input("up-bud-voice","contents"),
    State("up-bud-voice","filename"),
    prevent_initial_call=True
)
def up_voice(contents, filename):
    df = parse_upload(contents, filename)
    dff = _budget_normalize_voice(df)
    if dff.empty: return [], [], None
    return dff.to_dict("records"), lock_variance_cols(pretty_columns(dff)), dff.to_dict("records")

@app.callback(
    Output("tbl-bud-bo", "data", allow_duplicate=True),
    Output("tbl-bud-bo", "columns", allow_duplicate=True),
    Output("store-bud-bo", "data", allow_duplicate=True),
    Input("up-bud-bo","contents"),
    State("up-bud-bo","filename"),
    prevent_initial_call=True
)
def up_bo(contents, filename):
    df = parse_upload(contents, filename)
    dff = _budget_normalize_bo(df)
    if dff.empty: return [], [], None
    return dff.to_dict("records"), lock_variance_cols(pretty_columns(dff)), dff.to_dict("records")

@app.callback(
    Output("tbl-bud-chat", "data", allow_duplicate=True),
    Output("tbl-bud-chat", "columns", allow_duplicate=True),
    Output("store-bud-chat", "data", allow_duplicate=True),
    Input("up-bud-chat","contents"),
    State("up-bud-chat","filename"),
    prevent_initial_call=True
)
def up_chat(contents, filename):
    df = parse_upload(contents, filename)
    dff = _budget_normalize_chat(df)
    if dff.empty: return [], [], None
    return dff.to_dict("records"), lock_variance_cols(pretty_columns(dff)), dff.to_dict("records")

@app.callback(
    Output("tbl-bud-ob", "data", allow_duplicate=True),
    Output("tbl-bud-ob", "columns", allow_duplicate=True),
    Output("store-bud-ob", "data", allow_duplicate=True),
    Input("up-bud-ob","contents"),
    State("up-bud-ob","filename"),
    prevent_initial_call=True
)
def up_ob(contents, filename):
    df = parse_upload(contents, filename)
    dff = _budget_normalize_ob(df)
    if dff.empty: return [], [], None
    return dff.to_dict("records"), lock_variance_cols(pretty_columns(dff)), dff.to_dict("records")

# ---- Save budgets ----

@app.callback(
    Output("msg-save-bud-voice", "children"),
    Input("btn-save-bud-voice", "n_clicks"),
    State("bud-ba", "value"), State("bud-subba", "value"), State("bud-channel", "value"), State("bud-site", "value"),
    State("store-bud-voice", "data"),
    prevent_initial_call=True,
)
def save_voice_budget(_n, ba, subba, channel, site, store):
    if not (ba and subba and channel and site):
        return "Pick BA, Sub BA, Channel and Site first."
    dff = pd.DataFrame(store or [])
    if dff.empty:
        return "Nothing to save."
    key = _canon_scope(ba, subba, "Voice", site)
    save_timeseries("voice_budget", key, dff)
    _save_budget_hc_timeseries(key, dff)
    aht = dff[["week", "budget_aht_sec"]].rename(columns={"budget_aht_sec": "aht_sec"})
    save_timeseries("voice_planned_aht", key, aht)
    return f"Saved Voice budget for {key} (" + str(len(dff)) + " rows)."

@app.callback(
    Output("msg-save-bud-bo", "children"),
    Input("btn-save-bud-bo", "n_clicks"),
    State("bud-ba", "value"), State("bud-subba", "value"), State("bud-channel", "value"), State("bud-site", "value"),
    State("store-bud-bo", "data"),
    prevent_initial_call=True,
)
def save_bo_budget(_n, ba, subba, channel, site, store):
    if not (ba and subba and channel and site):
        return "Pick BA, Sub BA, Channel and Site first."
    dff = pd.DataFrame(store or [])
    if dff.empty:
        return "Nothing to save."
    key = _canon_scope(ba, subba, "Back Office", site)
    save_timeseries("bo_budget", key, dff)
    _save_budget_hc_timeseries(key, dff)
    sut = dff[["week", "budget_sut_sec"]].rename(columns={"budget_sut_sec": "sut_sec"})
    save_timeseries("bo_planned_sut", key, sut)
    return f"Saved Back Office budget for {key} (" + str(len(dff)) + " rows)."

# ---------------------- Download templates (new clubbed) ----------------------


@app.callback(
    Output("msg-save-bud-chat", "children"),
    Input("btn-save-bud-chat", "n_clicks"),
    State("bud-ba", "value"), State("bud-subba", "value"), State("bud-channel", "value"), State("bud-site", "value"),
    State("store-bud-chat", "data"),
    prevent_initial_call=True,
)
def save_chat_budget(_n, ba, subba, channel, site, store):
    if not (ba and subba and channel and site):
        return "Pick BA, Sub BA, Channel and Site first."
    dff = pd.DataFrame(store or [])
    if dff.empty:
        return "Nothing to save."
    key = _canon_scope(ba, subba, "Chat", site)
    save_timeseries("chat_budget", key, dff)
    _save_budget_hc_timeseries(key, dff)
    aht = dff[["week", "budget_aht_sec"]].rename(columns={"budget_aht_sec": "aht_sec"})
    save_timeseries("chat_planned_aht", key, aht)
    return f"Saved Chat budget for {key} (" + str(len(dff)) + " rows)."

@app.callback(
    Output("msg-save-bud-ob", "children"),
    Input("btn-save-bud-ob", "n_clicks"),
    State("bud-ba", "value"), State("bud-subba", "value"), State("bud-channel", "value"), State("bud-site", "value"),
    State("store-bud-ob", "data"),
    prevent_initial_call=True,
)
def save_ob_budget(_n, ba, subba, channel, site, store):
    if not (ba and subba and channel and site):
        return "Pick BA, Sub BA, Channel and Site first."
    dff = pd.DataFrame(store or [])
    if dff.empty:
        return "Nothing to save."
    key = _canon_scope(ba, subba, "Outbound", site)
    save_timeseries("ob_budget", key, dff)
    _save_budget_hc_timeseries(key, dff)
    aht = dff[["week", "budget_aht_sec"]].rename(columns={"budget_aht_sec": "aht_sec"})
    save_timeseries("ob_planned_aht", key, aht)
    return f"Saved Outbound budget for {key} (" + str(len(dff)) + " rows)."
