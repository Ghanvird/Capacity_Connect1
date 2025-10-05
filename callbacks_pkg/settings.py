from __future__ import annotations
import dash
from dash import html, dcc, dash_table, Output, Input, State, callback
try:
    from dash import ctx
except ImportError:
    from dash import callback_context as ctx
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import datetime as dt
from app_instance import app
from dash.exceptions import PreventUpdate
from cap_store import save_defaults, save_scoped_settings, save_timeseries, save_holidays, resolve_holidays
from common import _bas_from_headcount, _bo_tactical_canon, _read_upload_to_df, _preview_cols_data, _coerce_time, _minutes_to_seconds, _sbas_from_headcount, _sites_for_ba_location, _voice_tactical_canon, headcount_template_df, parse_upload, pretty_columns, voice_forecast_template_df, voice_actual_template_df, bo_forecast_template_df, bo_actual_template_df, sidebar_component, _canon_scope, _all_locations, _lobs_for_ba_sba, resolve_settings, load_defaults, DEFAULT_SETTINGS
from plan_detail._common import current_user_fallback
try:
    from auth import get_user_role, can_save_settings
except Exception:
    def get_user_role(_): return 'viewer'
    def can_save_settings(role): return role in ('admin','planner')


@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("btn-save-settings", "n_clicks"),
    Input("btn-save-holidays", "n_clicks"),
    Input("btn-save-headcount", "n_clicks"),
    Input("btn-save-voice-forecast", "n_clicks"),
    Input("btn-save-voice-actual", "n_clicks"),
    Input("btn-save-bo-forecast", "n_clicks"),
    Input("btn-save-bo-actual", "n_clicks"),
    Input("btn-save-chat-forecast", "n_clicks"),
    Input("btn-save-chat-actual", "n_clicks"),
    Input("btn-save-ob-forecast", "n_clicks"),
    Input("btn-save-ob-actual", "n_clicks"),
    Input("btn-save-voice-forecast1", "n_clicks"),
    Input("btn-save-bo-forecast1", "n_clicks"),
    Input("btn-save-chat-forecast1", "n_clicks"),
    Input("btn-save-ob-forecast1", "n_clicks"),
    Input("up-voice-forecast1", "contents"),
    Input("up-bo-forecast1", "contents"),
    Input("up-holidays", "contents"),
    Input("up-voice-forecast", "contents"),
    Input("up-voice-actual", "contents"),
    Input("up-bo-forecast", "contents"),
    Input("up-bo-actual", "contents"),
    Input("up-chat-forecast", "contents"),
    Input("up-chat-actual", "contents"),
    Input("up-ob-forecast", "contents"),
    Input("up-ob-actual", "contents"),
    Input("up-headcount", "contents"),
    Input("up-chat-forecast1", "contents"),
    Input("up-ob-forecast1", "contents"),
    prevent_initial_call=True,
)
def _settings_show_loader(*_):
    trig = getattr(ctx, "triggered_id", None)
    if not trig:
        raise PreventUpdate
    triggered_value = ctx.triggered[0].get("value") if ctx.triggered else None
    if trig.startswith("up-") and not triggered_value:
        raise PreventUpdate
    return True


@app.callback(
    Output("global-loading", "data", allow_duplicate=True),
    Input("voice-forecast-msg1", "children"),
    Input("bo-forecast-msg1", "children"),
    Input("holidays-msg", "children"),
    Input("voice-forecast-msg", "children"),
    Input("voice-actual-msg", "children"),
    Input("bo-forecast-msg", "children"),
    Input("bo-actual-msg", "children"),
    Input("chat-forecast-msg", "children"),
    Input("chat-actual-msg", "children"),
    Input("ob-forecast-msg", "children"),
    Input("ob-actual-msg", "children"),
    Input("ob-forecast-msg1", "children"),
    Input("chat-forecast-msg1", "children"),
    Input("hc-msg", "children"),
    Input("settings-save-msg", "children"),
    prevent_initial_call=True,
)
def _settings_hide_loader(*_):
    return False

@app.callback(
    Output("dl-voice-forecast-tmpl1", "data"),
    Input("btn-dl-voice-forecast-tmpl1", "n_clicks"),
    prevent_initial_call=True
)
def dl_voice_tactical_template(n):
    if not n: raise PreventUpdate
    df = pd.DataFrame({
        "date": ["2025-07-07","2025-07-07"],
        "interval": ["09:00","09:30"],  # 30-min
        "volume": [120, 115],
        "aht_sec": [360, 355],
    })
    return dcc.send_data_frame(df.to_csv, "voice_tactical_template.csv", index=False)

@app.callback(
    Output("dl-bo-forecast-tmpl1", "data"),
    Input("btn-dl-bo-forecast-tmpl1", "n_clicks"),
    prevent_initial_call=True
)
def dl_bo_tactical_template(n):
    if not n: raise PreventUpdate
    df = pd.DataFrame({
        "date": ["2025-07-07","2025-07-08"],
        "items": [5000, 5200],
        "sut_sec": [540, 540],
    })
    return dcc.send_data_frame(df.to_csv, "bo_tactical_template.csv", index=False)

# ====================== PREVIEW (on upload) ======================

@app.callback(
    Output("tbl-voice-forecast1", "columns"),
    Output("tbl-voice-forecast1", "data"),
    Output("voice-forecast-msg1", "children", allow_duplicate=True),
    Input("up-voice-forecast1", "contents"),
    State("up-voice-forecast1", "filename"),
    prevent_initial_call=True
)
def preview_voice_tactical(contents, filename):
    if not contents: raise PreventUpdate
    df = _read_upload_to_df(contents, filename)
    cols, data = _preview_cols_data(df)
    return cols, data, f"Loaded {len(df)} rows from {filename or 'upload'}."

@app.callback(
    Output("tbl-bo-forecast1", "columns"),
    Output("tbl-bo-forecast1", "data"),
    Output("bo-forecast-msg1", "children", allow_duplicate=True),
    Input("up-bo-forecast1", "contents"),
    State("up-bo-forecast1", "filename"),
    prevent_initial_call=True
)
def preview_bo_tactical(contents, filename):
    if not contents: raise PreventUpdate
    df = _read_upload_to_df(contents, filename)
    cols, data = _preview_cols_data(df)
    return cols, data, f"Loaded {len(df)} rows from {filename or 'upload'}."

@app.callback(
    Output("tbl-holidays", "columns"),
    Output("tbl-holidays", "data"),
    Output("holidays-msg", "children", allow_duplicate=True),
    Input("up-holidays", "contents"),
    State("up-holidays", "filename"),
    prevent_initial_call=True
)
def preview_holidays(contents, filename):
    if not contents:
        raise PreventUpdate
    df = parse_upload(contents, filename)
    if df is None or df.empty:
        return [], [], "Could not read file"
    try:
        norm = _normalize_holiday_df(df)
    except ValueError as exc:
        return [], [], str(exc)
    cols = pretty_columns(norm)
    return cols, norm.to_dict("records"), f"Loaded {len(norm)} holiday dates from {filename or 'upload'}."

@app.callback(
    Output("tbl-holidays", "data", allow_duplicate=True),
    Output("tbl-holidays", "columns", allow_duplicate=True),
    Input("set-scope", "value"),
    Input("set-location", "value"),
    Input("set-ba", "value"),
    Input("set-subba", "value"),
    Input("set-lob", "value"),
    Input("set-site-hier", "value"),
    prevent_initial_call=True
)
def load_saved_holidays(scope, location, ba, sba, lob, site):
    scope_val = (scope or "global").strip().lower()
    if scope_val == "location":
        loc = (location or "").strip()
        if not loc:
            return dash.no_update, dash.no_update
        df = resolve_holidays(location=loc)
    elif scope_val == "hier":
        if not (ba and sba and lob):
            return dash.no_update, dash.no_update
        df = resolve_holidays(ba=ba, subba=sba, lob=lob, site=site, location=location)
    else:
        df = resolve_holidays()
    if not isinstance(df, pd.DataFrame) or df.empty:
        return [], []
    out = df.copy()
    if "name" not in out.columns:
        out["name"] = ""
    out = out[["date", "name"]]
    cols = pretty_columns(out)
    return out.to_dict("records"), cols

@app.callback(
    Output("dl-holidays-template", "data"),
    Input("btn-dl-holidays-template", "n_clicks"),
    prevent_initial_call=True
)
def download_holiday_template(n):
    if not n:
        raise PreventUpdate
    sample = pd.DataFrame({
        "date": [pd.Timestamp.today().normalize().date(), pd.Timestamp.today().normalize().date() + pd.Timedelta(days=30)],
        "name": ["Holiday 1", "Holiday 2"],
    })
    return dcc.send_data_frame(sample.to_csv, "holiday_template.csv", index=False)

@app.callback(
    Output("holidays-msg", "children", allow_duplicate=True),
    Input("btn-save-holidays", "n_clicks"),
    State("tbl-holidays", "data"),
    State("set-scope", "value"),
    State("set-location", "value"),
    State("set-ba", "value"),
    State("set-subba", "value"),
    State("set-lob", "value"),
    State("set-site-hier", "value"),
    prevent_initial_call=True
)
def save_holidays_callback(n, rows, scope, location, ba, sba, lob, site):
    if not n:
        raise PreventUpdate
    user = current_user_fallback()
    if not can_save_settings(get_user_role(user)):
        return "Insufficient permissions to save settings.", False
    df = pd.DataFrame(rows or [])
    if df.empty:
        norm = pd.DataFrame(columns=["date","name"])
    else:
        if "date" not in df.columns:
            try:
                norm = _normalize_holiday_df(df)
            except ValueError as exc:
                return str(exc), False
        else:
            norm = df.copy()
            norm["date"] = pd.to_datetime(norm["date"], errors="coerce")
            norm = norm.dropna(subset=["date"])
            if norm.empty:
                norm = pd.DataFrame(columns=["date","name"])
            else:
                norm["date"] = norm["date"].dt.date.astype(str)
                if "name" in norm.columns:
                    norm["name"] = norm["name"].astype(str).str.strip()
                else:
                    norm["name"] = ""
                norm = norm.drop_duplicates(subset=["date"], keep="last")
                norm = norm[["date","name"]]
    scope_val = (scope or "global").strip().lower()
    if scope_val == "location":
        loc = (location or "").strip()
        if not loc:
            return "Select a location before saving holidays.", False
        save_holidays("location", loc, norm)
        label = f"location {loc}"
    elif scope_val == "hier":
        if not (ba and sba and lob):
            return "Pick Business Area, Sub Business Area and Channel before saving holidays.", False
        sk = _canon_scope(ba, sba, lob, site)
        save_holidays("hier", sk, norm)
        label = f"scope {sk}"
    else:
        save_holidays("global", "global", norm)
        label = "global scope"
    count = len(norm)
    plural = "s" if count != 1 else ""
    return f"Saved {count} holiday date{plural} for {label}.", False
# ====================== SAVE ======================

@app.callback(
    Output("voice-forecast-msg1", "children", allow_duplicate=True),
    Input("btn-save-voice-forecast1", "n_clicks"),
    State("up-voice-forecast1", "contents"),
    State("up-voice-forecast1", "filename"),
    State("set-ba", "value"),
    State("set-subba", "value"),
    State("set-lob", "value"),
    State("set-site-hier", "value"),
    prevent_initial_call=True,
)
def save_voice_tactical(n, contents, filename, ba, subba, channel, site):
    if not n: raise PreventUpdate
    if not contents: return "No file uploaded.", False
    if not (ba and subba and channel and site):
        return "Pick BA, Sub BA, Channel and Site first.", False
    scope_key = _canon_scope(ba, subba, channel, site)
    raw = _read_upload_to_df(contents, filename)
    vol_df, aht_df, dbg = _voice_tactical_canon(raw)
    save_timeseries("voice_tactical_volume", scope_key, vol_df)
    if not aht_df.empty:
        save_timeseries("voice_tactical_aht", scope_key, aht_df)
    return f"Saved voice_tactical_volume ({len(vol_df)}) and voice_tactical_aht ({len(aht_df)}) for scope {scope_key}. {dbg}", False

@app.callback(
    Output("bo-forecast-msg1", "children", allow_duplicate=True),
    Input("btn-save-bo-forecast1", "n_clicks"),
    State("up-bo-forecast1", "contents"),
    State("up-bo-forecast1", "filename"),
    State("set-ba", "value"),
    State("set-subba", "value"),
    State("set-lob", "value"),
    State("set-site-hier", "value"),
    prevent_initial_call=True,
)
def save_bo_tactical(n, contents, filename, ba, subba, channel, site):
    if not n: raise PreventUpdate
    if not contents: return "No file uploaded.", False
    if not (ba and subba and channel and site):
        return "Pick BA, Sub BA, Channel and Site first.", False
    scope_key = _canon_scope(ba, subba, channel, site)
    raw = _read_upload_to_df(contents, filename)
    vol_df, sut_df, dbg = _bo_tactical_canon(raw)
    save_timeseries("bo_tactical_volume", scope_key, vol_df)
    if not sut_df.empty:
        save_timeseries("bo_tactical_sut", scope_key, sut_df)
    return f"Saved bo_tactical_volume ({len(vol_df)}) and bo_tactical_sut ({len(sut_df)}) for scope {scope_key}. {dbg}", False


# ---- Scope option chaining ------------------------------------------------------------------------

@app.callback(Output("dl-voice-forecast-tmpl","data"), Input("btn-dl-voice-forecast-tmpl","n_clicks"), prevent_initial_call=True)
def dl_voice_fc_tmpl(_n): return dcc.send_data_frame(voice_forecast_template_df().to_csv, "voice_forecast_template.csv", index=False)

@app.callback(Output("dl-voice-actual-tmpl","data"), Input("btn-dl-voice-actual-tmpl","n_clicks"), prevent_initial_call=True)
def dl_voice_ac_tmpl(_n): return dcc.send_data_frame(voice_actual_template_df().to_csv, "voice_actual_template.csv", index=False)

@app.callback(Output("dl-bo-forecast-tmpl","data"), Input("btn-dl-bo-forecast-tmpl","n_clicks"), prevent_initial_call=True)
def dl_bo_fc_tmpl(_n): return dcc.send_data_frame(bo_forecast_template_df().to_csv, "backoffice_forecast_template.csv", index=False)

@app.callback(Output("dl-bo-actual-tmpl","data"), Input("btn-dl-bo-actual-tmpl","n_clicks"), prevent_initial_call=True)
def dl_bo_ac_tmpl(_n): return dcc.send_data_frame(bo_actual_template_df().to_csv, "backoffice_actual_template.csv", index=False)

# ---------------------- Upload previews (clubbed) ----------------------

@app.callback(Output("dl-chat-forecast-tmpl","data"), Input("btn-dl-chat-forecast-tmpl","n_clicks"), prevent_initial_call=True)
def dl_chat_fc_tmpl(_n):
    df = pd.DataFrame({"date":[], "items":[], "aht_sec":[]})
    return dcc.send_data_frame(df.to_csv, "chat_template.csv", index=False)

@app.callback(Output("dl-chat-actual-tmpl","data"), Input("btn-dl-chat-actual-tmpl","n_clicks"), prevent_initial_call=True)
def dl_chat_ac_tmpl(_n):
    df = pd.DataFrame({"date":[], "items":[], "aht_sec":[]})
    return dcc.send_data_frame(df.to_csv, "chat_actual_template.csv", index=False)

@app.callback(Output("dl-ob-forecast-tmpl","data"), Input("btn-dl-ob-forecast-tmpl","n_clicks"), prevent_initial_call=True)
def dl_ob_fc_tmpl(_n):
    df = pd.DataFrame({"date":[], "opc":[], "connect_rate":[], "rpc_rate":[], "aht_sec":[]})
    return dcc.send_data_frame(df.to_csv, "outbound_template.csv", index=False)

@app.callback(Output("dl-ob-actual-tmpl","data"), Input("btn-dl-ob-actual-tmpl","n_clicks"), prevent_initial_call=True)
def dl_ob_ac_tmpl(_n):
    df = pd.DataFrame({"date":[], "opc":[], "connect_rate":[], "rpc_rate":[], "aht_sec":[]})
    return dcc.send_data_frame(df.to_csv, "outbound_actual_template.csv", index=False)
@app.callback(Output("tbl-voice-forecast","data"), Output("tbl-voice-forecast","columns"), Output("voice-forecast-msg","children", allow_duplicate=True),
          Input("up-voice-forecast","contents"), State("up-voice-forecast","filename"), prevent_initial_call=True)
def up_voice_forecast(contents, filename):
    df = parse_upload(contents, filename)
    if df.empty: return [], [], "Could not read file"

    L = {c.lower(): c for c in df.columns}
    date_c     = L.get("date")
    ivl_c      = L.get("interval") or L.get("interval start") or L.get("interval_start")
    vol_c      = L.get("forecast volume") or L.get("volume")
    aht_c      = L.get("forecast aht") or L.get("aht")

    if not (date_c and ivl_c and vol_c):
        return [], [], "Need at least Date, Interval (or Interval Start), and Volume"

    dff = pd.DataFrame({
        "date": pd.to_datetime(df[date_c], errors="coerce").dt.date.astype(str),
        "interval": df[ivl_c].map(_coerce_time),
        "volume": pd.to_numeric(df[vol_c], errors="coerce").fillna(0.0),
    })
    if aht_c:
        # Smart seconds coercion: treat numbers as seconds; convert only time-like strings or explicit minutes
        def _smart_to_seconds(x):
            s = str(x).strip().lower()
            if not s:
                return None
            if ":" in s:
                # Try MM:SS first; fallback to HH:MM as minutes if second part >= 60
                try:
                    a, b = s.split(":", 1)
                    a = float(a); b = float(b)
                    return a*60.0 + b if b < 60 else (a*60.0 + b) * 60.0
                except Exception:
                    return None
            if s.endswith("m") or "min" in s:
                try:
                    return float(s.rstrip("m").replace("min","")) * 60.0
                except Exception:
                    return None
            try:
                v = float(s)
                # Heuristic: small numbers are minutes; otherwise seconds
                return v*60.0 if v <= 20 else v
            except Exception:
                return None
        dff["aht_sec"] = df[aht_c].apply(_smart_to_seconds)
    else:
        dff["aht_sec"] = 300.0  # sensible default

    for extra in ["Business Area","Sub Business Area","Channel"]:
        if extra in df.columns:
            dff[extra] = df[extra]

    cols = ["date","interval","volume","aht_sec","Business Area","Sub Business Area","Channel"]
    dff = dff[[c for c in cols if c in dff.columns]]

    return dff.to_dict("records"), pretty_columns(dff), f"Loaded {len(dff)} rows"

@app.callback(Output("tbl-voice-actual","data"), Output("tbl-voice-actual","columns"), Output("voice-actual-msg","children", allow_duplicate=True),
          Input("up-voice-actual","contents"), State("up-voice-actual","filename"), prevent_initial_call=True)
def up_voice_actual(contents, filename):
    df = parse_upload(contents, filename)
    if df.empty: return [], [], "Could not read file"

    L = {c.lower(): c for c in df.columns}
    date_c     = L.get("date")
    ivl_c      = L.get("interval") or L.get("interval start") or L.get("interval_start")
    vol_c      = L.get("actual volume") or L.get("volume")
    aht_c      = L.get("actual aht") or L.get("aht")

    if not (date_c and ivl_c and vol_c):
        return [], [], "Need at least Date, Interval (or Interval Start), and Volume"

    dff = pd.DataFrame({
        "date": pd.to_datetime(df[date_c], errors="coerce").dt.date.astype(str),
        "interval": df[ivl_c].map(_coerce_time),
        "volume": pd.to_numeric(df[vol_c], errors="coerce").fillna(0.0),
    })
    if aht_c:
        def _smart_to_seconds(x):
            s = str(x).strip().lower()
            if not s:
                return None
            if ":" in s:
                try:
                    a, b = s.split(":", 1)
                    a = float(a); b = float(b)
                    return a*60.0 + b if b < 60 else (a*60.0 + b) * 60.0
                except Exception:
                    return None
            if s.endswith("m") or "min" in s:
                try:
                    return float(s.rstrip("m").replace("min","")) * 60.0
                except Exception:
                    return None
            try:
                v = float(s)
                return v*60.0 if v <= 20 else v
            except Exception:
                return None
        dff["aht_sec"] = df[aht_c].apply(_smart_to_seconds)
    else:
        dff["aht_sec"] = 300.0

    for extra in ["Business Area","Sub Business Area","Channel"]:
        if extra in df.columns:
            dff[extra] = df[extra]

    cols = ["date","interval","volume","aht_sec","Business Area","Sub Business Area","Channel"]
    dff = dff[[c for c in cols if c in dff.columns]]

    return dff.to_dict("records"), pretty_columns(dff), f"Loaded {len(dff)} rows"

@app.callback(Output("tbl-bo-forecast","data"), Output("tbl-bo-forecast","columns"), Output("bo-forecast-msg","children", allow_duplicate=True),
          Input("up-bo-forecast","contents"), State("up-bo-forecast","filename"), prevent_initial_call=True)
def up_bo_forecast(contents, filename):
    df = parse_upload(contents, filename)
    if df.empty: return [], [], "Could not read file"

    L = {c.lower(): c for c in df.columns}
    date_c = L.get("date")
    vol_c  = L.get("forecast volume") or L.get("volume") or L.get("items")
    sut_c  = L.get("forecast sut") or L.get("sut") or L.get("durationseconds")

    if not (date_c and vol_c and sut_c):
        return [], [], "Need Date, Volume/Items and SUT/DurationSeconds"

    dff = pd.DataFrame({
        "date": pd.to_datetime(df[date_c], errors="coerce").dt.date.astype(str),
        "items": pd.to_numeric(df[vol_c], errors="coerce").fillna(0.0),
        "sut_sec": pd.to_numeric(df[sut_c], errors="coerce").fillna(0.0),
    })

    for extra in ["Business Area","Sub Business Area","Channel"]:
        if extra in df.columns:
            dff[extra] = df[extra]

    cols = ["date","items","sut_sec","Business Area","Sub Business Area","Channel"]
    dff = dff[[c for c in cols if c in dff.columns]]

    return dff.to_dict("records"), pretty_columns(dff), f"Loaded {len(dff)} rows"

@app.callback(Output("tbl-bo-actual","data"), Output("tbl-bo-actual","columns"), Output("bo-actual-msg","children", allow_duplicate=True),
          Input("up-bo-actual","contents"), State("up-bo-actual","filename"), prevent_initial_call=True)
def up_bo_actual(contents, filename):
    df = parse_upload(contents, filename)
    if df.empty: return [], [], "Could not read file"

    L = {c.lower(): c for c in df.columns}
    date_c = L.get("date")
    vol_c  = L.get("actual volume") or L.get("volume") or L.get("items")
    sut_c  = L.get("actual sut") or L.get("sut") or L.get("durationseconds")

    if not (date_c and vol_c and sut_c):
        return [], [], "Need Date, Volume/Items and SUT/DurationSeconds"

    dff = pd.DataFrame({
        "date": pd.to_datetime(df[date_c], errors="coerce").dt.date.astype(str),
        "items": pd.to_numeric(df[vol_c], errors="coerce").fillna(0.0),
        "sut_sec": pd.to_numeric(df[sut_c], errors="coerce").fillna(0.0),
    })

    for extra in ["Business Area","Sub Business Area","Channel"]:
        if extra in df.columns:
            dff[extra] = df[extra]

    cols = ["date","items","sut_sec","Business Area","Sub Business Area","Channel"]
    dff = dff[[c for c in cols if c in dff.columns]]

    return dff.to_dict("records"), pretty_columns(dff), f"Loaded {len(dff)} rows"

# ---------------------- Save (per-scope, clubbed) ----------------------
def _scope_guard(scope, ba, sba, ch, site):
    if scope != "hier":
        return False, "Switch scope to Business Area ▶ Sub Business Area ▶ Channel ▶ Site."
    if not (ba and sba and ch and site):
        return False, "Pick BA, Sub BA, Channel and Site first."
    return True, ""

def _normalize_holiday_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize various holiday file formats to [date,name].
    - Accepts date in columns: date, holiday date, holiday_date, day, dt, event date, date of holiday
    - Parses dates robustly: standard parse, then day-first fallback, then Excel-serial fallback.
    - Name column optional: name/holiday/description/label/event.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["date","name"])
    data = df.copy()
    data.columns = [str(c) for c in data.columns]
    cols = {str(c).strip().lower(): c for c in data.columns}
    date_col = None
    for key in ("date","holiday date","holiday_date","day","dt","event date","date of holiday","date_of_holiday"):
        date_col = cols.get(key)
        if date_col:
            break
    if not date_col:
        raise ValueError("Holiday file must include a 'date' column.")
    name_col = None
    for key in ("name","holiday","description","label","event"):
        name_col = cols.get(key)
        if name_col:
            break
    ser = data[date_col]
    # First pass: standard parse
    parsed = pd.to_datetime(ser, errors="coerce")
    # If mostly NaT, try dayfirst
    if parsed.isna().mean() >= 0.8:
        parsed = pd.to_datetime(ser, errors="coerce", dayfirst=True)
    # If still mostly NaT, try Excel serials
    if parsed.isna().mean() >= 0.8:
        try:
            parsed = pd.to_datetime(pd.to_numeric(ser, errors="coerce"), unit="d", origin="1899-12-30", errors="coerce")
        except Exception:
            pass
    out = pd.DataFrame({"date": parsed})
    out = out.dropna(subset=["date"]) 
    if out.empty:
        return pd.DataFrame(columns=["date","name"])
    out["date"] = out["date"].dt.date.astype(str)
    if name_col:
        out["name"] = data[name_col].astype(str).str.strip()
    else:
        out["name"] = ""
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out[["date","name"]]

@app.callback(Output("voice-forecast-msg","children", allow_duplicate=True),
          Input("btn-save-voice-forecast","n_clicks"),
          State("tbl-voice-forecast","data"),
          State("set-scope","value"), State("set-ba","value"), State("set-subba","value"),
          State("set-lob","value"), State("set-site-hier","value"),
          prevent_initial_call=True)
def save_voice_forecast(_n, preview_rows, scope, ba, sba, ch, site):
    ok, msg = _scope_guard(scope, ba, sba, ch, site)
    if not ok: return msg
    sk = _canon_scope(ba, sba, ch, site)
    df = pd.DataFrame(preview_rows or [])
    if df.empty or not {'date','interval','volume','aht_sec'}.issubset(df.columns):
        return 'Missing required columns for Voice (date/interval/volume/aht_sec)'
    save_timeseries('voice_forecast_volume', sk, df[['date','interval','volume']].copy())
    save_timeseries('voice_forecast_aht',    sk, df[['date','interval','aht_sec']].copy())
    return f"Saved VOICE forecast ({len(df)}) for {sk}", False

@app.callback(Output("voice-actual-msg","children", allow_duplicate=True),
          Input("btn-save-voice-actual","n_clicks"),
          State("tbl-voice-actual","data"),
          State("set-scope","value"), State("set-ba","value"), State("set-subba","value"),
          State("set-lob","value"), State("set-site-hier","value"),
          prevent_initial_call=True)
def save_voice_actual(_n, preview_rows, scope, ba, sba, ch, site):
    ok, msg = _scope_guard(scope, ba, sba, ch, site)
    if not ok: return msg
    sk = _canon_scope(ba, sba, ch, site)
    df = pd.DataFrame(preview_rows or [])
    if df.empty or not {'date','interval','volume','aht_sec'}.issubset(df.columns):
        return 'Missing required columns for Voice (date/interval/volume/aht_sec)'
    save_timeseries('voice_actual_volume', sk, df[['date','interval','volume']].copy())
    save_timeseries('voice_actual_aht',    sk, df[['date','interval','aht_sec']].copy())
    return f"Saved VOICE actual ({len(df)}) for {sk}", False

@app.callback(Output("bo-forecast-msg","children", allow_duplicate=True),
          Input("btn-save-bo-forecast","n_clicks"),
          State("tbl-bo-forecast","data"),
          State("set-scope","value"), State("set-ba","value"), State("set-subba","value"),
          State("set-lob","value"), State("set-site-hier","value"),
          prevent_initial_call=True)
def save_bo_forecast(_n, preview_rows, scope, ba, sba, ch, site):
    ok, msg = _scope_guard(scope, ba, sba, ch, site)
    if not ok: return msg
    sk = _canon_scope(ba, sba, ch, site)
    df = pd.DataFrame(preview_rows or [])
    if df.empty or not {'date','items','sut_sec'}.issubset(df.columns):
        return 'Missing required columns (date/items/sut_sec)'
    save_timeseries('bo_forecast_volume', sk, df[['date','items']].rename(columns={'items':'volume'}))
    save_timeseries('bo_forecast_sut',    sk, df[['date','sut_sec']].copy())
    return f"Saved BO forecast ({len(df)}) for {sk}", False

@app.callback(Output("bo-actual-msg","children", allow_duplicate=True),
          Input("btn-save-bo-actual","n_clicks"),
          State("tbl-bo-actual","data"),
          State("set-scope","value"), State("set-ba","value"), State("set-subba","value"),
          State("set-lob","value"), State("set-site-hier","value"),
          prevent_initial_call=True)
def save_bo_actual(_n, preview_rows, scope, ba, sba, ch, site):
    ok, msg = _scope_guard(scope, ba, sba, ch, site)
    if not ok: return msg
    sk = _canon_scope(ba, sba, ch, site)
    df = pd.DataFrame(preview_rows or [])
    if df.empty or not {'date','items','sut_sec'}.issubset(df.columns):
        return 'Missing required columns (date/items/sut_sec)'
    save_timeseries('bo_actual_volume', sk, df[['date','items']].rename(columns={'items':'volume'}))
    save_timeseries('bo_actual_sut',    sk, df[['date','sut_sec']].copy())
    return f"Saved BO actual ({len(df)}) for {sk}", False

@app.callback(
    Output("sidebar_collapsed","data"),
    Input("btn-burger-top","n_clicks"),
    State("sidebar_collapsed","data"),
    prevent_initial_call=True
)
def toggle_sidebar(n, collapsed):
    if not n: raise PreventUpdate
    return not bool(collapsed)

@app.callback(Output("app-wrapper","className"), Input("sidebar_collapsed","data"))
def set_wrapper_class(collapsed):
    return "sidebar-collapsed" if collapsed else "sidebar-expanded"

@app.callback(Output("sidebar","children"), Input("sidebar_collapsed","data"))
def render_sidebar(collapsed):
    return sidebar_component(bool(collapsed)).children

# ---------------------- Settings: Headcount-only dynamic sources ----------------------

@app.callback(
    Output("row-location", "style"),
    Output("row-hier", "style"),
    Input("set-scope", "value"),
)
def _toggle_scope_rows(scope):
    if scope == "location":
        return ({"display":"flex"}, {"display":"none"})
    if scope == "hier":
        return ({"display":"none"}, {"display":"flex"})
    return ({"display":"none"}, {"display":"none"})

@app.callback(
    Output("set-ba","options"),
    Output("set-ba","value"),
    Output("set-location","options"),
    Output("set-location","value"),
    Input("url-router","pathname"),
    prevent_initial_call=False
)
def settings_enter(path):
    if (path or "").rstrip("/") != "/settings":
        raise PreventUpdate
    bas  = _bas_from_headcount()
    locs = _all_locations()  # from Position Location Country
    ba_val  = bas[0]  if bas  else None
    loc_val = locs[0] if locs else None
    return (
        [{"label": b, "value": b} for b in bas], ba_val,
        [{"label": l, "value": l} for l in locs], loc_val
    )

@app.callback(
    Output("set-subba","options", allow_duplicate=True),
    Output("set-subba","value", allow_duplicate=True),
    Input("set-ba","value"),
    prevent_initial_call=True,
)
def settings_fill_sba(ba):
    sbas = _sbas_from_headcount(ba) if ba else []
    return [{"label": s, "value": s} for s in sbas], (sbas[0] if sbas else None)

@app.callback(
    Output("set-site-hier","options"),
    Output("set-site-hier","value"),
    Input("set-ba","value"),
    Input("set-location","value"),
    prevent_initial_call=False
)
def settings_fill_site_hier(ba, location):
    sites = _sites_for_ba_location(ba, location) if ba else []
    return [{"label": s, "value": s} for s in sites], (sites[0] if sites else None)

@app.callback(
    Output("set-lob","options"),
    Output("set-lob","value"),
    Input("set-ba","value"), Input("set-subba","value")
)
def settings_fill_lob(ba, sba):
    lobs = _lobs_for_ba_sba(ba, sba) if (ba and sba) else []
    return [{"label": l, "value": l} for l in lobs], (lobs[0] if lobs else None)

@app.callback(
    Output("set-interval","value"),
    Output("set-hours","value"),
    Output("set-shrink","value"),
    Output("set-sl","value"),
    Output("set-slsec","value"),
    Output("set-occ","value"),
    Output("set-utilbo","value"),
    Output("set-utilob","value"),
    Output("set-shrink-chat","value"),
    Output("set-shrink-ob","value"),
    Output("set-utilchat","value"),
    Output("set-chatcc","value"),
    Output("set-bo-model","value"),
    Output("set-bo-tat","value"),
    Output("set-bo-wd","value"),
    Output("set-bo-hpd","value"),
    Output("set-bo-shrink","value"),
    Output("set-nest-weeks","value"),
    Output("set-sda-weeks","value"),
    Output("set-throughput-train","value"),
    Output("set-throughput-nest","value"),
    Output("settings-scope-note","children"),
    Input("set-scope","value"),
    Input("set-ba","value"),
    Input("set-subba","value"),
    Input("set-lob","value"),
    Input("set-location","value"),
    Input("set-site-hier","value"),
    prevent_initial_call=False
)
def load_for_scope(scope, ba, subba, channel, location_only, site_hier):
    # most specific first
    if scope == "hier" and ba and subba and channel and site_hier:
        s = resolve_settings(ba=ba, subba=subba, lob=channel, location=site_hier)
        note = f"Scope: {ba} › {subba} › {channel} › {site_hier}"
    elif scope == "location" and location_only:
        s = resolve_settings(location=location_only)
        note = f"Scope: Location = {location_only}"
    else:
        s = load_defaults()
        note = "Scope: Global defaults"

    s = (s or DEFAULT_SETTINGS)
    return (
        int(s.get("interval_minutes", 30)),
        float(s.get("hours_per_fte", 8.0)),
        float(s.get("shrinkage_pct", 0.30)) * 100.0,
        float(s.get("target_sl", 0.80)) * 100.0,
        int(s.get("sl_seconds", 20)),
        float(s.get("occupancy_cap_voice", 0.85)) * 100.0,
        float(s.get("util_bo", 0.85)) * 100.0,
        float(s.get("util_ob", 0.85)) * 100.0,
        float(s.get("chat_shrinkage_pct", s.get("shrinkage_pct", 0.0))) * 100.0,
        float(s.get("ob_shrinkage_pct",   s.get("shrinkage_pct", 0.0))) * 100.0,
        float(s.get("util_chat", s.get("util_bo", 0.85))) * 100.0,
        float(s.get("chat_concurrency", 1.5)),
        (s.get("bo_capacity_model","tat")),
        float(s.get("bo_tat_days", 5)),
        int(s.get("bo_workdays_per_week", 5)),
        float(s.get("bo_hours_per_day", s.get("hours_per_fte", 8))),
        float(s.get("bo_shrinkage_pct", s.get("shrinkage_pct", 0))) * 100.0,
        int(s.get("nesting_weeks", s.get("default_nesting_weeks", 0) or 0)),
        int(s.get("sda_weeks", s.get("default_sda_weeks", 0) or 0)),
        float(str(s.get("throughput_train_pct", 100)).replace('%','')),
        float(str(s.get("throughput_nest_pct", 100)).replace('%','')),
        note,
    )

# --------- Dynamic week inputs for Nesting ---------
@app.callback(
    Output("nest-login-container","children"),
    Output("nest-aht-container","children"),
    Input("set-nest-weeks","value"),
    Input("set-scope","value"),
    Input("set-ba","value"), Input("set-subba","value"),
    Input("set-lob","value"), Input("set-location","value"), Input("set-site-hier","value"),
)
def _render_nest_inputs(n_weeks, scope, ba, subba, lob, location_only, site_hier):
    try:
        n = int(n_weeks or 0)
    except Exception:
        n = 0
    # Resolve current settings for prefill
    if scope == "hier" and ba and subba and lob and site_hier:
        s = resolve_settings(ba=ba, subba=subba, lob=lob, location=site_hier)
    elif scope == "location" and location_only:
        s = resolve_settings(location=location_only)
    else:
        s = load_defaults()
    s = s or DEFAULT_SETTINGS

    def _to_list(val):
        if val is None: return []
        if isinstance(val, (list, tuple, np.ndarray)): return [float(str(x).replace('%','')) for x in val]
        return [float(str(x).strip().replace('%','')) for x in str(val).split(',') if str(x).strip() != '']

    login_vals = _to_list(s.get("nesting_productivity_pct"))
    aht_vals   = _to_list(s.get("nesting_aht_uplift_pct"))
    # Build components
    login_children = []
    aht_children   = []
    for i in range(1, n+1):
        lv = login_vals[i-1] if i-1 < len(login_vals) else 100.0
        av = aht_vals[i-1]   if i-1 < len(aht_vals)   else 0.0
        login_children.append(
            dbc.InputGroup([
                dbc.InputGroupText(f"Week {i} Login %"),
                dbc.Input(id={"group":"nest","kind":"login","index": i}, type="number", step=1, value=lv)
            ], className="mb-2")
        )
        aht_children.append(
            dbc.InputGroup([
                dbc.InputGroupText(f"Week {i} AHT Uplift %"),
                dbc.Input(id={"group":"nest","kind":"aht","index": i}, type="number", step=1, value=av)
            ], className="mb-2")
        )
    return login_children, aht_children

# --------- Dynamic week inputs for SDA ---------
@app.callback(
    Output("sda-login-container","children"),
    Output("sda-aht-container","children"),
    Input("set-sda-weeks","value"),
    Input("set-scope","value"),
    Input("set-ba","value"), Input("set-subba","value"),
    Input("set-lob","value"), Input("set-location","value"), Input("set-site-hier","value"),
)
def _render_sda_inputs(n_weeks, scope, ba, subba, lob, location_only, site_hier):
    try:
        n = int(n_weeks or 0)
    except Exception:
        n = 0
    if scope == "hier" and ba and subba and lob and site_hier:
        s = resolve_settings(ba=ba, subba=subba, lob=lob, location=site_hier)
    elif scope == "location" and location_only:
        s = resolve_settings(location=location_only)
    else:
        s = load_defaults()
    s = s or DEFAULT_SETTINGS

    def _to_list(val):
        if val is None: return []
        if isinstance(val, (list, tuple, np.ndarray)): return [float(str(x).replace('%','')) for x in val]
        return [float(str(x).strip().replace('%','')) for x in str(val).split(',') if str(x).strip() != '']

    login_vals = _to_list(s.get("sda_productivity_pct"))
    aht_vals   = _to_list(s.get("sda_aht_uplift_pct"))
    login_children = []
    aht_children   = []
    for i in range(1, n+1):
        lv = login_vals[i-1] if i-1 < len(login_vals) else 100.0
        av = aht_vals[i-1]   if i-1 < len(aht_vals)   else 0.0
        login_children.append(
            dbc.InputGroup([
                dbc.InputGroupText(f"Week {i} Login %"),
                dbc.Input(id={"group":"sda","kind":"login","index": i}, type="number", step=1, value=lv)
            ], className="mb-2")
        )
        aht_children.append(
            dbc.InputGroup([
                dbc.InputGroupText(f"Week {i} AHT Uplift %"),
                dbc.Input(id={"group":"sda","kind":"aht","index": i}, type="number", step=1, value=av)
            ], className="mb-2")
        )
    return login_children, aht_children

from dash import ALL

@app.callback(
    Output("settings-save-msg","children", allow_duplicate=True),
    Input("btn-save-settings","n_clicks"),
    State("set-scope","value"),
    State("set-ba","value"), State("set-subba","value"),
    State("set-lob","value"),
    State("set-location","value"), State("set-site-hier","value"),
    State("set-interval","value"), State("set-hours","value"),
    State("set-shrink","value"), State("set-sl","value"), State("set-slsec","value"),
    State("set-occ","value"), State("set-utilbo","value"), State("set-utilob","value"), State("set-shrink-chat","value"), State("set-shrink-ob","value"), State("set-utilchat","value"), State("set-chatcc","value"),
    State("set-bo-model","value"), State("set-bo-tat","value"),
    State("set-bo-wd","value"), State("set-bo-hpd","value"), State("set-bo-shrink","value"),
    State("set-nest-weeks","value"),
    State({'group':'nest','kind':'login','index': ALL}, 'value'),
    State({'group':'nest','kind':'aht','index': ALL}, 'value'),
    State("set-sda-weeks","value"),
    State({'group':'sda','kind':'login','index': ALL}, 'value'),
    State({'group':'sda','kind':'aht','index': ALL}, 'value'),
    State("set-throughput-train","value"),
    State("set-throughput-nest","value"),
    prevent_initial_call=True
)
def save_scoped(n, scope, ba, subba, channel, location_only, site_hier,
                ivl, hrs, shr, sl, slsec, occ, utilbo, utilob, shrink_chat, shrink_ob, utilchat, chatcc,
                bo_model, bo_tat, bo_wd, bo_hpd, bo_shrink,
                nest_weeks, nest_login_list, nest_aht_list,
                sda_weeks, sda_login_list, sda_aht_list,
                t_train, t_nest):
    if not n:
        raise PreventUpdate
    # role guard
    user = current_user_fallback()
    if not can_save_settings(get_user_role(user)):
        return "Insufficient permissions to save settings.", False

    # Normalize dynamic lists to CSV strings
    def _to_csv(vals, n):
        arr = list(vals or [])
        arr = ["" if v is None else str(v) for v in arr[: int(n or 0)]]
        return ",".join(arr)

    payload = {
        "interval_minutes":     int(ivl or 30),
        "interval_sec":         int(ivl or 30) * 60,
        "hours_per_fte":        float(hrs or 8.0),
        "shrinkage_pct":        float(shr or 0) / 100.0,
        "target_sl":            float(sl or 80) / 100.0,
        "sl_seconds":           int(slsec or 20),
        "occupancy_cap_voice":  float(occ or 85) / 100.0,
        "util_bo":              float(utilbo or 85) / 100.0,
        "util_ob":              float(utilob or 85) / 100.0,
        "chat_shrinkage_pct":  float(shrink_chat or 0) / 100.0,
        "ob_shrinkage_pct":    float(shrink_ob or 0) / 100.0,
        "util_chat":           float(utilchat or 85) / 100.0,
        "chat_concurrency":     float(chatcc or 1.5),
        "bo_capacity_model":    (bo_model or "tat").lower(),
        "bo_tat_days":          float(bo_tat or 5),
        "bo_workdays_per_week": int(bo_wd or 5),
        "bo_hours_per_day":     float(bo_hpd or (hrs or 8.0)),
        "bo_shrinkage_pct":     float(bo_shrink or 0) / 100.0,
        # Learning curve + SDA + Throughput
        "nesting_weeks":        int(nest_weeks or 0),
        "nesting_productivity_pct": _to_csv(nest_login_list, nest_weeks),
        "nesting_aht_uplift_pct":   _to_csv(nest_aht_list,   nest_weeks),
        "sda_weeks":            int(sda_weeks or 0),
        "sda_productivity_pct": _to_csv(sda_login_list, sda_weeks),
        "sda_aht_uplift_pct":   _to_csv(sda_aht_list,   sda_weeks),
        "throughput_train_pct": float(t_train or 100),
        "throughput_nest_pct":  float(t_nest or 100),
    }

    if scope == "global":
        save_defaults(payload)
        return "Saved global defaults", False

    if scope == "location":
        if not location_only:
            return "Select a location to save.", False
        save_scoped_settings("location", location_only, payload)
        return f"Saved for location: {location_only}", False
    if scope == "hier":
        if not (ba and subba and channel and site_hier):
            return "Pick BA/SubBA/Channel/Site to save.", False
        key = f"{ba}|{subba}|{channel}|{site_hier}"
        save_scoped_settings("hier", key, payload)
        return f"Saved for {ba} ? {subba} ? {channel} ? {site_hier}", False
    return "", False

@app.callback(Output("set-location-hint","children"), Input("set-location","value"))
def _loc_hint(val):
    return "", False if not val else f"Using Position Location Country = {val}"


# ---------------------- Roster (unchanged) ----------------------
def build_roster_template_wide(start_date: dt.date, end_date: dt.date, include_sample: bool = False) -> pd.DataFrame:
    base_cols = [
        "BRID", "Name", "Team Manager",
        "Business Area", "Sub Business Area", "LOB",
        "Site", "Location", "Country"
    ]
    if not isinstance(start_date, dt.date):
        start_date = pd.to_datetime(start_date).date()
    if not isinstance(end_date, dt.date):
        end_date = pd.to_datetime(end_date).date()
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    date_cols = [(start_date + dt.timedelta(days=i)).isoformat()
                 for i in range((end_date - start_date).days + 1)]
    cols = base_cols + date_cols
    df = pd.DataFrame(columns=cols)

    if include_sample and date_cols:
        r1 = {c: "" for c in cols}
        r1.update({
            "BRID": "IN0001", "Name": "Asha Rao", "Team Manager": "Priyanka Menon",
            "Business Area": "Retail", "Sub Business Area": "Cards", "LOB": "Back Office",
            "Site": "Chennai", "Location": "IN-Chennai", "Country": "India",
            date_cols[0]: "09:00-17:30"
        })
        r2 = {c: "" for c in cols}
        r2.update({
            "BRID": "UK0002", "Name": "Alex Doe", "Team Manager": "Chris Lee",
            "Business Area": "Retail", "Sub Business Area": "Cards", "LOB": "Voice",
            "Site": "Glasgow", "Location": "UK-Glasgow", "Country": "UK",
            date_cols[0]: "Leave"
        })
        if len(date_cols) > 1:
            r1[date_cols[1]] = "10:00-18:00"
        df = pd.DataFrame([r1, r2])[cols]
    return df

def normalize_roster_wide(df_wide: pd.DataFrame) -> pd.DataFrame:
    if df_wide is None or df_wide.empty:
        return pd.DataFrame(columns=[
            "BRID","Name","Team Manager","Business Area","Sub Business Area",
            "LOB","Site","Location","Country","date","entry"
        ])
    id_cols = ["BRID","Name","Team Manager","Business Area","Sub Business Area","LOB","Site","Location","Country"]
    id_cols = [c for c in id_cols if c in df_wide.columns]
    date_cols = [c for c in df_wide.columns if c not in id_cols]
    long = df_wide.melt(id_vars=id_cols, value_vars=date_cols, var_name="date", value_name="entry")
    long["entry"] = long["entry"].fillna("").astype(str).str.strip()
    long = long[long["entry"] != ""]
    long["date"] = pd.to_datetime(long["date"], errors="coerce", dayfirst=True).dt.date
    long = long[pd.notna(long["date"])]
    long["is_leave"] = long["entry"].str.lower().isin({"leave","l","off","pto"})
    return long

@app.callback(
    Output("tbl-headcount-preview","data"),
    Output("tbl-headcount-preview","columns"),
    Output("hc-msg", "children", allow_duplicate=True),
    Input("up-headcount","contents"),
    State("up-headcount","filename"),
    prevent_initial_call=True
)
def hc_preview(contents, filename):
    df = parse_upload(contents, filename)
    if df is None or df.empty:
        return [], [], "", None
    wanted = ["BRID","Full Name","Line Manager BRID","Line Manager Full Name",
              "Business Area","Sub Business Area","LOB","Site","Location"]
    cols = [c for c in df.columns if str(c).strip() in wanted] or list(df.columns)[:12]
    dff = df[cols].copy()
    return dff.to_dict("records"), pretty_columns(dff), f"Preview loaded: {len(df)} rows", None

@app.callback(
    Output("hc-msg","children", allow_duplicate=True),
    Input("btn-save-headcount","n_clicks"),
    State("up-headcount","contents"),
    State("up-headcount","filename"),
    prevent_initial_call=True
)
def hc_save(n, contents, filename):
    if not n:
        raise PreventUpdate
    df = parse_upload(contents, filename)
    if df is None or df.empty:
        return "No data to save.", False
    try:
        from cap_store import save_headcount_df
        count = save_headcount_df(df)
        return f"Saved headcount: {count} rows."
    except Exception as e:
        return f"Error saving headcount: {e}"

@app.callback(Output("dl-hc-template","data"),
          Input("btn-dl-hc-template","n_clicks"),
          prevent_initial_call=True)
def dl_hc_tmpl(_n):
    return dcc.send_data_frame(headcount_template_df().to_csv, "headcount_template.csv", index=False)

# Auto-refresh the Default Settings scope dropdowns after Headcount save
@app.callback(
    Output("set-ba","options", allow_duplicate=True),
    Output("set-ba","value", allow_duplicate=True),
    Output("set-location","options", allow_duplicate=True),
    Output("set-location","value", allow_duplicate=True),
    Input("hc-msg", "children"),
    prevent_initial_call=True,
)
def _refresh_scope_after_headcount(_msg):
    # When headcount is saved, refresh BA and Location options from the new mapping
    try:
        bas  = _bas_from_headcount() or []
    except Exception:
        bas = []
    try:
        locs = _all_locations() or []
    except Exception:
        locs = []
    return (
        [{"label": b, "value": b} for b in bas], (bas[0] if bas else None),
        [{"label": l, "value": l} for l in locs], (locs[0] if locs else None),
    )

# Also refresh scope immediately on upload preview (without needing save)
@app.callback(
    Output("set-ba","options", allow_duplicate=True),
    Output("set-ba","value", allow_duplicate=True),
    Output("set-location","options", allow_duplicate=True),
    Output("set-location","value", allow_duplicate=True),
    Input("up-headcount", "contents"),
    State("up-headcount", "filename"),
    prevent_initial_call=True,
)
def _preview_scope_from_headcount(contents, filename):
    if not contents:
        raise PreventUpdate
    try:
        df = parse_upload(contents, filename)
    except Exception:
        df = None
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise PreventUpdate
    L = {str(c).strip().lower(): c for c in df.columns}
    # BA column candidates
    c_ba = L.get("journey") or L.get("business area") or L.get("current org unit description") or L.get("level 0")
    # Location/Country column candidates (Position Location Country preferred)
    c_loc = L.get("position_location_country") or L.get("location country") or L.get("country") or L.get("location")
    def _unique(col):
        try:
            vals = df[col].dropna().astype(str).str.strip().replace({"": np.nan}).dropna().drop_duplicates().sort_values()
            return vals.tolist()
        except Exception:
            return []
    bas = _unique(c_ba) if c_ba else []
    locs = _unique(c_loc) if c_loc else []
    return (
        [{"label": b, "value": b} for b in bas], (bas[0] if bas else None),
        [{"label": l, "value": l} for l in locs], (locs[0] if locs else None),
    )

# === Shrinkage RAW: Download templates ===
@app.callback(Output("tbl-chat-forecast","data"), Output("tbl-chat-forecast","columns"), Output("chat-forecast-msg","children", allow_duplicate=True),
          Input("up-chat-forecast","contents"), State("up-chat-forecast","filename"), prevent_initial_call=True)
def up_chat_forecast(contents, filename):
    df = parse_upload(contents, filename)
    if df is None or df.empty: return [], [], "Could not read file"
    L = {c.lower(): c for c in df.columns}
    c_date = L.get("date"); c_items = L.get("items") or L.get("volume"); c_aht = L.get("aht_sec") or L.get("aht")
    if not (c_date and c_items): return [], [], "Need Date and Items"
    out = pd.DataFrame({
        "date": pd.to_datetime(df[c_date], errors="coerce").dt.date.astype(str),
        "items": pd.to_numeric(df[c_items], errors="coerce").fillna(0.0),
        "aht_sec": pd.to_numeric(df[c_aht], errors="coerce").fillna(240.0) if c_aht else 240.0,
    })
    for extra in ["Business Area","Sub Business Area","Channel"]:
        if extra in df.columns: out[extra] = df[extra]
    cols = ["date","items","aht_sec","Business Area","Sub Business Area","Channel"]
    out = out[[c for c in cols if c in out.columns]]
    return out.to_dict("records"), pretty_columns(out), f"Loaded {len(out)} rows"

@app.callback(Output("tbl-chat-actual","data"), Output("tbl-chat-actual","columns"), Output("chat-actual-msg","children", allow_duplicate=True),
          Input("up-chat-actual","contents"), State("up-chat-actual","filename"), prevent_initial_call=True)
def up_chat_actual(contents, filename):
    return up_chat_forecast(contents, filename)

@app.callback(Output("chat-forecast-msg","children", allow_duplicate=True),
          Input("btn-save-chat-forecast","n_clicks"),
          State("tbl-chat-forecast","data"),
          State("set-scope","value"), State("set-ba","value"), State("set-subba","value"), State("set-lob","value"), State("set-site-hier","value"),
          prevent_initial_call=True)
def save_chat_forecast(_n, rows, scope, ba, sba, ch, site):
    ok, msg = _scope_guard(scope, ba, sba, ch, site)
    if not ok: return msg
    df = pd.DataFrame(rows or [])
    if df.empty or not {'date','items','aht_sec'}.issubset(df.columns):
        return 'Missing required columns (date/items/aht_sec)', False
    sk = _canon_scope(ba, sba, ch, site)
    save_timeseries('chat_forecast_volume', sk, df[['date','items']].rename(columns={'items':'volume'}))
    save_timeseries('chat_forecast_aht',    sk, df[['date','aht_sec']])
    return f"Saved CHAT forecast ({len(df)}) for {sk}", False

@app.callback(Output("chat-actual-msg","children", allow_duplicate=True),
          Input("btn-save-chat-actual","n_clicks"),
          State("tbl-chat-actual","data"),
          State("set-scope","value"), State("set-ba","value"), State("set-subba","value"), State("set-lob","value"), State("set-site-hier","value"),
          prevent_initial_call=True)
def save_chat_actual(_n, rows, scope, ba, sba, ch, site):
    ok, msg = _scope_guard(scope, ba, sba, ch, site)
    if not ok: return msg
    df = pd.DataFrame(rows or [])
    if df.empty or not {'date','items','aht_sec'}.issubset(df.columns):
        return 'Missing required columns (date/items/aht_sec)', False
    sk = _canon_scope(ba, sba, ch, site)
    save_timeseries('chat_actual_volume', sk, df[['date','items']].rename(columns={'items':'volume'}))
    save_timeseries('chat_actual_aht',    sk, df[['date','aht_sec']])
    return f"Saved CHAT actual ({len(df)}) for {sk}", False

@app.callback(Output("tbl-ob-forecast","data"), Output("tbl-ob-forecast","columns"), Output("ob-forecast-msg","children", allow_duplicate=True),
          Input("up-ob-forecast","contents"), State("up-ob-forecast","filename"), prevent_initial_call=True)
def up_ob_forecast(contents, filename):
    df = parse_upload(contents, filename)
    if df is None or df.empty: return [], [], "Could not read file"
    L = {c.lower(): c for c in df.columns}
    c_date = L.get('date'); c_opc = L.get('opc') or L.get('dials') or L.get('calls')
    c_conn = L.get('connect_rate') or L.get('connect%') or L.get('connect pct')
    c_rpc  = L.get('rpc'); c_rpcr = L.get('rpc_rate') or L.get('rpc%')
    c_aht  = L.get('aht_sec') or L.get('aht')
    if not c_date or not (c_opc or (c_conn and (c_rpc or c_rpcr))):
        return [], [], 'Need Date, and either OPC or Connect Rate + (RPC or RPC Rate)'
    out = pd.DataFrame({'date': pd.to_datetime(df[c_date], errors='coerce').dt.date.astype(str)})
    if c_opc: out['opc'] = pd.to_numeric(df[c_opc], errors='coerce').fillna(0.0)
    if c_conn: out['connect_rate'] = pd.to_numeric(df[c_conn], errors='coerce').fillna(0.0)
    if c_rpc: out['rpc'] = pd.to_numeric(df[c_rpc], errors='coerce').fillna(0.0)
    if c_rpcr: out['rpc_rate'] = pd.to_numeric(df[c_rpcr], errors='coerce').fillna(0.0)
    out['aht_sec'] = pd.to_numeric(df[c_aht], errors='coerce').fillna(240.0) if c_aht else 240.0
    for extra in ["Business Area","Sub Business Area","Channel"]:
        if extra in df.columns: out[extra] = df[extra]
    cols = ['date','opc','connect_rate','rpc','rpc_rate','aht_sec','Business Area','Sub Business Area','Channel']
    out = out[[c for c in cols if c in out.columns]]
    return out.to_dict('records'), pretty_columns(out), f"Loaded {len(out)} rows"

@app.callback(Output("tbl-ob-actual","data"), Output("tbl-ob-actual","columns"), Output("ob-actual-msg","children", allow_duplicate=True),
          Input("up-ob-actual","contents"), State("up-ob-actual","filename"), prevent_initial_call=True)
def up_ob_actual(contents, filename):
    return up_ob_forecast(contents, filename)

@app.callback(Output("ob-forecast-msg","children", allow_duplicate=True),
          Input("btn-save-ob-forecast","n_clicks"),
          State("tbl-ob-forecast","data"),
          State("set-scope","value"), State("set-ba","value"), State("set-subba","value"), State("set-lob","value"), State("set-site-hier","value"),
          prevent_initial_call=True)
def save_ob_forecast(_n, rows, scope, ba, sba, ch, site):
    ok, msg = _scope_guard(scope, ba, sba, ch, site)
    if not ok: return msg
    df = pd.DataFrame(rows or [])
    if df.empty or 'date' not in df.columns or 'aht_sec' not in df.columns:
        return 'Missing required columns (date/aht_sec)', False
    sk = _canon_scope(ba, sba, ch, site)
    # Save each metric if present
    def save_if(name):
        if name in df.columns:
            save_timeseries(f'ob_forecast_{name}', sk, df[['date', name]].copy())
    for name in ['opc','connect_rate','rpc','rpc_rate','aht_sec']:
        save_if(name)
    return f"Saved OUTBOUND forecast ({len(df)}) for {sk}", False

@app.callback(Output("ob-actual-msg","children", allow_duplicate=True),
          Input("btn-save-ob-actual","n_clicks"),
          State("tbl-ob-actual","data"),
          State("set-scope","value"), State("set-ba","value"), State("set-subba","value"), State("set-lob","value"), State("set-site-hier","value"),
          prevent_initial_call=True)
def save_ob_actual(_n, rows, scope, ba, sba, ch, site):
    ok, msg = _scope_guard(scope, ba, sba, ch, site)
    if not ok: return msg
    df = pd.DataFrame(rows or [])
    if df.empty or 'date' not in df.columns or 'aht_sec' not in df.columns:
        return 'Missing required columns (date/aht_sec)', False
    sk = _canon_scope(ba, sba, ch, site)
    def save_if(name):
        if name in df.columns:
            save_timeseries(f'ob_actual_{name}', sk, df[['date', name]].copy())
    for name in ['opc','connect_rate','rpc','rpc_rate','aht_sec']:
        save_if(name)
    return f"Saved OUTBOUND actual ({len(df)}) for {sk}", False

# Tactical by scope: Chat and Outbound
@app.callback(Output("tbl-chat-forecast1","data"), Output("tbl-chat-forecast1","columns"), Output("chat-forecast-msg1","children", allow_duplicate=True),
          Input("up-chat-forecast1","contents"), State("up-chat-forecast1","filename"), prevent_initial_call=True)
def up_chat_forecast1(contents, filename):
    return up_chat_forecast(contents, filename)

@app.callback(Output("chat-forecast-msg1","children", allow_duplicate=True),
          Input("btn-save-chat-forecast1","n_clicks"),
          State("tbl-chat-forecast1","data"),
          State("set-scope","value"), State("set-ba","value"), State("set-subba","value"), State("set-lob","value"), State("set-site-hier","value"),
          prevent_initial_call=True)
def save_chat_forecast1(_n, rows, scope, ba, sba, ch, site):
    ok, msg = _scope_guard(scope, ba, sba, ch, site)
    if not ok: return msg
    df = pd.DataFrame(rows or [])
    if df.empty or not {'date','items','aht_sec'}.issubset(df.columns):
        return 'Missing required columns (date/items/aht_sec)', False
    sk = _canon_scope(ba, sba, ch, site)
    save_timeseries('chat_tactical_volume', sk, df[['date','items']].rename(columns={'items':'volume'}))
    save_timeseries('chat_tactical_aht',    sk, df[['date','aht_sec']])
    return f"Saved CHAT tactical ({len(df)}) for {sk}"

@app.callback(Output("tbl-ob-forecast1","data"), Output("tbl-ob-forecast1","columns"), Output("ob-forecast-msg1","children", allow_duplicate=True),
          Input("up-ob-forecast1","contents"), State("up-ob-forecast1","filename"), prevent_initial_call=True)
def up_ob_forecast1(contents, filename):
    return up_ob_forecast(contents, filename)

@app.callback(Output("ob-forecast-msg1","children", allow_duplicate=True),
          Input("btn-save-ob-forecast1","n_clicks"),
          State("tbl-ob-forecast1","data"),
          State("set-scope","value"), State("set-ba","value"), State("set-subba","value"), State("set-lob","value"), State("set-site-hier","value"),
          prevent_initial_call=True)
def save_ob_forecast1(_n, rows, scope, ba, sba, ch, site):
    ok, msg = _scope_guard(scope, ba, sba, ch, site)
    if not ok: return msg
    df = pd.DataFrame(rows or [])
    if df.empty or 'date' not in df.columns or 'aht_sec' not in df.columns:
        return 'Missing required columns (date/aht_sec)', False
    sk = _canon_scope(ba, sba, ch, site)
    def save_if(name):
        if name in df.columns:
            save_timeseries(f'ob_tactical_{name}', sk, df[['date', name]].copy())
    for name in ['opc','connect_rate','rpc','rpc_rate','aht_sec']:
        save_if(name)
    return f"Saved OUTBOUND tactical ({len(df)}) for {sk}"
