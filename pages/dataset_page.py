from __future__ import annotations
from dash import html, dcc, dash_table, Output, Input, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta

from app_instance import app
from common import header_bar, pretty_columns, CHANNEL_LIST
from cap_store import (
    load_timeseries_any,
    load_roster, load_hiring, resolve_settings
)
from common import _hcu_df, _hcu_cols
from capacity_core import required_fte_daily, supply_fte_daily, make_voice_sample, make_backoffice_sample


def _today_range(days: int = 56) -> tuple[date, date]:
    end = date.today(); start = end - timedelta(days=days)
    return start, end


def _hc_dim_df() -> pd.DataFrame:
    df = _hcu_df()
    if df is None or df.empty:
        return pd.DataFrame(columns=["Business Area","Sub Business Area","Channel"])
    C = _hcu_cols(df)
    cols = {"Business Area": C.get("ba"), "Sub Business Area": C.get("sba"), "Channel": C.get("lob")}
    out = pd.DataFrame()
    for k, c in cols.items():
        out[k] = df[c].astype(str) if c and c in df.columns else ""
    for k in out.columns:
        out[k] = out[k].fillna("").astype(str).str.strip()
    return out


def _scope_keys_from_headcount() -> list[str]:
    m = _hc_dim_df()
    if m.empty:
        return []
    d = m[["Business Area","Sub Business Area","Channel"]].copy()
    d = d[(d["Business Area"]!="") & (d["Sub Business Area"]!="")]
    d["Channel"] = d["Channel"].replace("", "Voice")
    d["sk"] = (d["Business Area"].str.strip() + "|" + d["Sub Business Area"].str.strip() + "|" + d["Channel"].str.strip()).str.lower()
    return sorted(d["sk"].dropna().unique().tolist())


def _load_voice(scopes: list[str], pref: str = "auto") -> pd.DataFrame:
    pref = (pref or "auto").lower()
    if pref == "forecast":
        vol = load_timeseries_any("voice_forecast_volume", scopes)
        aht = load_timeseries_any("voice_forecast_aht", scopes)
        if vol.empty:
            vol = load_timeseries_any("voice_actual_volume", scopes)
        if aht.empty:
            aht = load_timeseries_any("voice_actual_aht", scopes)
    else:  # auto/actual
        vol = load_timeseries_any("voice_actual_volume", scopes)
        aht = load_timeseries_any("voice_actual_aht", scopes)
        if vol.empty:
            vol = load_timeseries_any("voice_forecast_volume", scopes)
        if aht.empty:
            aht = load_timeseries_any("voice_forecast_aht", scopes)
    if vol.empty:
        vol = load_timeseries_any("voice_tactical_volume", scopes)
    if aht.empty:
        aht = load_timeseries_any("voice_tactical_aht", scopes)
    if vol.empty:
        return make_voice_sample()
    vc = {c.lower(): c for c in vol.columns}; ac = {c.lower(): c for c in aht.columns}
    dv = vc.get("date", "date"); da = ac.get("date", "date")
    iv = vc.get("interval"); ia = ac.get("interval")
    if iv and ia:
        df = pd.merge(vol.rename(columns={vc.get("volume","volume"):"volume"}),
                      aht.rename(columns={ac.get("aht_sec","aht_sec"):"aht_sec"}),
                      left_on=[dv, iv], right_on=[da, ia], how="outer")
        df = df.rename(columns={dv:"date", iv:"interval"})
    else:
        df = pd.merge(vol.rename(columns={vc.get("volume","volume"):"volume"}),
                      aht.rename(columns={ac.get("aht_sec","aht_sec"):"aht_sec"}),
                      left_on=[dv], right_on=[da], how="left")
        df = df.rename(columns={dv:"date"}); df["interval"] = None
    if "aht_sec" not in df:
        df["aht_sec"] = 300.0
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df


def _load_bo(scopes: list[str], pref: str = "auto") -> pd.DataFrame:
    pref = (pref or "auto").lower()
    if pref == "forecast":
        vol = load_timeseries_any("bo_forecast_volume", scopes)
        sut = load_timeseries_any("bo_forecast_sut", scopes)
        if vol.empty:
            vol = load_timeseries_any("bo_actual_volume", scopes)
        if sut.empty:
            sut = load_timeseries_any("bo_actual_sut", scopes)
    else:  # auto/actual
        vol = load_timeseries_any("bo_actual_volume", scopes)
        sut = load_timeseries_any("bo_actual_sut", scopes)
        if vol.empty:
            vol = load_timeseries_any("bo_forecast_volume", scopes)
        if sut.empty:
            sut = load_timeseries_any("bo_forecast_sut", scopes)
    if vol.empty:
        vol = load_timeseries_any("bo_tactical_volume", scopes)
    if sut.empty:
        sut = load_timeseries_any("bo_tactical_sut", scopes)
    if vol.empty:
        return make_backoffice_sample()
    vc = {c.lower(): c for c in vol.columns}; sc = {c.lower(): c for c in sut.columns}
    dv = vc.get("date","date"); ds = sc.get("date","date")
    df = pd.merge(vol.rename(columns={vc.get("items","items"):"items"}),
                  sut.rename(columns={sc.get("sut_sec","sut_sec"):"sut_sec"}),
                  left_on=[dv], right_on=[ds], how="left").rename(columns={dv:"date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    if "sut_sec" not in df:
        df["sut_sec"] = 600.0
    return df


def page_dataset():
    # Option pools from headcount
    dims = _hc_dim_df()
    opts_ba = sorted([x for x in dims["Business Area"].unique().tolist() if x])
    opts_sba = sorted([x for x in dims["Sub Business Area"].unique().tolist() if x])
    opts_loc = sorted([x for x in (dims["Location"].unique().tolist() if "Location" in dims.columns else []) if x])
    opts_ch = sorted([x for x in (dims["Channel"].unique().tolist() if "Channel" in dims.columns else []) if x]) or list(CHANNEL_LIST)
    opts_site = sorted([x for x in (dims["Site"].unique().tolist() if "Site" in dims.columns else []) if x])

    start, end = _today_range(56)

    return html.Div(dbc.Container([
        header_bar(),
        dbc.Card(dbc.CardBody([
            html.H5("Planner Dataset — Inputs Snapshot"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(id="ds-ba", options=[{"label":x, "value":x} for x in opts_ba], multi=True, placeholder="Business Area"), md=3),
                dbc.Col(dcc.Dropdown(id="ds-sba", options=[{"label":x, "value":x} for x in opts_sba], multi=True, placeholder="Sub Business Area"), md=3),
                dbc.Col(dcc.Dropdown(id="ds-ch", options=[{"label":x, "value":x} for x in opts_ch], multi=True, placeholder="Channel"), md=3),
                dbc.Col(dcc.RadioItems(id="ds-series", options=[{"label":" Auto","value":"auto"},{"label":" Actual","value":"actual"},{"label":" Forecast","value":"forecast"}], value="auto", inline=True), md=3),
            ], className="g-2"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(id="ds-loc", options=[{"label":x, "value":x} for x in opts_loc], multi=True, placeholder="Location"), md=3),
                dbc.Col(dcc.Dropdown(id="ds-site", options=[{"label":x, "value":x} for x in opts_site], multi=True, placeholder="Site"), md=3),
                dbc.Col(dcc.DatePickerRange(id="ds-dates", className="date-compact", start_date=start, end_date=end, display_format="YYYY-MM-DD"), md=6),
            ], className="g-2 mt-1"),
        ]), className="mb-3"),

        html.Div(dash_table.DataTable(id="dataset-table", page_size=12, style_table={"overflowX":"auto"}, style_as_list_view=True, style_header={"textTransform": "none"}), className="loading-block"),
        html.Div(dcc.Graph(id="dataset-fig", className="mt-3"), className="loading-block"),
    ], fluid=True), className="loading-page")


@app.callback(
    Output("ds-sba", "options"), Output("ds-sba", "value"),
    Input("ds-ba", "value"), State("ds-sba", "value"), prevent_initial_call=False
)
def _ds_dep_sba(ba_vals, sba_curr):
    m = _hc_dim_df(); df = m.copy()
    if ba_vals:
        sel = {str(x).strip().lower() for x in (ba_vals if isinstance(ba_vals, list) else [ba_vals])}
        df = df[df["Business Area"].astype(str).str.strip().str.lower().isin(sel)]
    sba_list = sorted([x for x in df["Sub Business Area"].astype(str).dropna().unique().tolist() if x])
    opts = [{"label":x, "value":x} for x in sba_list]
    curr = sba_curr if isinstance(sba_curr, list) else ([sba_curr] if sba_curr else [])
    return opts, [x for x in curr if x in sba_list]


@app.callback(
    Output("ds-ch", "options"), Output("ds-ch", "value"),
    Input("ds-ba", "value"), Input("ds-sba", "value"), State("ds-ch", "value"), prevent_initial_call=False
)
def _ds_dep_channel(ba_vals, sba_vals, ch_curr):
    m = _hc_dim_df(); df = m.copy()
    if ba_vals:
        df = df[df["Business Area"].astype(str).str.strip().str.lower().isin({str(x).strip().lower() for x in (ba_vals if isinstance(ba_vals, list) else [ba_vals])})]
    if sba_vals:
        df = df[df["Sub Business Area"].astype(str).str.strip().str.lower().isin({str(x).strip().lower() for x in (sba_vals if isinstance(sba_vals, list) else [sba_vals])})]
    ch_list = sorted([x for x in df["Channel"].astype(str).dropna().unique().tolist() if x]) or list(CHANNEL_LIST)
    opts = [{"label":x, "value":x} for x in ch_list]
    curr = ch_curr if isinstance(ch_curr, list) else ([ch_curr] if ch_curr else [])
    return opts, [x for x in curr if x in ch_list]


@app.callback(
    Output("ds-site", "options"), Output("ds-site", "value"),
    Input("ds-ba", "value"), Input("ds-sba", "value"), Input("ds-ch", "value"), Input("ds-loc", "value"),
    State("ds-site", "value"), prevent_initial_call=False
)
def _ds_dep_site(ba_vals, sba_vals, ch_vals, loc_vals, site_curr):
    df = _hc_dim_df().copy()
    def f(col, vals):
        nonlocal df
        if vals:
            df = df[df[col].astype(str).str.strip().str.lower().isin({str(x).strip().lower() for x in (vals if isinstance(vals, list) else [vals])})]
    f("Business Area", ba_vals); f("Sub Business Area", sba_vals); f("Channel", ch_vals); f("Location", loc_vals)
    site_list = sorted([x for x in df.get("Site", pd.Series([], dtype=str)).astype(str).dropna().str.strip().unique().tolist() if x])
    loc_set = set([str(x).strip().lower() for x in df.get("Location", pd.Series([], dtype=str)).astype(str).dropna().unique().tolist()])
    country_block = {"india","uk","united kingdom","great britain","england","scotland","wales"}
    site_list = [s for s in site_list if s and (sl := s.strip().lower()) not in loc_set and sl not in country_block]
    opts = [{"label":x, "value":x} for x in site_list]
    curr = site_curr if isinstance(site_curr, list) else ([site_curr] if site_curr else [])
    return opts, [x for x in curr if x in site_list]


@app.callback(
    Output("dataset-table", "data"), Output("dataset-table", "columns"), Output("dataset-fig", "figure"),
    Input("ds-dates", "start_date"), Input("ds-dates", "end_date"), Input("ds-series", "value"),
    Input("ds-ba", "value"), Input("ds-sba", "value"), Input("ds-ch", "value"), Input("ds-loc", "value"), Input("ds-site", "value"),
    prevent_initial_call=False
)
def _ds_refresh(s, e, pref, ba, sba, ch, loc, site):
    try:
        start = pd.to_datetime(s).date() if s else _today_range(56)[0]
        end = pd.to_datetime(e).date() if e else _today_range(56)[1]
    except Exception:
        start, end = _today_range(56)

    dims = _hc_dim_df().copy()
    def f(col, vals):
        nonlocal dims
        if vals:
            dims = dims[dims[col].astype(str).str.strip().str.lower().isin({str(x).strip().lower() for x in (vals if isinstance(vals, list) else [vals])})]
    f("Business Area", ba); f("Sub Business Area", sba); f("Channel", ch); f("Location", loc); f("Site", site)
    if dims.empty:
        return [], pretty_columns(["date","program","total_req_fte","supply_fte","staffing_pct"]), px.line(title="No data")

    d = dims[["Business Area","Sub Business Area","Channel"]].copy()
    d["Channel"] = d["Channel"].replace("", "Voice")
    d["sk"] = (d["Business Area"].str.strip() + "|" + d["Sub Business Area"].str.strip() + "|" + d["Channel"].str.strip()).str.lower()
    scopes = sorted(d["sk"].dropna().unique().tolist())

    voice = _load_voice(scopes, pref=pref)
    bo    = _load_bo(scopes, pref=pref)
    if not voice.empty:
        voice["date"] = pd.to_datetime(voice["date"], errors="coerce").dt.date
        voice = voice[pd.notna(voice["date"])]
        voice = voice[(voice["date"] >= start) & (voice["date"] <= end)]
    if not bo.empty:
        bo["date"] = pd.to_datetime(bo["date"], errors="coerce").dt.date
        bo = bo[pd.notna(bo["date"])]
        bo = bo[(bo["date"] >= start) & (bo["date"] <= end)]

    settings = resolve_settings()
    req_df = required_fte_daily(voice, bo, pd.DataFrame(), settings)
    if not req_df.empty:
        req_df = req_df.groupby(["date","program"], as_index=False)["total_req_fte"].sum()

    roster = load_roster(); roster = roster if isinstance(roster, pd.DataFrame) else pd.DataFrame()
    hiring = load_hiring(); hiring = hiring if isinstance(hiring, pd.DataFrame) else pd.DataFrame()
    sup_df = supply_fte_daily(roster, hiring)
    if not sup_df.empty:
        sup_df = sup_df.groupby(["date","program"], as_index=False)["supply_fte"].sum()

    df = pd.merge(req_df, sup_df, on=["date","program"], how="outer").fillna({"total_req_fte":0.0, "supply_fte":0.0})
    df["staffing_pct"] = np.where(df["total_req_fte"]>0, (df["supply_fte"] / df["total_req_fte"]) * 100.0, np.nan)
    df = df.sort_values(["date","program"]) if not df.empty else pd.DataFrame(columns=["date","program","total_req_fte","supply_fte","staffing_pct"])

    daily = df.groupby("date", as_index=False)[["total_req_fte","supply_fte"]].sum() if not df.empty else pd.DataFrame(columns=["date","total_req_fte","supply_fte"])
    if not daily.empty:
        # Ensure numeric dtypes for wide-form plotting
        for c in ["total_req_fte","supply_fte"]:
            daily[c] = pd.to_numeric(daily[c], errors="coerce")
    fig = px.line(daily, x="date", y=["total_req_fte","supply_fte"], markers=True, title=f"Requirements vs Supply ({pref.title() if pref!='auto' else 'Auto'})") if not daily.empty else px.line(title="No data")

    return df.to_dict("records"), pretty_columns(df), fig



