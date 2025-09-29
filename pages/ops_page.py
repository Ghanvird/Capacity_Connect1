from __future__ import annotations
from dash import html, dcc, dash_table, Output, Input, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

from app_instance import app
from common import header_bar
from cap_store import (
    load_timeseries_any,
    load_roster, load_hiring, resolve_settings
)
from common import _hcu_df, _hcu_cols, CHANNEL_LIST
from capacity_core import required_fte_daily, voice_requirements_interval, make_voice_sample, make_backoffice_sample, supply_fte_daily


def _today_range(default_days: int = 28) -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=default_days)
    return start.isoformat(), end.isoformat()


def _hc_dim_df() -> pd.DataFrame:
    df = _hcu_df()
    if df is None or df.empty:
        # Minimal scaffold to avoid empty UI
        return pd.DataFrame([
            {"Business Area": "Retail", "Sub Business Area": "Cards", "Channel": "Voice", "Location": "India", "Site": "Chennai"},
            {"Business Area": "Retail", "Sub Business Area": "Cards", "Channel": "Back Office", "Location": "UK", "Site": "Glasgow"},
        ])
    C = _hcu_cols(df)
    cols = {}
    cols["Business Area"] = C.get("ba")
    cols["Sub Business Area"] = C.get("sba")
    cols["Channel"] = C.get("lob")
    cols["Location"] = C.get("loc")
    cols["Site"] = C.get("site")
    out = pd.DataFrame()
    for k, c in cols.items():
        if c and c in df.columns:
            out[k] = df[c].astype(str)
        else:
            out[k] = ""
    # normalize blanks
    for k in out.columns:
        out[k] = out[k].fillna("").astype(str).str.strip()
    return out


def _scope_keys_from_filters(ba, sba, ch, site, loc) -> pd.DataFrame:
    df = _hc_dim_df().copy()
    # Apply filters if provided
    def _apply(col, vals):
        nonlocal df
        if vals:
            s = {str(v).strip().lower() for v in (vals if isinstance(vals, list) else [vals]) if str(v).strip()}
            if s:
                df = df[df[col].astype(str).str.strip().str.lower().isin(s)]
    _apply("Business Area", ba)
    _apply("Sub Business Area", sba)
    _apply("Channel", ch)
    _apply("Site", site)
    _apply("Location", loc)
    if df.empty:
        return pd.DataFrame(columns=["ba","sba","ch","loc","site","sk"])
    # Determine channel set: selected, else unique from HC, else CHANNEL_LIST
    if ch:
        if isinstance(ch, list):
            ch_list = [str(x).strip() for x in ch if str(x).strip()]
        else:
            ch_list = [str(ch).strip()]
    else:
        uniq = sorted([x for x in df["Channel"].astype(str).str.strip().unique().tolist() if x])
        ch_list = uniq or list(CHANNEL_LIST)

    rows = []
    for _, r in df.iterrows():
        ba_v = str(r.get("Business Area", "")).strip()
        sba_v = str(r.get("Sub Business Area", "")).strip()
        loc_v = str(r.get("Location", "")).strip()
        site_v= str(r.get("Site", "")).strip()
        for ch_v in ch_list:
            sk = f"{ba_v}|{sba_v}|{str(ch_v).strip()}".lower()
            rows.append({"ba": ba_v, "sba": sba_v, "ch": str(ch_v), "loc": loc_v, "site": site_v, "sk": sk})
    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["sk","site","loc"]) if not out.empty else out
    return out[["ba","sba","ch","loc","site","sk"]]


def _load_voice(scopes: list[str]) -> pd.DataFrame:
    # Try actual -> forecast -> tactical -> sample
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
        return make_voice_sample()  # fallback
    # Merge volume + aht on date + interval when present
    vc = {c.lower(): c for c in vol.columns}
    ac = {c.lower(): c for c in aht.columns}
    date_v = vc.get("date", "date"); date_a = ac.get("date", "date")
    ivl_v  = vc.get("interval"); ivl_a  = ac.get("interval")
    if ivl_v and ivl_a:
        df = pd.merge(vol.rename(columns={vc.get("volume","volume"):"volume"}),
                      aht.rename(columns={ac.get("aht_sec","aht_sec"):"aht_sec"}),
                      left_on=[date_v, ivl_v], right_on=[date_a, ivl_a], how="outer")
        df = df.rename(columns={date_v: "date", ivl_v: "interval"})
    else:
        df = pd.merge(vol.rename(columns={vc.get("volume","volume"):"volume"}),
                      aht.rename(columns={ac.get("aht_sec","aht_sec"):"aht_sec"}),
                      left_on=[date_v], right_on=[date_a], how="left")
        df = df.rename(columns={date_v: "date"})
        df["interval"] = None
    if "aht_sec" not in df:
        df["aht_sec"] = 300.0
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df


def _load_bo(scopes: list[str]) -> pd.DataFrame:
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
    vc = {c.lower(): c for c in vol.columns}
    sc = {c.lower(): c for c in sut.columns}
    d_v = vc.get("date","date"); d_s = sc.get("date","date")
    df = pd.merge(vol.rename(columns={vc.get("items","items"):"items"}),
                  sut.rename(columns={sc.get("sut_sec","sut_sec"):"sut_sec"}),
                  left_on=[d_v], right_on=[d_s], how="left").rename(columns={d_v:"date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    if "sut_sec" not in df:
        df["sut_sec"] = 600.0
    return df


def _agg_by_grain(df: pd.DataFrame, date_col: str, grain: str) -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce").dt.date
    if grain in ("D", "daily"):
        d["bucket"] = d[date_col]
    elif grain in ("W", "weekly"):
        d["bucket"] = pd.to_datetime(d[date_col]).dt.to_period("W-MON").dt.start_time.dt.date
    elif grain in ("M", "monthly"):
        d["bucket"] = pd.to_datetime(d[date_col]).dt.to_period("M").dt.start_time.dt.date
    elif grain in ("Q", "quarterly"):
        d["bucket"] = pd.to_datetime(d[date_col]).dt.to_period("Q").dt.start_time.dt.date
    elif grain in ("Y", "yearly"):
        d["bucket"] = pd.to_datetime(d[date_col]).dt.to_period("Y").dt.start_time.dt.date
    else:
        # fallback to daily
        d["bucket"] = d[date_col]
    return d


def page_ops():
    # Build initial option pools
    m = _hc_dim_df()
    opts_ba = sorted(m["Business Area"].astype(str).dropna().unique().tolist())
    opts_sba = sorted(m["Sub Business Area"].astype(str).dropna().unique().tolist())
    opts_loc = sorted(m["Location"].astype(str).dropna().unique().tolist())
    # Channels (fallback to default list)
    opts_ch = sorted([x for x in m["Channel"].astype(str).dropna().unique().tolist() if x]) or list(CHANNEL_LIST)
    # Site options minus any values equal to Location or known country synonyms
    raw_sites = sorted([x for x in m["Site"].astype(str).dropna().unique().tolist() if x]) if "Site" in m.columns else []
    loc_set0 = {str(x).strip().lower() for x in opts_loc}
    country_block = {"india", "uk", "united kingdom", "great britain", "england", "scotland", "wales"}
    opts_site = [
        s for s in raw_sites
        if str(s).strip()
        and (ls := str(s).strip().lower()) not in loc_set0
        and ls not in country_block
    ]

    start, end = _today_range(28)

    return html.Div(dbc.Container([
        header_bar(),
        dbc.Card(dbc.CardBody([
            html.H5("Operational Dashboard"),
            dbc.Row([
                dbc.Col(dcc.DatePickerRange(id="ops-dates", className="date-compact", start_date=start, end_date=end, display_format="YYYY-MM-DD"), md=2, className="jkl"),
                dbc.Col(dcc.Dropdown(id="ops-grain", options=[
                    {"label":"Interval","value":"interval"},
                    {"label":"Daily","value":"D"},{"label":"Weekly","value":"W"},
                    {"label":"Monthly","value":"M"},{"label":"Quarterly","value":"Q"},{"label":"Yearly","value":"Y"},
                ], value="D", clearable=False), md=2, className="jkl"),
                dbc.Col(dcc.Dropdown(id="ops-ba", options=[{"label":x, "value":x} for x in opts_ba], multi=True, placeholder="Business Area"), md=2, className="jkl"),
                dbc.Col(dcc.Dropdown(id="ops-sba", options=[{"label":x, "value":x} for x in opts_sba], multi=True, placeholder="Sub Business Area"), md=2, className="jkl"),
                dbc.Col(dcc.Dropdown(id="ops-ch", options=[{"label":x, "value":x} for x in opts_ch], multi=True, placeholder="Channel"), md=2, className="jkl"),
            ], className="g-2"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(id="ops-loc", options=[{"label":x, "value":x} for x in opts_loc], multi=True, placeholder="Location"), md=4),
                dbc.Col(dcc.Dropdown(id="ops-site", options=[{"label":x, "value":x} for x in opts_site], multi=True, placeholder="Site"), md=4),
                dbc.Col(dbc.Button("Reset Filters", id="ops-reset", color="secondary", outline=True, className="w-100"), md=4),
            ], className="g-2 mt-1"),
        ]), className="mb-3"),

        # KPI style summary (Req vs Supply)
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Div("Required FTE", className="kpi-head"),
                    html.Div(id="ops-kpi-req", className="display-6")
                ]), className="edge-teal"), md=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Div("Supply FTE", className="kpi-head"),
                    html.Div(id="ops-kpi-sup", className="display-6")
                ]), className="edge-blue"), md=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Div("Gap (Req - Sup)", className="kpi-head"),
                    html.Div(id="ops-kpi-gap", className="display-6")
                ]), className="edge-red"), md=3),
            ], className="g-3 mb-3"),
        ])),
        
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id="ops-line-req-sup", config={"displayModeBar": False}), className="loading-block"), md=6),
                dbc.Col(html.Div(dcc.Graph(id="ops-bar-volume", config={"displayModeBar": False}), className="loading-block"), md=6),
            ], className="g-3"),
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id="ops-pie-channel", config={"displayModeBar": False}), className="loading-block"), md=4),
                dbc.Col(html.Div(dcc.Graph(id="ops-col-site", config={"displayModeBar": False}), className="loading-block"), md=4),
                dbc.Col(html.Div(dcc.Graph(id="ops-waterfall-gap", config={"displayModeBar": False}), className="loading-block"), md=4),
            ], className="g-3 mb-3"),
        ])),
        

        dbc.Card(dbc.CardBody([
            html.H6("Summary Table"),
            html.Div(dash_table.DataTable(id="ops-table-summary", page_size=10, style_table={"overflowX":"auto"}, style_as_list_view=True), className="loading-block")
        ]))
    ], fluid=True), className="loading-page")


@app.callback(
    Output("ops-ba", "value"), Output("ops-sba", "value", allow_duplicate=True), Output("ops-ch", "value", allow_duplicate=True),
    Output("ops-site", "value", allow_duplicate=True), Output("ops-loc", "value"),
    Input("ops-reset", "n_clicks"), prevent_initial_call=True
)
def _reset_filters(_n):
    return [], [], [], [], []


@app.callback(
    Output("ops-sba", "options"), Output("ops-sba", "value"),
    Input("ops-ba", "value"),
    State("ops-sba", "value"),
    prevent_initial_call=False
)
def _dep_sba(ba_vals, sba_curr):
    df = _hc_dim_df().copy()
    if ba_vals:
        sel = {str(x).strip().lower() for x in (ba_vals if isinstance(ba_vals, list) else [ba_vals])}
        df = df[df["Business Area"].astype(str).str.strip().str.lower().isin(sel)]
    sba_list = sorted(df["Sub Business Area"].astype(str).dropna().unique().tolist())
    opts = [{"label": x, "value": x} for x in sba_list]
    curr = sba_curr if isinstance(sba_curr, list) else ([sba_curr] if sba_curr else [])
    new_val = [x for x in curr if x in sba_list]
    return opts, new_val


@app.callback(
    Output("ops-ch", "options"), Output("ops-ch", "value"),
    Input("ops-ba", "value"), Input("ops-sba", "value"),
    State("ops-ch", "value"),
    prevent_initial_call=False
)
def _dep_channel(ba_vals, sba_vals, ch_curr):
    m = _hc_dim_df(); df = m.copy()
    if ba_vals:
        sel_ba = {str(x).strip().lower() for x in (ba_vals if isinstance(ba_vals, list) else [ba_vals])}
        df = df[df["Business Area"].astype(str).str.strip().str.lower().isin(sel_ba)]
    if sba_vals:
        sel_sba = {str(x).strip().lower() for x in (sba_vals if isinstance(sba_vals, list) else [sba_vals])}
        df = df[df["Sub Business Area"].astype(str).str.strip().str.lower().isin(sel_sba)]
    try:
        from common import CHANNEL_LIST as _CHAN
    except Exception:
        _CHAN = ["Voice", "Back Office", "Outbound", "Blended", "Chat", "MessageUs"]
    ch_list = sorted(df["Channel"].astype(str).dropna().unique().tolist()) or list(_CHAN)
    opts = [{"label": x, "value": x} for x in ch_list]
    curr = ch_curr if isinstance(ch_curr, list) else ([ch_curr] if ch_curr else [])
    new_val = [x for x in curr if x in ch_list]
    return opts, new_val


@app.callback(
    Output("ops-site", "options"), Output("ops-site", "value"),
    Input("ops-ba", "value"), Input("ops-sba", "value"), Input("ops-ch", "value"),
    Input("ops-loc", "value"),
    State("ops-site", "value"),
    prevent_initial_call=False
)
def _dep_site(ba_vals, sba_vals, ch_vals, loc_vals, site_curr):
    df = _hc_dim_df().copy()
    # Apply hierarchical filters
    def _flt(col, vals):
        nonlocal df
        if vals:
            sel = {str(x).strip().lower() for x in (vals if isinstance(vals, list) else [vals])}
            df = df[df[col].astype(str).str.strip().str.lower().isin(sel)]
    _flt("Business Area", ba_vals)
    _flt("Sub Business Area", sba_vals)
    _flt("Channel", ch_vals)
    _flt("Location", loc_vals)

    # Build site list; guard against datasets where Site accidentally contains Location values
    site_list = sorted(df["Site"].astype(str).dropna().str.strip().unique().tolist()) if "Site" in df.columns else []
    loc_set = set(df["Location"].astype(str).str.strip().str.lower().unique().tolist()) if "Location" in df.columns else set()
    country_block = {"india", "uk", "united kingdom", "great britain", "england", "scotland", "wales"}
    site_list = [s for s in site_list if s and (sl := s.strip().lower()) not in loc_set and sl not in country_block]

    opts = [{"label": x, "value": x} for x in site_list]
    curr = site_curr if isinstance(site_curr, list) else ([site_curr] if site_curr else [])
    new_val = [x for x in curr if x in site_list]
    return opts, new_val


@app.callback(
    Output("ops-line-req-sup", "figure"),
    Output("ops-bar-volume", "figure"),
    Output("ops-pie-channel", "figure"),
    Output("ops-col-site", "figure"),
    Output("ops-waterfall-gap", "figure"),
    Output("ops-table-summary", "data"),
    Output("ops-table-summary", "columns"),
    Output("ops-kpi-req", "children"),
    Output("ops-kpi-sup", "children"),
    Output("ops-kpi-gap", "children"),
    Input("ops-dates", "start_date"), Input("ops-dates", "end_date"),
    Input("ops-grain", "value"),
    Input("ops-ba", "value"), Input("ops-sba", "value"), Input("ops-ch", "value"),
    Input("ops-site", "value"), Input("ops-loc", "value"),
    prevent_initial_call=False
)
def _refresh_ops(s, e, grain, ba, sba, ch, site, loc):
    try:
        s = s or _today_range(28)[0]
        e = e or _today_range(28)[1]
        start = pd.to_datetime(s).date(); end = pd.to_datetime(e).date()
    except Exception:
        start, end = date.today() - timedelta(days=28), date.today()

    map_df = _scope_keys_from_filters(ba, sba, ch, site, loc)
    scopes = map_df["sk"].unique().tolist()

    voice = _load_voice(scopes)
    bo = _load_bo(scopes)
    # Restrict by date
    if not voice.empty:
        voice["date"] = pd.to_datetime(voice["date"], errors="coerce").dt.date
        voice = voice[pd.notna(voice["date"])]
        voice = voice[(voice["date"] >= start) & (voice["date"] <= end)]
    if not bo.empty:
        bo["date"] = pd.to_datetime(bo["date"], errors="coerce").dt.date
        bo = bo[pd.notna(bo["date"])]
        bo = bo[(bo["date"] >= start) & (bo["date"] <= end)]

    # Settings (pick most specific single selection if available)
    ba_arg = (ba[0] if isinstance(ba, list) and ba else None)
    sba_arg = (sba[0] if isinstance(sba, list) and sba else None)
    ch_arg = (ch[0] if isinstance(ch, list) and ch else None)
    loc_arg = (loc[0] if isinstance(loc, list) and loc else None)
    settings = resolve_settings(location=loc_arg, ba=ba_arg, subba=sba_arg, lob=ch_arg)

    # Required FTE (daily)
    req_day = required_fte_daily(voice, bo, pd.DataFrame(), settings)
    if not req_day.empty:
        req_day = req_day.groupby("date", as_index=False)["total_req_fte"].sum()

    # Supply FTE (daily, from roster + hiring)
    roster = load_roster()
    if not isinstance(roster, pd.DataFrame):
        roster = pd.DataFrame()
    hiring = load_hiring()
    if not isinstance(hiring, pd.DataFrame):
        hiring = pd.DataFrame()
    # Try simple column-based filtering when possible
    def _maybe_filter(dff: pd.DataFrame) -> pd.DataFrame:
        if dff is None or dff.empty:
            return pd.DataFrame()
        dd = dff.copy()
        def f(col, vals):
            nonlocal dd
            if col in dd.columns and vals:
                sset = set([str(x) for x in (vals if isinstance(vals, list) else [vals])])
                dd = dd[dd[col].astype(str).isin(sset)]
        # common columns across uploads
        f("program", ba)
        f("Business Area", ba)
        f("Sub Business Area", sba)
        f("LOB", ch); f("Channel", ch)
        f("site", site); f("Site", site)
        f("location", loc); f("Location", loc); f("country", loc); f("Country", loc)
        return dd

    roster_f = _maybe_filter(roster)
    hiring_f = _maybe_filter(hiring)
    sd = supply_fte_daily(roster_f if isinstance(roster_f, pd.DataFrame) else pd.DataFrame(),
                          hiring_f if isinstance(hiring_f, pd.DataFrame) else pd.DataFrame())
    if not sd.empty:
        sd = sd.groupby("date", as_index=False)["supply_fte"].sum()

    # KPI values
    kpi_req = float(req_day["total_req_fte"].sum()) if isinstance(req_day, pd.DataFrame) and not req_day.empty else 0.0
    kpi_sup = float(sd["supply_fte"].sum()) if isinstance(sd, pd.DataFrame) and not sd.empty else 0.0
    kpi_gap = kpi_req - kpi_sup

    # Line chart (Req vs Supply by time)
    if grain == "interval" and not voice.empty and "interval" in voice.columns and voice["interval"].notna().any():
        # Interval view (voice agents required)
        vi = voice_requirements_interval(voice, settings)
        if not vi.empty:
            vi["ts"] = pd.to_datetime(vi["date"]).astype(str) + " " + vi["interval"].astype(str)
            fig_line = px.line(vi, x="ts", y="agents_req", color_discrete_sequence=["#2563eb"], title="Agents Required by Interval")
        else:
            fig_line = px.line(title="No interval data")
    else:
        # Aggregate by requested grain
        r = req_day.copy() if isinstance(req_day, pd.DataFrame) else pd.DataFrame(columns=["date","total_req_fte"])
        s_df = sd.copy() if isinstance(sd, pd.DataFrame) else pd.DataFrame(columns=["date","supply_fte"])
        if not r.empty:
            r = _agg_by_grain(r, "date", grain).groupby("bucket", as_index=False)["total_req_fte"].sum()
        if not s_df.empty:
            s_df = _agg_by_grain(s_df, "date", grain).groupby("bucket", as_index=False)["supply_fte"].sum()
        fig_line = px.line(title="Requirements vs Supply")
        if not r.empty:
            fig_line.add_scatter(x=r["bucket"], y=r["total_req_fte"], name="Required FTE")
        if not s_df.empty:
            fig_line.add_scatter(x=s_df["bucket"], y=s_df["supply_fte"], name="Supply FTE")

    # Bar: Volume/Items over time (stacked)
    bv = pd.DataFrame()
    if isinstance(voice, pd.DataFrame) and not voice.empty:
        v_day = voice.copy()
        if grain == "interval" and "interval" in v_day.columns and v_day["interval"].notna().any():
            v_day["bucket"] = v_day["date"].astype(str) + " " + v_day["interval"].astype(str)
        else:
            v_day = _agg_by_grain(v_day, "date", grain)
        v_agg = v_day.groupby("bucket", as_index=False)["volume"].sum().rename(columns={"volume":"Voice Calls"})
        bv = v_agg
    if isinstance(bo, pd.DataFrame) and not bo.empty:
        b_day = _agg_by_grain(bo.copy(), "date", grain)
        b_agg = b_day.groupby("bucket", as_index=False)["items"].sum().rename(columns={"items":"BO Items"})
        bv = b_agg if bv.empty else pd.merge(bv, b_agg, on="bucket", how="outer")
    fig_bar = px.bar(title="Workload by Time")
    if not bv.empty:
        bvm = bv.melt(id_vars=["bucket"], var_name="Metric", value_name="Value").fillna(0)
        fig_bar = px.bar(bvm, x="bucket", y="Value", color="Metric", barmode="stack")

    # Pie by Channel (share of volume/items)
    # Use scope_key attached to timeseries to infer channel
    def _attach_channel(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in is None or df_in.empty: return pd.DataFrame(columns=["ch","val"])
        d = df_in.copy()
        if "scope_key" in d.columns:
            d["ch"] = d["scope_key"].astype(str).str.split("|").str[2]
        else:
            d["ch"] = "All"
        return d
    vc = _attach_channel(voice)
    bc = _attach_channel(bo)
    pie_df = pd.DataFrame(columns=["ch","val"]) 
    if not vc.empty:
        pie_df = vc.groupby("ch", as_index=False)["volume"].sum().rename(columns={"volume":"val"})
    if not bc.empty:
        bo_p = bc.groupby("ch", as_index=False)["items"].sum().rename(columns={"items":"val"})
        pie_df = bo_p if pie_df.empty else pd.concat([pie_df, bo_p]).groupby("ch", as_index=False)["val"].sum()
    fig_pie = px.pie(values="val", names="ch", data_frame=pie_df, title="Workload Share by Channel") if not pie_df.empty else px.pie(title="No data")

    # Column chart by Site
    if not map_df.empty:
        # Map totals per scope to site using mapping
        key_map = map_df.drop_duplicates(subset=["sk","site"])[["sk","site"]]
        site_df = pd.DataFrame(columns=["site","val"])
        if "scope_key" in voice.columns:
            v_site = voice.groupby("scope_key", as_index=False)["volume"].sum().merge(key_map, left_on="scope_key", right_on="sk", how="left")
            v_site = v_site.groupby("site", as_index=False)["volume"].sum().rename(columns={"volume":"val"})
            site_df = v_site
        if "scope_key" in bo.columns:
            b_site = bo.groupby("scope_key", as_index=False)["items"].sum().merge(key_map, left_on="scope_key", right_on="sk", how="left")
            b_site = b_site.groupby("site", as_index=False)["items"].sum().rename(columns={"items":"val"})
            site_df = b_site if site_df.empty else pd.concat([site_df, b_site]).groupby("site", as_index=False)["val"].sum()
        fig_site = px.bar(site_df, x="site", y="val", title="Workload by Site") if not site_df.empty else px.bar(title="No site mapping")
    else:
        fig_site = px.bar(title="No mapping")

    # Waterfall: Required vs Supply vs Gap
    # Waterfall using graph_objects (px.waterfall not available in some Plotly versions)
    wf_df = pd.DataFrame({
        "label": ["Required", "Supply", "Gap"],
        "value": [kpi_req, -kpi_sup, 0.0],  # last bar computed as total
        "measure": ["relative", "relative", "total"],
    })
    try:
        fig_wf = go.Figure(go.Waterfall(
            x=wf_df["label"],
            y=wf_df["value"],
            measure=wf_df["measure"],
            connector={"line": {"color": "#8da2b1"}},
        ))
        fig_wf.update_layout(title="Requirement vs Supply (Waterfall)")
    except Exception:
        # Fallback to simple bar if Waterfall trace is unavailable
        fig_wf = px.bar(wf_df, x="label", y="value", title="Requirement vs Supply")

    # Summary table by BA/SBA/Channel/Site
    summary = []
    if not map_df.empty:
        # Aggregate by scope key then map to dims
        v_sum = voice.groupby("scope_key", as_index=False)["volume"].sum() if not voice.empty and "scope_key" in voice.columns else pd.DataFrame(columns=["scope_key","volume"])
        b_sum = bo.groupby("scope_key", as_index=False)["items"].sum() if not bo.empty and "scope_key" in bo.columns else pd.DataFrame(columns=["scope_key","items"])
        sk_map = map_df.drop_duplicates(subset=["sk","ba","sba","ch","site","loc"]).rename(columns={"sk":"scope_key"})
        merged = sk_map.merge(v_sum, on="scope_key", how="left").merge(b_sum, on="scope_key", how="left")
        merged["volume"] = merged["volume"].fillna(0)
        merged["items"] = merged["items"].fillna(0)
        tbl = merged.groupby(["ba","sba","ch","site","loc"], as_index=False)[["volume","items"]].sum()
        summary = tbl.to_dict("records")
        columns = [{"name": c, "id": c} for c in ["ba","sba","ch","site","loc","volume","items"]]
    else:
        columns = [{"name": c, "id": c} for c in ["ba","sba","ch","site","loc","volume","items"]]

    return (
        fig_line, fig_bar, fig_pie, fig_site, fig_wf,
        summary, columns,
        f"{kpi_req:.1f}", f"{kpi_sup:.1f}", f"{kpi_gap:.1f}"
    )



