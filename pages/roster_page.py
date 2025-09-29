from __future__ import annotations
from dash import html, dcc, dash_table, Output, Input, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from app_instance import app
from common import *  # noqa

def page_roster():
    df_wide_db = load_roster_wide()
    if df_wide_db is None or df_wide_db.empty:
        df_wide_db = pd.DataFrame()
    df_long_db = load_roster_long()
    if df_long_db is None or df_long_db.empty:
        df_long_db = pd.DataFrame()

    return html.Div(dbc.Container([
        header_bar(),

        dcc.Store(id="roster_wide_store", data=(df_wide_db.to_dict("records") if not df_wide_db.empty else [])),
        dcc.Store(id="roster_long_store", data=(df_long_db.to_dict("records") if not df_long_db.empty else [])),
        dcc.Store(id="roster_pid"),

        dbc.Card(dbc.CardBody([
            html.H5("Download Roster Template"),
            dbc.Row([
                dbc.Col(dcc.DatePickerRange(
                    id="roster-template-dates",
                    className="date-compact",
                    display_format="YYYY-MM-DD"
                ), md=4),
                dbc.Col(dbc.Button("Download Empty Template (CSV)", id="btn-dl-roster-template", color="secondary", className="w-100"), md=4),
                dbc.Col(dbc.Button("Download Sample", id="btn-dl-roster-sample", outline=True, color="secondary", className="w-100"), md=4),
            ], className="my-2"),
            dcc.Download(id="dl-roster-template"),
            dcc.Download(id="dl-roster-sample"),
        ]), className="mb-3"),

        dbc.Card(dbc.CardBody([
            html.H5("Upload Filled Roster"),
            html.Div(id="roster-plan-msg", className="text-muted small"),
            dbc.Row([
                dbc.Col(dcc.Upload(
                    id="up-roster-wide",
                    children=html.Div(["⬆️ Drag & drop or click to upload CSV"]),
                    multiple=False, className="upload-box"
                ), md=6),
                dbc.Col(dbc.Button("Save", id="btn-save-roster-wide", color="primary"), md=2),
                dbc.Col(html.Div(html.Span(id="roster-save-msg", className="text-success"), className="loading-block"), md=4),
            ], className="my-2"),
            html.Div(dash_table.DataTable(
                id="tbl-roster-wide",
                data=df_wide_db.to_dict("records"),
                columns=pretty_columns(df_wide_db if not df_wide_db.empty else
                    ["BRID","Name","Team Manager","Business Area","Sub Business Area","LOB","Site","Location","Country"]),
                editable=True, row_deletable=True, page_size=10,
                style_table={"overflowX":"auto"},
                style_as_list_view=True,
                style_header={"textTransform": "none"},
            ), className="loading-block"),
            html.Div(html.Div(id="roster-wide-msg", className="text-muted mt-1"), className="loading-block"),
        ]), className="mb-3"),

        dbc.Card(dbc.CardBody([
            html.H5("Normalized Schedule Preview"),
            dbc.Row([
                dbc.Col(dcc.DatePickerRange(
                    id="roster-preview-dates",
                    className="date-compact",
                    display_format="YYYY-MM-DD"
                ), md=4),
            ], className="my-2"),
            html.Div(dash_table.DataTable(
                id="tbl-roster-long",
                data=(df_long_db.to_dict("records") if not df_long_db.empty else []),
                columns=pretty_columns(df_long_db if not df_long_db.empty else
                    ["BRID","Name","Team Manager","Business Area","Sub Business Area","LOB","Site","Location","Country","date","entry","is_leave"]),
                page_size=12,
                style_table={"overflowX":"auto"},
                style_as_list_view=True,
                style_header={"textTransform": "none"},
            ), className="loading-block"),
        ]), className="mb-3"),

        dbc.Card(dbc.CardBody([
            html.H6("Bulk edit helpers"),
            dbc.Row([
                dbc.Col(dcc.DatePickerRange(
                    id="clear-range",
                    className="date-compact",
                    display_format="YYYY-MM-DD"
                ), md=5),
                dbc.Col(dcc.Dropdown(
                    id="clear-brids",
                    multi=True,
                    placeholder="Limit to BRIDs (optional)"
                ), md=4),
                dbc.Col(dcc.RadioItems(
                    id="clear-action",
                    options=[{"label": " Clear", "value": "blank"},
                             {"label": " Leave", "value": "Leave"},
                             {"label": " OFF", "value": "OFF"}],
                    value="blank",
                ), md=3),
            ], className="g-2"),
            dbc.Button("Apply to range", id="btn-apply-clear", color="danger", outline=True, className="mt-2"),
            html.Div(html.Div(id="bulk-clear-msg", className="text-muted small mt-2"), className="loading-block")
        ]), className="mb-3"),

    ], fluid=True), className="loading-page")



