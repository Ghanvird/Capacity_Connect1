from __future__ import annotations
from dash import html, dcc, dash_table, Output, Input, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from app_instance import app
from common import *  # noqa

def page_budget():
    return html.Div(dbc.Container([
        header_bar(),
        html.H4("Budgets", className="ghanii"),
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Business Area"),
                    dcc.Dropdown(id="bud-ba", options=[], placeholder="Select Business Area"),
                ], md=3),
                dbc.Col([
                    html.Label("Sub Business Area"),
                    dcc.Dropdown(id="bud-subba", options=[], placeholder="Select Sub Business Area"),
                ], md=3),
                dbc.Col([
                    html.Label("Channel"),
                    dcc.Dropdown(
                    id="bud-channel",
                    options=[{"label": c, "value": c} for c in CHANNEL_LIST],
                    value="Voice",
                    placeholder="Select Channel"
                    ),
                ], md=3),
                dbc.Col([
                    html.Label("Site"),
                    dcc.Dropdown(id="bud-site", options=[], placeholder="Select Site"),
                ], md=3),
                ], className="gy-2 my-1"),

            dbc.Tabs(id="bud-tabs", active_tab="bud-voice", children=[

                # VOICE
                dbc.Tab(label="Voice budget", tab_id="bud-voice", children=[
                    dcc.Store(id="store-bud-voice"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Start week (Monday)"),
                            dcc.DatePickerSingle(id="bud-voice-start", display_format="YYYY-MM-DD")
                        ], md=3),
                        dbc.Col([
                            html.Label("Weeks"),
                            dbc.Input(id="bud-voice-weeks", type="number", value=8, min=1, step=1)
                        ], md=2),
                        dbc.Col(dbc.Button("Download Template", id="btn-bud-voice-tmpl", outline=True, className="mt-4"), md=3),
                        dbc.Col(dcc.Upload(id="up-bud-voice", children=html.Div(["⬆️ Upload CSV/XLSX"]), multiple=False,
                                        className="upload-box mt-4"), md=4),
                    ], className="my-2"),
                    dcc.Download(id="dl-bud-voice"),
                    html.Div(dash_table.DataTable(
                        id="tbl-bud-voice",
                        page_size=10, editable=True, style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ), className="loading-block"),
                    dbc.Row([
                        dbc.Col(dbc.Button("Save Voice Budget", id="btn-save-bud-voice", color="primary"), md=3),
                        dbc.Col(html.Div(html.Span(id="msg-save-bud-voice", className="text-success"), className="loading-block"), md=9),
                    ], className="mt-2"),
                ]),

                # BACK OFFICE
                dbc.Tab(label="Back Office budget", tab_id="bud-bo", children=[
                    dcc.Store(id="store-bud-bo"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Start week (Monday)"),
                            dcc.DatePickerSingle(id="bud-bo-start", display_format="YYYY-MM-DD")
                        ], md=3),
                        dbc.Col([
                            html.Label("Weeks"),
                            dbc.Input(id="bud-bo-weeks", type="number", value=8, min=1, step=1)
                        ], md=2),
                        dbc.Col(dbc.Button("Download Template", id="btn-bud-bo-tmpl", outline=True, className="mt-4"), md=3),
                        dbc.Col(dcc.Upload(id="up-bud-bo", children=html.Div(["⬆️ Upload CSV/XLSX"]), multiple=False,
                                        className="upload-box mt-4"), md=4),
                    ], className="my-2"),
                    dcc.Download(id="dl-bud-bo"),
                    html.Div(dash_table.DataTable(
                        id="tbl-bud-bo",
                        page_size=10, editable=True, style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ), className="loading-block"),
                    dbc.Row([
                        dbc.Col(dbc.Button("Save Back Office Budget", id="btn-save-bud-bo", color="primary"), md=3),
                        dbc.Col(html.Div(html.Span(id="msg-save-bud-bo", className="text-success"), className="loading-block"), md=9),
                    ], className="mt-2"),
                ]),

                # CHAT
                dbc.Tab(label="Chat budget", tab_id="bud-chat", children=[
                    dcc.Store(id="store-bud-chat"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Start week (Monday)"),
                            dcc.DatePickerSingle(id="bud-chat-start", display_format="YYYY-MM-DD")
                        ], md=3),
                        dbc.Col([
                            html.Label("Weeks"),
                            dbc.Input(id="bud-chat-weeks", type="number", value=8, min=1, step=1)
                        ], md=2),
                        dbc.Col(dbc.Button("Download Template", id="btn-bud-chat-tmpl", outline=True, className="mt-4"), md=3),
                        dbc.Col(dcc.Upload(id="up-bud-chat", children=html.Div(["⬆️ Upload CSV/XLSX"]), multiple=False,
                                        className="upload-box mt-4"), md=4),
                    ], className="my-2"),
                    dcc.Download(id="dl-bud-chat"),
                    html.Div(dash_table.DataTable(
                        id="tbl-bud-chat",
                        page_size=10, editable=True, style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ), className="loading-block"),
                    dbc.Row([
                        dbc.Col(dbc.Button("Save Chat Budget", id="btn-save-bud-chat", color="primary"), md=3),
                        dbc.Col(html.Div(html.Span(id="msg-save-bud-chat", className="text-success"), className="loading-block"), md=9),
                    ], className="mt-2"),
                ]),

                # OUTBOUND
                dbc.Tab(label="Outbound budget", tab_id="bud-ob", children=[
                    dcc.Store(id="store-bud-ob"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Start week (Monday)"),
                            dcc.DatePickerSingle(id="bud-ob-start", display_format="YYYY-MM-DD")
                        ], md=3),
                        dbc.Col([
                            html.Label("Weeks"),
                            dbc.Input(id="bud-ob-weeks", type="number", value=8, min=1, step=1)
                        ], md=2),
                        dbc.Col(dbc.Button("Download Template", id="btn-bud-ob-tmpl", outline=True, className="mt-4"), md=3),
                        dbc.Col(dcc.Upload(id="up-bud-ob", children=html.Div(["⬆️ Upload CSV/XLSX"]), multiple=False,
                                        className="upload-box mt-4"), md=4),
                    ], className="my-2"),
                    dcc.Download(id="dl-bud-ob"),
                    html.Div(dash_table.DataTable(
                        id="tbl-bud-ob",
                        page_size=10, editable=True, style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ), className="loading-block"),
                    dbc.Row([
                        dbc.Col(dbc.Button("Save Outbound Budget", id="btn-save-bud-ob", color="primary"), md=3),
                        dbc.Col(html.Div(html.Span(id="msg-save-bud-ob", className="text-success"), className="loading-block"), md=9),
                    ], className="mt-2"),
                ]),
            ])
        ])),        
    ], fluid=True), className="loading-page")



