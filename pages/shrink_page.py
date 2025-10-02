from __future__ import annotations
from dash import html, dcc, dash_table, Output, Input, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from app_instance import app
from common import *  # noqa

def page_shrink_attr():
    shr = normalize_shrink_weekly(load_shrinkage())
    att = load_attrition()
    return html.Div(dbc.Container([
        header_bar(),
        dcc.Interval(id="shr-msg-timer", interval=5000, n_intervals=0, disabled=True),
        dbc.Tabs(id="shr-tabs", active_tab="tab-shrink", children=[
            dbc.Tab(label="Shrinkage", tab_id="tab-shrink", children=[
                # inner tabs: Manual (existing), Back Office (Raw), Voice (Raw)
                dbc.Tabs(id="shr-inner-tabs", active_tab="inner-manual", children=[
                    dbc.Tab(label="Weekly Shrink %", tab_id="inner-manual", children=[
                        html.Div(dash_table.DataTable(
                            id="tbl-shrink",
                            columns=shrink_weekly_columns(),
                            data=shr.to_dict("records"),
                            editable=True, row_deletable=True, page_size=10,
                            style_table={"overflowX":"auto"}, style_as_list_view=True,
                            style_header={"textTransform":"none"},
                        ), className="loading-block"),
                        html.Div(dcc.Graph(id="fig-shrink", style={"height":"280px"}, config={"displayModeBar": False}), className="loading-block"),
                    ]),

                    dbc.Tab(label="Voice Shrinkage (Raw)", tab_id="inner-voice", children=[
                        dcc.Store(id="voice-shr-raw-store"),
                        dbc.Row([
                            dbc.Col(dcc.Upload(id="up-shr-voice-raw",
                                               children=html.Div(["⬆️ Upload Voice Shrinkage (HH:MM) XLSX/CSV"]),
                                               multiple=False, className="upload-box"), md=6),
                            dbc.Col(dbc.Button("Save Voice Shrinkage", id="btn-save-shr-voice-raw", color="primary", className="w-100"), md=3),
                            dbc.Col(dbc.Button("Download Alvaria Template", id="btn-dl-shr-voice-template", outline=True, color="secondary", className="w-100"), md=3),
                        ], className="my-2"),
                        dcc.Download(id="dl-shr-voice-template"),
                        html.H6("Uploaded (normalized)"),
                        html.Div(dash_table.DataTable(id="tbl-shr-voice-raw", page_size=8, style_table={"overflowX":"auto"},
                                             style_as_list_view=True, style_header={"textTransform":"none"}), className="loading-block"),
                        html.H6("Daily Summary (derived)"),
                        html.Div(dash_table.DataTable(id="tbl-shr-voice-sum", page_size=8, style_table={"overflowX":"auto"},
                                             style_as_list_view=True, style_header={"textTransform":"none"}), className="loading-block"),
                        html.Div(html.Div(id="voice-shr-save-msg", className="text-success mt-2"), className="loading-block"),
                    ]),

                    dbc.Tab(label="Back Office Shrinkage (Raw)", tab_id="inner-bo", children=[
                        dcc.Store(id="bo-shr-raw-store"),
                        dbc.Row([
                            dbc.Col(dcc.Upload(id="up-shr-bo-raw",
                                               children=html.Div(["⬆️ Upload Back Office Shrinkage (Voice HH:MM or BO seconds) XLSX/CSV"]),
                                               multiple=False, className="upload-box"), md=6),
                            dbc.Col(dbc.Button("Save Back Office Shrinkage", id="btn-save-shr-bo-raw", color="primary", className="w-100"), md=3),
                            dbc.Col(html.Div([
                                dbc.Button("Download Alvaria Template", id="btn-dl-shr-bo-voice-template", outline=True, color="secondary", className="w-100"),
                                dbc.Button("Download Control IQ Template", id="btn-dl-shr-bo-template", outline=True, color="secondary", className="w-100 mt-2"),
                            ]), md=3),
                        ], className="my-2"),
                        dcc.Download(id="dl-shr-bo-voice-template"),
                        dcc.Download(id="dl-shr-bo-template"),
                        html.H6("Uploaded (normalized)"),
                        html.Div(dash_table.DataTable(id="tbl-shr-bo-raw", page_size=8, style_table={"overflowX":"auto"},
                                             style_as_list_view=True, style_header={"textTransform":"none"}), className="loading-block"),
                        html.H6("Daily Summary (derived)"),
                        html.Div(dash_table.DataTable(id="tbl-shr-bo-sum", page_size=8, style_table={"overflowX":"auto"},
                                             style_as_list_view=True, style_header={"textTransform":"none"}), className="loading-block"),
                        html.Div(html.Div(id="bo-shr-save-msg", className="text-success mt-2"), className="loading-block"),
                    ]),

                    dbc.Tab(label="Chat Shrinkage (Raw)", tab_id="inner-chat", children=[
                        dcc.Store(id="chat-shr-raw-store"),
                        dbc.Row([
                            dbc.Col(dcc.Upload(id="up-shr-chat-raw",
                                               children=html.Div(["⬆️ Upload Chat Shrinkage (Voice HH:MM or BO seconds) XLSX/CSV"]),
                                               multiple=False, className="upload-box"), md=6),
                            dbc.Col(dbc.Button("Save Chat Shrinkage", id="btn-save-shr-chat-raw", color="primary", className="w-100"), md=3),
                            dbc.Col(html.Div([
                                dbc.Button("Download Alvaria Template", id="btn-dl-shr-chat-voice-template", outline=True, color="secondary", className="w-100"),
                                dbc.Button("Download Control IQ Template", id="btn-dl-shr-chat-bo-template", outline=True, color="secondary", className="w-100 mt-2"),
                            ]), md=3),
                        ], className="my-2"),
                        dcc.Download(id="dl-shr-chat-voice-template"),
                        dcc.Download(id="dl-shr-chat-bo-template"),
                        html.H6("Uploaded (normalized)"),
                        html.Div(dash_table.DataTable(id="tbl-shr-chat-raw", page_size=8, style_table={"overflowX":"auto"},
                                             style_as_list_view=True, style_header={"textTransform":"none"}), className="loading-block"),
                        html.H6("Daily Summary (derived)"),
                        html.Div(dash_table.DataTable(id="tbl-shr-chat-sum", page_size=8, style_table={"overflowX":"auto"},
                                             style_as_list_view=True, style_header={"textTransform":"none"}), className="loading-block"),
                        html.Div(html.Div(id="chat-shr-save-msg", className="text-success mt-2"), className="loading-block"),
                    ]),

                    dbc.Tab(label="Outbound Shrinkage (Raw)", tab_id="inner-ob", children=[
                        dcc.Store(id="ob-shr-raw-store"),
                        dbc.Row([
                            dbc.Col(dcc.Upload(id="up-shr-ob-raw",
                                               children=html.Div(["⬆️ Upload Outbound Shrinkage (Voice HH:MM or BO seconds) XLSX/CSV"]),
                                               multiple=False, className="upload-box"), md=6),
                            dbc.Col(dbc.Button("Save Outbound Shrinkage", id="btn-save-shr-ob-raw", color="primary", className="w-100"), md=3),
                            dbc.Col(html.Div([
                                dbc.Button("Download Alvaria Template", id="btn-dl-shr-ob-voice-template", outline=True, color="secondary", className="w-100"),
                                dbc.Button("Download Control IQ Template", id="btn-dl-shr-ob-bo-template", outline=True, color="secondary", className="w-100 mt-2"),
                            ]), md=3),
                        ], className="my-2"),
                        dcc.Download(id="dl-shr-ob-voice-template"),
                        dcc.Download(id="dl-shr-ob-bo-template"),
                        html.H6("Uploaded (normalized)"),
                        html.Div(dash_table.DataTable(id="tbl-shr-ob-raw", page_size=8, style_table={"overflowX":"auto"},
                                             style_as_list_view=True, style_header={"textTransform":"none"}), className="loading-block"),
                        html.H6("Daily Summary (derived)"),
                        html.Div(dash_table.DataTable(id="tbl-shr-ob-sum", page_size=8, style_table={"overflowX":"auto"},
                                             style_as_list_view=True, style_header={"textTransform":"none"}), className="loading-block"),
                        html.Div(html.Div(id="ob-shr-save-msg", className="text-success mt-2"), className="loading-block"),
                    ]),
                ])
            ]),

            # Removed legacy placeholders (no longer needed)

            dbc.Tab(label="Attrition", tab_id="tab-attr", children=[
                dbc.Row([
                    dbc.Col(dcc.Upload(id="up-attr", children=html.Div(["⬆️ Drag & drop or click to upload XLSX"]), multiple=False, className="upload-box"), md=4),
                    dbc.Col(dbc.Button("Save", id="btn-save-attr", color="primary"), md=2),
                    dbc.Col(html.Div(html.Span(id="attr-save-msg", className="text-success"), className="loading-block"), md=3),
                    dbc.Col(dbc.Button("Download Leavers Sample", id="btn-dl-attr", outline=True, color="secondary"), md=3),
                ], className="my-2"),
                dcc.Download(id="dl-attr-sample"),
                dcc.Store(id="attr_raw_store"),
                html.Div(dash_table.DataTable(
                    id="tbl-attr-shrink",
                    columns=pretty_columns(att if not att.empty else ["week","attrition_pct","program"]),
                    data=att.to_dict("records"),
                    editable=True, row_deletable=True, page_size=10,
                    style_table={"overflowX":"auto"}, style_as_list_view=True,
                    style_header={"textTransform":"none"},
                ), className="loading-block"),
                html.Div(dcc.Graph(id="fig-attr", style={"height":"280px"}, config={"displayModeBar": False}), className="loading-block"),
            ]),
        ])
    ], fluid=True), className="loading-page")
