from __future__ import annotations
from dash import html, dcc, dash_table, Output, Input, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from app_instance import app
from common import *  # noqa

def page_default_settings():
    base = load_defaults() or DEFAULT_SETTINGS
    return html.Div(dbc.Container([
        header_bar(),

        dbc.Card(dbc.CardBody([
            html.H5("Default Settings - Scope"),

            # Scope picker
            dbc.Row([
                dbc.Col(dbc.RadioItems(
                    id="set-scope",
                    options=[
                        {"label": " Global", "value": "global"},
                        {"label": " Location (Country)", "value": "location"},
                        {"label": " Business Area ▶ Sub Business Area ▶ Channel ▶ Site", "value": "hier"},
                    ],
                    value="global",
                    style={"display": "flex", "gap": "10px"}
                ), md=12),
            ], className="mb-2"),

            # Location-only scope row  (uses Position Location Country)
            dbc.Row([
                dbc.Col(dbc.Label("Location / Country"), md=2),
                dbc.Col(dcc.Dropdown(id="set-location", placeholder="Select Country (from headcount)"), md=4),
                dbc.Col(html.Div(id="set-location-hint", className="text-muted small"), md=6),
            ], id="row-location", className="mb-1", style={"display": "none"}),

            # Hierarchical scope row (BA > SubBA > Channel > Site)
            dbc.Row([
                dbc.Col(dbc.Label("Business Area"), md=2),
                dbc.Col(dcc.Dropdown(id="set-ba",        placeholder="Business Area"),         md=3),
                dbc.Col(dcc.Dropdown(id="set-subba",     placeholder="Sub Business Area"),     md=3),
                dbc.Col(dcc.Dropdown(id="set-lob",       placeholder="Channel"),               md=2),
                dbc.Col(dcc.Dropdown(id="set-site-hier", placeholder="Site (Building)"),       md=2),
            ], id="row-hier", className="mb-1", style={"display":"none"}),

            html.Div(id="settings-scope-note-hint", className="text-muted small mt-2"),
        ]), className="mb-3"),


        dbc.Card(
            dbc.CardBody([
                html.H5("Parameters"),
                dbc.Row([
                    # --- Col 1 ----------------------------------------------------------
                    dbc.Col([
                        dbc.Label("Interval Minutes (Voice)"),
                        dbc.Input(
                            id="set-interval", type="number", min=5, max=120, step=5,
                            value=int(base.get("interval_minutes", 30))
                        ),

                        dbc.Label("Work Hours per FTE / Day", className="mt-3"),
                        dbc.Input(
                            id="set-hours", type="number", min=1, max=12, step=0.25,
                            value=float(base.get("hours_per_fte", 8))
                        ),

                        dbc.Label("Shrinkage % (0-100)", className="mt-3"),
                        dbc.Input(
                            id="set-shrink", type="number", min=0, max=100, step=0.5,
                            value=float(100 * float(base.get("shrinkage_pct", 0)))
                        ),
                    ], md=6),

                    # --- Col 2 (Voice targets) -----------------------------------------
                    dbc.Col([
                        html.Strong("Voice Targets - "),
                        dbc.Label("Service Level % (0-100)", className="mt-2"),
                        dbc.Input(
                            id="set-sl", type="number", min=50, max=99, step=1,
                            value=int(100 * float(base.get("target_sl", 0.80)))
                        ),

                        dbc.Label("Service Level T (seconds)", className="mt-3"),
                        dbc.Input(
                            id="set-slsec", type="number", min=1, max=120, step=1,
                            value=int(base.get("sl_seconds", 20))
                        ),

                        dbc.Label("Max Occupancy % (Voice)", className="mt-3"),
                        dbc.Input(
                            id="set-occ", type="number", min=60, max=100, step=1,
                            value=int(100 * float(base.get("occupancy_cap_voice", 0.85)))
                        ),
                    ], md=6),
                        html.Hr(className="my-3"),
                    # --- Col 3 (BO + Productivity) -------------------------------------
                    dbc.Col([
                        html.Strong("Back Office (TAT / Capacity) - "),

                        dbc.Label("Capacity Model", className="mt-2"),
                        dcc.Dropdown(
                            id="set-bo-model",
                            options=[{"label": "TAT (Within X days)", "value": "tat"},
                                    {"label": "Erlang (queueing)",     "value": "erlang"}],
                            value=(base.get("bo_capacity_model") or "tat").lower(),
                            clearable=False
                        ),

                        dbc.Label("TAT (days)", className="mt-3"),
                        dbc.Input(
                            id="set-bo-tat", type="number", min=1, max=30, step=1,
                            value=float(base.get("bo_tat_days", 5))
                        ),

                        dbc.Label("Workdays per Week", className="mt-3"),
                        dbc.Input(
                            id="set-bo-wd", type="number", min=1, max=7, step=1,
                            value=int(base.get("bo_workdays_per_week", 5))
                        ),
                        dbc.Label("Work Hours / Day (BO)", className="mt-3"),
                        dbc.Input(
                            id="set-bo-hpd", type="number", min=1, max=12, step=0.25,
                            value=float(base.get("bo_hours_per_day", base.get("hours_per_fte", 8)))
                        ),
                        dbc.Label("Chat Concurrency", className="mt-3"),
                        dbc.Input(
                            id="set-chatcc", type="number", min=1, max=10, step=0.1,
                            value=float(base.get("chat_concurrency", 1.5))
                        ),
                        dbc.Label("Utilization % (Chat)", className="mt-3"),
                        dbc.Input(
                            id="set-utilchat", type="number", min=50, max=100, step=1,
                            value=int(100 * float(base.get("util_chat", base.get("util_bo", 0.85))))
                        ),
                    ], md=6),
                    dbc.Col([
                        dbc.Label("BO Shrinkage % (0-100)", className="mt-3"),
                        dbc.Input(
                            id="set-bo-shrink", type="number", min=0, max=100, step=0.5,
                            value=float(100 * float(base.get("bo_shrinkage_pct", base.get("shrinkage_pct", 0))))
                        ),
                        html.Strong("Productivity - "),
                        dbc.Label("Occupancy % (Back Office)", className="mt-2"),
                        dbc.Input(
                            id="set-utilbo", type="number", min=50, max=100, step=1,
                            value=int(100 * float(base.get("util_bo", 0.85)))
                        ),
                        dbc.Label("Utilization % (Outbound)", className="mt-3"),
                        dbc.Input(
                            id="set-utilob", type="number", min=50, max=100, step=1,
                            value=int(100 * float(base.get("util_ob", 0.85)))
                        ),

                        dbc.Label("Shrinkage % (Chat) (0-100)", className="mt-3"),
                        dbc.Input(
                            id="set-shrink-chat", type="number", min=0, max=100, step=0.5,
                            value=float(100 * float(base.get("chat_shrinkage_pct", base.get("shrinkage_pct", 0))))
                        ),

                        dbc.Label("Shrinkage % (Outbound) (0-100)", className="mt-3"),
                        dbc.Input(
                            id="set-shrink-ob", type="number", min=0, max=100, step=0.5,
                            value=float(100 * float(base.get("ob_shrinkage_pct", base.get("shrinkage_pct", 0))))
                        ),
                        
                        html.Div(id="settings-scope-note", className="text-muted small mt-3"),
                    ], md=6),
                ], className="g-3"),

                html.Hr(className="my-3"),
                html.H5("Learning Curve & SDA"),
                dbc.Row([
                    dbc.Col([
                        html.Strong("Nesting (OJT)"),
                        dbc.Label("Weeks", className="mt-2"),
                        dbc.Input(id="set-nest-weeks", type="number", min=0, step=1,
                                  value=int(base.get("nesting_weeks", base.get("default_nesting_weeks", 0)))),
                        html.Div(id="nest-login-container", className="mt-2"),
                        html.Div(id="nest-aht-container", className="mt-2"),
                    ], md=6),
                    dbc.Col([
                        html.Strong("SDA (Phase 2)"),
                        dbc.Label("Weeks", className="mt-2"),
                        dbc.Input(id="set-sda-weeks", type="number", min=0, step=1,
                                  value=int(base.get("sda_weeks", base.get("default_sda_weeks", 0)))),
                        html.Div(id="sda-login-container", className="mt-2"),
                        html.Div(id="sda-aht-container", className="mt-2"),
                    ], md=6),
                ], className="g-3"),

                dbc.Row([
                    dbc.Col(dbc.InputGroup([
                        dbc.InputGroupText("Throughput Train %"),
                        dbc.Input(id="set-throughput-train", type="number", step=1,
                                  value=float(str(base.get("throughput_train_pct", 100)).replace('%','')))
                    ]), md=6),
                    dbc.Col(dbc.InputGroup([
                        dbc.InputGroupText("Throughput Nest %"),
                        dbc.Input(id="set-throughput-nest", type="number", step=1,
                                  value=float(str(base.get("throughput_nest_pct", 100)).replace('%','')))
                    ]), md=6),
                ], className="g-3 mt-2"),

                html.Div([
                    dbc.Button("Save Settings", id="btn-save-settings", color="primary", className="me-2"),
                    html.Span(id="settings-save-msg", className="text-success ms-2")
                ], className="mt-3"),
            ])
        ),


        # ====== CLUBBED TABS: Voice & Back Office only ======
        dbc.Card(dbc.CardBody([
            html.H5("Upload Volume & AHT/SUT (by scope)"),
            html.Div("Voice uses 30-min intervals; Back Office uses daily totals.", className="text-muted small mb-2"),
            dbc.Alert("Alert:- Uploads are saved to the selected scope. Even if your file includes Business Area, Sub Business Area, and Channel, please choose the scope above first.",color="light"),
            dbc.Tabs(id="vol-tabs", children=[

                # ---------- Voice (Forecast + Actual) ----------
                dcc.Tab(label="Voice", value="tab-voice", children=[
                    html.H6("Forecast"),
                    dbc.Row([
                        dbc.Col(dcc.Upload(
                            id="up-voice-forecast",
                            children=html.Div(["⬆️ Upload Voice Forecast (XLSX/CSV)"]),
                            multiple=False, className="upload-box"
                        ), md=6),
                        dbc.Col(dbc.Button("Save Voice Forecast", id="btn-save-voice-forecast", color="primary", className="w-100"), md=3),
                        dbc.Col(dbc.Button("Download Template", id="btn-dl-voice-forecast-tmpl", outline=True, color="secondary", className="w-100"), md=3),
                    ], className="my-2"),
                    dcc.Download(id="dl-voice-forecast-tmpl"),
                    html.Div(dash_table.DataTable(id="tbl-voice-forecast", page_size=10,
                        style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ), className="loading-block"),
                    html.Div(html.Div(id="voice-forecast-msg", className="text-success mt-1"), className="loading-block"),

                    html.Hr(),
                    html.H6("Actual"),
                    dbc.Row([
                        dbc.Col(dcc.Upload(
                            id="up-voice-actual",
                            children=html.Div(["⬆️ Upload Voice Actual (XLSX/CSV)"]),
                            multiple=False, className="upload-box"
                        ), md=6),
                        dbc.Col(dbc.Button("Save Voice Actual", id="btn-save-voice-actual", color="primary", className="w-100"), md=3),
                        dbc.Col(dbc.Button("Download Template", id="btn-dl-voice-actual-tmpl", outline=True, color="secondary", className="w-100"), md=3),
                    ], className="my-2"),
                    dcc.Download(id="dl-voice-actual-tmpl"),
                    html.Div(dash_table.DataTable(id="tbl-voice-actual", page_size=10,
                        style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ), className="loading-block"),
                    html.Div(html.Div(id="voice-actual-msg", className="text-success mt-1"), className="loading-block"),
                ]),

                # ---------- Back Office (Forecast + Actual) ----------
                dcc.Tab(label="Back Office", value="tab-bo", children=[
                    html.H6("Forecast"),
                    dbc.Row([
                        dbc.Col(dcc.Upload(
                            id="up-bo-forecast",
                            children=html.Div(["⬆️ Upload Back Office Forecast (XLSX/CSV)"]),
                            multiple=False, className="upload-box"
                        ), md=6),
                        dbc.Col(dbc.Button("Save BO Forecast", id="btn-save-bo-forecast", color="primary", className="w-100"), md=3),
                        dbc.Col(dbc.Button("Download Template", id="btn-dl-bo-forecast-tmpl", outline=True, color="secondary", className="w-100"), md=3),
                    ], className="my-2"),
                    dcc.Download(id="dl-bo-forecast-tmpl"),
                    html.Div(dash_table.DataTable(id="tbl-bo-forecast", page_size=10,
                        style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ), className="loading-block"),
                    html.Div(html.Div(id="bo-forecast-msg", className="text-success mt-1"), className="loading-block"),

                    html.Hr(),
                    html.H6("Actual"),
                    dbc.Row([
                        dbc.Col(dcc.Upload(
                            id="up-bo-actual",
                            children=html.Div(["⬆️ Upload Back Office Actual (XLSX/CSV)"]),
                            multiple=False, className="upload-box"
                        ), md=6),
                        dbc.Col(dbc.Button("Save BO Actual", id="btn-save-bo-actual", color="primary", className="w-100"), md=3),
                        dbc.Col(dbc.Button("Download Template", id="btn-dl-bo-actual-tmpl", outline=True, color="secondary", className="w-100"), md=3),
                    ], className="my-2"),
                    dcc.Download(id="dl-bo-actual-tmpl"),
                    html.Div(dash_table.DataTable(id="tbl-bo-actual", page_size=10,
                        style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ), className="loading-block"),
                    html.Div(html.Div(id="bo-actual-msg", className="text-success mt-1"), className="loading-block"),
                ]),

                # ---------- Chat (Forecast + Actual) ----------
                dcc.Tab(label="Chat", value="tab-chat", children=[
                    html.H6("Forecast"),
                    dbc.Row([
                        dbc.Col(dcc.Upload(
                            id="up-chat-forecast",
                            children=html.Div(["⬆️ Upload Chat Forecast (XLSX/CSV)"]),
                            multiple=False, className="upload-box"
                        ), md=6),
                        dbc.Col(dbc.Button("Save Chat Forecast", id="btn-save-chat-forecast", color="primary", className="w-100"), md=3),
                        dbc.Col(dbc.Button("Download Template", id="btn-dl-chat-forecast-tmpl", outline=True, color="secondary", className="w-100"), md=3),
                    ], className="my-2"),
                    dcc.Download(id="dl-chat-forecast-tmpl"),
                    html.Div(dash_table.DataTable(id="tbl-chat-forecast", page_size=10,
                        style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ), className="loading-block"),
                    html.Div(html.Div(id="chat-forecast-msg", className="text-success mt-1"), className="loading-block"),

                    html.Hr(),
                    html.H6("Actual"),
                    dbc.Row([
                        dbc.Col(dcc.Upload(
                            id="up-chat-actual",
                            children=html.Div(["⬆️ Upload Chat Actual (XLSX/CSV)"]),
                            multiple=False, className="upload-box"
                        ), md=6),
                        dbc.Col(dbc.Button("Save Chat Actual", id="btn-save-chat-actual", color="primary", className="w-100"), md=3),
                        dbc.Col(dbc.Button("Download Template", id="btn-dl-chat-actual-tmpl", outline=True, color="secondary", className="w-100"), md=3),
                    ], className="my-2"),
                    dcc.Download(id="dl-chat-actual-tmpl"),
                    html.Div(dash_table.DataTable(id="tbl-chat-actual", page_size=10,
                        style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ), className="loading-block"),
                    html.Div(html.Div(id="chat-actual-msg", className="text-success mt-1"), className="loading-block"),
                ]),

                # ---------- Outbound (Forecast + Actual) ----------
                dcc.Tab(label="Outbound", value="tab-ob", children=[
                    html.H6("Forecast"),
                    dbc.Row([
                        dbc.Col(dcc.Upload(
                            id="up-ob-forecast",
                            children=html.Div(["⬆️ Upload Outbound Forecast (XLSX/CSV)"]),
                            multiple=False, className="upload-box"
                        ), md=6),
                        dbc.Col(dbc.Button("Save Outbound Forecast", id="btn-save-ob-forecast", color="primary", className="w-100"), md=3),
               	        dbc.Col(dbc.Button("Download Template", id="btn-dl-ob-forecast-tmpl", outline=True, color="secondary", className="w-100"), md=3),
                    ], className="my-2"),
                    dcc.Download(id="dl-ob-forecast-tmpl"),
                    html.Div(dash_table.DataTable(id="tbl-ob-forecast", page_size=10,
                        style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ), className="loading-block"),
                    html.Div(html.Div(id="ob-forecast-msg", className="text-success mt-1"), className="loading-block"),

                    html.Hr(),
                    html.H6("Actual"),
                    dbc.Row([
                        dbc.Col(dcc.Upload(
                            id="up-ob-actual",
                            children=html.Div(["⬆️ Upload Outbound Actual (XLSX/CSV)"]),
                            multiple=False, className="upload-box"
                        ), md=6),
                        dbc.Col(dbc.Button("Save Outbound Actual", id="btn-save-ob-actual", color="primary", className="w-100"), md=3),
                        dbc.Col(dbc.Button("Download Template", id="btn-dl-ob-actual-tmpl", outline=True, color="secondary", className="w-100"), md=3),
                    ], className="my-2"),
                    dcc.Download(id="dl-ob-actual-tmpl"),
                    html.Div(dash_table.DataTable(id="tbl-ob-actual", page_size=10,
                        style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ), className="loading-block"),
                    html.Div(html.Div(id="ob-actual-msg", className="text-success mt-1"), className="loading-block"),
                ]),
            ])
        ]), className="mb-3"),

        # ===== Tactical upload remains =====
        dbc.Card(dbc.CardBody([
            html.H5("Upload Tactical Volume (by scope)"),
            html.Div("Voice uses 30-min intervals; Back Office uses daily totals.", className="text-muted small mb-2"),
            dbc.Alert("Alert:- Uploads are saved to the selected scope. Even if your file includes Business Area, Sub Business Area, and Channel, please choose the scope above first.",color="light"),
            dbc.Tabs(id="vol-tabs1", children=[

                # ---------- Voice (Forecast) ----------
                dcc.Tab(label="Voice", value="tab-voice1", children=[
                    html.H6("Forecast"),
                    dbc.Row([
                        dbc.Col(dcc.Upload(
                            id="up-voice-forecast1",
                            children=html.Div(["⬆️ Upload Voice Forecast (XLSX/CSV)"]),
                            multiple=False, className="upload-box"
                        ), md=6),
                        dbc.Col(dbc.Button("Save Voice Forecast", id="btn-save-voice-forecast1", color="primary", className="w-100"), md=3),
                        dbc.Col(dbc.Button("Download Template", id="btn-dl-voice-forecast-tmpl1", outline=True, color="secondary", className="w-100"), md=3),
                    ], className="my-2"),
                    dcc.Download(id="dl-voice-forecast-tmpl1"),
                    html.Div(dash_table.DataTable(id="tbl-voice-forecast1", page_size=10,
                        style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ), className="loading-block"),
                    html.Div(html.Div(id="voice-forecast-msg1", className="text-success mt-1"), className="loading-block"),
                ]),

                # ---------- Back Office (Forecast) ----------
                dcc.Tab(label="Back Office", value="tab-bo1", children=[
                    html.H6("Forecast"),
                    dbc.Row([
                        dbc.Col(dcc.Upload(
                            id="up-bo-forecast1",
                            children=html.Div(["⬆️ Upload Back Office Forecast (XLSX/CSV)"]),
                            multiple=False, className="upload-box"
                        ), md=6),
                        dbc.Col(dbc.Button("Save BO Forecast", id="btn-save-bo-forecast1", color="primary", className="w-100"), md=3),
                        dbc.Col(dbc.Button("Download Template", id="btn-dl-bo-forecast-tmpl1", outline=True, color="secondary", className="w-100"), md=3),
                    ], className="my-2"),
                    dcc.Download(id="dl-bo-forecast-tmpl1"),
                    html.Div(dash_table.DataTable(id="tbl-bo-forecast1", page_size=10,
                        style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ), className="loading-block"),
                    html.Div(html.Div(id="bo-forecast-msg1", className="text-success mt-1"), className="loading-block"),
                ]),

                # ---------- Chat (Forecast by scope) ----------
                dcc.Tab(label="Chat", value="tab-chat1", children=[
                    html.H6("Forecast"),
                    dbc.Row([
                        dbc.Col(dcc.Upload(
                            id="up-chat-forecast1",
                            children=html.Div(["⬆️ Upload Chat Forecast (XLSX/CSV)"]),
                            multiple=False, className="upload-box"
                        ), md=6),
                        dbc.Col(dbc.Button("Save Chat Forecast", id="btn-save-chat-forecast1", color="primary", className="w-100"), md=3),
                        dbc.Col(dbc.Button("Download Template", id="btn-dl-chat-forecast-tmpl1", outline=True, color="secondary", className="w-100"), md=3),
                    ], className="my-2"),
                    dcc.Download(id="dl-chat-forecast-tmpl1"),
                    html.Div(dash_table.DataTable(id="tbl-chat-forecast1", page_size=10,
                        style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ), className="loading-block"),
                    html.Div(html.Div(id="chat-forecast-msg1", className="text-success mt-1"), className="loading-block"),
                ]),

                # ---------- Outbound (Forecast by scope) ----------
                dcc.Tab(label="Outbound", value="tab-ob1", children=[
                    html.H6("Forecast"),
                    dbc.Row([
                        dbc.Col(dcc.Upload(
                            id="up-ob-forecast1",
                            children=html.Div(["⬆️ Upload Outbound Forecast (XLSX/CSV)"]),
                            multiple=False, className="upload-box"
                        ), md=6),
                        dbc.Col(dbc.Button("Save Outbound Forecast", id="btn-save-ob-forecast1", color="primary", className="w-100"), md=3),
                        dbc.Col(dbc.Button("Download Template", id="btn-dl-ob-forecast-tmpl1", outline=True, color="secondary", className="w-100"), md=3),
                    ], className="my-2"),
                    dcc.Download(id="dl-ob-forecast-tmpl1"),
                    html.Div(dash_table.DataTable(id="tbl-ob-forecast1", page_size=10,
                        style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ), className="loading-block"),
                    html.Div(html.Div(id="ob-forecast-msg1", className="text-success mt-1"), className="loading-block"),
                ]),
            ])
        ]), className="mb-3"),        

        # ===== Holiday Calendar =====
        dbc.Card(dbc.CardBody([
            html.H5("Holiday Calendar"),
            html.Div("Upload site/location specific holidays to exclude from working days.", className="text-muted small mb-2"),
            dbc.Row([
                dbc.Col(dcc.Upload(
                    id="up-holidays",
                    children=html.Div(["⬆️ Upload Holiday List (XLSX/CSV)"]),
                    multiple=False, className="upload-box"
                ), md=6),
                dbc.Col(dbc.Button("Save Holidays", id="btn-save-holidays", color="primary", className="w-100"), md=3),
                dbc.Col(dbc.Button("Download Template", id="btn-dl-holidays-template", outline=True, color="secondary", className="w-100"), md=3),
            ], className="my-2"),
            dcc.Download(id="dl-holidays-template"),
            html.Div(dash_table.DataTable(
                id="tbl-holidays", page_size=10,
                # Hide headers until user uploads or saved holidays load
                columns=[],
                data=[],
                style_table={"overflowX":"auto"}, style_as_list_view=True,
                style_header={"textTransform":"none"}
            ), className="loading-block"),
            html.Div(html.Div(id="holidays-msg", className="text-success mt-1"), className="loading-block"),
        ]), className="mb-3"),
        # ===== Headcount upload remains =====
        dbc.Card(dbc.CardBody([
            html.H5("Headcount Update - BRID Mapping"),
            html.Div("Upload the latest headcount file to keep BRID - Manager/Hierarchy mappings in sync.", className="text-muted small mb-2"),

            dbc.Row([
                dbc.Col(dcc.Upload(
                    id="up-headcount",
                    children=html.Div(["⬆️ Upload Headcount XLSX"]),
                    multiple=False, className="upload-box"
                ), md=6),
                dbc.Col(dbc.Button("Save Headcount", id="btn-save-headcount", color="primary", className="w-100"), md=3),
                dbc.Col(dbc.Button("Download Template", id="btn-dl-hc-template", outline=True, color="secondary", className="w-100"), md=3),
            ], className="my-2"),
            dcc.Download(id="dl-hc-template"),
            html.Div(dash_table.DataTable(
                id="tbl-headcount-preview", page_size=8,
                style_table={"overflowX":"auto"}, style_as_list_view=True,
                style_header={"textTransform":"none"}
            ), className="loading-block"),
            html.Div(html.Div(id="hc-msg", className="text-success mt-1"), className="loading-block"),
        ]), className="mb-3"),
    ], fluid=True), className="loading-page")