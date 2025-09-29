from __future__ import annotations
import html
from dash import dcc, html
import dash_bootstrap_components as dbc

from common import header_bar

def page_forecast():
    return html.Div(dbc.Container([
        header_bar(),], fluid=True), className="loading-page")