from __future__ import annotations
import html
from dash import dcc, html
from planning_workspace import planning_layout

def page_planning():
    return html.Div(planning_layout(), className="loading-page")



