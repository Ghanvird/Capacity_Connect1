from __future__ import annotations
import urllib.parse as _url
import pandas as _pd
from dash import html, dcc, Input, Output, State, callback
from plan_detail._common import get_plan


def expand_week_bar_component() -> html.Div:
    return html.Div(id="fw-expand-bar", className="mb-2")


def _week_ids_from_columns(cols) -> list[str]:
    ids: list[str] = []
    try:
        for c in (cols or []):
            cid = c.get("id") if isinstance(c, dict) else None
            if not cid or cid == "metric":
                continue
            try:
                t = _pd.to_datetime(cid, errors="coerce")
                if _pd.isna(t):
                    continue
                ids.append(str(_pd.Timestamp(t).date()))
            except Exception:
                continue
    except Exception:
        return []
    return ids


@callback(
    Output("fw-expand-bar", "children"),
    Input("tbl-fw", "columns"),
    State("plan-detail-id", "data"),
    prevent_initial_call=False,
)
def _render_expand_bar(cols, pid):
    week_ids = _week_ids_from_columns(cols)
    if not week_ids:
        return html.Div()
    # plan scope for link prefilter
    ba = sba = ch = None
    try:
        if isinstance(pid, int) or (isinstance(pid, str) and not str(pid).startswith("ba::")):
            p = get_plan(pid) or {}
            ba = p.get("vertical"); sba = p.get("sub_ba"); ch = (p.get("channel") or p.get("lob") or "").split(",")[0].strip()
    except Exception:
        pass

    def _link_for_week(w: str):
        qs = {"start": w, "end": str(_pd.to_datetime(w) + _pd.to_timedelta(6, unit="D"))[:10]}
        if ba:  qs["ba"] = ba
        if sba: qs["sba"] = sba
        if ch:  qs["ch"] = ch
        href = "/capacity-drill?" + _url.urlencode(qs)
        return dcc.Link(html.Button("+", title=f"Expand week {w}", className="btn btn-sm btn-outline-secondary me-1"), href=href, target="_blank")

    items = []
    for w in week_ids:
        # Only show the + button; omit the week text since it's already in headers
        items.append(_link_for_week(w))
    return html.Div([html.Span("Expand week: ", className="me-2"), *items])
