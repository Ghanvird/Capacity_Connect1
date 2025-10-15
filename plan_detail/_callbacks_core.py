# file: plan_detail/_callbacks_core.py
from __future__ import annotations
from dash import dcc, html, dash_table, Input, Output, State, ctx, no_update
import dash
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# UI-only bits we reuse (overlay style + pid resolver)
from cap_store import load_timeseries
from plan_detail._ui import _auto_dates, _pid_from_layout, _update_roster_by_class
from plan_store import get_plan, list_plans
import datetime as dt
# Everything data/logic comes from _common (so we don't depend on _ui for logic)
from ._common import (  # noqa
    _month_cols, save_df, load_df, _save_table,
    _roster_columns, _bulkfile_columns,
    _week_span, _week_cols, _format_crumb, _ROSTER_REQUIRED_IDS, _parse_upload,
    _blank_grid, _scope_key, _monday, _hier_from_hcu, _build_global_hierarchy,
    _load_hcu_df, _lower_map, CHANNEL_DEFAULTS, _load_index,
    # New Hire helpers centralized in _common
    load_nh_classes, save_nh_classes, next_class_reference, current_user_fallback, save_plan_meta, get_plan_meta, clone_plan,
)

from ._calc import _fill_tables_fixed
from ._fill_tables_fixed_monthly import _fill_tables_fixed_monthly
from ._fill_tables_fixed_daily import _fill_tables_fixed_daily
from ._fill_tables_fixed_interval import _fill_tables_fixed_interval
from ._grain_cols import day_cols_for_weeks, interval_cols_for_day


def _verify_storage(pid: str | int, when: str) -> list[str]:
    """Console log of rowcounts for all plan tables (WHY: trace vanishing)."""
    suffixes = ["fw", "hc", "attr", "shr", "train", "ratio", "seat",
                "bva", "nh", "emp", "bulk_files", "notes"]
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    logs: list[str] = [f"[verify_storage][{ts}][{when}] pid={pid}"]
    for s in suffixes:
        key = f"plan_{pid}_{s}"
        try:
            df = load_df(key)
            n = len(df) if isinstance(df, pd.DataFrame) else -1
            logs.append(f" - {key}: {n} rows")
        except Exception as e:
            logs.append(f" - {key}: ERROR {e}")
    for line in logs:
        print(line)
    return logs


def _coalesce(pid: str | int, suffix: str, data) -> pd.DataFrame:
    """
    Return the DataFrame we *should* save.
    WHY: If user hits Save while tables haven't hydrated yet, the State may be []/None.
         In that case, keep existing persisted data instead of overwriting to empty.
    """
    incoming = pd.DataFrame(data or [])
    if incoming.empty:
        existing = load_df(f"plan_{pid}_{suffix}")
        if isinstance(existing, pd.DataFrame) and not existing.empty:
            return existing
    return incoming



def _pick(d, *keys, default=""):
    """Return the first non-empty value for any of the given keys (case-insensitive)."""
    if not isinstance(d, dict):
        return default
    low = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        v = low.get(str(k).lower())
        if v not in (None, "", "null", "None"):
            return v
    return default

def _fmt_date(val):
    try:
        return pd.to_datetime(val).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""
    
def register_plan_detail_core(app: dash.Dash):
        def _selected_rows(data, selected_rows):
            df = pd.DataFrame(data or [])
            if df.empty or not selected_rows:
                return df, []
            idx = [i for i in selected_rows if 0 <= i < len(df)]
            return df, idx
        # Remove
        @app.callback(
            Output("plan-msg", "children", allow_duplicate=True),
            Output("plan-msg-timer", "disabled", allow_duplicate=True),
            Input("plan-detail-id", "data"),
            prevent_initial_call=True,
        )
        def _clear_plan_msg_on_enter(_pid):
            return "", True
        
        @app.callback(
            Output("modal-remove", "is_open"),
            Input("btn-emp-remove", "n_clicks"),
            Input("btn-remove-cancel", "n_clicks"),
            prevent_initial_call=True
        )
        def _open_remove(n_open, n_cancel):
            t = ctx.triggered_id
            return True if t == "btn-emp-remove" else False

        # Hide Employee Roster action buttons in BA roll-up view
        @app.callback(
            Output("btn-emp-add",   "style", allow_duplicate=True),
            Output("btn-emp-tp",    "style", allow_duplicate=True),
            Output("btn-emp-loa",   "style", allow_duplicate=True),
            Output("btn-emp-back",  "style", allow_duplicate=True),
            Output("btn-emp-term",  "style", allow_duplicate=True),
            Output("btn-emp-ftp",   "style", allow_duplicate=True),
            Output("btn-emp-undo",  "style", allow_duplicate=True),
            Output("btn-emp-class", "style", allow_duplicate=True),
            Output("btn-emp-remove","style", allow_duplicate=True),
            Input("plan-detail-id", "data"),
            prevent_initial_call=True,
        )
        def _hide_emp_buttons(pid):
            hide = {"display": "none"}
            show = {}
            if isinstance(pid, str) and str(pid).startswith("ba::"):
                return hide, hide, hide, hide, hide, hide, hide, hide, hide
            return show, show, show, show, show, show, show, show, show

        # Hide New Hire actions in BA roll-up view
        @app.callback(
            Output("btn-nh-add",     "style", allow_duplicate=True),
            Output("btn-nh-details", "style", allow_duplicate=True),
            Input("plan-detail-id", "data"),
            prevent_initial_call=True,
        )
        def _hide_nh_buttons(pid):
            hide = {"display": "none"}
            show = {}
            if isinstance(pid, str) and str(pid).startswith("ba::"):
                return hide, hide
            return show, show

        @app.callback(
            Output("tbl-emp-roster", "data", allow_duplicate=True),
            Output("lbl-emp-total", "children", allow_duplicate=True),
            Output("modal-remove", "is_open", allow_duplicate=True),
            Output("plan-msg", "children", allow_duplicate=True),
            Output("plan-msg-timer", "disabled", allow_duplicate=True),
            Output("plan-refresh-tick", "data", allow_duplicate=True),
            Output("emp-undo", "data", allow_duplicate=True),
            Input("btn-remove-ok", "n_clicks"),
            State("tbl-emp-roster", "data"),
            State("tbl-emp-roster", "selected_rows"),
            State("plan-detail-id", "data"),
            State("plan-refresh-tick", "data"),
            prevent_initial_call=True
        )
        def _do_remove(n, data, selected_rows, pid, plan_refresh_tick):
            df, idx = _selected_rows(data, selected_rows)
            if df.empty or not idx:
                raise dash.exceptions.PreventUpdate
            keep = df.drop(index=idx).reset_index(drop=True)
            save_df(f"plan_{pid}_emp", keep)
            save_plan_meta(pid, {"last_updated_on": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                     "last_updated_by": current_user_fallback()})
            next_tick = int(plan_refresh_tick or 0) + 1
            undo_snap = {"data": df.to_dict("records"), "ts": pd.Timestamp.utcnow().isoformat(timespec="seconds")}

            return keep.to_dict("records"), f"Total: {len(keep):02d} Records", False, "Removed ✅", False, next_tick, undo_snap

        # Change Class
        @app.callback(
            Output("modal-class", "is_open"),
            Output("class-change-hint", "children"),
            Input("btn-emp-class", "n_clicks"),
            Input("btn-class-cancel", "n_clicks"),
            State("tbl-emp-roster", "data"),
            State("tbl-emp-roster", "selected_rows"),
            prevent_initial_call=True
        )
        def _open_class(n_open, n_cancel, data, sel):
            t = ctx.triggered_id
            if t == "btn-emp-class":
                df, idx = _selected_rows(data, sel)
                who = ", ".join(df.loc[idx, "name"].astype(str).tolist()) if not df.empty and idx else ""
                return True, f"Selected: {who}"
            return False, ""

        @app.callback(
            Output("tbl-emp-roster", "data", allow_duplicate=True),
            Output("modal-class", "is_open", allow_duplicate=True),
            Output("plan-msg", "children", allow_duplicate=True),
            Output("plan-msg-timer", "disabled", allow_duplicate=True),
            Output("plan-refresh-tick", "data", allow_duplicate=True),
            Output("emp-undo", "data", allow_duplicate=True),
            Input("btn-class-save", "n_clicks"),
            State("inp-class-ref", "value"),
            State("tbl-emp-roster", "data"),
            State("tbl-emp-roster", "selected_rows"),
            State("plan-detail-id", "data"),
            State("plan-refresh-tick", "data"),
            prevent_initial_call=True
        )
        def _do_class(n, cref, data, sel, pid, tick):
            if not n or not cref:
                raise dash.exceptions.PreventUpdate
            df, idx = _selected_rows(data, sel)
            if df.empty or not idx:
                raise dash.exceptions.PreventUpdate
            before = df.copy()
            df.loc[idx, "class_ref"] = cref
            save_df(f"plan_{pid}_emp", df)
            save_plan_meta(pid, {"last_updated_on": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                     "last_updated_by": current_user_fallback()})
            next_tick = int(tick or 0) + 1
            undo_snap = {"data": before.to_dict("records"), "ts": pd.Timestamp.utcnow().isoformat(timespec="seconds")}

            return df.to_dict("records"), False, "Class updated ✅", False, next_tick, undo_snap

        # Populate class-ref options when class modal opens
        @app.callback(
            Output("inp-class-ref", "options"),
            Input("modal-class", "is_open"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True
        )
        def _class_ref_options(is_open, pid):
            if not is_open:
                raise dash.exceptions.PreventUpdate
            try:
                df = load_nh_classes(str(pid))
            except Exception:
                df = pd.DataFrame()
            if isinstance(df, pd.DataFrame) and not df.empty and "class_reference" in df.columns:
                vals = df["class_reference"].dropna().astype(str).str.strip().unique().tolist()
                return [{"label": v, "value": v} for v in vals]
            return []

        # FT/PT
        @app.callback(
            Output("modal-ftp", "is_open"),
            Output("ftp-who", "children"),
            Input("btn-emp-ftp", "n_clicks"),
            Input("btn-ftp-cancel", "n_clicks"),
            State("tbl-emp-roster", "data"),
            State("tbl-emp-roster", "selected_rows"),
            prevent_initial_call=True
        )
        def _open_ftp(n_open, n_cancel, data, sel):
            t = ctx.triggered_id
            if t == "btn-emp-ftp":
                df, idx = _selected_rows(data, sel)
                who = ", ".join(df.loc[idx, "name"].astype(str).tolist()) if not df.empty and idx else ""
                return True, f"Selected: {who}"
            return False, ""

        @app.callback(
            Output("tbl-emp-roster", "data", allow_duplicate=True),
            Output("modal-ftp", "is_open", allow_duplicate=True),
            Output("plan-msg", "children", allow_duplicate=True),
            Output("plan-msg-timer", "disabled", allow_duplicate=True),
            Input("btn-ftp-save", "n_clicks"),
            State("inp-ftp-date", "date"),
            State("inp-ftp-hours", "value"),
            State("tbl-emp-roster", "data"),
            State("tbl-emp-roster", "selected_rows"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True
        )
        def _do_ftp(n, date, hours, data, sel, pid):
            if not n:
                raise dash.exceptions.PreventUpdate
            df, idx = _selected_rows(data, sel)
            if df.empty or not idx:
                raise dash.exceptions.PreventUpdate
            for i in idx:
                cur = str(df.at[i, "ftpt_status"] or "")
                if cur.lower().startswith("full"):
                    df.at[i, "ftpt_status"] = "Part-time"
                    if hours: df.at[i, "ftpt_hours"] = hours
                else:
                    df.at[i, "ftpt_status"] = "Full-time"
                    df.at[i, "ftpt_hours"] = hours or ""
            save_df(f"plan_{pid}_emp", df)
            save_plan_meta(pid, {"last_updated_on": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                     "last_updated_by": current_user_fallback()})

            return df.to_dict("records"), False, "FT/PT updated ✅", False

        # Move to LOA
        @app.callback(
            Output("modal-loa", "is_open"),
            Output("loa-who", "children"),
            Input("btn-emp-loa", "n_clicks"),
            Input("btn-loa-cancel", "n_clicks"),
            State("tbl-emp-roster", "data"),
            State("tbl-emp-roster", "selected_rows"),
            prevent_initial_call=True
        )
        def _open_loa(n_open, n_cancel, data, sel):
            t = ctx.triggered_id
            if t == "btn-emp-loa":
                df, idx = _selected_rows(data, sel)
                who = ", ".join(df.loc[idx, "name"].astype(str).tolist()) if not df.empty and idx else ""
                return True, f"Selected: {who}"
            return False, ""

        @app.callback(
            Output("tbl-emp-roster", "data", allow_duplicate=True),
            Output("modal-loa", "is_open", allow_duplicate=True),
            Output("plan-msg", "children", allow_duplicate=True),
            Output("plan-msg-timer", "disabled", allow_duplicate=True),
            Input("btn-loa-save", "n_clicks"),
            State("inp-loa-date", "date"),
            State("tbl-emp-roster", "data"),
            State("tbl-emp-roster", "selected_rows"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True
        )
        def _do_loa(n, date, data, sel, pid):
            if not n or not date:
                raise dash.exceptions.PreventUpdate
            df, idx = _selected_rows(data, sel)
            if df.empty or not idx:
                raise dash.exceptions.PreventUpdate
            monday = _monday(date).isoformat()
            for i in idx:
                df.at[i, "loa_date"] = monday
                df.at[i, "current_status"] = "Moved to LOA"
                df.at[i, "work_status"] = "Moved to LOA"
            save_df(f"plan_{pid}_emp", df)
            save_plan_meta(pid, {"last_updated_on": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                     "last_updated_by": current_user_fallback()})

            return df.to_dict("records"), False, "Moved to LOA ✅", False

        # Back from LOA
        @app.callback(
            Output("modal-back", "is_open"),
            Output("back-who", "children"),
            Input("btn-emp-back", "n_clicks"),
            Input("btn-back-cancel", "n_clicks"),
            State("tbl-emp-roster", "data"),
            State("tbl-emp-roster", "selected_rows"),
            prevent_initial_call=True
        )
        def _open_back(n_open, n_cancel, data, sel):
            t = ctx.triggered_id
            if t == "btn-emp-back":
                df, idx = _selected_rows(data, sel)
                who = ", ".join(df.loc[idx, "name"].astype(str).tolist()) if not df.empty and idx else ""
                return True, f"Selected: {who}"
            return False, ""

        @app.callback(
            Output("tbl-emp-roster", "data", allow_duplicate=True),
            Output("modal-back", "is_open", allow_duplicate=True),
            Output("plan-msg", "children", allow_duplicate=True),
            Output("plan-msg-timer", "disabled", allow_duplicate=True),
            Input("btn-back-save", "n_clicks"),
            State("inp-back-date", "date"),
            State("tbl-emp-roster", "data"),
            State("tbl-emp-roster", "selected_rows"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True
        )
        def _do_back(n, date, data, sel, pid):
            if not n or not date:
                raise dash.exceptions.PreventUpdate
            df, idx = _selected_rows(data, sel)
            if df.empty or not idx:
                raise dash.exceptions.PreventUpdate
            monday = _monday(date).isoformat()
            for i in idx:
                df.at[i, "back_from_loa_date"] = monday
                df.at[i, "current_status"] = "Production"
                df.at[i, "work_status"] = "Production"
            save_df(f"plan_{pid}_emp", df)
            save_plan_meta(pid, {"last_updated_on": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                     "last_updated_by": current_user_fallback()})

            return df.to_dict("records"), False, "Back from LOA ✅", False

        # Terminate
        @app.callback(
            Output("modal-term", "is_open"),
            Output("term-who", "children"),
            Input("btn-emp-term", "n_clicks"),
            Input("btn-term-cancel", "n_clicks"),
            State("tbl-emp-roster", "data"),
            State("tbl-emp-roster", "selected_rows"),
            prevent_initial_call=True
        )
        def _open_term(n_open, n_cancel, data, sel):
            t = ctx.triggered_id
            if t == "btn-emp-term":
                df, idx = _selected_rows(data, sel)
                who = ", ".join(df.loc[idx, "name"].astype(str).tolist()) if not df.empty and idx else ""
                return True, f"Selected: {who}"
            return False, ""

        @app.callback(
            Output("tbl-emp-roster", "data", allow_duplicate=True),
            Output("modal-term", "is_open", allow_duplicate=True),
            Output("plan-msg", "children", allow_duplicate=True),
            Output("plan-msg-timer", "disabled", allow_duplicate=True),
            Input("btn-term-save", "n_clicks"),
            State("inp-term-date", "date"),
            State("tbl-emp-roster", "data"),
            State("tbl-emp-roster", "selected_rows"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True
        )
        def _do_term(n, date, data, sel, pid):
            if not n or not date:
                raise dash.exceptions.PreventUpdate
            df, idx = _selected_rows(data, sel)
            if df.empty or not idx:
                raise dash.exceptions.PreventUpdate
            for i in idx:
                df.at[i, "terminate_date"] = pd.to_datetime(date).date().isoformat()
                df.at[i, "current_status"] = "Terminated"
                df.at[i, "work_status"] = "Terminated"
            save_df(f"plan_{pid}_emp", df)
            save_plan_meta(pid, {"last_updated_on": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                     "last_updated_by": current_user_fallback()})

            return df.to_dict("records"), False, "Terminated ✅", False

        # Transfer & Promotion " open
        @app.callback(
            Output("modal-tp", "is_open"),
            Output("tp-hier-map", "data"),
            Output("tp-current", "data"),
            Output("tp-who", "children"),
            Input("btn-emp-tp", "n_clicks"),
            Input("btn-tp-cancel", "n_clicks"),
            State("plan-detail-id", "data"),
            State("tbl-emp-roster", "data"),
            State("tbl-emp-roster", "selected_rows"),
            prevent_initial_call=True
        )
        def _open_tp(n_open, n_cancel, pid, data, sel):
            t = ctx.triggered_id
            if t == "btn-emp-tp":
                # ?? Prefer Headcount Update (Journey / Level 3 / Position Location Building Description)
                hier = _hier_from_hcu()
                if not hier.get("ba"):  # fallback if HCU missing/empty
                    hier = _build_global_hierarchy()

                p = get_plan(pid) or {}
                cur = dict(
                    ba=p.get("business_area","") or p.get("plan_ba","") or p.get("vertical","") or p.get("ba",""),
                    subba=p.get("sub_business_area","") or p.get("plan_sub_ba","") or p.get("sub_ba","") or p.get("subba",""),
                    lob=p.get("lob","") or p.get("channel",""),
                    site=p.get("site",""),
                )

                df, idx = _selected_rows(data, sel)
                who = ", ".join(df.loc[idx, "name"].astype(str).tolist()) if not df.empty and idx else ""
                return True, hier, cur, f"Selected: {who}"
            return False, {}, {}, ""

        @app.callback(
            Output("tp-ba", "options"),
            Output("tp-ba", "value"),
            Input("tp-hier-map", "data"),
            State("tp-current", "data"),
            prevent_initial_call=False
        )
        def _tp_fill_ba(hmap, cur):
            bas = list((hmap or {}).get("ba") or [])
            opts = [{"label": b, "value": b} for b in bas]
            cur_ba = (cur or {}).get("ba")
            val = cur_ba if cur_ba in bas else (bas[0] if bas else None)
            return opts, val

        @app.callback(
            Output("tp-subba", "options"),
            Output("tp-subba", "value"),
            Input("tp-ba", "value"),
            State("tp-hier-map", "data"),
            State("tp-current", "data"),
            prevent_initial_call=False
        )
        def _tp_fill_subba(ba_val, hmap, cur):
            sub_map = dict((hmap or {}).get("subba") or {})
            # tolerant key match for BA
            key = None
            if ba_val:
                for k in sub_map.keys():
                    if str(k).strip().lower() == str(ba_val).strip().lower():
                        key = k; break
            subs = list(sub_map.get(key, [])) if key else []
            opts = [{"label": s, "value": s} for s in subs]
            cur_sub = (cur or {}).get("subba")
            val = cur_sub if cur_sub in subs else (subs[0] if subs else None)
            return opts, val

        @app.callback(
            Output("tp-lob", "options"),
            Output("tp-lob", "value"),
            Input("tp-ba", "value"),
            Input("tp-subba", "value"),
            State("tp-hier-map", "data"),
            State("tp-current", "data"),
            prevent_initial_call=False
        )
        def _tp_fill_lob(ba_val, sub_val, hmap, cur):
            lob_map = dict((hmap or {}).get("lob") or {})
            # tolerant composite key
            key = None
            if ba_val and sub_val:
                target = f"{str(ba_val).strip().lower()}|{str(sub_val).strip().lower()}"
                for k in lob_map.keys():
                    if str(k).strip().lower() == target:
                        key = k; break
            lobs = list(lob_map.get(key, CHANNEL_DEFAULTS))
            opts = [{"label": l, "value": l} for l in lobs]
            cur_lob = (cur or {}).get("lob")
            val = cur_lob if cur_lob in lobs else (lobs[0] if lobs else None)
            return opts, val


        # TP: mirror values to twp-* for the "Transfer with Promotion" tab on open/change

        @app.callback(
            Output("twp-ba", "options"), Output("twp-ba", "value"),
            Output("twp-subba", "options"), Output("twp-subba", "value"),
            Output("twp-lob", "options"), Output("twp-lob", "value"),
            Input("tp-ba", "options"), Input("tp-ba", "value"),
            Input("tp-subba", "options"), Input("tp-subba", "value"),
            Input("tp-lob", "options"), Input("tp-lob", "value"),
            prevent_initial_call=False

        )
        def _mirror_tp_to_twp(ba_opts, ba_val, sub_opts, sub_val, lob_opts, lob_val):
            return ba_opts or [], ba_val, sub_opts or [], sub_val, lob_opts or [], lob_val

        # TP: simple Site picker " gather unique sites we know (from roster wide/long + plan site)
        @app.callback(
            Output("tp-site", "options"), Output("tp-site", "value"),
            Output("twp-site", "options"), Output("twp-site", "value"),
            Input("tp-current", "data"),
            prevent_initial_call=False
        )
        def _fill_sites(cur):
            # Collect sites from the Headcount Update upload only
            sites: set[str] = set()
            try:
                hcu = _load_hcu_df()  # ? load_headcount() under the hood
                if isinstance(hcu, pd.DataFrame) and not hcu.empty:
                    L = _lower_map(hcu)
                    # primary column, plus a few tolerant aliases
                    site_col = (
                        L.get("position location building description")
                        or L.get("position_location_building_description")
                        or L.get("building description")
                        or L.get("site")
                    )
                    if site_col:
                        s = (
                            hcu[site_col]
                            .dropna()
                            .astype(str)
                            .str.strip()
                            .replace({"": np.nan})
                            .dropna()
                            .unique()
                            .tolist()
                        )
                        sites |= set(s)
            except Exception:
                pass

            # Always include current plan site (so user sees what's already set)
            cur_site = (cur or {}).get("site")
            if cur_site:
                sites.add(str(cur_site).strip())

            opts = [{"label": s, "value": s} for s in sorted(sites)]
            # Prefer current plan site if it exists in options; else first option (if any)
            val = cur_site if (cur_site in sites) else (sorted(sites)[0] if sites else None)
            return opts, val, opts, val



        # TP: Save (applies to selected rows)

        @app.callback(
            Output("tbl-emp-roster", "data", allow_duplicate=True),
            Output("modal-tp", "is_open", allow_duplicate=True),
            Output("plan-msg", "children", allow_duplicate=True),
            Output("plan-msg-timer", "disabled", allow_duplicate=True),
            Input("btn-tp-save", "n_clicks"),
            State("tp-active-tab", "value"),
            # transfer
            State("tp-ba", "value"), State("tp-subba", "value"), State("tp-lob", "value"), State("tp-site", "value"),
            State("tp-transfer-type", "value"),
            State("tp-new-class", "value"),
            State("tp-class-ref", "value"),
            State("tp-date-from", "date"), State("tp-date-to", "date"),
            # promotion
            State("promo-type", "value"), State("promo-role", "value"),
            State("promo-date-from", "date"), State("promo-date-to", "date"),
            # both
            State("twp-ba", "value"), State("twp-subba", "value"),
            State("twp-lob", "value"), State("twp-site", "value"),
            State("twp-type", "value"), State("twp-new-class", "value"),
            State("twp-class-ref", "value"), State("twp-role", "value"),
            State("twp-date-from", "date"), State("twp-date-to", "date"),
            # selection + data
            State("tbl-emp-roster", "data"),
            State("tbl-emp-roster", "selected_rows"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True
        )
        def _tp_save(n, tab,
                     t_ba, t_sub, t_lob, t_site, t_type, t_newcls, t_clref, t_from, t_to,
                     p_type, p_role, p_from, p_to,
                     b_ba, b_sub, b_lob, b_site, b_type, b_newcls, b_clref, b_role, b_from, b_to,
                     data, sel, pid):
            if not n:
                raise dash.exceptions.PreventUpdate
            df, idx = _selected_rows(data, sel)
            if df.empty or not idx:
                raise dash.exceptions.PreventUpdate

            def _apply_transfer(I, ba, sub, lob, site, typ, newclass, cref, dfrom, dto):
                if ba:   df.at[I, "biz_area"] = ba
                if sub:  df.at[I, "sub_biz_area"] = sub
                if lob:  df.at[I, "lob"] = lob
                if site: df.at[I, "site"] = site
                if newclass and cref:
                    df.at[I, "class_ref"] = cref
                    if dfrom: df.at[I, "training_start"] = pd.to_datetime(dfrom).date().isoformat()
                df.at[I, "current_status"] = "Interim Transfer" if typ == "interim" else "Transferred"
                df.at[I, "work_status"] = df.at[I, "current_status"]

            def _apply_promo(I, typ, role, dfrom, dto):
                if role: df.at[I, "role"] = role
                df.at[I, "current_status"] = "Promotion (Temp)" if typ == "interim" else "Promotion"
                df.at[I, "work_status"] = "Production"

            if tab == "tp-transfer":
                for I in idx:
                    _apply_transfer(I, t_ba, t_sub, t_lob, t_site, (t_type or "perm"),
                                    bool(t_newcls), t_clref, t_from, t_to)
            elif tab == "tp-promo":
                for I in idx:
                    _apply_promo(I, (p_type or "perm"), p_role, p_from, p_to)
            else:  # both
                for I in idx:
                    _apply_transfer(I, b_ba, b_sub, b_lob, b_site, (b_type or "perm"),
                                    bool(b_newcls), b_clref, b_from, b_to)
                    _apply_promo(I, (b_type or "perm"), b_role, b_from, b_to)
            save_df(f"plan_{pid}_emp", df)
            save_plan_meta(pid, {"last_updated_on": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                     "last_updated_by": current_user_fallback()})

            return df.to_dict("records"), False, "Transfer / Promotion saved ✅", False

        #_________________________________End Employee Roster Modals & Actions __________________________________

        @app.callback(
            Output("plan-detail-id", "data", allow_duplicate=True),
            Input("url-router", "pathname"),
            prevent_initial_call=True,
        )
        def _capture_pid(pathname):
            path = (pathname or "").rstrip("/")
            if not path.startswith("/plan/"):
                raise dash.exceptions.PreventUpdate
            # BA roll-up route: /plan/ba/<ba>
            if path.startswith("/plan/ba/"):
                try:
                    from urllib.parse import unquote as _unq
                    ba = _unq(path.split("/plan/ba/", 1)[-1])
                except Exception:
                    ba = path.split("/plan/ba/", 1)[-1]
                return f"ba::{ba}"
            try:
                return int(path.rsplit("/", 1)[-1])
            except Exception:
                return no_update

        @app.callback(
            Output("plan-hdr-name", "children"),
            Output("plan-type", "data"),
            Output("plan-weeks", "data"),
            Output("tbl-fw", "columns"),
            Output("tbl-hc", "columns"),
            Output("tbl-attr", "columns"),
            Output("tbl-shr", "columns"),
            Output("tbl-train", "columns"),
            Output("tbl-ratio", "columns"),
            Output("tbl-seat", "columns"),
            Output("tbl-bva", "columns"),
            Output("tbl-nh", "columns"),
            Output("tbl-emp-roster", "columns"),
            Output("tbl-bulk-files", "columns"),
            Output("tbl-notes", "columns"),
            Input("plan-detail-id", "data"),
            Input("plan-grain", "data"),
            State("url-router", "pathname"),
            prevent_initial_call=False,
        )
        def _init_cols(pid, grain, pathname):
            path = (pathname or "").rstrip("/")
            if not path.startswith("/plan/"):
                raise dash.exceptions.PreventUpdate

            # BA roll-up synthetic pid
            if isinstance(pid, str) and pid.startswith("ba::"):
                from ba_rollup_plan import month_cols_for_ba, week_cols_for_ba
                ba = pid.split("ba::", 1)[-1]
                g = str(grain or 'week').lower()
                if g == 'month':
                    cols, ids = month_cols_for_ba(ba, status_filter="current")
                elif g == 'day':
                    wcols, wids = week_cols_for_ba(ba, status_filter="current")
                    _weeks = [c["id"] for c in wcols if c.get("id") != "metric"]
                    cols, ids = day_cols_for_weeks(_weeks)
                elif g == 'interval':
                    # Use first week Monday for header date to match interval calculations
                    try:
                        first_week = next((c["id"] for c in wcols if c.get("id") != "metric"), None)
                        _monday = pd.to_datetime(first_week).date() if first_week else None
                    except Exception:
                        _monday = None
                    cols, ids = interval_cols_for_day(_monday)
                else:
                    cols, ids = week_cols_for_ba(ba, status_filter="current")
                # Name and type
                name  = f"{ba}"
                ptype = "Volume Based"
                notes_cols = [{"name": "Date", "id": "when"},{"name": "User", "id": "user"},{"name": "Note", "id": "note"}]
                return (
                    name, ptype, ids,
                    cols, cols, cols, cols, cols, cols, cols, cols, cols,
                    _roster_columns(), _bulkfile_columns(), notes_cols
                )

            if not isinstance(pid, int):
                raise dash.exceptions.PreventUpdate

            p = get_plan(pid) or {}
            name  = p.get("plan_name")  or f"Plan {pid}"
            ptype = p.get("plan_type")  or "Volume Based"

            weeks = _week_span(p.get("start_week"), p.get("end_week"))
            g = str(grain or 'week').lower()
            if g == 'month':
                cols, week_ids = _month_cols(weeks)
            elif g == 'day':
                cols, week_ids = day_cols_for_weeks(weeks)
            elif g == 'interval':
                # Match the header date to the representative Monday used in calculations
                try:
                    _monday = pd.to_datetime(weeks[0]).date() if weeks else None
                except Exception:
                    _monday = None
                cols, ivl_ids = interval_cols_for_day(_monday)
                week_ids = ivl_ids
            else:
                cols, week_ids = _week_cols(weeks)

            notes_cols = [{"name": "Date", "id": "when"},{"name": "User", "id": "user"},{"name": "Note", "id": "note"}]

            return (
                name, ptype, week_ids,
                cols, cols, cols, cols, cols, cols, cols, cols, cols,
                _roster_columns(), _bulkfile_columns(), notes_cols
            )

        # existing callback _fill_tables(...)
        @app.callback(
            Output("plan-upper", "children"),
            Output("tbl-fw", "data"), Output("tbl-hc", "data", allow_duplicate=True),
            Output("tbl-attr", "data"), Output("tbl-shr", "data"),
            Output("tbl-train", "data"), Output("tbl-ratio", "data"),
            Output("tbl-seat", "data"), Output("tbl-bva", "data"),
            Output("tbl-nh", "data"),
            Output("tbl-emp-roster", "data"),
            Output("tbl-bulk-files", "data"),
            Output("tbl-notes", "data"),
            Output("plan-loading", "data", allow_duplicate=True),
            Input("plan-type", "data"),
            State("plan-detail-id", "data"),
            State("tbl-fw", "columns"),
            Input("plan-refresh-tick", "data"),
            State("plan-grain", "data"),
            State("plan-whatif", "data"),            # what-if deltas
            Input("fw-backlog-carryover", "value"),  # backlog toggle now triggers refresh
            prevent_initial_call=True,
        )
        def _fill_tables(ptype, pid, fw_cols, _tick, grain, whatif, backlog_toggle):
            w = dict(whatif or {})
            try:
                w["backlog_carryover"] = bool(backlog_toggle)
            except Exception:
                pass

            g = str(grain or 'week').lower()

            # BA roll-up synthetic pid supports weekly/monthly, reusing same tabs
            if isinstance(pid, str) and pid.startswith("ba::"):
                from ba_rollup_plan import compute_ba_rollup_tables
                ba = pid.split("ba::", 1)[-1]
                results = compute_ba_rollup_tables(ba, fw_cols, whatif=w, status_filter="current", grain=g)
                return (*results, False)

            if g == 'month':
                results = _fill_tables_fixed_monthly(ptype, pid, fw_cols, _tick, w)
            elif g == 'day':
                results = _fill_tables_fixed_daily(ptype, pid, fw_cols, _tick, w)
            elif g == 'interval':
                results = _fill_tables_fixed_interval(ptype, pid, fw_cols, _tick, w)
            else:
                results = _fill_tables_fixed(ptype, pid, fw_cols, _tick, w, grain='week')

            return (*results, False)

        # def _fill_tables(ptype, pid, fw_cols, _tick, grain, whatif, backlog_toggle):   # <-- ADD param
        #     w = dict(whatif or {})
        #     try:
        #         w["backlog_carryover"] = bool(backlog_toggle)
        #     except Exception:
        #         pass
        #     results = _fill_tables_fixed(ptype, pid, fw_cols, _tick, w, grain=grain or 'week')  # <-- pass

        #     # If monthly grain requested, aggregate weekly tables to monthly ids
        #     if str(grain or 'week').lower() == 'month':
        #         import pandas as _pd
        #         def _to_month_ids(cols):
        #             mids = []
        #             for c in cols:
        #                 try:
        #                     t = _pd.to_datetime(c)
        #                     m = _pd.Timestamp(t).to_period('M').to_timestamp().date().isoformat()
        #                 except Exception:
        #                     m = None
        #                 mids.append(m)
        #             return mids
        #         def _agg(df, month_ids):
        #             if not isinstance(df, _pd.DataFrame) or df.empty:
        #                 return df
        #             wk_cols = [c for c in df.columns if c != 'metric']
        #             wk2m = dict(zip(wk_cols, _to_month_ids(wk_cols)))
        #             def is_pct(name: str) -> bool:
        #                 s = str(name).lower()
        #                 return ('%' in s) or ('aht/sut' in s) or ('utilization' in s) or ('occupancy' in s)
        #             rows = []
        #             for _, r in df.iterrows():
        #                 name = r.get('metric')
        #                 sums = {m:0.0 for m in month_ids}
        #                 counts = {m:0 for m in month_ids}
        #                 for w in wk_cols:
        #                     m = wk2m.get(w)
        #                     if m not in sums: continue
        #                     try:
        #                         v = float(_pd.to_numeric(r.get(w), errors='coerce'))
        #                     except Exception:
        #                         v = 0.0
        #                     sums[m] += v
        #                     counts[m] += 1
        #                 outrow = {'metric': name}
        #                 for m in month_ids:
        #                     if is_pct(name):
        #                         c = max(1, counts.get(m,1))
        #                         outrow[m] = sums[m] / c
        #                     else:
        #                         outrow[m] = sums[m]
        #                 rows.append(outrow)
        #             cols = ['metric'] + month_ids
        #             return _pd.DataFrame(rows)[cols]

        #         # month ids from fw_cols order
        #         month_ids = [c.get('id') for c in (fw_cols or []) if c.get('id') != 'metric']
        #         upper, fw, hc, att, shr, trn, rat, seat, bva, nh, roster_df, bulk_files_df, notes_df = results
        #         fw  = _agg(fw, month_ids)
        #         hc  = _agg(hc, month_ids)
        #         att = _agg(att, month_ids)
        #         shr = _agg(shr, month_ids)
        #         trn = _agg(trn, month_ids)
        #         rat = _agg(rat, month_ids)
        #         seat= _agg(seat, month_ids)
        #         bva = _agg(bva, month_ids)
        #         nh  = _agg(nh, month_ids)
        #         results = (upper, fw, hc, att, shr, trn, rat, seat, bva, nh, roster_df, bulk_files_df, notes_df)

        #     return (*results, False)

        # Toggle Week/Month via the UI switch
        @app.callback(
            Output("plan-grain", "data", allow_duplicate=True),
            Input("toggle-switch", "value"),
            prevent_initial_call=True
        )
        def _set_grain(val):
            try:
                return 'month' if bool(val) else 'week'
            except Exception:
                return 'week'

        # Explicit view buttons for Day / Interval (options panel)
        @app.callback(
            Output("plan-grain", "data", allow_duplicate=True),
            Input("opt-view-day", "n_clicks"),
            prevent_initial_call=True,
        )
        def _set_day_view(n):
            if not n:
                raise dash.exceptions.PreventUpdate
            return 'day'

        @app.callback(
            Output("plan-grain", "data", allow_duplicate=True),
            Input("opt-view-interval", "n_clicks"),
            prevent_initial_call=True,
        )
        def _set_interval_view(n):
            if not n:
                raise dash.exceptions.PreventUpdate
            return 'interval'


        # Save all tabs
        # @app.callback(
        #     Output("plan-msg", "children"),
        #     Output("plan-msg-timer", "disabled"),
        #     Output("plan-refresh-tick", "data"),
        #     Output("global-loading", "data"),
        #     Input("btn-plan-save", "n_clicks"),
        #     State("plan-detail-id", "data"),
        #     State("tbl-fw", "data"), State("tbl-hc", "data"),
        #     State("tbl-attr", "data"), State("tbl-shr", "data"),
        #     State("tbl-train", "data"), State("tbl-ratio", "data"),
        #     State("tbl-seat", "data"), State("tbl-bva", "data"),
        #     State("tbl-nh", "data"), State("tbl-emp-roster", "data"),
        #     State("tbl-bulk-files", "data"),
        #     State("tbl-notes", "data"),
        #     State("plan-refresh-tick", "data"),
        #     prevent_initial_call=False
        # )
        # def _save(_n, pid, fw, hc, attr, shr, trn, rat, seat, bva, nh, emp, bulk_files, notes, tick):
        #     if pid:
        #         _verify_storage(pid, when="pre-save")

        #     if ctx.triggered_id != "btn-plan-save":
        #         raise dash.exceptions.PreventUpdate
        #     if not pid:
        #         raise dash.exceptions.PreventUpdate
        #     # role guard: only admin/planner may save
        #     try:
        #         from auth import get_user_role
        #         from plan_detail._common import current_user_fallback
        #         if get_user_role(current_user_fallback()) not in ("admin","planner"):
        #             # message, timer_disabled, refresh_tick (no_update), global-loading off
        #             return "Insufficient permissions to save.", False, no_update, False
        #     except Exception:
        #         pass
        #     _save_table(pid, "fw",         pd.DataFrame(fw or []))
        #     _save_table(pid, "hc",         pd.DataFrame(hc or []))
        #     _save_table(pid, "attr",       pd.DataFrame(attr or []))
        #     _save_table(pid, "shr",        pd.DataFrame(shr or []))
        #     _save_table(pid, "train",      pd.DataFrame(trn or []))
        #     _save_table(pid, "ratio",      pd.DataFrame(rat or []))
        #     _save_table(pid, "seat",       pd.DataFrame(seat or []))
        #     _save_table(pid, "bva",        pd.DataFrame(bva or []))
        #     _save_table(pid, "nh",         pd.DataFrame(nh or []))
        #     _save_table(pid, "emp",        pd.DataFrame(emp or []))
        #     _save_table(pid, "bulk_files", pd.DataFrame(bulk_files or []))
        #     _save_table(pid, "notes",      pd.DataFrame(notes or []))
        #     # Update plan meta so header reflects manual save
        #     try:
        #         from plan_detail._common import save_plan_meta, current_user_fallback
        #         save_plan_meta(pid, {
        #             "last_updated_on": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        #             "last_updated_by": current_user_fallback(),
        #         })
        #     except Exception:
        #         pass
        #     try:
        #         next_tick = int(tick or 0) + 1
        #     except Exception:
        #         next_tick = 1
        #     # Debug (post-save): confirm counts after write
        #     _verify_storage(pid, when="post-save")
        #     return "Saved", False, next_tick, False

        @app.callback(
            Output("plan-msg", "children", allow_duplicate=True),
            Output("plan-msg-timer", "disabled", allow_duplicate=True),
            Output("plan-refresh-tick", "data", allow_duplicate=True),
            Output("global-loading", "data"),
            Input("btn-plan-save", "n_clicks"),
            State("plan-detail-id", "data"),
            State("tbl-fw", "data"), State("tbl-hc", "data"),
            State("tbl-attr", "data"), State("tbl-shr", "data"),
            State("tbl-train", "data"), State("tbl-ratio", "data"),
            State("tbl-seat", "data"), State("tbl-bva", "data"),
            State("tbl-nh", "data"), State("tbl-emp-roster", "data"),
            State("tbl-bulk-files", "data"),
            State("tbl-notes", "data"),
            State("plan-refresh-tick", "data"),
            State("plan-hydrated","data"),
            prevent_initial_call=True,
        )
        def _save(_n, pid, fw, hc, attr, shr, trn, rat, seat, bva, nh, emp, bulk_files, notes, tick, plan_hydrated):
            # Only act on explicit Save button
            if ctx.triggered_id != "btn-plan-save":
                raise dash.exceptions.PreventUpdate
            if not pid:
                raise dash.exceptions.PreventUpdate
            if not plan_hydrated:
                return "Tables still loading - try in a moment.", False, no_update, False

            try:
                from auth import get_user_role
                if get_user_role(current_user_fallback()) not in ("admin", "planner"):
                    return "Insufficient permissions to save.", False, no_update, False
            except Exception:
                pass

            # DEBUG: pre-save snapshot
            _verify_storage(pid, when="pre-save")

            # Coalesce each incoming table with existing persisted data if empty
            fw_df   = _coalesce(pid, "fw",         fw)
            hc_df   = _coalesce(pid, "hc",         hc)
            attr_df = _coalesce(pid, "attr",       attr)
            shr_df  = _coalesce(pid, "shr",        shr)
            trn_df  = _coalesce(pid, "train",      trn)
            rat_df  = _coalesce(pid, "ratio",      rat)
            seat_df = _coalesce(pid, "seat",       seat)
            bva_df  = _coalesce(pid, "bva",        bva)
            nh_df   = _coalesce(pid, "nh",         nh)
            emp_df  = _coalesce(pid, "emp",        emp)
            bf_df   = _coalesce(pid, "bulk_files", bulk_files)
            note_df = _coalesce(pid, "notes",      notes)

            # Persist all (unchanged API)
            _save_table(pid, "fw",         fw_df)
            _save_table(pid, "hc",         hc_df)
            _save_table(pid, "attr",       attr_df)
            _save_table(pid, "shr",        shr_df)
            _save_table(pid, "train",      trn_df)
            _save_table(pid, "ratio",      rat_df)
            _save_table(pid, "seat",       seat_df)
            _save_table(pid, "bva",        bva_df)
            _save_table(pid, "nh",         nh_df)
            _save_table(pid, "emp",        emp_df)
            _save_table(pid, "bulk_files", bf_df)
            _save_table(pid, "notes",      note_df)

            # Update plan meta (unchanged)
            try:
                save_plan_meta(pid, {
                    "last_updated_on": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                    "last_updated_by": current_user_fallback(),
                })
            except Exception:
                pass

            try:
                next_tick = int(tick or 0) + 1
            except Exception:
                next_tick = 1

            # DEBUG: post-save snapshot
            _verify_storage(pid, when="post-save")

            return "Saved", False, next_tick, False

        # Turn on global overlay immediately on Save click
        @app.callback(
            Output("global-loading", "data", allow_duplicate=True),
            Input("btn-plan-save", "n_clicks"),
            prevent_initial_call=True
        )
        def _show_glob_loading(n):
            if not n:
                raise dash.exceptions.PreventUpdate
            return True

        # Upper collapse
        @app.callback(
            Output("plan-upper-collapsed", "data"),
            Output("plan-upper", "style"),
            Output("plan-hdr-collapse", "children"),
            Input("plan-hdr-collapse", "n_clicks"),
            State("plan-upper-collapsed", "data"),
            prevent_initial_call=False
        )
        def _toggle_upper(n_clicks, collapsed):
            collapsed = bool(collapsed)
            if n_clicks:
                collapsed = not collapsed
            style = {"display": "none"} if collapsed else {"display": "block"}
            icon = "▲" if collapsed else "▼"
            return collapsed, style, icon

        # Refresh trigger
        @app.callback(
            Output("plan-msg", "children", allow_duplicate=True),
            Output("plan-msg-timer", "disabled", allow_duplicate=True),
            Output("plan-refresh-tick", "data", allow_duplicate=True),
            Output("plan-loading", "data", allow_duplicate=True), 
            Input("btn-plan-refresh", "n_clicks"),
            State("plan-refresh-tick", "data"),
            prevent_initial_call=True
        )
        def _refresh_msg(_n, tick):
            tick = int(tick or 0) + 1
            return "Refreshed ✅", False, tick, True

        # Clear banner after 5s
        @app.callback(
            Output("plan-msg", "children", allow_duplicate=True),
            Output("plan-msg-timer", "disabled", allow_duplicate=True),
            Input("plan-msg-timer", "n_intervals"),
            prevent_initial_call=True
        )
        def _clear_msg(_ticks):
            return "", True

        # Enable action buttons only when rows selected
        @app.callback(
            Output("btn-emp-tp",    "disabled"),
            Output("btn-emp-loa",   "disabled"),
            Output("btn-emp-back",  "disabled"),
            Output("btn-emp-term",  "disabled"),
            Output("btn-emp-ftp",   "disabled"),
            Output("btn-emp-undo",  "disabled"),
            Output("btn-emp-class", "disabled"),
            Output("btn-emp-remove","disabled"),
            Input("tbl-emp-roster", "data"),
            Input("tbl-emp-roster", "selected_rows"),
            Input("emp-undo", "data"),
            prevent_initial_call=False
        )
        def _toggle_roster_buttons(data, selected_rows, undo_store):
            has_rows = bool(data) and len(data) > 0
            has_sel  = has_rows and bool(selected_rows)
            disabled = not has_sel
            undo_disabled = not(bool(undo_store) and bool((undo_store or {}).get("data")))
            return (disabled, disabled, disabled, disabled, disabled, undo_disabled, disabled, disabled)

        @app.callback(
            Output("tbl-emp-roster", "row_selectable"),
            Output("tbl-emp-roster", "selected_rows"),
            Input("tbl-emp-roster", "data"),
            prevent_initial_call=False
        )
        def _roster_selectability(data):
            has_rows = bool(data) and len(data) > 0
            return ("multi" if has_rows else False, [] if not has_rows else no_update)

        # "+ Add New" modal open/crumb
        @app.callback(
            Output("modal-emp-add", "is_open"),
            Output("modal-roster-crumb", "children"),
            Input("btn-emp-add", "n_clicks"),
            Input("btn-emp-modal-cancel", "n_clicks"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True
        )
        def _modal_toggle(n_add, n_cancel, pid):
            trigger = ctx.triggered_id
            if trigger == "btn-emp-add":
                p = get_plan(pid) or {}
                crumb = _format_crumb(p)
                return True, crumb
            return False, ""

        # Add employee Save
        @app.callback(
            Output("tbl-emp-roster", "data", allow_duplicate=True),
            Output("lbl-emp-total", "children", allow_duplicate=True),
            Output("modal-emp-add", "is_open", allow_duplicate=True),
            Output("plan-msg", "children", allow_duplicate=True),
            Output("plan-msg-timer", "disabled", allow_duplicate=True),
            Output("plan-refresh-tick", "data", allow_duplicate=True),
            Output("emp-undo", "data", allow_duplicate=True),
            Input("btn-emp-modal-save", "n_clicks"),
            State("tbl-emp-roster", "data"),
            State("inp-brid", "value"), State("inp-name", "value"),
            State("inp-ftpt", "value"), State("inp-role", "value"),
            State("inp-prod-date", "date"), State("inp-tl", "value"), State("inp-avp", "value"),
            State("inp-ftpt-hours", "value"),
            State("inp-work-status", "value"), State("inp-current-status", "value"),
            State("inp-term-date", "date"),
            State("plan-detail-id", "data"),
            State("plan-refresh-tick", "data"),
            prevent_initial_call=True
        )
        def _add_emp(_n, data, brid, name, ftpt, role, prod_date, tl, avp, ftpt_hours, work_status, current_status, term_date, pid, plan_refresh_tick):
            data = data or []
            if not brid or any(str(r.get("brid", "")).strip() == str(brid).strip() for r in data):
                return data, f"Total: {len(data):02d} Records", False, "BRID exists or missing", False, no_update, no_update

            p = get_plan(pid) or {}
            r = {cid: "" for cid in _ROSTER_REQUIRED_IDS}
            r.update({
                "brid": brid,
                "name": name or "",
                "ftpt_status": ftpt or "",
                "ftpt_hours": (str(ftpt_hours) if ftpt_hours not in (None, "") else ""),
                "role": role or "Agent",
                "production_start": prod_date or "",
                "team_leader": tl or "",
                "avp": avp or "",
                "work_status": (work_status or "Production"),
                "current_status": (current_status or "Production"),
                "terminate_date": (pd.to_datetime(term_date).date().isoformat() if term_date else ""),
                "biz_area": (p.get("business_area") or p.get("plan_ba") or p.get("vertical") or "").strip(),
                "sub_biz_area": (p.get("sub_business_area") or p.get("plan_sub_ba") or p.get("sub_ba") or p.get("subba") or "").strip(),
                "lob": ((p.get("lob") or p.get("channel") or "").strip().title()),
                "site": (p.get("site") or "").strip(),
            })
            new = data + [r]
            save_df(f"plan_{pid}_emp", pd.DataFrame(new))
            save_plan_meta(pid, {"last_updated_on": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                     "last_updated_by": current_user_fallback()})

            next_tick = int(plan_refresh_tick or 0) + 1
            undo_snap = {"data": (data or []), "ts": pd.Timestamp.utcnow().isoformat(timespec="seconds")}
            return new, f"Total: {len(new):02d} Records", False, "Employee added ✅", False, next_tick, undo_snap

        @app.callback(
            Output("lbl-emp-total", "children"),
            Input("tbl-emp-roster", "data"),
            prevent_initial_call=False
        )
        def _update_emp_total(data):
            df = pd.DataFrame(data or [])
            if "brid" in df.columns:
                s = df["brid"].astype(str).str.strip()
                n = s.replace({"": np.nan, "nan": np.nan}).nunique(dropna=True)
            else:
                n = len(df)
            return f"Total: {int(n):02d} Records"

        # (helpers removed ? bump/snapshot logic folded into primary callbacks)

        # Undo roster change (apply previous snapshot)
        @app.callback(
            Output("tbl-emp-roster", "data", allow_duplicate=True),
            Output("lbl-emp-total", "children", allow_duplicate=True),
            Output("plan-msg", "children", allow_duplicate=True),
            Output("plan-msg-timer", "disabled", allow_duplicate=True),
            Output("emp-undo", "data", allow_duplicate=True),
            Input("btn-emp-undo", "n_clicks"),
            State("emp-undo", "data"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True
        )
        def _undo_roster(n, snap, pid):
            if not n or not snap:
                raise dash.exceptions.PreventUpdate
            prev = (snap or {}).get("data") or []
            save_df(f"plan_{pid}_emp", pd.DataFrame(prev))
            save_plan_meta(pid, {"last_updated_on": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                      "last_updated_by": current_user_fallback()})
            return prev, f"Total: {len(prev):02d} Records", "Undo applied ✅", False, {}

        # Workstatus dataset download
        @app.callback(
            Output("dl-workstatus", "data"),
            Input("btn-emp-dl", "n_clicks"),
            State("tbl-emp-roster", "data"),
            prevent_initial_call=True
        )
        def _download_workstatus(_n, data):
            df = pd.DataFrame(data or [])
            return dcc.send_data_frame(df.to_csv, "workstatus_dataset.csv", index=False)

        # Bulk upload ingest
        @app.callback(
            Output("tbl-bulk-files", "data", allow_duplicate=True),
            Output("tbl-emp-roster", "data", allow_duplicate=True),
            Output("lbl-emp-total", "children", allow_duplicate=True),
            Output("plan-msg", "children", allow_duplicate=True),
            Output("plan-msg-timer", "disabled", allow_duplicate=True),
            Input("up-roster-bulk", "contents"),
            State("up-roster-bulk", "filename"),
            State("tbl-bulk-files", "data"),
            State("tbl-emp-roster", "data"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True
        )
        def _ingest_bulk(contents, filename, files_data, roster_data, pid):
            if not contents:
                raise dash.exceptions.PreventUpdate
            files_data = files_data or []
            roster_data = roster_data or []

            recs_df, ledger = _parse_upload(contents, filename)
            files_data.append(ledger or {"file_name": filename, "ext":"", "size_kb":0, "is_valid":"No", "status":"Invalid"})

            if not recs_df.empty and "brid" in recs_df.columns:
                existing = {str(r.get("brid","")).strip(): i for i, r in enumerate(roster_data)}
                for _, row in recs_df.iterrows():
                    key = str(row.get("brid","")).strip()
                    if not key:
                        continue
                    if key in existing:
                        roster_data[existing[key]].update({k: row.get(k, roster_data[existing[key]].get(k)) for k in _ROSTER_REQUIRED_IDS})
                    else:
                        new_row = {cid: row.get(cid, "") for cid in _ROSTER_REQUIRED_IDS}
                        roster_data.append(new_row)

                save_df(f"plan_{pid}_emp", pd.DataFrame(roster_data))
                save_df(f"plan_{pid}_bulk_files", pd.DataFrame(files_data))
                save_plan_meta(pid, {"last_updated_on": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                     "last_updated_by": current_user_fallback()})


                return (files_data, roster_data, f"Total: {len(roster_data):02d} Records", "Bulk file loaded ✅", False)

            save_df(f"plan_{pid}_bulk_files", pd.DataFrame(files_data))
            save_plan_meta(pid, {"last_updated_on": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                     "last_updated_by": current_user_fallback()})

            return (files_data, roster_data, f"Total: {len(roster_data):02d} Records", "Bulk file invalid", False)

        @app.callback(
            Output("dl-template", "data"),
            Input("btn-template-dl", "n_clicks"),
            prevent_initial_call=True
        )
        def _download_template(_n):
            cols = [c["name"] for c in _roster_columns()]
            # Provide two example rows to guide users
            ex1 = {
                "BRID": "IN0001",
                "Name": "Asha Rao",
                "Class Reference": "NH-2024-11-01",
                "Work Status": "Production",
                "Role": "Agent",
                "FT/PT Status": "Full-time",
                "FT/PT Hours": 40,
                "Current Status": "Production",
                "Training Start": "2024-11-01",
                "Training End": "2024-11-07",
                "Nesting Start": "2024-11-08",
                "Nesting End": "2024-11-15",
                "Production Start": "2024-11-18",
                "Terminate Date": "",
                "Team Leader": "Priyanka Menon",
                "AVP": "Anil Sharma",
                "Business Area": "Bereavement and Delegated Authority",
                "Sub Business Area": "Bereavement",
                "LOB": "Back Office",
                "LOA Date": "",
                "Back from LOA Date": "",
                "Site": "Candor TechSpace, Noida",
            }
            ex2 = {
                "BRID": "UK0002",
                "Name": "Alex Doe",
                "Class Reference": "",
                "Work Status": "Production",
                "Role": "CSA",
                "FT/PT Status": "Part-time",
                "FT/PT Hours": 20,
                "Current Status": "Production",
                "Training Start": "",
                "Training End": "",
                "Nesting Start": "",
                "Nesting End": "",
                "Production Start": "2023-06-01",
                "Terminate Date": "2024-09-30",
                "Team Leader": "Chris Lee",
                "AVP": "Samantha Jones",
                "Business Area": "Retail",
                "Sub Business Area": "Cards",
                "LOB": "Voice",
                "LOA Date": "",
                "Back from LOA Date": "",
                "Site": "Manchester, 4 Piccadilly Place",
            }
            rows = []
            for ex in (ex1, ex2):
                # ensure all expected columns exist (blank when not provided)
                r = {c: "" for c in cols}
                r.update(ex)
                rows.append(r)
            df = pd.DataFrame(rows, columns=cols)
            return dcc.send_data_frame(df.to_csv, "employee_roster_template.csv", index=False)

        # Notes save
        @app.callback(
            Output("tbl-notes", "data", allow_duplicate=True),
            Output("plan-msg", "children", allow_duplicate=True),
            Output("plan-msg-timer", "disabled", allow_duplicate=True),
            Input("btn-note-save", "n_clicks"),
            State("notes-input", "value"),
            State("tbl-notes", "data"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True
        )
        def _save_note(_n, text, data, pid):
            if not (text and text.strip()):
                raise dash.exceptions.PreventUpdate
            data = data or []
            stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
            user = (current_user_fallback() or "User")
            row = {"when": stamp, "user": str(user), "note": text.strip()}
            data = [row] + data
            save_df(f"plan_{pid}_notes", pd.DataFrame(data))
            save_plan_meta(pid, {"last_updated_on": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                     "last_updated_by": current_user_fallback()})

            return data, "Note saved ✅", False
    
        @app.callback(
            Output("plan-loading", "data"),
            Input("plan-detail-id", "data"),
            prevent_initial_call=False
        )
        def _start_page_loading(_pid):
            # Show overlay as soon as we land on /plan/<id>
            return True
    
        @app.callback(
            Output("global-loading", "data", allow_duplicate=True),
            Input("plan-loading", "data"),
            prevent_initial_call=True
        )
        def _sync_global_loading(is_loading):
            return bool(is_loading)

        @app.callback(
            Output("tbl-hc","data", allow_duplicate=True),
            Output("tbl-hc","columns", allow_duplicate=True),
            Input("set-scope","value"),
            Input("set-ba","value"), Input("set-subba","value"), Input("set-lob","value"),
            State("plan-weeks","data"),
            State("tbl-emp-roster","data"),
            prevent_initial_call=True
        )
        def hc_tab_data(scope, ba, sba, lob, week_ids, roster_rows):
            from dash import no_update
            import pandas as pd

            if scope != "hier" or not (ba and sba and lob):
                return [], no_update

            week_ids = list(week_ids or [])
            metrics = ["Budget HC (#)","Planned HC (#)","Actual Agent HC (#)","SME Billable HC (#)","Variance (#)"]
            df = _blank_grid(metrics, week_ids)  # your existing helper

            # tiny local helpers
            def _set(metric, w, val): df.loc[df["metric"]==metric, w] = float(val or 0)
            def _get(metric, w):
                ser = df.loc[df["metric"]==metric, w]
                return float(ser.iloc[0]) if len(ser) and pd.notna(ser.iloc[0]) else 0.0

            # ---- Budget / Planned from saved timeseries (unchanged) ----
            key = _scope_key(ba, sba, lob)
            bud = load_timeseries("hc_budget",  key) or pd.DataFrame(columns=["week","headcount"])
            pla = load_timeseries("hc_planned", key) or pd.DataFrame(columns=["week","headcount"])

            if isinstance(bud, pd.DataFrame) and not bud.empty:
                for _, r in bud.iterrows():
                    w = str(r.get("week",""))
                    if w in week_ids:
                        _set("Budget HC (#)", w, r.get("headcount", 0))

            if isinstance(pla, pd.DataFrame) and not pla.empty:
                for _, r in pla.iterrows():
                    w = str(r.get("week",""))
                    if w in week_ids:
                        _set("Planned HC (#)", w, r.get("headcount", 0))

            for w in week_ids:  # Planned fallback to Budget
                if _get("Planned HC (#)", w) == 0:
                    _set("Planned HC (#)", w, _get("Budget HC (#)", w))

            # ---- Actuals from Employee Roster sub-tab (what you asked) ----
            r = pd.DataFrame(roster_rows or [])
            if not r.empty:
                L = {str(c).strip().lower(): c for c in r.columns}
                # core columns (be liberal with names)
                brid_c = L.get("brid") or L.get("employee id") or L.get("employee_id")
                role_c = L.get("role") or L.get("position group") or L.get("position description")
                ba_c   = L.get("business area") or L.get("ba") or L.get("vertical")
                sba_c  = L.get("sub business area") or L.get("level 3") or L.get("level_3") or L.get("subba") or L.get("sub_ba")
                lob_c  = L.get("lob") or L.get("channel") or L.get("program")
                cur_c  = L.get("current status") or L.get("current_status") or L.get("status")
                work_c = L.get("work status") or L.get("work_status")

                if brid_c and role_c:
                    r[brid_c] = r[brid_c].astype(str).str.strip()

                    # scope filter (only apply if the columns exist)
                    def _match(col, val):
                        if not col or col not in r.columns: return True
                        return r[col].astype(str).str.strip().str.lower() == (val or "").strip().lower()
                    r = r[_match(ba_c, ba) & _match(sba_c, sba) & _match(lob_c, lob)]

                    # effective status: Current Status else Work Status
                    if cur_c and cur_c in r.columns:
                        eff = r[cur_c].astype(str)
                        if work_c and work_c in r.columns:
                            eff = eff.where(eff.str.strip()!="", r[work_c].astype(str))
                    else:
                        eff = r[work_c].astype(str) if work_c and work_c in r.columns else ""

                    eff = eff.str.strip().str.lower()
                    is_prod = eff.eq("production")

                    role_txt = r[role_c].astype(str).str.strip().str.lower()
                    is_agent = role_txt.str.contains(r"\bagent\b")
                    is_sme   = role_txt.str.contains(r"\bsme\b")

                    # distinct BRIDs
                    agent_cnt = (
                        r.loc[is_prod & is_agent, brid_c]
                        .dropna().astype(str).str.strip().nunique()
                    )
                    sme_cnt = (
                        r.loc[is_prod & is_sme, brid_c]
                        .dropna().astype(str).str.strip().nunique()
                    )

                    for w in week_ids:
                        _set("Actual Agent HC (#)", w, agent_cnt)
                        _set("SME Billable HC (#)", w, sme_cnt)

            # ---- Variance = Actual Agent - Budget (leave this as-is unless you want to include SMEs) ----
            for w in week_ids:
                _set("Variance (#)", w, _get("Actual Agent HC (#)", w) - _get("Budget HC (#)", w))

            # finalize types/columns
            for w in week_ids:
                df[w] = pd.to_numeric(df[w], errors="coerce").fillna(0).round(0).astype(int)
            cols = [{"name":"Metric","id":"metric","editable":False}] + [{"name": w, "id": w} for w in week_ids]
            return df.to_dict("records"), cols

        # ------------ New Hire: modal toggle ------------
        @app.callback(
            Output("nh-modal", "is_open", allow_duplicate=True),
            Output("nh-form-error", "children"),
            Output("nh-form-class-ref", "value"),
            Output("nh-form-source-id", "value"),
            Input("btn-nh-add", "n_clicks"),
            Input("btn-nh-cancel", "n_clicks"),
            Input("btn-nh-save", "n_clicks"),
            State("nh-modal", "is_open"),
            State("plan-detail-id", "data"),   # <-- get PID here
            prevent_initial_call=True
        )
        def nh_modal_toggle(n_add, n_cancel, n_save, is_open, pid):
            trig = ctx.triggered_id
            if trig == "btn-nh-add":
                pid = str(pid or "")
                key = f"plan_{pid}_nh_classes"
                df = load_df(key)
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame()
                # next_class_reference + current_user_fallback come from _common (wildcard import)
                return True, "", next_class_reference(pid, df), current_user_fallback()
            if trig == "btn-nh-cancel":
                return False, "", no_update, no_update
            return is_open, "", no_update, no_update

        # ------------ New Hire: save / upsert class ------------
        @app.callback(
            Output("store-nh-classes", "data", allow_duplicate=True),
            Output("tbl-nh-recent", "data", allow_duplicate=True),
            Output("tbl-nh-recent", "columns", allow_duplicate=True),
            Output("nh-msg", "children", allow_duplicate=True),
            Output("nh-modal", "is_open", allow_duplicate=True),
            Output("plan-refresh-tick", "data", allow_duplicate=True),   # NEW
            Input("btn-nh-save", "n_clicks"),
            State("nh-form-class-ref", "value"),
            State("nh-form-source-id", "value"),
            State("nh-form-emp-type", "value"),
            State("nh-form-status", "value"),
            State("nh-form-class-type", "value"),
            State("nh-form-class-level", "value"),
            State("nh-form-grads", "value"),
            State("nh-form-billable", "value"),
            State("nh-form-train-weeks", "value"),
            State("nh-form-nest-weeks", "value"),
            State("nh-form-date-induction", "date"),
            State("nh-form-date-training", "date"),
            State("nh-form-date-nesting", "date"),
            State("nh-form-date-production", "date"),
            State("plan-detail-id", "data"),
            State("plan-refresh-tick", "data"),                           # NEW
            prevent_initial_call=True
        )
        def nh_save(_, class_ref, source_id, emp_type, status, class_type, class_level,
                    grads_needed, billable_hc, train_weeks, nest_weeks,
                    dt_induction, dt_training, dt_nesting, dt_production,
                    pid, tick):

            if not class_ref or not dt_training:
                return no_update, no_update, no_update, "Class Reference and Training Starts On are required.", True

            pid = str(pid or "")
            key = f"plan_{pid}_nh_classes"
            df = load_df(key)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame()

            # dates ? ISO strings
            ts, te, ns, ne, ps = _auto_dates(
                training_start=dt_training,
                training_weeks=train_weeks,
                nesting_start=dt_nesting,
                nesting_weeks=nest_weeks,
                production_start=dt_production
            )

            rec = {
                "class_reference": str(class_ref).strip(),
                "source_system_id": (source_id or current_user_fallback()),
                "emp_type": (emp_type or "full-time"),
                "status": (status or "tentative"),
                "class_type": class_type,
                "class_level": class_level,
                "grads_needed": int(grads_needed or 0),
                "billable_hc": int(billable_hc or 0),
                "training_weeks": int(train_weeks or 0),
                "nesting_weeks": int(nest_weeks or 0),
                "induction_start": dt_induction,
                "training_start": ts,
                "training_end": te,
                "nesting_start": ns,
                "nesting_end": ne,
                "production_start": ps,
                "created_by": current_user_fallback(),
                "created_ts": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
            }

            key_lower = rec["class_reference"].lower()
            if "class_reference" in df.columns:
                mask = df["class_reference"].astype(str).str.strip().str.lower().eq(key_lower)
            else:
                mask = pd.Series(False, index=df.index)

            if mask.any():
                for k, v in rec.items():
                    df.loc[mask, k] = v
            else:
                df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)

            save_df(key, df)  # single source of truth
            save_plan_meta(pid, {"last_updated_on": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                     "last_updated_by": current_user_fallback()})

            n_updated = _update_roster_by_class(pid, rec["class_reference"], ts, te, ns, ne, ps)

            preview = df.sort_values("created_ts", ascending=False).head(5)
            cols = [{"name": c.replace("_", " ").title(), "id": c} for c in preview.columns]
            msg = f"Saved class '{rec['class_reference']}'. Roster rows updated: {n_updated}."
            next_tick = int(tick or 0) + 1
            return df.to_dict("records"), preview.to_dict("records"), cols, msg, False, next_tick

        # ------------ New Hire: page seed / refresh ------------
        @app.callback(
            Output("tbl-nh-recent", "data", allow_duplicate=True),
            Output("tbl-nh-recent", "columns", allow_duplicate=True),
            Output("store-nh-classes", "data", allow_duplicate=True),
            Input("plan-refresh-tick", "data"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True
        )
        def nh_seed(_tick, pid):
            pid = str(pid or "")
            key = f"plan_{pid}_nh_classes"
            df = load_df(key)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame()
            preview = df.copy()
            if "created_ts" in preview.columns:
                preview = preview.sort_values("created_ts", ascending=False)
            preview = preview.head(5)
            cols = [{"name": c.replace("_", " ").title(), "id": c} for c in preview.columns]
            return preview.to_dict("records"), cols, df.to_dict("records")

        # ------------ New Hire: confirm selected classes ------------
        @app.callback(
            Output("tbl-nh-details", "data", allow_duplicate=True),
            Output("store-nh-classes", "data", allow_duplicate=True),
            Output("nh-msg", "children", allow_duplicate=True),
            Output("nh-details-modal", "is_open", allow_duplicate=True),
            Input("btn-nh-confirm", "n_clicks"),
            State("tbl-nh-details", "data"),
            State("tbl-nh-details", "selected_rows"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True
        )
        def _nh_confirm(n, details_rows, selected_rows, pid):
            if not n or not details_rows or not selected_rows:
                raise dash.exceptions.PreventUpdate

            pid = str(pid or "")
            key = f"plan_{pid}_nh_classes"

            # Load full dataset
            all_df = load_df(key)
            if not isinstance(all_df, pd.DataFrame):
                all_df = pd.DataFrame()

            # Find class_reference column in both tables
            det = pd.DataFrame(details_rows or [])
            det_low = {c.lower(): c for c in det.columns}
            all_low = {c.lower(): c for c in all_df.columns}

            det_cref = det_low.get("class_reference") or det_low.get("class reference")
            all_cref = all_low.get("class_reference") or all_low.get("class reference")

            if not (det_cref and all_cref):
                raise dash.exceptions.PreventUpdate

            # Build selection set
            chosen = []
            for i in selected_rows:
                if 0 <= i < len(det):
                    v = det.at[i, det_cref]
                    if pd.notna(v) and str(v).strip():
                        chosen.append(str(v).strip().lower())
            if not chosen:
                raise dash.exceptions.PreventUpdate

            # Update status to Confirmed for selected classes
            all_df[all_cref] = all_df[all_cref].astype(str)
            mask = all_df[all_cref].str.strip().str.lower().isin(chosen)
            n_upd = int(mask.sum())
            if n_upd > 0:
                all_df.loc[mask, "status"] = "confirmed"
                save_df(key, all_df)
                save_plan_meta(pid, {"last_updated_on": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                     "last_updated_by": current_user_fallback()})

            # Refresh the details preview (top 5 recent)
            preview = all_df.copy()
            if "created_ts" in preview.columns:
                preview = preview.sort_values("created_ts", ascending=False)
            preview = preview.head(5)

            msg = f"Marked {n_upd} class(es) as Confirmed ✅"
            return preview.to_dict("records"), all_df.to_dict("records"), msg, False


        # Enable/disable the Confirm button based on selection
        @app.callback(
            Output("btn-nh-confirm", "disabled"),
            Input("tbl-nh-details", "selected_rows"),
            Input("tbl-nh-details", "data"),
            prevent_initial_call=False
        )
        def _toggle_nh_confirm(sel, data):
            has = bool(data) and bool(sel)
            return not has  # disabled when nothing selected


        # Confirm the selected NH classes
        @app.callback(
            Output("tbl-nh-details", "data"),
            Output("tbl-nh-recent", "data"),
            Output("tbl-nh-recent", "columns"),
            Output("store-nh-classes", "data"),
            Output("nh-msg", "children"),
            Input("btn-nh-confirm", "n_clicks"),
            State("tbl-nh-details", "data"),
            State("tbl-nh-details", "selected_rows"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True
        )
        def _nh_confirm_selected(_n, data, selected_rows, pid):
            if not _n or not data or not selected_rows:
                raise dash.exceptions.PreventUpdate

            pid = str(pid or "")
            key = f"plan_{pid}_nh_classes"

            # Load full dataset, upsert statuses for selected class_references
            df_full = load_df(key)
            if not isinstance(df_full, pd.DataFrame):
                df_full = pd.DataFrame()

            selected = pd.DataFrame(data).iloc[selected_rows]
            to_confirm = set(selected["class_reference"].astype(str))

            if not to_confirm:
                raise dash.exceptions.PreventUpdate

            if "class_reference" in df_full.columns:
                mask = df_full["class_reference"].astype(str).isin(to_confirm)
                df_full.loc[mask, "status"] = "confirmed"

                save_df(key, df_full)
                save_plan_meta(pid, {"last_updated_on": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                     "last_updated_by": current_user_fallback()})
                # refresh details table + recent preview
                details = df_full.sort_values("created_ts", ascending=False)
                cols = [{"name": c.replace("_", " ").title(), "id": c} for c in details.columns]
                preview = details.head(5)

                msg = f"Confirmed {mask.sum()} class(es) ✅"
                return (
                    details.to_dict("records"),
                    preview.to_dict("records"),
                    cols,
                    df_full.to_dict("records"),
                    msg
                )

            # nothing to match
            raise dash.exceptions.PreventUpdate


        # ------------ New Hire: Training Details modal ------------
        @app.callback(
            Output("nh-details-modal", "is_open"),
            Output("tbl-nh-details", "data", allow_duplicate=True),
            Output("tbl-nh-details", "columns"),
            Output("tbl-nh-details", "row_selectable"),
            Output("tbl-nh-details", "selected_rows"),
            Input("btn-nh-details", "n_clicks"),
            Input("btn-nh-details-close", "n_clicks"),
            State("plan-detail-id", "data"),
            State("nh-details-modal", "is_open"),
            prevent_initial_call=True
        )
        def nh_details(n_open, n_close, pid, is_open):
            t = ctx.triggered_id
            if t == "btn-nh-details-close":
                return False, dash.no_update, dash.no_update, dash.no_update, dash.no_update

            key = f"plan_{pid}_nh_classes"
            df = load_df(key)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame()

            cols_out = [{"name": c.replace("_"," ").title(), "id": c} for c in df.columns]
            # ? enable selection
            selectable = "multi" if not df.empty else False
            return True, df.to_dict("records"), cols_out, selectable, []


        # ------------ New Hire: download ------------
        @app.callback(
            Output("dl-nh-dataset", "data"),
            Input("btn-nh-dl", "n_clicks"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True
        )
        def nh_download(_n, pid):
            key = f"plan_{pid}_nh_classes"
            df = load_df(key)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame()
            return dcc.send_data_frame(df.to_csv, f"new_hire_classes_{pid}.csv", index=False)

        # ---- Options panel open/close + header info ---------------------------------
        @app.callback(
            Output("plan-opt-canvas", "is_open", allow_duplicate=True),
            Input("plan-opt-toggle", "n_clicks"),
            State("plan-opt-canvas", "is_open"),
            prevent_initial_call=True
        )
        def _toggle_opts(n, is_open):
            if not n:
                raise dash.exceptions.PreventUpdate
            return not bool(is_open)

        @app.callback(
            Output("plan-opt-canvas", "is_open"),
            Output("opt-plan-type", "children"),
            Output("opt-plan-start", "children"),
            Output("opt-plan-end", "children"),
            Input("plan-opt-toggle", "n_clicks"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True
        )
        def _opt_toggle(_n, pid):
            p = get_plan(pid) or {}
            return True, (p.get("plan_type") or "Volume Based"), (p.get("start_week") or ""), (p.get("end_week") or "")
        
        # --- Options panel header (Type / Start / End) ---
        @app.callback(
            Output("opt-plan-type", "children", allow_duplicate=True),
            Output("opt-plan-start", "children", allow_duplicate=True),
            Output("opt-plan-end", "children", allow_duplicate=True),
            Input("plan-detail-id", "data"),
            prevent_initial_call=True
        )
        def _opt_fill(pid):
            p = get_plan(pid) or {}
            pt = p.get("plan_type") or "Volume Based"
            sw = p.get("start_week") or ""
            ew = p.get("end_week") or ""
            # normalize to Monday YYYY-MM-DD if possible
            try: sw = _monday(sw).isoformat() if sw else ""
            except Exception: pass
            try: ew = _monday(ew).isoformat() if ew else ""
            except Exception: pass
            return pt, (sw or ''), (ew or '')


        # View-between-weeks modal
        @app.callback(
            Output("opt-view-modal", "is_open"),
            Input("opt-view", "n_clicks"),
            Input("opt-view-cancel", "n_clicks"),
            prevent_initial_call=True
        )
        def _view_modal(n1, n2):
            t = ctx.triggered_id
            return True if t == "opt-view" else False

        # Apply view-between-weeks: re-generate all columns for that range and trigger refresh
        @app.callback(
            Output("tbl-fw", "columns", allow_duplicate=True),
            Output("tbl-hc", "columns", allow_duplicate=True),
            Output("tbl-attr", "columns", allow_duplicate=True),
            Output("tbl-shr", "columns", allow_duplicate=True),
            Output("tbl-train", "columns", allow_duplicate=True),
            Output("tbl-ratio", "columns", allow_duplicate=True),
            Output("tbl-seat", "columns", allow_duplicate=True),
            Output("tbl-bva", "columns", allow_duplicate=True),
            Output("tbl-nh", "columns", allow_duplicate=True),
            Output("plan-refresh-tick", "data", allow_duplicate=True),
            Input("opt-view-save", "n_clicks"),
            State("opt-view-from", "date"),
            State("opt-view-to", "date"),
            prevent_initial_call=True
        )
        def _view_apply(_n, dfrom, dto):
            if not (dfrom and dto):
                raise dash.exceptions.PreventUpdate
            from ._common import _week_span, _week_cols
            span = _week_span(dfrom, dto)
            cols, _ = _week_cols(span)
            tick = int(pd.Timestamp.utcnow().timestamp())  # cheap trigger
            return (cols,)*9 + (tick,)

        # Save As (simple version: duplicates current plan rows into a new plan entry)
        @app.callback(
            Output("opt-saveas-modal", "is_open"),
            Input("opt-saveas", "n_clicks"),
            Input("opt-saveas-cancel", "n_clicks"),
            prevent_initial_call=True
        )
        def _saveas_open(n1, n2):
            t = ctx.triggered_id
            return True if t == "opt-saveas" else False

        @app.callback(
            Output("opt-saveas-msg", "children"),
            Input("opt-saveas-save", "n_clicks"),
            State("opt-saveas-name", "value"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True
        )
        def _saveas_do(_n, name, pid):
            if not name:
                return "Enter a name."
            # minimal versioning " copy plan meta + all plan_{pid}_* tables to a new pseudo-id
            new_pid = clone_plan(pid, name)
            return f"Saved as '{name}' (id={new_pid})."

        # Export " one Excel with all lower-tab sheets
        @app.callback(
            Output("dl-plan-export", "data"),
            Input("opt-export", "n_clicks"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True
        )
        def _export_plan(_n, pid):
            import io
            import pandas as pd
            buf = io.BytesIO()
            upper_df = None
            try:
                meta = get_plan(pid) or {}
                plan_type = meta.get("plan_type") or "Volume Based"
                fw_df = load_df(f"plan_{pid}_fw")
                fw_cols = [{"name": "Metric", "id": "metric"}]
                if isinstance(fw_df, pd.DataFrame) and not fw_df.empty:
                    for col in fw_df.columns:
                        if col == "metric":
                            continue
                        fw_cols.append({"name": str(col), "id": col})
                upper, *_ = _fill_tables_fixed(plan_type, pid, fw_cols, 0, whatif={}, grain='week')
                upper_data = getattr(upper, "data", None)
                if isinstance(upper_data, list) and upper_data:
                    upper_df = pd.DataFrame(upper_data)
            except Exception:
                upper_df = None

            keys = ["fw","hc","attr","shr","train","ratio","seat","bva","nh","emp","bulk_files","notes"]
            with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
                if isinstance(upper_df, pd.DataFrame) and not upper_df.empty:
                    upper_df.to_excel(xw, sheet_name="upper", index=False)
                for k in keys:
                    try:
                        df = load_df(f"plan_{pid}_{k}")
                    except Exception:
                        df = pd.DataFrame()
                    (df if isinstance(df, pd.DataFrame) else pd.DataFrame()).to_excel(xw, sheet_name=k[:31], index=False)
            buf.seek(0)
            return dcc.send_bytes(buf.read(), f"plan_{pid}_export.xlsx")

        # Extend plan end-week by N weeks
        @app.callback(
            Output("opt-extend-modal", "is_open"),
            Input("opt-extend", "n_clicks"),
            Input("opt-extend-cancel", "n_clicks"),
            prevent_initial_call=True
        )
        def _extend_open(n1, n2):
            t = ctx.triggered_id
            return True if t == "opt-extend" else False

        @app.callback(
            Output("opt-extend-msg", "children"),
            Output("plan-refresh-tick", "data", allow_duplicate=True),
            Input("opt-extend-save", "n_clicks"),
            State("opt-extend-weeks", "value"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True
        )
        def _extend_do(_n, add_weeks, pid):
            if not int(add_weeks or 0):
                return "Enter weeks > 0.", no_update
            from plan_store import extend_plan_weeks
            extend_plan_weeks(pid, int(add_weeks))
            return f"Extended by {int(add_weeks)} week(s).", int(pd.Timestamp.utcnow().timestamp())

        # Switch / Compare shells (wired to open/close)

        @app.callback(
            Output("opt-switch-modal", "is_open", allow_duplicate=True),
            Input("opt-switch", "n_clicks"),
            Input("opt-switch-close", "n_clicks"),
            prevent_initial_call=True,
        )
        def _switch_open(n1, n2):
            return True if ctx.triggered_id == "opt-switch" else False

        @app.callback(
            Output("opt-switch-select", "options", allow_duplicate=True),
            Output("opt-switch-select", "value", allow_duplicate=True),
            Output("opt-switch-warning", "children",allow_duplicate=True),
            Input("opt-switch-modal", "is_open"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True,
        )
        def _switch_options(is_open, current_pid):
            if not is_open:
                raise dash.exceptions.PreventUpdate
            try:
                current_pid = int(current_pid) if current_pid is not None else None
            except Exception:
                current_pid = None

            def _norm(val):
                return str(val).strip().lower() if val not in (None, '') else ''

            def _channels(val):
                if val is None:
                    return tuple()
                if isinstance(val, (list, tuple)):
                    parts = [str(v).strip().lower() for v in val if str(v).strip()]
                else:
                    parts = [p.strip().lower() for p in str(val).split(',') if p.strip()]
                return tuple(sorted(set(parts)))

            current = get_plan(current_pid) or {}
            scope = {
                'vertical': _norm(current.get('vertical')),
                'sub_ba': _norm(current.get('sub_ba')),
                'location': _norm(current.get('location')),
                'site': _norm(current.get('site')),
                'channel': _channels(current.get('channel')),
            }

            lookup = {}
            try:
                for plan in list_plans(include_deleted=False):
                    try:
                        pid = int(plan.get('id'))
                    except Exception:
                        continue
                    lookup.setdefault(pid, {}).update(plan)
            except Exception:
                pass
            try:
                df = _load_index()
                if isinstance(df, pd.DataFrame) and not df.empty and 'plan_id' in df.columns:
                    meta = df.copy()
                    meta['plan_id'] = pd.to_numeric(meta['plan_id'], errors='coerce')
                    meta = meta.dropna(subset=['plan_id'])
                    for _, row in meta.iterrows():
                        pid = int(row['plan_id'])
                        lookup.setdefault(pid, {}).update(dict(row))
            except Exception:
                pass

            options = []
            for pid, info in lookup.items():
                if pid == current_pid:
                    continue
                if _norm(info.get('vertical')) != scope['vertical']:
                    continue
                if _norm(info.get('sub_ba')) != scope['sub_ba']:
                    continue
                if _norm(info.get('location')) != scope['location']:
                    continue
                if _norm(info.get('site')) != scope['site']:
                    continue
                if _channels(info.get('channel')) != scope['channel']:
                    continue
                name = info.get('plan_name') or info.get('name') or f"Plan {pid}"
                status = info.get('status') or ('current' if info.get('is_current') else 'history')
                label = f"{name} (id={pid}, status={status})"
                options.append({'label': label, 'value': pid})

            if not options:
                return [], None, "No other plans with the same scope were found."

            options = sorted(options, key=lambda o: o['label'].lower())
            first_value = options[0]['value']
            return options, first_value, ''

        @app.callback(
            Output("opt-switch-apply", "disabled", allow_duplicate=True),
            Input("opt-switch-select", "value"),
            prevent_initial_call=True,
        )
        def _switch_toggle(value):
            return not bool(value)

        @app.callback(
            Output("plan-detail-id", "data", allow_duplicate=True),
            Output("plan-refresh-tick", "data", allow_duplicate=True),
            Output("opt-switch-modal", "is_open", allow_duplicate=True),
            Output("plan-opt-canvas", "is_open",allow_duplicate=True),
            Output("opt-switch-warning", "children", allow_duplicate=True),
            Input("opt-switch-apply", "n_clicks"),
            State("opt-switch-select", "value"),
            prevent_initial_call=True,
        )
        def _switch_apply(n_clicks, selected_pid):
            if not n_clicks or not selected_pid:
                raise dash.exceptions.PreventUpdate
            try:
                new_pid = int(selected_pid)
            except Exception:
                return no_update, no_update, False, no_update, "Invalid plan selection."
            try:
                tick = int(pd.Timestamp.utcnow().timestamp())
            except Exception:
                tick = 0
            return new_pid, tick, False, False, ''



        @app.callback(Output("opt-compare-modal","is_open"),
                    Input("opt-compare","n_clicks"), Input("opt-compare-close","n_clicks"),
                    prevent_initial_call=True)
        def _compare_open(n1,n2): return True if ctx.triggered_id=="opt-compare" else False

        @app.callback(
            Output("opt-compare-body", "children"),
            Input("opt-compare-modal", "is_open"),
            State("plan-detail-id", "data"),
            State("plan-refresh-tick", "data"),
            prevent_initial_call=True,
        )
        def _compare_body(is_open, pid, _tick):
            if not is_open:
                raise dash.exceptions.PreventUpdate
            try:
                current = get_plan(pid) or {}
            except Exception:
                current = {}
            if not current:
                return html.Div("Plan details not available for comparison.", className="text-muted")

            def _norm_text(val):
                return (str(val).strip().lower() if val not in (None, "") else "")

            def _norm_channels(val):
                if val is None:
                    return tuple()
                if isinstance(val, (list, tuple)):
                    parts = [str(v).strip().lower() for v in val if str(v).strip()]
                else:
                    parts = [p.strip().lower() for p in str(val).split(",") if p.strip()]
                return tuple(sorted(set(parts)))

            scope = {
                "vertical": _norm_text(current.get("vertical")),
                "sub_ba": _norm_text(current.get("sub_ba")),
                "location": _norm_text(current.get("location")),
                "site": _norm_text(current.get("site")),
                "channel": _norm_channels(current.get("channel")),
            }

            candidates = {}
            try:
                for plan in list_plans(include_deleted=False):
                    candidates[int(plan.get("id"))] = plan
            except Exception:
                pass
            try:
                df = _load_index()
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df = df.copy()
                    if "plan_id" in df.columns:
                        df["plan_id"] = pd.to_numeric(df["plan_id"], errors="coerce")
                        df = df.dropna(subset=["plan_id"])
                        for _, row in df.iterrows():
                            candidates[int(row["plan_id"])] = dict(row)
            except Exception:
                pass

            options = []
            for cand_id, row in candidates.items():
                if cand_id == int(pid):
                    continue
                if _norm_text(row.get("vertical")) != scope["vertical"]:
                    continue
                if _norm_text(row.get("sub_ba")) != scope["sub_ba"]:
                    continue
                if _norm_text(row.get("location")) != scope["location"]:
                    continue
                if _norm_text(row.get("site")) != scope["site"]:
                    continue
                if _norm_channels(row.get("channel")) != scope["channel"]:
                    continue
                name = row.get("plan_name") or row.get("name") or "Unnamed"
                status = row.get("status") or ("current" if row.get("is_current") else "history")
                label = f"{name} (id={cand_id}, status={status})"
                options.append({"label": label, "value": cand_id})

            options = sorted(options, key=lambda o: o["label"].lower())
            dropdown = dcc.Dropdown(
                id="opt-compare-select",
                options=options,
                placeholder="Choose a plan",
                className="mb-2",
                disabled=not options,
            )
            compare_btn = dbc.Button(
                "Compare",
                id="opt-compare-run",
                color="primary",
                disabled=True,
                className="mb-2",
            )
            notice = html.Div(
                "No other plans with the same scope were found.",
                className="text-muted",
            ) if not options else html.P(
                "Select a plan to compare with the current one.",
                className="mb-2"
            )
            warning = html.Div(id="opt-compare-warning", className="text-danger small mb-2")
            return html.Div([
                notice,
                dropdown,
                warning,
                compare_btn,
            ])

        @app.callback(
            Output("opt-compare-run", "disabled"),
            Input("opt-compare-select", "value"),
            Input("opt-compare-modal", "is_open"),
            prevent_initial_call=True,
        )
        def _compare_toggle(value, is_open):
            if not is_open:
                raise dash.exceptions.PreventUpdate
            return not bool(value)

        @app.callback(
            Output("opt-compare-result", "children", allow_duplicate=True),
            Output("opt-compare-result-modal", "is_open", allow_duplicate=True),
            Output("opt-compare-modal", "is_open", allow_duplicate=True),
            Output("plan-opt-canvas", "is_open",allow_duplicate=True),
            Output("opt-compare-warning", "children", allow_duplicate=True),
            Input("opt-compare-run", "n_clicks"),
            State("opt-compare-select", "value"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True,
        )
        def _compare_result(n_clicks, compare_id, current_pid):
            if not n_clicks:
                raise dash.exceptions.PreventUpdate
            if not compare_id:
                return no_update, no_update, True, no_update, "Select a plan to compare."
            try:
                compare_pid = int(compare_id)
                current_pid = int(current_pid)
            except Exception:
                return no_update, no_update, True, no_update, "Invalid plan selection."

            def _collect_upper(pid):
                meta = get_plan(pid) or {}
                plan_type = meta.get("plan_type") or "Volume Based"
                fw = load_df(f"plan_{pid}_fw")
                if isinstance(fw, pd.DataFrame) and not fw.empty:
                    cols = [c for c in fw.columns if c != "metric"]
                else:
                    cols = _week_span(meta.get("start_week"), meta.get("end_week"))
                fw_cols = [{"name": "Metric", "id": "metric"}] + [{"name": c, "id": c} for c in cols]
                upper, *_ = _fill_tables_fixed(plan_type, pid, fw_cols, 0, whatif={}, grain='week')
                df = pd.DataFrame(upper.data)
                return df.set_index('metric') if not df.empty else pd.DataFrame()

            curr_df = _collect_upper(current_pid)
            comp_df = _collect_upper(compare_pid)
            if curr_df.empty or comp_df.empty:
                return html.Div("Missing summary data for comparison.", className="text-warning"), True, False, False, ""

            merged_cols = sorted(set(curr_df.columns) | set(comp_df.columns))
            curr_df = curr_df.reindex(columns=merged_cols).fillna('')
            comp_df = comp_df.reindex(index=curr_df.index, columns=merged_cols).fillna('')
            diff_df = pd.DataFrame(index=curr_df.index)
            for col in merged_cols:
                curr_vals = pd.to_numeric(curr_df[col], errors='coerce')
                comp_vals = pd.to_numeric(comp_df[col], errors='coerce')
                diff = curr_vals - comp_vals
                diff_df[col] = diff.where(curr_vals.notna() & comp_vals.notna(), '')

            def _table(title, df):
                records = df.reset_index().to_dict("records")
                return html.Div([
                    html.H6(title),
                    dash_table.DataTable(
                        columns=[{"name": "Metric", "id": "metric"}] + [{"name": c, "id": c} for c in df.columns],
                        data=records,
                        style_as_list_view=True,
                        page_size=10,
                    )
                ], className="mb-3")

            layout = html.Div([
                html.Div([
                    _table("Current Plan", curr_df),
                    _table("Comparison Plan", comp_df),
                    _table("Delta (Current - Comparison)", diff_df),
                ], className="compare-tables", style={"maxWidth": "960px", "margin": "0 auto"})
            ], className="compare-modal-result")

            return layout, True, False, False, ""
        @app.callback(
            Output("opt-compare-result-modal", "is_open", allow_duplicate=True),
            Input("opt-compare-result-close", "n_clicks"),
            State("opt-compare-result-modal", "is_open"),
            prevent_initial_call=True,
        )
        def _compare_result_close(n, is_open):
            if not n:
                raise dash.exceptions.PreventUpdate
            return False


        # What-If store
        @app.callback(
            Output("plan-whatif", "data"),
            Input("whatif-apply", "n_clicks"),
            Input("whatif-clear", "n_clicks"),
            State("whatif-aht-delta", "value"),
            State("whatif-shr-delta", "value"),
            State("whatif-attr-delta", "value"),
            State("whatif-vol-delta", "value"),
            prevent_initial_call=True
        )
        def _whatif(n_apply, n_clear, aht, shr, attr, vol):
            t = ctx.triggered_id
            if t == "whatif-clear":
                return {}
            # Live What-if data passed to _fill_tables_fixed
            def _f(v, d=0.0):
                try:
                    return float(v)
                except Exception:
                    return d
            return dict(
                aht_delta=_f(aht, 0.0),
                shrink_delta=_f(shr, 0.0),
                attr_delta=_f(attr, 0.0),
                vol_delta=_f(vol, 0.0),
            )

        # Delete plan (admin only)
        @app.callback(
            Output("plan-msg", "children", allow_duplicate=True),
            Output("url-router", "href", allow_duplicate=True),
            Input("opt-delete", "n_clicks"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True
        )
        def _plan_delete(n, pid):
            if not n or not pid:
                raise dash.exceptions.PreventUpdate
            try:
                from plan_detail._common import current_user_fallback
                from auth import get_user_role, can_delete_plans
                if not can_delete_plans(get_user_role(current_user_fallback())):
                    return "Insufficient permissions to delete plan.", dash.no_update
            except Exception:
                return "Insufficient permissions to delete plan.", dash.no_update
            try:
                from plan_store import delete_plan
                delete_plan(int(pid))
                return f"Plan {pid} deleted ✅", "/"
            except Exception as e:
                return f"Delete failed: {e}", dash.no_update


        # ------------ What-If: Apply filters ------------
        @app.callback(
            Output("plan-msg", "children", allow_duplicate=True),
            Output("plan-msg-timer", "disabled", allow_duplicate=True),
            Output("plan-refresh-tick", "data", allow_duplicate=True),
            Input("whatif-apply", "n_clicks"),
            State("plan-detail-id", "data"),
            State("whatif-aht-delta", "value"),
            State("whatif-shr-delta", "value"),
            State("whatif-attr-delta", "value"),
            State("whatif-vol-delta", "value"),     # <-- volume % (new)
            prevent_initial_call=True
        )
        def _whatif_apply(n, pid, aht_delta, shr_delta, attr_delta, vol_delta):
            if not n or not pid:
                raise dash.exceptions.PreventUpdate
            pid = str(pid)

            def f(v, d=0.0):
                try: return float(v)
                except Exception: return d

            overrides = {
                "aht_delta":    f(aht_delta, 0.0),   # % ()
                "shrink_delta": f(shr_delta, 0.0),   # % ()
                "attr_delta":   f(attr_delta, 0.0),  # HC (+/-) per week
                "vol_delta":    f(vol_delta, 0.0),   # % () applied to forecast + req_w_forecast
            }

            rec = {
                "ts": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                "start_week": "",
                "end_week": "",
                "overrides": overrides,
            }
            save_df(f"plan_{pid}_whatif", pd.DataFrame([rec]))
            save_plan_meta(pid, {"last_updated_on": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                     "last_updated_by": current_user_fallback()})


            try:
                tick = int(ctx.states.get("plan-refresh-tick.data") or 0)
            except Exception:
                tick = 0
            return "What-if applied ✅", False, tick + 1

        @app.callback(
            Output("plan-msg", "children", allow_duplicate=True),
            Output("plan-msg-timer", "disabled", allow_duplicate=True),
            Output("plan-refresh-tick", "data", allow_duplicate=True),
            Input("whatif-clear", "n_clicks"),
            State("plan-detail-id", "data"),
            prevent_initial_call=True
        )
        def _whatif_clear(n, pid):
            if not n or not pid:
                raise dash.exceptions.PreventUpdate

            pid = str(pid)
            rec = {"ts": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                "start_week": "", "end_week": "", "overrides": {}}
            save_df(f"plan_{pid}_whatif", pd.DataFrame([rec]))
            save_plan_meta(pid, {"last_updated_on": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                     "last_updated_by": current_user_fallback()})


            try:
                tick = int(ctx.states.get("plan-refresh-tick.data") or 0)
            except Exception:
                tick = 0
            return "What-if cleared ✅", False, tick + 1

        # Reset What-If input fields to 0 on Clear button
        @app.callback(
            Output("whatif-aht-delta", "value"),
            Output("whatif-shr-delta", "value"),
            Output("whatif-attr-delta", "value"),
            Output("whatif-vol-delta", "value"),
            Input("whatif-clear", "n_clicks"),
            prevent_initial_call=True,
        )
        def _whatif_reset_inputs(n):
            if not n:
                raise dash.exceptions.PreventUpdate
            return 0, 0, 0, 0

        # Keep What-If inputs in sync with store: when cleared ({}), force zeros
        @app.callback(
            Output("whatif-aht-delta", "value", allow_duplicate=True),
            Output("whatif-shr-delta", "value", allow_duplicate=True),
            Output("whatif-attr-delta", "value", allow_duplicate=True),
            Output("whatif-vol-delta", "value", allow_duplicate=True),
            Input("plan-whatif", "data"),
            prevent_initial_call=True,
        )
        def _whatif_sync_inputs(store):
            try:
                d = dict(store or {})
            except Exception:
                d = {}
            if not d:
                return 0, 0, 0, 0
            def f(v):
                try:
                    return float(v)
                except Exception:
                    return 0
            return f(d.get("aht_delta")), f(d.get("shrink_delta")), f(d.get("attr_delta")), f(d.get("vol_delta"))

        @app.callback(
            Output("opt-created-by", "children", allow_duplicate=True),
            Output("opt-created-on", "children", allow_duplicate=True),
            Output("opt-updated-by", "children", allow_duplicate=True),
            Output("opt-updated-on", "children", allow_duplicate=True),
            Input("plan-detail-id", "data"),
            Input("plan-refresh-tick", "data"),   # so it refreshes after saves/what-ifs
            Input("plan-opt-canvas", "is_open"),   # refresh values when panel opens
            prevent_initial_call=True
        )
        def _opt_meta(pid, _tick, _is_open):
            try:
                pid = int(pid)
            except Exception:
                return "","","",""

            row = get_plan(pid) or {}            # DB row: has owner/created_at/updated_at
            meta = get_plan_meta(pid) or {}      # meta row: has created_by/on, last_updated_*

            created_by = meta.get("created_by") or row.get("owner") or ""
            created_on = meta.get("created_on") or row.get("created_at") or ""
            updated_by = meta.get("last_updated_by") or meta.get("updated_by") or ""
            updated_on = meta.get("last_updated_on") or row.get("updated_at") or ""

            def nice(ts):
                try:
                    return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    return ts or ""

            return created_by, nice(created_on), updated_by, nice(updated_on)
        
        @app.callback(
            Output("opt-created-by", "children"),
            Output("opt-created-on", "children"),
            Output("opt-updated-by", "children"),
            Output("opt-updated-on", "children"),
            Input("plan-detail-id", "data"),
            Input("plan-refresh-tick", "data"),
            Input("plan-opt-canvas", "is_open"),   # refresh values when panel opens
            prevent_initial_call=True,
        )
        def _hdr_meta(pid, _tick, _is_open):
            try:
                pid_i = int(pid)
            except Exception:
                return "", "", "", ""

            row = get_plan(pid_i) or {}
            meta = get_plan_meta(pid_i) or {}

            # Prefer plan meta; fall back to plan row fields
            created_by = meta.get("created_by") or row.get("owner") or row.get("created_by") or ""
            created_on = meta.get("created_on") or row.get("created_at") or ""
            updated_by = (
                meta.get("last_updated_by")
                or meta.get("updated_by")
                or row.get("updated_by")
                or created_by
                or ""
            )
            updated_on = meta.get("last_updated_on") or row.get("updated_at") or ""

            def fmt(ts):
                try:
                    return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M UTC")
                except Exception:
                    return ""

            return created_by, fmt(created_on), updated_by, fmt(updated_on)
            
        @app.callback(
            Output("plan-msg", "children", allow_duplicate=True),
            Output("plan-msg-timer", "disabled", allow_duplicate=True),
            Output("plan-refresh-tick", "data", allow_duplicate=True),
            Input("lc-ovr-save", "n_clicks"),
            State("plan-detail-id", "data"),
            State("lc-ovr-start-week", "date"),
            State("lc-ovr-nesting-weeks", "value"),
            State("lc-ovr-nesting-prod", "value"),
            State("lc-ovr-nesting-aht", "value"),
            State("lc-ovr-sda-weeks", "value"),
            State("lc-ovr-sda-prod", "value"),
            State("lc-ovr-sda-aht", "value"),
            State("lc-ovr-throughput-train", "value"),
            State("lc-ovr-throughput-nest", "value"),
            State("plan-refresh-tick", "data"),
            prevent_initial_call=True
        )
        def _lc_save(n, pid, w_start, nest_w, nest_prod, nest_aht, sda_w, sda_prod, sda_aht, t_train, t_nest, tick):
            if not n or not pid:
                raise dash.exceptions.PreventUpdate
            pid = str(pid)
            rec = {
                "start_week": w_start,
                "nesting_weeks": nest_w,
                "nesting_prod_pct": nest_prod,
                "nesting_aht_uplift_pct": nest_aht,
                "sda_weeks": sda_w,
                "sda_prod_pct": sda_prod,
                "sda_aht_uplift_pct": sda_aht,
                "throughput_train_pct": t_train,
                "throughput_nest_pct": t_nest,
                "ts": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
            }
            try:
                df = load_df(f"plan_{pid}_lc_overrides")
            except Exception:
                df = None
            import pandas as _pd
            if not isinstance(df, _pd.DataFrame):
                df = _pd.DataFrame()
            df = _pd.concat([df, _pd.DataFrame([rec])], ignore_index=True)
            if "start_week" in df.columns:
                try:
                    df = df.dropna(subset=["start_week"]).sort_values("start_week")
                    df = df.drop_duplicates(subset=["start_week"], keep="last")
                except Exception:
                    pass
            save_df(f"plan_{pid}_lc_overrides", df)
            save_plan_meta(pid, {"last_updated_on": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                                 "last_updated_by": current_user_fallback()})
            try:
                t = int(tick or 0)
            except Exception:
                t = 0
            return "Learning curve saved ✅", False, t + 1
        
        @app.callback(
            Output("plan-debug", "data"),
            Input("plan-detail-id", "data"),
            Input("plan-refresh-tick", "data"),
            prevent_initial_call=False,
        )
        def _verify_on_enter(pid, _tick):
            if not pid:
                raise dash.exceptions.PreventUpdate
            logs = _verify_storage(pid, when="enter/tick")
            return {"pid": pid, "logs": logs}
        
        @app.callback(
            Output("plan-msg", "children", allow_duplicate=True),
            Output("plan-msg-timer", "disabled", allow_duplicate=True),
            Input("url-router", "pathname"),
            prevent_initial_call=True,
        )
        def _clear_msg_on_path(pathname: str | None):
            if not pathname or not str(pathname).startswith("/plan"):
                # Ignore non-plan routes
                raise PreventUpdate
            # Clear immediately on entering /plan
            return "", True
        
        @app.callback(
            Output("plan-msg", "style", allow_duplicate=True),
            Input("plan-loading", "data"),
            prevent_initial_call=True,
        )
        def _hide_msg_while_loading(is_loading: bool | None):
            if is_loading:
                # Hide completely during overlay
                return {"display": "none"}
            # Show again when overlay is off
            return {"display": "block"}
        @app.callback(
            Output("plan-hydrated", "data"),
            Input("url-router", "pathname"),
            prevent_initial_call=False,
        )
        def _hydrate_reset_on_route(pathname: str | None):
            if not pathname or not str(pathname).startswith("/plan"):
                raise PreventUpdate
            return False

        # B) Flip hydration True when overlay finishes AND table stores have mounted
        @app.callback(
            Output("plan-hydrated", "data", allow_duplicate=True),
            Input("global-loading", "data"),
            State("plan-detail-id", "data"),
            State("tbl-fw", "data"), State("tbl-hc", "data"),
            State("tbl-attr", "data"), State("tbl-shr", "data"),
            State("tbl-train", "data"), State("tbl-ratio", "data"),
            State("tbl-seat", "data"), State("tbl-bva", "data"),
            State("tbl-nh", "data"), State("tbl-emp-roster", "data"),
            prevent_initial_call=True,
        )
        def _hydrate_when_loaded(is_loading, pid, fw, hc, attr, shr, trn, rat, seat, bva, nh, emp):
            if not pid:
                return False
            if is_loading:
                return False
            # Treat "hydrated" as: at least one table's .data is not None (even if empty list)
            mounted_any = any(v is not None for v in (fw, hc, attr, shr, trn, rat, seat, bva, nh, emp))
            return bool(mounted_any)


        def _coerce_float(val) -> float | None:
            try:
                out = float(pd.to_numeric(val, errors="coerce"))
                if pd.isna(out):
                    return None
                return out
            except Exception:
                return None
        
        def _autosave_seat(df: pd.DataFrame) -> pd.DataFrame:
            if not isinstance(df, pd.DataFrame) or df.empty or "metric" not in df.columns:
                return df
            seat_idx = df.set_index("metric", drop=False)
            week_cols = [c for c in df.columns if c != "metric"]
            for w in week_cols:
                req = _coerce_float(seat_idx.at["Seats Required (#)", w]) if "Seats Required (#)" in seat_idx.index else None
                avail = _coerce_float(seat_idx.at["Seats Available (#)", w]) if "Seats Available (#)" in seat_idx.index else None
                req = req if req is not None else 0.0
                avail = avail if avail is not None else 0.0
                util_pct = (req / avail * 100.0) if avail > 0 else 0.0
                variance_pct = ((avail - req) / avail * 100.0) if avail > 0 else 0.0
                if "Seat Utilization %" in seat_idx.index:
                    seat_idx.at["Seat Utilization %", w] = round(max(0.0, util_pct), 1)
                if "Variance vs Available (pp)" in seat_idx.index:
                    seat_idx.at["Variance vs Available (pp)", w] = round(variance_pct, 1)
            return seat_idx.reset_index(drop=True)
        
        def _autosave_ratio(df: pd.DataFrame) -> pd.DataFrame:
            if not isinstance(df, pd.DataFrame) or df.empty or "metric" not in df.columns:
                return df
            ratio_idx = df.set_index("metric", drop=False)
            week_cols = [c for c in df.columns if c != "metric"]
            for w in week_cols:
                planned = _coerce_float(ratio_idx.at["Planned TL/Agent Ratio", w]) if "Planned TL/Agent Ratio" in ratio_idx.index else None
                actual = _coerce_float(ratio_idx.at["Actual TL/Agent Ratio", w]) if "Actual TL/Agent Ratio" in ratio_idx.index else None
                if planned is not None and "Planned TL/Agent Ratio" in ratio_idx.index:
                    ratio_idx.at["Planned TL/Agent Ratio", w] = round(planned, 1)
                if actual is not None and "Actual TL/Agent Ratio" in ratio_idx.index:
                    ratio_idx.at["Actual TL/Agent Ratio", w] = round(actual, 1)
                if planned is not None and actual is not None and "Variance" in ratio_idx.index:
                    ratio_idx.at["Variance", w] = round(actual - planned, 1)
            return ratio_idx.reset_index(drop=True)
        
        def _autosave_attr(df: pd.DataFrame, pid: int, hc_data: list | None = None) -> pd.DataFrame:
            if not isinstance(df, pd.DataFrame) or df.empty or "metric" not in df.columns:
                return df
            attr_idx = df.set_index("metric", drop=False)
            week_cols = [c for c in df.columns if c != "metric"]

            # Prefer the live HC grid from the page if provided; fall back to persisted store.
            hc_df = None
            try:
                if isinstance(hc_data, list) and hc_data:
                    hc_df = pd.DataFrame(hc_data)
                else:
                    hc_df = load_df(f"plan_{pid}_hc")
            except Exception:
                hc_df = None

            def _extract_hc_map(metric_name: str) -> dict[str, float]:
                out: dict[str, float] = {}
                if not isinstance(hc_df, pd.DataFrame) or hc_df.empty or "metric" not in hc_df.columns:
                    return out
                try:
                    row = hc_df.set_index("metric", drop=False).loc[metric_name]
                except Exception:
                    return out
                for col in week_cols:
                    val = _coerce_float(row.get(col))
                    if val is not None:
                        out[col] = val
                return out

            hc_plan_map = _extract_hc_map("Planned/Tactical HC (#)")
            if not hc_plan_map:
                hc_plan_map = _extract_hc_map("Budgeted HC (#)")
            hc_actual_map = _extract_hc_map("Actual Agent HC (#)")

            def _prev_week_id(label: str) -> str | None:
                try:
                    t = pd.to_datetime(label, errors="coerce")
                    if pd.isna(t):
                        return None
                    prev = pd.Timestamp(t).normalize() - pd.Timedelta(days=7)
                    return prev.date().isoformat()
                except Exception:
                    return None

            variance_label = None
            for cand in ("Variance vs Planned (pp)", "Variance vs Planned"):
                if cand in attr_idx.index:
                    variance_label = cand
                    break

            for w in week_cols:
                planned_hc = _coerce_float(attr_idx.at["Planned Attrition HC (#)", w]) if "Planned Attrition HC (#)" in attr_idx.index else None
                actual_hc = _coerce_float(attr_idx.at["Actual Attrition HC (#)", w]) if "Actual Attrition HC (#)" in attr_idx.index else None

                planned_pct = None
                if planned_hc is not None:
                    # Baseline: previous week's Actual HC; fallback to current week starting HC
                    prev_w = _prev_week_id(w)
                    denom = None
                    if prev_w:
                        denom = hc_actual_map.get(prev_w)
                    if denom in (None, 0):
                        denom = hc_plan_map.get(w)
                    if denom in (None, 0):
                        denom = hc_actual_map.get(w)
                    if denom and denom > 0:
                        planned_pct = round((planned_hc / denom) * 100.0, 1)
                        if "Planned Attrition %" in attr_idx.index:
                            attr_idx.at["Planned Attrition %", w] = planned_pct
                if planned_pct is None and "Planned Attrition %" in attr_idx.index:
                    tmp = _coerce_float(attr_idx.at["Planned Attrition %", w])
                    if tmp is not None:
                        planned_pct = round(tmp, 1)
                        attr_idx.at["Planned Attrition %", w] = planned_pct

                actual_pct = None
                if actual_hc is not None:
                    denom = hc_actual_map.get(w)
                    if denom in (None, 0):
                        denom = hc_plan_map.get(w)
                    if denom and denom > 0:
                        actual_pct = round((actual_hc / denom) * 100.0, 1)
                        if "Actual Attrition %" in attr_idx.index:
                            attr_idx.at["Actual Attrition %", w] = actual_pct
                if actual_pct is None and "Actual Attrition %" in attr_idx.index:
                    tmp = _coerce_float(attr_idx.at["Actual Attrition %", w])
                    if tmp is not None:
                        actual_pct = round(tmp, 1)
                        attr_idx.at["Actual Attrition %", w] = actual_pct

                if variance_label and planned_pct is not None and actual_pct is not None:
                    attr_idx.at[variance_label, w] = round(actual_pct - planned_pct, 1)

            return attr_idx.reset_index(drop=True)
        
        @app.callback(
            Output("plan-refresh-tick", "data", allow_duplicate=True),
            Output("plan-msg", "children", allow_duplicate=True),
            Output("plan-msg-timer", "disabled", allow_duplicate=True),
            Input("tbl-attr", "data_timestamp"),
            Input("tbl-shr", "data_timestamp"),
            Input("tbl-seat", "data_timestamp"),
            Input("tbl-ratio", "data_timestamp"),
            Input("tbl-nh",   "data_timestamp"),
            State("plan-detail-id", "data"),
            State("plan-hydrated", "data"),
            State("tbl-attr", "data"),
            State("tbl-shr", "data"),
            State("tbl-seat", "data"),
            State("tbl-ratio", "data"),
            State("tbl-nh",   "data"),
            State("tbl-hc", "data"),
            State("plan-refresh-tick", "data"),
            prevent_initial_call=True
        )
        def _autosave_lower_tables(ts_attr, ts_shr, ts_seat, ts_ratio, ts_nh, pid, hydrated, attr_data, shr_data, seat_data, ratio_data, nh_data, hc_live, tick):
            trig = ctx.triggered_id
            if not pid or not hydrated or trig is None:
                raise dash.exceptions.PreventUpdate
            trig_ts = ctx.triggered[0].get("value") if ctx.triggered else None
            if trig_ts in (None, 0):
                raise dash.exceptions.PreventUpdate
            suffix_map = {
                "tbl-attr": ("attr", attr_data, "Attrition"),
                "tbl-shr": ("shr", shr_data, "Shrinkage"),
                "tbl-seat": ("seat", seat_data, "Seat Utilization"),
                "tbl-ratio": ("ratio", ratio_data, "Ratios"),
                "tbl-nh":   ("nh",   nh_data,   "New Hire"),
            }
            suffix, payload, label = suffix_map.get(trig, (None, None, None))
            if not suffix:
                raise dash.exceptions.PreventUpdate
            df = pd.DataFrame(payload or [])
            if suffix == "seat":
                df = _autosave_seat(df)
            elif suffix == "ratio":
                df = _autosave_ratio(df)
            elif suffix == "attr":
                df = _autosave_attr(df, pid, hc_data=hc_live)
            elif suffix == "nh":
                # Compute Recruitment Achievement from Planned/Actual
                if isinstance(df, pd.DataFrame) and not df.empty and "metric" in df.columns:
                    nh_idx = df.set_index("metric", drop=False)
                    week_cols = [c for c in df.columns if c != "metric"]
                    for w in week_cols:
                        try:
                            plan = float(pd.to_numeric(nh_idx.at["Planned New Hire HC (#)", w], errors="coerce")) if "Planned New Hire HC (#)" in nh_idx.index else 0.0
                        except Exception:
                            plan = 0.0
                        try:
                            act = float(pd.to_numeric(nh_idx.at["Actual New Hire HC (#)", w], errors="coerce")) if "Actual New Hire HC (#)" in nh_idx.index else 0.0
                        except Exception:
                            act = 0.0
                        val = 0.0 if plan <= 0 else (100.0 * act / plan)
                        if "Recruitment Achievement" in nh_idx.index:
                            nh_idx.at["Recruitment Achievement", w] = round(val, 1)
                    df = nh_idx.reset_index(drop=True)
            _save_table(pid, suffix, df)
            save_plan_meta(pid, {
                "last_updated_on": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                "last_updated_by": current_user_fallback(),
            })
            try:
                next_tick = int(tick or 0) + 1
            except Exception:
                next_tick = 1
            msg = f"{label} auto-saved."
            return next_tick, msg, False

        # C) Disable Save button until hydrated
        # @app.callback(
        #     Output("btn-plan-save", "disabled"),
        #     Input("plan-hydrated", "data"),
        #     prevent_initial_call=False,
        # )
        # def _toggle_save_disabled(hydrated):
        #     return not bool(hydrated)


        # D) Refresh plan tables when roster changes (e.g., Terminate/LOA/Back)
        @app.callback(
            Output("plan-refresh-tick", "data", allow_duplicate=True),
            Input("tbl-emp-roster", "data_timestamp"),
            State("plan-refresh-tick", "data"),
            prevent_initial_call=True
        )
        def _bump_tick_on_roster(ts, tick):
            try:
                if ts in (None, 0):
                    raise dash.exceptions.PreventUpdate
                return int(tick or 0) + 1
            except Exception:
                return 1
