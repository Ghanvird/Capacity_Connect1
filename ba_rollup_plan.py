from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import dash
from dash import dash_table

from plan_store import list_plans
from cap_db import load_df
from plan_detail._common import _roster_columns, _ROSTER_REQUIRED_IDS
from plan_detail._common import _week_span, _month_cols, _week_cols
from plan_detail._calc import _fill_tables_fixed
from plan_detail._fill_tables_fixed_monthly import _fill_tables_fixed_monthly


def _safe_df(records) -> pd.DataFrame:
    try:
        df = pd.DataFrame(records or [])
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def month_cols_for_ba(ba: str, status_filter: str = "current") -> Tuple[list, list[str]]:
    plans = list_plans(vertical=ba, status_filter=status_filter) or []
    if not plans:
        # default to current month only
        today = pd.Timestamp.today().normalize()
        start = today.to_period("M").to_timestamp().date().isoformat()
        cols = [{"name": "Metric", "id": "metric", "editable": False}, {"name": f"Actual\n{today.strftime('%b %Y')}", "id": start}]
        return cols, [start]
    starts = [str(p.get("start_week") or "").strip() for p in plans]
    ends   = [str(p.get("end_week") or "").strip() for p in plans]
    starts = [s for s in starts if s]
    ends   = [e for e in ends if e]
    if not starts or not ends:
        return month_cols_for_ba(ba, status_filter)
    start = min(starts)
    end   = max(ends)
    weeks = _week_span(start, end)
    cols, mids = _month_cols(weeks)
    return cols, mids

def week_cols_for_ba(ba: str, status_filter: str = "current") -> Tuple[list, list[str]]:
    plans = list_plans(vertical=ba, status_filter=status_filter) or []
    if not plans:
        today = pd.Timestamp.today().normalize()
        weeks = [today.to_period("W-MON").to_timestamp().date().isoformat()]
        cols, ids = _week_cols(weeks)
        return cols, ids
    starts = [str(p.get("start_week") or "").strip() for p in plans]
    ends   = [str(p.get("end_week") or "").strip() for p in plans]
    starts = [s for s in starts if s]
    ends   = [e for e in ends if e]
    if not starts or not ends:
        today = pd.Timestamp.today().normalize()
        weeks = [today.to_period("W-MON").to_timestamp().date().isoformat()]
        return _week_cols(weeks)
    start = min(starts)
    end   = max(ends)
    weeks = _week_span(start, end)
    cols, wids = _week_cols(weeks)
    return cols, wids


def _merge_metric_tables(dfs: List[pd.DataFrame], ids: List[str], weighted: dict | None = None, lookups: List[pd.DataFrame] | None = None) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame(columns=["metric"] + ids)

    # normalize
    norm: List[pd.DataFrame] = []
    for df in dfs:
        d = df.copy()
        if d.empty:
            continue
        if "metric" not in d.columns:
            continue
        for c in ids:
            if c not in d.columns:
                d[c] = 0.0
        cols = ["metric"] + ids
        d = d[cols]
        norm.append(d)
    if not norm:
        return pd.DataFrame(columns=["metric"] + ids)

    # union of metrics
    metrics = []
    seen = set()
    for d in norm:
        for m in d["metric"].astype(str):
            if m not in seen:
                seen.add(m); metrics.append(m)

    def is_avg_metric(name: str) -> bool:
        s = str(name).lower()
        return ("%" in s) or ("aht/sut" in s) or ("utilization" in s) or ("occupancy" in s) or ("ratio" in s) or ("service level" in s)

    out_rows = []
    weighted = weighted or {}
    lookups = lookups or []
    for m in metrics:
        row = {"metric": m}
        for col in ids:
            nums = []
            wgts = []
            for i, d in enumerate(norm):
                try:
                    ser = d.loc[d["metric"].astype(str) == m, col]
                    if ser.empty:
                        continue
                    v = float(pd.to_numeric(ser.iloc[0], errors="coerce"))
                    if m in weighted:
                        lk = weighted[m]
                        lkdf = lookups[i] if i < len(lookups) and isinstance(lookups[i], pd.DataFrame) else d
                        wser = lkdf.loc[lkdf["metric"].astype(str) == lk, col] if "metric" in lkdf.columns else pd.Series([], dtype=float)
                        if wser.empty:
                            continue
                        w = float(pd.to_numeric(wser.iloc[0], errors="coerce"))
                        nums.append(v * max(0.0, w))
                        wgts.append(max(0.0, w))
                    else:
                        nums.append(v)
                except Exception:
                    pass
            if m in weighted:
                tot_w = sum(wgts) if wgts else 0.0
                row[col] = (sum(nums) / tot_w) if tot_w > 0 else 0.0
            else:
                row[col] = sum(nums)/len(nums) if (nums and is_avg_metric(m)) else (sum(nums) if nums else 0.0)
        out_rows.append(row)
    return pd.DataFrame(out_rows)[["metric"] + ids]


def compute_ba_rollup_monthly_tables(ba: str, fw_cols: list[dict], whatif=None, status_filter: str = "current"):
    # Columns → month ids
    month_ids = [c.get("id") for c in fw_cols if c.get("id") != "metric"]

    # For each plan under BA, compute monthly tables, then merge
    plans = list_plans(vertical=ba, status_filter=status_filter) or []
    if not plans:
        empty_cols = [{"name": "Metric", "id": "metric", "editable": False}] + [c for c in fw_cols if c.get("id") != "metric"]
        empty_upper = dash_table.DataTable(id="tbl-upper", data=[], columns=empty_cols)
        empty = pd.DataFrame({"metric": []})
        return (
            empty_upper,
            empty.to_dict("records"), empty.to_dict("records"), empty.to_dict("records"), empty.to_dict("records"),
            empty.to_dict("records"), empty.to_dict("records"), empty.to_dict("records"), empty.to_dict("records"), empty.to_dict("records"),
            [], [], [],
        )

    # Use Volume Based type for rows spec; actual numbers are merged
    ptype = "Volume Based"
    fw_list = []
    hc_list = []
    att_list = []
    shr_list = []
    trn_list = []
    rat_list = []
    seat_list= []
    bva_list = []
    nh_list  = []
    upper_list: List[pd.DataFrame] = []
    upper_list: List[pd.DataFrame] = []
    upper_list: List[pd.DataFrame] = []

    for p in plans:
        pid = p.get("id")
        if pid is None:
            continue
        try:
            upper, fw, hc, att, shr, trn, rat, seat, bva, nh, _ro, _bf, _nt = _fill_tables_fixed_monthly(ptype, pid, fw_cols, _tick=0, whatif=whatif)
            fw_list.append(_safe_df(fw))
            hc_list.append(_safe_df(hc))
            att_list.append(_safe_df(att))
            shr_list.append(_safe_df(shr))
            trn_list.append(_safe_df(trn))
            rat_list.append(_safe_df(rat))
            seat_list.append(_safe_df(seat))
            bva_list.append(_safe_df(bva))
            nh_list.append(_safe_df(nh))
        except Exception:
            continue

    fw_agg  = _merge_metric_tables(
        fw_list,
        month_ids,
        weighted={"Actual AHT/SUT": "Actual Volume", "Forecast AHT/SUT": "Forecast", "Budgeted AHT/SUT": "Forecast"},
        lookups=fw_list,
    )
    hc_agg  = _merge_metric_tables(hc_list, month_ids)
    att_agg = _merge_metric_tables(att_list, month_ids)
    shr_agg = _merge_metric_tables(shr_list, month_ids)
    trn_agg = _merge_metric_tables(trn_list, month_ids)
    rat_agg = _merge_metric_tables(rat_list, month_ids)
    seat_agg= _merge_metric_tables(seat_list, month_ids)
    bva_agg = _merge_metric_tables(bva_list, month_ids)
    nh_agg  = _merge_metric_tables(nh_list, month_ids)

    # Simple upper built from available rows
    def _row(df: pd.DataFrame, name: str) -> dict:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}
        d = df.copy()
        d["metric"] = d["metric"].astype(str).str.strip()
        m = d[d["metric"] == name]
        return m.iloc[0].to_dict() if not m.empty else {"metric": name, **{m: 0.0 for m in month_ids}}

    req_row = _row(fw_agg, "FTE Required (#)")
    if not req_row or all(k == "metric" or (isinstance(v, (int,float)) and v == 0) for k,v in req_row.items()):
        # try billable variant
        req_row = _row(fw_agg, "Billable FTE Required (#)")
    sup_row = _row(hc_agg, "Planned/Tactical HC (#)")
    over = {"metric": "FTE Over/Under (#)"}
    for m in month_ids:
        try:
            over[m] = float(sup_row.get(m, 0.0)) - float(req_row.get(m, 0.0))
        except Exception:
            over[m] = 0.0

    upper_df = pd.DataFrame([
        {"metric": "FTE Required (@ Forecast)", **{m: req_row.get(m, 0.0) for m in month_ids}},
        {"metric": "Projected Supply HC",     **{m: sup_row.get(m, 0.0) for m in month_ids}},
        over,
    ])

    upper = dash_table.DataTable(
        id="tbl-upper",
        data=upper_df.to_dict("records"),
        columns=[{"name": "Metric", "id": "metric", "editable": False}] + [{"name": c["name"], "id": c["id"]} for c in fw_cols if c.get("id") != "metric"],
        editable=False,
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={"whiteSpace": "pre"},
    )

    return (
        upper,
        (fw_agg.to_dict("records") if not fw_agg.empty else []),
        (hc_agg.to_dict("records") if not hc_agg.empty else []),
        (att_agg.to_dict("records") if not att_agg.empty else []),
        (shr_agg.to_dict("records") if not shr_agg.empty else []),
        (trn_agg.to_dict("records") if not trn_agg.empty else []),
        (rat_agg.to_dict("records") if not rat_agg.empty else []),
        (seat_agg.to_dict("records") if not seat_agg.empty else []),
        (bva_agg.to_dict("records") if not bva_agg.empty else []),
        (nh_agg.to_dict("records") if not nh_agg.empty else []),
        [], [], [],
    )


def compute_ba_rollup_tables(ba: str, fw_cols: list[dict], whatif=None, status_filter: str = "current", grain: str = "week"):
    ids = [c.get("id") for c in fw_cols if c.get("id") != "metric"]
    plans = list_plans(vertical=ba, status_filter=status_filter) or []
    if not plans:
        empty_cols = [{"name": "Metric", "id": "metric", "editable": False}] + [c for c in fw_cols if c.get("id") != "metric"]
        empty_upper = dash_table.DataTable(id="tbl-upper", data=[], columns=empty_cols)
        empty = pd.DataFrame({"metric": []})
        return (
            empty_upper,
            empty.to_dict("records"), empty.to_dict("records"), empty.to_dict("records"), empty.to_dict("records"),
            empty.to_dict("records"), empty.to_dict("records"), empty.to_dict("records"), empty.to_dict("records"), empty.to_dict("records"),
            [], [], [],
        )

    ptype = "Volume Based"
    fw_list = []
    hc_list = []
    att_list = []
    shr_list = []
    trn_list = []
    rat_list = []
    seat_list= []
    bva_list = []
    nh_list  = []
    upper_list: List[pd.DataFrame] = []

    for p in plans:
        pid = p.get("id")
        if pid is None:
            continue
        try:
            if str(grain or 'week').lower() == 'month':
                upper, fw, hc, att, shr, trn, rat, seat, bva, nh, _ro, _bf, _nt = _fill_tables_fixed_monthly(ptype, pid, fw_cols, _tick=0, whatif=whatif)
            else:
                upper, fw, hc, att, shr, trn, rat, seat, bva, nh, _ro, _bf, _nt = _fill_tables_fixed(ptype, pid, fw_cols, _tick=0, whatif=whatif, grain='week')
            fw_list.append(_safe_df(fw))
            hc_list.append(_safe_df(hc))
            att_list.append(_safe_df(att))
            shr_list.append(_safe_df(shr))
            trn_list.append(_safe_df(trn))
            rat_list.append(_safe_df(rat))
            seat_list.append(_safe_df(seat))
            bva_list.append(_safe_df(bva))
            nh_list.append(_safe_df(nh))
            # Extract upper table data from the DataTable component
            try:
                comp = upper
                props = getattr(comp, 'to_plotly_json', lambda: {})()
                data = (props.get('props', {}) or {}).get('data')
                if isinstance(data, list):
                    up = pd.DataFrame(data)
                    if not up.empty:
                        upper_list.append(up)
            except Exception:
                pass
        except Exception:
            continue

    fw_agg  = _merge_metric_tables(
        fw_list,
        ids,
        weighted={"Actual AHT/SUT": "Actual Volume", "Forecast AHT/SUT": "Forecast", "Budgeted AHT/SUT": "Forecast"},
        lookups=fw_list,
    )
    hc_agg  = _merge_metric_tables(hc_list, ids)
    att_agg = _merge_metric_tables(att_list, ids)
    shr_agg = _merge_metric_tables(shr_list, ids)
    trn_agg = _merge_metric_tables(trn_list, ids)
    rat_agg = _merge_metric_tables(rat_list, ids)
    seat_agg= _merge_metric_tables(seat_list, ids)
    bva_agg = _merge_metric_tables(bva_list, ids)
    nh_agg  = _merge_metric_tables(nh_list, ids)

    # ---------- Upper parity aggregation ----------
    def _upper_row_sum(label: str) -> dict:
        out = {"metric": label}
        for col in ids:
            s = 0.0
            for up in upper_list:
                try:
                    m = up[up["metric"].astype(str).str.strip() == label]
                    if not m.empty:
                        v = float(pd.to_numeric(m.iloc[0].get(col), errors="coerce"))
                        s += v
                except Exception:
                    pass
            out[col] = s
        return out

    # Required @ Forecast/Actual
    reqF = _upper_row_sum("FTE Required @ Forecast Volume")
    reqA = _upper_row_sum("FTE Required @ Actual Volume")
    # Tactical/Budget deltas can be summed directly
    ou_tac = _upper_row_sum("FTE Over/Under Tactical Vs Actual")
    ou_bud = _upper_row_sum("FTE Over/Under Budgeted Vs Actual")
    # Supply and Handling Capacity
    sup   = _upper_row_sum("Projected Supply HC")
    hcap  = _upper_row_sum("Projected Handling Capacity (#)")

    # MTP vs Actual = sum(reqF) - sum(reqA)
    ou_mtp = {"metric": "FTE Over/Under MTP Vs Actual"}
    for col in ids:
        ou_mtp[col] = float(reqF.get(col, 0.0)) - float(reqA.get(col, 0.0))

    # Projected Service Level = volume-weighted average of child SL (weights = Forecast per child)
    sl = {"metric": "Projected Service Level"}
    for col in ids:
        num = 0.0; den = 0.0
        for up, fw in zip(upper_list, fw_list):
            try:
                m = up[up["metric"].astype(str).str.strip() == "Projected Service Level"]
                if m.empty:
                    continue
                sl_val = float(pd.to_numeric(m.iloc[0].get(col), errors="coerce"))
            except Exception:
                continue
            # weight from Forecast volume for the same child
            try:
                fw_m = fw[fw["metric"].astype(str).str.strip() == "Forecast"]
                w = float(pd.to_numeric(fw_m.iloc[0].get(col), errors="coerce")) if not fw_m.empty else 0.0
            except Exception:
                w = 0.0
            if w > 0:
                num += sl_val * w; den += w
        sl[col] = (num/den) if den > 0 else 0.0

    # Compose upper DF with full parity rows
    upper_rows = [reqF, reqA, ou_mtp, ou_tac, ou_bud, sup, hcap, sl]
    upper_df = pd.DataFrame(upper_rows)

    # Format: SL to 1 decimal, others to integer (parity with sub‑BA plans)
    if isinstance(upper_df, pd.DataFrame) and not upper_df.empty:
        for col in ids:
            if col not in upper_df.columns:
                continue
            mask_sl = upper_df["metric"].astype(str).str.strip().eq("Projected Service Level")
            mask_not_sl = ~mask_sl
            upper_df.loc[mask_sl, col] = pd.to_numeric(upper_df.loc[mask_sl, col], errors="coerce").fillna(0.0).round(1)
            upper_df.loc[mask_not_sl, col] = pd.to_numeric(upper_df.loc[mask_not_sl, col], errors="coerce").fillna(0.0).round(0).astype(int)

    upper = dash_table.DataTable(
        id="tbl-upper",
        data=upper_df.to_dict("records"),
        columns=[{"name": "Metric", "id": "metric", "editable": False}] + [{"name": c["name"], "id": c["id"]} for c in fw_cols if c.get("id") != "metric"],
        editable=False,
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={"whiteSpace": "pre"},
    )

    # --------- Consolidated Employee Roster (union of all child plans) ---------
    roster_frames: List[pd.DataFrame] = []
    for p in plans:
        pid = p.get("id")
        try:
            df = load_df(f"plan_{pid}_emp")
            if isinstance(df, pd.DataFrame) and not df.empty:
                # keep only required ids; add missing
                d = df.copy()
                for cid in _ROSTER_REQUIRED_IDS:
                    if cid not in d.columns:
                        d[cid] = ""
                roster_frames.append(d[_ROSTER_REQUIRED_IDS])
        except Exception:
            pass
    if roster_frames:
        roster_all = pd.concat(roster_frames, ignore_index=True)
        # de-dup by BRID if present
        brid_col = "brid" if "brid" in roster_all.columns else None
        if brid_col:
            roster_all = roster_all.drop_duplicates(subset=[brid_col], keep="last")
        roster_data = roster_all.to_dict("records")
    else:
        roster_data = []

    return (
        upper,
        (fw_agg.to_dict("records") if not fw_agg.empty else []),
        (hc_agg.to_dict("records") if not hc_agg.empty else []),
        (att_agg.to_dict("records") if not att_agg.empty else []),
        (shr_agg.to_dict("records") if not shr_agg.empty else []),
        (trn_agg.to_dict("records") if not trn_agg.empty else []),
        (rat_agg.to_dict("records") if not rat_agg.empty else []),
        (seat_agg.to_dict("records") if not seat_agg.empty else []),
        (bva_agg.to_dict("records") if not bva_agg.empty else []),
        (nh_agg.to_dict("records") if not nh_agg.empty else []),
        roster_data, [], [],
    )
