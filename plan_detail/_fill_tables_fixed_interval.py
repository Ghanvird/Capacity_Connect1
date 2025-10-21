from __future__ import annotations
import math
import re
import pandas as pd
import numpy as np
import datetime as dt
from typing import List, Tuple, Optional
from dash import dash_table

from plan_store import get_plan
from cap_store import resolve_settings, load_roster_long
from ._common import _week_span, _scope_key, _assemble_voice, _assemble_bo, _assemble_ob, _assemble_chat, _load_ts_with_fallback
from ._calc import _fill_tables_fixed
from ._grain_cols import interval_cols_for_day
from capacity_core import voice_requirements_interval, min_agents, _ivl_minutes_from_str


def _broadcast_daily_to_intervals(df: pd.DataFrame, interval_ids: List[str]) -> pd.DataFrame:
    """Convert a daily-matrix DataFrame (metric + date cols) to an interval-matrix
    for a single representative day without aggregating/splitting values.

    Behavior:
    - For percent/rate/AHT/SUT-like rows: replicate the day's value to every interval column.
    - For other rows: also replicate the day's value to every interval column (no equal distribution).
      This avoids manufacturing interval-level splits and keeps day-level inputs "as-is".
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["metric"] + list(interval_ids))

    # Identify daily columns (ISO dates)
    day_cols = [c for c in df.columns if c != "metric"]
    if not day_cols:
        return pd.DataFrame(columns=["metric"] + list(interval_ids))

    # Use the first day as representative
    rep_day = day_cols[0]

    out = pd.DataFrame({"metric": df["metric"].astype(str).tolist()})
    zeros = pd.DataFrame(0.0, index=out.index, columns=interval_ids)
    out = pd.concat([out, zeros], axis=1)

    def is_percent_like(name: str) -> bool:
        s = str(name).strip().lower()
        return (
            ("%" in s)
            or ("service level" in s)
            or ("occupancy" in s)
            or ("utilization" in s)
            or ("efficiency" in s)
            or ("aht/sut" in s)
        )

    for i, row in df.iterrows():
        name = str(row.get("metric", ""))
        try:
            val = float(pd.to_numeric(row.get(rep_day), errors="coerce"))
        except Exception:
            val = 0.0
        # Replicate for all rows (no equal distribution)
        for ivl in interval_ids:
            out.at[i, ivl] = val

    return out[["metric"] + list(interval_ids)]


def _round_one_decimal(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        if c == "metric":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").round(1)
    return out


def _make_upper_table(upper_ivl: pd.DataFrame, ivl_cols: List[dict]):
    if not isinstance(upper_ivl, pd.DataFrame) or upper_ivl.empty:
        upper_ivl = pd.DataFrame({"metric": []})
    # Round to 1 decimal place for all numeric cells
    upper_ivl = _round_one_decimal(upper_ivl)
    return dash_table.DataTable(
        id="tbl-upper",
        data=upper_ivl.to_dict("records"),
        columns=[{"name": "Metric", "id": "metric", "editable": False}] + [
            {"name": c["name"], "id": c["id"]} for c in ivl_cols if c["id"] != "metric"
        ],
        editable=False,
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={"whiteSpace": "pre"},
    )


def _fill_tables_fixed_interval(ptype, pid, _fw_cols_unused, _tick, whatif=None, ivl_min: int = 30, sel_date: Optional[str] = None):
    """Interval view (representative day):
    1) Reuse weekly engine then daily reshape
    2) Split first-day values across HH:MM intervals uniformly (percent-like rows replicated)
    """
    p = get_plan(pid) or {}
    weeks = _week_span(p.get("start_week"), p.get("end_week"))

    # Build weekly, daily first
    weekly_fw_cols = [{"name": "Metric", "id": "metric", "editable": False}] + [{"name": w, "id": w} for w in weeks]
    weekly = _fill_tables_fixed(ptype, pid, weekly_fw_cols, _tick, whatif=whatif, grain='week')
    (
        upper_w,
        fw_w, hc_w, att_w, shr_w, trn_w, rat_w, seat_w, bva_w, nh_w,
        roster_rec, bulk_files_rec, notes_rec,
    ) = weekly

    # Weekly -> daily (representative day = selected date or first week Monday)
    if sel_date:
        try:
            ref_day = pd.to_datetime(sel_date).date()
        except Exception:
            ref_day = pd.to_datetime(weeks[0]).date() if weeks else dt.date.today()
    else:
        ref_day = pd.to_datetime(weeks[0]).date() if weeks else dt.date.today()
    rep_day = ref_day.isoformat()
    # Build simple daily frames by copying weekly Monday values
    def weekly_col_to_day(recs):
        df = pd.DataFrame(recs or [])
        out = pd.DataFrame({"metric": df.get("metric", pd.Series(dtype="object"))})
        if rep_day not in out.columns:
            out[rep_day] = 0.0
        if not isinstance(df, pd.DataFrame) or df.empty:
            return out[["metric", rep_day]]
        if weeks:
            w0 = weeks[0]
        else:
            w0 = rep_day
        if w0 in df.columns:
            out[rep_day] = pd.to_numeric(df[w0], errors="coerce").fillna(0.0)
        else:
            # copy first numeric col
            wn = [c for c in df.columns if c != "metric"]
            if wn:
                out[rep_day] = pd.to_numeric(df[wn[0]], errors="coerce").fillna(0.0)
        return out[["metric", rep_day]]

    fw_d   = weekly_col_to_day(fw_w)
    hc_d   = weekly_col_to_day(hc_w)
    att_d  = weekly_col_to_day(att_w)
    shr_d  = weekly_col_to_day(shr_w)
    trn_d  = weekly_col_to_day(trn_w)
    rat_d  = weekly_col_to_day(rat_w)
    seat_d = weekly_col_to_day(seat_w)
    bva_d  = weekly_col_to_day(bva_w)
    nh_d   = weekly_col_to_day(nh_w)

    upper_df_w = pd.DataFrame(getattr(upper_w, 'data', None) or [])
    upper_d    = weekly_col_to_day(upper_df_w.to_dict("records"))

    # Determine coverage start based on uploaded interval series (earliest slot) or default to 08:00
    def _earliest_slot_from_df(df: pd.DataFrame) -> str | None:
        try:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return None
            d = df.copy()
            # Normalize date column name
            L = {str(c).strip().lower(): c for c in d.columns}
            c_date = L.get("date") or L.get("day")
            if c_date:
                d[c_date] = pd.to_datetime(d[c_date], errors="coerce").dt.date
                d = d[d[c_date].eq(ref_day)]
            # Require an interval-like column
            c_ivl = L.get("interval") or L.get("time")
            if not c_ivl or c_ivl not in d.columns or d[c_ivl].isna().all():
                return None
            labs = d[c_ivl].astype(str).str.slice(0, 5)
            labs = labs[labs.str.match(r"^\d{2}:\d{2}$")]
            if labs.empty:
                return None
            return labs.min()
        except Exception:
            return None

    def _infer_start_hhmm(plan_dict: dict) -> str:
        try:
            ch0 = (plan_dict.get("channel") or plan_dict.get("lob") or "").split(",")[0].strip().lower()
        except Exception:
            ch0 = ""
        # default to 08:00 if nothing found
        start = None
        try:
            if ch0 == "voice":
                sk = _scope_key(plan_dict.get("vertical"), plan_dict.get("sub_ba"), ch0)
                vF = _assemble_voice(sk, "forecast"); vA = _assemble_voice(sk, "actual")
                for dfx in (vF, vA):
                    start = start or _earliest_slot_from_df(dfx)
            elif ch0 == "chat":
                sk = _scope_key(plan_dict.get("vertical"), plan_dict.get("sub_ba"), ch0)
                volF = _load_ts_with_fallback("chat_forecast_volume", sk)
                volA = _load_ts_with_fallback("chat_actual_volume", sk)
                for dfx in (volF, volA):
                    start = start or _earliest_slot_from_df(dfx)
            elif ch0 in ("outbound", "ob"):
                sk = _scope_key(plan_dict.get("vertical"), plan_dict.get("sub_ba"), ch0)
                def _first_non_empty(keys: list[str]) -> pd.DataFrame:
                    for k in keys:
                        dfk = _load_ts_with_fallback(k, sk)
                        if isinstance(dfk, pd.DataFrame) and not dfk.empty:
                            return dfk
                    return pd.DataFrame()
                volF = _first_non_empty(["ob_forecast_opc","outbound_forecast_opc","ob_forecast_dials","outbound_forecast_dials","ob_forecast_calls"])
                volA = _first_non_empty(["ob_actual_opc","outbound_actual_opc","ob_actual_dials","outbound_actual_dials","ob_actual_calls"])
                for dfx in (volF, volA):
                    start = start or _earliest_slot_from_df(dfx)
        except Exception:
            start = None
        return start or "08:00"

    start_hhmm = _infer_start_hhmm(p)

    # Daily -> intervals using inferred start
    ivl_cols, ivl_ids = interval_cols_for_day(ref_day, ivl_min=ivl_min, start_hhmm=start_hhmm)
    # For interval-level FW table, do NOT distribute day values uniformly.
    # Initialize a blank interval matrix and later populate from uploaded interval series only.
    fw_i   = pd.DataFrame({"metric": fw_d.get("metric", pd.Series(dtype="object")).astype(str) if isinstance(fw_d, pd.DataFrame) else pd.Series([], dtype="object")})
    for _slot in ivl_ids:
        fw_i[_slot] = np.nan
    fw_i = _round_one_decimal(fw_i)
    hc_i   = _round_one_decimal(_broadcast_daily_to_intervals(hc_d,   ivl_ids))
    att_i  = _round_one_decimal(_broadcast_daily_to_intervals(att_d,  ivl_ids))
    shr_i  = _round_one_decimal(_broadcast_daily_to_intervals(shr_d,  ivl_ids))
    trn_i  = _round_one_decimal(_broadcast_daily_to_intervals(trn_d,  ivl_ids))
    rat_i  = _round_one_decimal(_broadcast_daily_to_intervals(rat_d,  ivl_ids))
    seat_i = _round_one_decimal(_broadcast_daily_to_intervals(seat_d, ivl_ids))
    bva_i  = _round_one_decimal(_broadcast_daily_to_intervals(bva_d,  ivl_ids))
    nh_i   = _round_one_decimal(_broadcast_daily_to_intervals(nh_d,   ivl_ids))

    # --- Populate FW (interval) from native interval time series when available ---
    def _write_row(df: pd.DataFrame, row_name: str, mapping: dict[str, float]) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty or not isinstance(mapping, dict) or not mapping:
            return df
        mser = df["metric"].astype(str).str.strip()
        if row_name not in mser.values:
            return df
        cols = [c for c in ivl_ids if (c in df.columns) and (c in mapping) and (mapping.get(c) is not None)]
        if cols:
            df.loc[mser == row_name, cols] = [[float(mapping[c]) for c in cols]]
        return df

    def _slot_series(df: pd.DataFrame, val_col: str) -> dict[str, float]:
        if not isinstance(df, pd.DataFrame) or df.empty or val_col not in df.columns:
            return {}
        d = df.copy()
        # tolerant column resolution
        L = {str(c).strip().lower(): c for c in d.columns}
        c_date = L.get("date") or L.get("day")
        c_ivl  = L.get("interval") or L.get("time")
        if not c_ivl:
            return {}
        if c_date:
            d[c_date] = pd.to_datetime(d[c_date], errors="coerce").dt.date
            d = d[d[c_date].eq(ref_day)]
        d = d.dropna(subset=[c_ivl])
        if d.empty:
            return {}
        labs = d[c_ivl].astype(str).str.slice(0,5)
        vals = pd.to_numeric(d.get(val_col), errors="coerce").fillna(0.0)
        g = pd.DataFrame({"lab": labs, "val": vals}).groupby("lab", as_index=True)["val"].sum()
        return {str(k): float(v) for k, v in g.to_dict().items()}

    def _slot_weighted(df: pd.DataFrame, vol_col: str, aht_col: str) -> dict[str, float]:
        if not isinstance(df, pd.DataFrame) or df.empty or vol_col not in df.columns:
            return {}
        d = df.copy(); L = {str(c).strip().lower(): c for c in d.columns}
        c_date = L.get("date") or L.get("day"); c_ivl = L.get("interval") or L.get("time")
        if not c_ivl:
            return {}
        if c_date:
            d[c_date] = pd.to_datetime(d[c_date], errors="coerce").dt.date
            d = d[d[c_date].eq(ref_day)]
        d = d.dropna(subset=[c_ivl])
        if d.empty:
            return {}
        labs = d[c_ivl].astype(str).str.slice(0,5)
        vol = pd.to_numeric(d.get(vol_col), errors="coerce").fillna(0.0)
        aht = pd.to_numeric(d.get(aht_col), errors="coerce").fillna(np.nan)
        num = (vol * aht)
        df2 = pd.DataFrame({"lab": labs, "num": num, "den": vol})
        g = df2.groupby("lab", as_index=True)[["num","den"]].sum()
        out: dict[str, float] = {}
        for k, row in g.iterrows():
            den = float(row.get("den", 0.0) or 0.0)
            out[str(k)] = float(row.get("num", 0.0) / den) if den > 0 else 0.0
        return out

    try:
        p = get_plan(pid) or {}
        ch0 = (p.get("channel") or p.get("lob") or "").split(",")[0].strip().lower()
        sk  = _scope_key(p.get("vertical"), p.get("sub_ba"), ch0)
        if ch0 == "voice":
            vF = _assemble_voice(sk, "forecast"); vA = _assemble_voice(sk, "actual"); vT = _assemble_voice(sk, "tactical")
            volF = _slot_series(vF, "volume"); volA = _slot_series(vA, "volume"); volT = _slot_series(vT, "volume")
            ahtF = _slot_weighted(vF, "volume", "aht_sec"); ahtA = _slot_weighted(vA, "volume", "aht_sec")
            fw_i = _write_row(fw_i, "Forecast", volF)
            fw_i = _write_row(fw_i, "Tactical Forecast", volT)
            fw_i = _write_row(fw_i, "Actual Volume", volA)
            fw_i = _write_row(fw_i, "Forecast AHT/SUT", ahtF)
            fw_i = _write_row(fw_i, "Actual AHT/SUT", ahtA)
            # fallback for plans with single AHT/SUT row
            if "AHT/SUT" in fw_i["metric"].astype(str).values and not any(x in fw_i["metric"].astype(str).values for x in ["Forecast AHT/SUT","Actual AHT/SUT"]):
                use_map = ahtF or ahtA or {}
                fw_i = _write_row(fw_i, "AHT/SUT", use_map)
        elif ch0 == "chat":
            # Chat: try native interval series; otherwise leave blank
            volF = _slot_series(_load_ts_with_fallback("chat_forecast_volume", sk), "items") or _slot_series(_load_ts_with_fallback("chat_forecast_volume", sk), "volume")
            volA = _slot_series(_load_ts_with_fallback("chat_actual_volume",   sk), "items") or _slot_series(_load_ts_with_fallback("chat_actual_volume",   sk), "volume")
            ahtF = _slot_series(_load_ts_with_fallback("chat_forecast_aht",    sk), "aht_sec") or _slot_series(_load_ts_with_fallback("chat_forecast_aht",    sk), "aht")
            ahtA = _slot_series(_load_ts_with_fallback("chat_actual_aht",      sk), "aht_sec") or _slot_series(_load_ts_with_fallback("chat_actual_aht",      sk), "aht")
            fw_i = _write_row(fw_i, "Forecast", volF)
            fw_i = _write_row(fw_i, "Actual Volume", volA)
            fw_i = _write_row(fw_i, "Forecast AHT/SUT", ahtF)
            fw_i = _write_row(fw_i, "Actual AHT/SUT", ahtA)
            if "AHT/SUT" in fw_i["metric"].astype(str).values and not any(x in fw_i["metric"].astype(str).values for x in ["Forecast AHT/SUT","Actual AHT/SUT"]):
                use_map = ahtF or ahtA or {}
                fw_i = _write_row(fw_i, "AHT/SUT", use_map)
        elif ch0 in ("outbound","ob"):
            def _first_non_empty(keys: list[str]) -> pd.DataFrame:
                for k in keys:
                    dfk = _load_ts_with_fallback(k, sk)
                    if isinstance(dfk, pd.DataFrame) and not dfk.empty:
                        return dfk
                return pd.DataFrame()
            vF = _first_non_empty(["ob_forecast_opc","outbound_forecast_opc","ob_forecast_dials","outbound_forecast_dials","ob_forecast_calls"]) 
            vA = _first_non_empty(["ob_actual_opc","outbound_actual_opc","ob_actual_dials","outbound_actual_dials","ob_actual_calls"]) 
            aF = _load_ts_with_fallback("ob_forecast_aht", sk)
            aA = _load_ts_with_fallback("ob_actual_aht",   sk)
            volF = _slot_series(vF, "opc") or _slot_series(vF, "dials") or _slot_series(vF, "calls") or _slot_series(vF, "volume")
            volA = _slot_series(vA, "opc") or _slot_series(vA, "dials") or _slot_series(vA, "calls") or _slot_series(vA, "volume")
            ahtF = _slot_series(aF, "aht_sec") or _slot_series(aF, "aht")
            ahtA = _slot_series(aA, "aht_sec") or _slot_series(aA, "aht")
            fw_i = _write_row(fw_i, "Forecast", volF)
            fw_i = _write_row(fw_i, "Actual Volume", volA)
            fw_i = _write_row(fw_i, "Forecast AHT/SUT", ahtF)
            fw_i = _write_row(fw_i, "Actual AHT/SUT", ahtA)
            if "AHT/SUT" in fw_i["metric"].astype(str).values and not any(x in fw_i["metric"].astype(str).values for x in ["Forecast AHT/SUT","Actual AHT/SUT"]):
                use_map = ahtF or ahtA or {}
                fw_i = _write_row(fw_i, "AHT/SUT", use_map)
        else:
            pass
    except Exception:
        pass

    # Build base upper grid for interval view from weekly upper metrics, replicated across interval slots
    week_cols_weekly = [c for c in upper_df_w.columns if c != 'metric'] if isinstance(upper_df_w, pd.DataFrame) else []
    def _week_monday_str(d: dt.date) -> str:
        try:
            return (d - dt.timedelta(days=d.weekday())).isoformat()
        except Exception:
            return None
    ref_week = _week_monday_str(ref_day)
    if isinstance(upper_df_w, pd.DataFrame) and not upper_df_w.empty and ref_week and (ref_week in week_cols_weekly):
        upper_all = pd.DataFrame({"metric": upper_df_w["metric"].astype(str).tolist()})
        for iv in ivl_ids:
            upper_all[iv] = 0.0
        for _, row in upper_df_w.iterrows():
            name = str(row.get('metric',''))
            try:
                val = float(pd.to_numeric(row.get(ref_week), errors='coerce'))
            except Exception:
                val = 0.0
            for iv in ivl_ids:
                upper_all.loc[upper_all["metric"].eq(name), iv] = val
    else:
        upper_all = pd.concat([
            pd.DataFrame({"metric": ["FTE Required @ Forecast Volume", "FTE Required @ Actual Volume"]}),
            pd.DataFrame(0.0, index=range(2), columns=ivl_ids)
        ], axis=1)

    # Upper (interval): compute agents per interval for the representative day per channel; fallback to weekly replicated
    try:
        p = get_plan(pid) or {}
        ch0 = (p.get("channel") or p.get("lob") or "").split(",")[0].strip().lower()
        if ch0 == "voice":
            sk = _scope_key(p.get("vertical"), p.get("sub_ba"), ch0)
            settings = resolve_settings(ba=p.get("vertical"), subba=p.get("sub_ba"), lob=ch0)
            vF = _assemble_voice(sk, "forecast"); vA = _assemble_voice(sk, "actual")
            def _agents(df):
                try:
                    iv = voice_requirements_interval(df, settings)
                    iv = iv.copy(); iv["date"] = pd.to_datetime(iv["date"], errors="coerce").dt.date
                    iv = iv[iv["date"].eq(ref_day)]
                    lab = iv.get("interval")
                    if lab is None: return {}
                    s = lab.astype(str).str.slice(0,5)
                    return dict(zip(s, pd.to_numeric(iv.get("agents_req"), errors="coerce").fillna(0.0)))
                except Exception:
                    return {}
            mF = _agents(vF)
            mA = _agents(vA if isinstance(vA, pd.DataFrame) and not vA.empty else vF)
            for k, v in mF.items():
                if k in upper_all.columns:
                    upper_all.loc[upper_all["metric"].eq("FTE Required @ Forecast Volume"), k] = float(v)
            for k, v in mA.items():
                if k in upper_all.columns:
                    upper_all.loc[upper_all["metric"].eq("FTE Required @ Actual Volume"), k] = float(v)
            # Also fill FW rows from native interval series for this day
            def _fill_fw_from_voice(df, row_name, col_name):
                if not isinstance(df, pd.DataFrame) or df.empty:
                    return
                d = df.copy(); d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date
                d = d[d["date"].eq(ref_day)]
                if "interval" not in d.columns:
                    return
                labs = d["interval"].astype(str).str.slice(0,5)
                vals = pd.to_numeric(d.get(col_name), errors="coerce").fillna(0.0).tolist()
                mapping = dict(zip(labs, vals))
                m = fw_i["metric"].astype(str).str.strip()
                if row_name in m.values:
                    for slot in ivl_ids:
                        if slot in fw_i.columns and slot in mapping:
                            fw_i.loc[m == row_name, slot] = float(mapping[slot])
            _fill_fw_from_voice(vF, "Forecast", "volume")
            _fill_fw_from_voice(vA, "Actual Volume", "volume")
            _fill_fw_from_voice(vF, "Forecast AHT/SUT", "aht_sec")
            _fill_fw_from_voice(vA, "Actual AHT/SUT", "aht_sec")
            # Interval-level capacity and service level using roster staffing (Voice)
            def _parse_hhmm_to_min(hhmm: str) -> int:
                try:
                    h, m = hhmm.split(":", 1)
                    return int(h) * 60 + int(m)
                except Exception:
                    return 0
            cov_start_min = _parse_hhmm_to_min(start_hhmm)
            ivl_sec = max(60, int(ivl_min) * 60)
            T_sec = int(float(settings.get("sl_seconds", 20) or 20))
            target_sl_pct = float(settings.get("target_sl", 0.8) or 0.8) * 100.0
            occ_cap = float(settings.get("occupancy_cap_voice", 0.85) or 0.85)
            def _erlang_c(A: float, N: int) -> float:
                if N <= 0: return 1.0
                if A <= 0: return 0.0
                if A >= N: return 1.0
                term = 1.0; ssum = term
                for k in range(1, N):
                    term *= A / k
                    ssum += term
                term *= A / N
                last = term * (N / (N - A))
                denom = ssum + last
                if denom <= 0: return 1.0
                p0 = 1.0 / denom
                return last * p0
            def _erlang_sl(calls: float, aht: float, agents: float) -> float:
                if aht <= 0 or ivl_sec <= 0 or agents <= 0: return 0.0
                if calls <= 0: return 1.0
                A = (calls * aht) / ivl_sec
                pw = _erlang_c(A, int(math.floor(agents)))
                return max(0.0, min(1.0, 1.0 - pw * math.exp(-max(0.0, (agents - A)) * (T_sec / max(1.0, aht)))))
            def _erlang_calls_capacity(agents: float, aht: float) -> float:
                if agents <= 0 or aht <= 0: return 0.0
                target = target_sl_pct / 100.0
                def sl_for(x: int) -> float:
                    return _erlang_sl(x, aht, agents)
                hi = max(1, int((agents * ivl_sec) / aht))
                if sl_for(hi) < target:
                    cap_hi = hi
                else:
                    lo = 0
                    cap_hi = hi
                    while sl_for(cap_hi) >= target and cap_hi < 10000000:
                        lo = cap_hi
                        cap_hi *= 2
                    while lo < cap_hi:
                        mid = (lo + cap_hi + 1) // 2
                        if sl_for(mid) >= target:
                            lo = mid
                        else:
                            cap_hi = mid - 1
                    cap_hi = lo
                occ_erlangs = occ_cap * agents
                occ_calls_cap = (occ_erlangs * ivl_sec) / max(1.0, aht)
                return float(min(cap_hi, occ_calls_cap))
            # staffing by interval slot from roster
            def _staff_by_slot_for_day() -> dict[str, float]:
                try:
                    rl = load_roster_long()
                except Exception:
                    return {}
                if not isinstance(rl, pd.DataFrame) or rl.empty:
                    return {}
                df = rl.copy()
                def _col(opts):
                    for c in opts:
                        if c in df.columns:
                            return c
                    return None
                c_ba  = _col(["Business Area","business area","vertical"]) 
                c_sba = _col(["Sub Business Area","sub business area","sub_ba"]) 
                c_lob = _col(["LOB","lob","Channel","channel"]) 
                c_site= _col(["Site","site","Location","location","Country","country"]) 
                BA  = p.get("vertical"); SBA = p.get("sub_ba"); LOB = (p.get("channel") or p.get("lob") or "").split(",")[0].strip()
                SITE= (p.get("site") or p.get("location") or p.get("country") or "").strip()
                def _match(series, val):
                    if not val or not isinstance(series, pd.Series):
                        return pd.Series([True]*len(series))
                    s = series.astype(str).str.strip().str.lower()
                    return s.eq(str(val).strip().lower())
                msk = pd.Series([True]*len(df))
                if c_ba:  msk &= _match(df[c_ba], BA)
                if c_sba and (SBA not in (None, "")): msk &= _match(df[c_sba], SBA)
                if c_lob: msk &= _match(df[c_lob], LOB)
                if c_site and (SITE not in (None, "")): msk &= _match(df[c_site], SITE)
                df = df[msk]
                if "is_leave" in df.columns:
                    df = df[~df["is_leave"].astype(bool)]
                if "date" not in df.columns or "entry" not in df.columns:
                    return {}
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
                df = df[df["date"].eq(ref_day)]
                if df.empty:
                    return {}
                slots = {lab: 0.0 for lab in ivl_ids}
                for _, rr in df.iterrows():
                    try:
                        sft = str(rr.get("entry", "")).strip()
                        m = re.match(r"^(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})$", sft)
                        if not m:
                            continue
                        sh, sm, eh, em = map(int, m.groups())
                        sh = min(23, max(0, sh)); eh = min(24, max(0, eh))
                        start_min = sh*60 + sm
                        end_min   = eh*60 + em
                        if end_min <= start_min:
                            end_min += 24*60
                        for idx2, lab2 in enumerate(ivl_ids):
                            slot_abs = cov_start_min + idx2*ivl_min
                            slot_rel = slot_abs
                            if slot_rel < start_min:
                                slot_rel += 24*60
                            if start_min <= slot_rel < end_min:
                                slots[lab2] = slots.get(lab2, 0.0) + 1.0
                    except Exception:
                        continue
                return slots
            staff_counts = _staff_by_slot_for_day()
            # extract per-slot forecast volume and aht
            def _slot_map(df: pd.DataFrame, val_col: str) -> dict[str, float]:
                if not isinstance(df, pd.DataFrame) or df.empty:
                    return {}
                d2 = df.copy(); d2["date"] = pd.to_datetime(d2.get("date"), errors="coerce").dt.date
                d2 = d2[d2["date"].eq(ref_day)]
                if "interval" not in d2.columns:
                    return {}
                labs2 = d2["interval"].astype(str).str.slice(0,5)
                vals2 = pd.to_numeric(d2.get(val_col), errors="coerce").fillna(0.0).tolist()
                return dict(zip(labs2, vals2))
            vol_map = _slot_map(vF, "volume")
            aht_map = _slot_map(vF, "aht_sec")
            # ensure rows exist
            if "Projected Handling Capacity (#)" not in upper_all["metric"].astype(str).values:
                upper_all = pd.concat([upper_all, pd.DataFrame({"metric":["Projected Handling Capacity (#)"]})], ignore_index=True)
                for iv in ivl_ids:
                    if iv not in upper_all.columns:
                        upper_all[iv] = 0.0
            if "Projected Service Level" not in upper_all["metric"].astype(str).values:
                upper_all = pd.concat([upper_all, pd.DataFrame({"metric":["Projected Service Level"]})], ignore_index=True)
                for iv in ivl_ids:
                    if iv not in upper_all.columns:
                        upper_all[iv] = 0.0
            mser = upper_all["metric"].astype(str)
            for lab in ivl_ids:
                agents = float(staff_counts.get(lab, 0.0) or 0.0)
                aht = float(aht_map.get(lab, aht_map.get(ivl_ids[0], 300.0)) or 300.0)
                calls = float(vol_map.get(lab, 0.0))
                cap_calls = _erlang_calls_capacity(agents, aht)
                sl_pct = 100.0 * _erlang_sl(calls, aht, agents)
                upper_all.loc[mser == "Projected Handling Capacity (#)", lab] = cap_calls
                upper_all.loc[mser == "Projected Service Level", lab] = sl_pct
        elif ch0 == "chat":
            sk = _scope_key(p.get("vertical"), p.get("sub_ba"), ch0)
            s = resolve_settings(ba=p.get("vertical"), subba=p.get("sub_ba"), lob=ch0)
            # Load raw timeseries (may be daily or interval)
            volF = _load_ts_with_fallback("chat_forecast_volume", sk)
            ahtF = _load_ts_with_fallback("chat_forecast_aht", sk)
            volA = _load_ts_with_fallback("chat_actual_volume", sk)
            ahtA = _load_ts_with_fallback("chat_actual_aht", sk)

            def _chat_agents(vol_df: pd.DataFrame, aht_df: pd.DataFrame) -> dict:
                try:
                    df = vol_df.copy() if isinstance(vol_df, pd.DataFrame) else pd.DataFrame()
                    if df.empty:
                        return {}
                    L = {str(c).strip().lower(): c for c in df.columns}
                    c_date = L.get("date") or L.get("day")
                    c_ivl  = L.get("interval") or L.get("time")
                    c_itm  = L.get("items") or L.get("volume") or L.get("chats") or L.get("txns") or L.get("transactions")
                    # aht
                    if isinstance(aht_df, pd.DataFrame) and not aht_df.empty:
                        ah = aht_df.copy(); LA = {str(c).strip().lower(): c for c in ah.columns}
                        c_ad = LA.get("date") or LA.get("day"); c_ai = LA.get("interval") or LA.get("time")
                        c_as = LA.get("aht_sec") or LA.get("aht") or LA.get("avg_aht")
                        if c_as:
                            if c_ad: ah[c_ad] = pd.to_datetime(ah[c_ad], errors="coerce").dt.date
                            join = [c for c in [c_ad, c_ai] if c]
                            if not join: join = [c_ad] if c_ad else []
                            if join:
                                df = df.merge(ah[[*join, c_as]], on=join, how="left")
                                df.rename(columns={c_as: "aht_sec"}, inplace=True)
                    if "aht_sec" not in df.columns:
                        df["aht_sec"] = float(s.get("chat_aht_sec", s.get("target_aht", 240)) or 240.0)
                    # normalize
                    df["date"] = pd.to_datetime(df[c_date] if c_date else df.get("date"), errors="coerce").dt.date
                    df["items"] = pd.to_numeric(df[c_itm] if c_itm else df.get("items"), errors="coerce").fillna(0.0)
                    df = df.dropna(subset=["date"]).copy()
                    df = df[df["date"].eq(ref_day)]
                    # If no interval column, distribute uniformly across coverage window
                    if not c_ivl or c_ivl not in df.columns or df[c_ivl].isna().all():
                        coverage_min = float(s.get("chat_coverage_minutes", s.get("hours_per_fte", 8.0) * 60.0) or 480.0)
                        ivm = int(float(s.get("chat_interval_minutes", s.get("interval_minutes", 30)) or 30))
                        n = max(1, int(round(coverage_min / ivm)))
                        slot_labels = interval_cols_for_day(ref_day, ivm, start_hhmm=start_hhmm)[1]
                        total = float(df["items"].sum())
                        per = total / float(n)
                        # compute agents per slot using concurrency and chat SL/occ caps
                        conc = float(s.get("chat_concurrency", 1.5) or 1.0)
                        target_sl = float(s.get("chat_target_sl", s.get("target_sl", 0.8)) or 0.8)
                        T_sec = float(s.get("chat_sl_seconds", s.get("sl_seconds", 20)) or 20.0)
                        occ_cap = s.get("occupancy_cap_chat", s.get("util_chat", s.get("occupancy_cap_voice", 0.85)))
                        occ_cap = float(occ_cap or 0.85)
                        out = {}
                        for lab in slot_labels:
                            N, *_ = min_agents(per, float(df["aht_sec"].iloc[0]) / max(conc, 1e-6), ivm, target_sl, T_sec, occ_cap)
                            out[lab] = float(N)
                        return out
                    # With intervals provided
                    df["interval"] = df[c_ivl].astype(str)
                    ivm = df["interval"].map(lambda ssv: _ivl_minutes_from_str(ssv, int(float(s.get("chat_interval_minutes", s.get("interval_minutes", 30)) or 30)))).fillna(30).astype(int)
                    conc = float(s.get("chat_concurrency", 1.5) or 1.0)
                    target_sl = float(s.get("chat_target_sl", s.get("target_sl", 0.8)) or 0.8)
                    T_sec = float(s.get("chat_sl_seconds", s.get("sl_seconds", 20)) or 20.0)
                    occ_cap = s.get("occupancy_cap_chat", s.get("util_chat", s.get("occupancy_cap_voice", 0.85)))
                    occ_cap = float(occ_cap or 0.85)
                    labs = df["interval"].astype(str).str.slice(0,5)
                    agents = []
                    for i, r in df.iterrows():
                        calls = float(r.get("items", 0.0) or 0.0)
                        aht = float(r.get("aht_sec", 0.0) or 0.0) / max(conc, 1e-6)
                        N, *_ = min_agents(calls, aht, int(ivm.iloc[i]), target_sl, T_sec, occ_cap)
                        agents.append(float(N))
                    return dict(zip(labs, agents))
                except Exception:
                    return {}
            mF = _chat_agents(volF, ahtF)
            mA = _chat_agents(volA if isinstance(volA, pd.DataFrame) and not volA.empty else volF,
                              ahtA if isinstance(ahtA, pd.DataFrame) and not ahtA.empty else ahtF)
            for k, v in mF.items():
                if k in upper_all.columns:
                    upper_all.loc[upper_all["metric"].eq("FTE Required @ Forecast Volume"), k] = float(v)
            for k, v in mA.items():
                if k in upper_all.columns:
                    upper_all.loc[upper_all["metric"].eq("FTE Required @ Actual Volume"), k] = float(v)
            # Fill FW rows from native interval series if present; fallback to uniform already set
            def _fill_fw_from_chat(vol_df, aht_df, row_name_vol, row_name_aht):
                if not isinstance(vol_df, pd.DataFrame) or vol_df.empty:
                    return
                d = vol_df.copy(); d["date"] = pd.to_datetime(d.get("date"), errors="coerce").dt.date
                d = d[d["date"].eq(ref_day)]
                if "interval" not in d.columns:
                    return
                labs = d["interval"].astype(str).str.slice(0,5)
                vol = pd.to_numeric(d.get("items"), errors="coerce").fillna(0.0).tolist()
                mapping_vol = dict(zip(labs, vol))
                m = fw_i["metric"].astype(str).str.strip()
                if row_name_vol in m.values:
                    for slot in ivl_ids:
                        if slot in mapping_vol and slot in fw_i.columns:
                            fw_i.loc[m == row_name_vol, slot] = float(mapping_vol[slot])
                if isinstance(aht_df, pd.DataFrame) and not aht_df.empty:
                    ah = aht_df.copy(); ah["date"] = pd.to_datetime(ah.get("date"), errors="coerce").dt.date
                    ah = ah[ah["date"].eq(ref_day)]
                    if "interval" in ah.columns:
                        labs2 = ah["interval"].astype(str).str.slice(0,5)
                        ahts  = pd.to_numeric(ah.filter(regex="aht", axis=1).iloc[:, -1], errors="coerce").fillna(0.0).tolist()
                        mapping_aht = dict(zip(labs2, ahts))
                        if row_name_aht in m.values:
                            for slot in ivl_ids:
                                if slot in mapping_aht and slot in fw_i.columns:
                                    fw_i.loc[m == row_name_aht, slot] = float(mapping_aht[slot])
            _fill_fw_from_chat(volF, ahtF, "Forecast", "Forecast AHT/SUT")
            _fill_fw_from_chat(volA if isinstance(volA, pd.DataFrame) and not volA.empty else volF,
                               ahtA if isinstance(ahtA, pd.DataFrame) and not ahtA.empty else ahtF,
                               "Actual Volume", "Actual AHT/SUT")
            # Interval-level capacity and service level (Chat)
            def _parse_hhmm_to_min(hhmm: str) -> int:
                try:
                    h, m = hhmm.split(":", 1)
                    return int(h) * 60 + int(m)
                except Exception:
                    return 0
            cov_start_min = _parse_hhmm_to_min(start_hhmm)
            ivl_sec = max(60, int(ivl_min) * 60)
            T_sec = int(float(s.get("chat_sl_seconds", s.get("sl_seconds", 20)) or 20))
            target_sl_pct = float(s.get("chat_target_sl", s.get("target_sl", 0.8)) or 0.8) * 100.0
            occ_cap = float(s.get("occupancy_cap_chat", s.get("util_chat", s.get("occupancy_cap_voice", 0.85))) or 0.85)
            conc = float(s.get("chat_concurrency", 1.5) or 1.0)
            def _erlang_c(A: float, N: int) -> float:
                if N <= 0: return 1.0
                if A <= 0: return 0.0
                if A >= N: return 1.0
                term = 1.0; ssum = term
                for k in range(1, N):
                    term *= A / k
                    ssum += term
                term *= A / N
                last = term * (N / (N - A))
                denom = ssum + last
                if denom <= 0: return 1.0
                p0 = 1.0 / denom
                return last * p0
            def _erlang_sl(calls: float, aht: float, agents: float) -> float:
                if aht <= 0 or ivl_sec <= 0 or agents <= 0: return 0.0
                if calls <= 0: return 1.0
                A = (calls * aht) / ivl_sec
                pw = _erlang_c(A, int(math.floor(agents)))
                return max(0.0, min(1.0, 1.0 - pw * math.exp(-max(0.0, (agents - A)) * (T_sec / max(1.0, aht)))))
            def _erlang_calls_capacity(agents: float, aht: float) -> float:
                if agents <= 0 or aht <= 0: return 0.0
                target = target_sl_pct / 100.0
                def sl_for(x: int) -> float:
                    return _erlang_sl(x, aht, agents)
                hi = max(1, int((agents * ivl_sec) / aht))
                if sl_for(hi) < target:
                    cap_hi = hi
                else:
                    lo = 0
                    cap_hi = hi
                    while sl_for(cap_hi) >= target and cap_hi < 10000000:
                        lo = cap_hi
                        cap_hi *= 2
                    while lo < cap_hi:
                        mid = (lo + cap_hi + 1) // 2
                        if sl_for(mid) >= target:
                            lo = mid
                        else:
                            cap_hi = mid - 1
                    cap_hi = lo
                occ_erlangs = occ_cap * agents
                occ_calls_cap = (occ_erlangs * ivl_sec) / max(1.0, aht)
                return float(min(cap_hi, occ_calls_cap))
            # staffing counts by slot
            def _staff_by_slot_for_day() -> dict[str, float]:
                try:
                    rl = load_roster_long()
                except Exception:
                    return {}
                if not isinstance(rl, pd.DataFrame) or rl.empty:
                    return {}
                df = rl.copy()
                def _col(opts):
                    for c in opts:
                        if c in df.columns:
                            return c
                    return None
                c_ba  = _col(["Business Area","business area","vertical"]) 
                c_sba = _col(["Sub Business Area","sub business area","sub_ba"]) 
                c_lob = _col(["LOB","lob","Channel","channel"]) 
                c_site= _col(["Site","site","Location","location","Country","country"]) 
                BA  = p.get("vertical"); SBA = p.get("sub_ba"); LOB = (p.get("channel") or p.get("lob") or "").split(",")[0].strip()
                SITE= (p.get("site") or p.get("location") or p.get("country") or "").strip()
                def _match(series, val):
                    if not val or not isinstance(series, pd.Series):
                        return pd.Series([True]*len(series))
                    s = series.astype(str).str.strip().str.lower()
                    return s.eq(str(val).strip().lower())
                msk = pd.Series([True]*len(df))
                if c_ba:  msk &= _match(df[c_ba], BA)
                if c_sba and (SBA not in (None, "")): msk &= _match(df[c_sba], SBA)
                if c_lob: msk &= _match(df[c_lob], LOB)
                if c_site and (SITE not in (None, "")): msk &= _match(df[c_site], SITE)
                df = df[msk]
                if "is_leave" in df.columns:
                    df = df[~df["is_leave"].astype(bool)]
                if "date" not in df.columns or "entry" not in df.columns:
                    return {}
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
                df = df[df["date"].eq(ref_day)]
                if df.empty:
                    return {}
                slots = {lab: 0.0 for lab in ivl_ids}
                for _, rr in df.iterrows():
                    try:
                        sft = str(rr.get("entry", "")).strip()
                        m = re.match(r"^(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})$", sft)
                        if not m:
                            continue
                        sh, sm, eh, em = map(int, m.groups())
                        sh = min(23, max(0, sh)); eh = min(24, max(0, eh))
                        start_min = sh*60 + sm
                        end_min   = eh*60 + em
                        if end_min <= start_min:
                            end_min += 24*60
                        for idx2, lab2 in enumerate(ivl_ids):
                            slot_abs = cov_start_min + idx2*ivl_min
                            slot_rel = slot_abs
                            if slot_rel < start_min:
                                slot_rel += 24*60
                            if start_min <= slot_rel < end_min:
                                slots[lab2] = slots.get(lab2, 0.0) + 1.0
                    except Exception:
                        continue
                return slots
            staff_counts = _staff_by_slot_for_day()
            # per-slot items and aht
            def _slot_map(df: pd.DataFrame, val_col: str) -> dict[str, float]:
                if not isinstance(df, pd.DataFrame) or df.empty:
                    return {}
                d2 = df.copy(); d2["date"] = pd.to_datetime(d2.get("date"), errors="coerce").dt.date
                d2 = d2[d2["date"].eq(ref_day)]
                if "interval" not in d2.columns:
                    return {}
                labs2 = d2["interval"].astype(str).str.slice(0,5)
                vals2 = pd.to_numeric(d2.get(val_col), errors="coerce").fillna(0.0).tolist()
                return dict(zip(labs2, vals2))
            vol_map = _slot_map(volF, "items")
            aht_map = _slot_map(ahtF, "aht_sec") if isinstance(ahtF, pd.DataFrame) else {}
            # ensure rows exist
            if "Projected Handling Capacity (#)" not in upper_all["metric"].astype(str).values:
                upper_all = pd.concat([upper_all, pd.DataFrame({"metric":["Projected Handling Capacity (#)"]})], ignore_index=True)
                for iv in ivl_ids:
                    if iv not in upper_all.columns:
                        upper_all[iv] = 0.0
            if "Projected Service Level" not in upper_all["metric"].astype(str).values:
                upper_all = pd.concat([upper_all, pd.DataFrame({"metric":["Projected Service Level"]})], ignore_index=True)
                for iv in ivl_ids:
                    if iv not in upper_all.columns:
                        upper_all[iv] = 0.0
            mser = upper_all["metric"].astype(str)
            for lab in ivl_ids:
                agents = float(staff_counts.get(lab, 0.0) or 0.0)
                aht_eff = float(aht_map.get(lab, aht_map.get(ivl_ids[0], 240.0)) or 240.0) / max(0.1, conc)
                calls = float(vol_map.get(lab, 0.0))
                cap_calls = _erlang_calls_capacity(agents, aht_eff)
                sl_pct = 100.0 * _erlang_sl(calls, aht_eff, agents)
                upper_all.loc[mser == "Projected Handling Capacity (#)", lab] = cap_calls
                upper_all.loc[mser == "Projected Service Level", lab] = sl_pct
        elif ch0 in ("outbound", "ob"):
            sk = _scope_key(p.get("vertical"), p.get("sub_ba"), ch0)
            s = resolve_settings(ba=p.get("vertical"), subba=p.get("sub_ba"), lob=ch0)
            # Load raw timeseries (multiple keys with fallbacks)
            def _first_non_empty(keys: list[str]) -> pd.DataFrame:
                for k in keys:
                    df = _load_ts_with_fallback(k, sk)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        return df.copy()
                return pd.DataFrame()
            volF = _first_non_empty(["ob_forecast_opc","outbound_forecast_opc","ob_forecast_dials","outbound_forecast_dials","ob_forecast_calls"]) 
            connF= _first_non_empty(["ob_forecast_connect_rate","outbound_forecast_connect_rate","ob_forecast_connect%"])
            rpcF = _first_non_empty(["ob_forecast_rpc"]) 
            rpcrF= _first_non_empty(["ob_forecast_rpc_rate"]) 
            ahtF = _first_non_empty(["ob_forecast_aht"]) 
            volA = _first_non_empty(["ob_actual_opc","outbound_actual_opc","ob_actual_dials","outbound_actual_dials","ob_actual_calls"]) 
            connA= _first_non_empty(["ob_actual_connect_rate","outbound_actual_connect_rate","ob_actual_connect%"])
            rpcA = _first_non_empty(["ob_actual_rpc"]) 
            rpcrA= _first_non_empty(["ob_actual_rpc_rate"]) 
            ahtA = _first_non_empty(["ob_actual_aht"]) 

            def _expected(vdf, cdf, rdf, rrd, adf):
                if not isinstance(vdf, pd.DataFrame) or vdf.empty:
                    return pd.DataFrame()
                d = vdf.copy(); LV = {str(c).strip().lower(): c for c in d.columns}
                c_date = LV.get("date") or LV.get("day"); c_ivl = LV.get("interval") or LV.get("time")
                c_opc  = LV.get("opc") or LV.get("dials") or LV.get("calls") or LV.get("volume")
                out = pd.DataFrame({
                    "date": pd.to_datetime(d[c_date] if c_date else d.get("date"), errors="coerce").dt.date,
                    "interval": (d[c_ivl].astype(str) if c_ivl else None),
                    "opc": pd.to_numeric(d[c_opc] if c_opc else d.get("opc"), errors="coerce").fillna(0.0),
                })
                # merge rates/rpc
                def _mrg(base, df, pick):
                    if not isinstance(df, pd.DataFrame) or df.empty:
                        return base
                    t = df.copy(); L = {str(c).strip().lower(): c for c in t.columns}
                    cd = L.get("date") or L.get("day"); ci = L.get("interval") or L.get("time"); cv = L.get(pick)
                    if cv is None:
                        cv = next((L.get(nm) for nm in [pick, pick.replace("_", " "), pick.replace("_", "")] if L.get(nm)), None)
                    if cv is None:
                        return base
                    if cd: t[cd] = pd.to_datetime(t[cd], errors="coerce").dt.date
                    join = [c for c in [cd, ci] if c]
                    if not join:
                        join = [cd] if cd else []
                    if not join:
                        return base
                    t = t[[*join, cv]].copy(); t.rename(columns={cv: pick}, inplace=True)
                    return base.merge(t, on=join, how="left")
                out = _mrg(out, cdf, "connect_rate")
                out = _mrg(out, rdf, "rpc")
                out = _mrg(out, rrd, "rpc_rate")
                out = _mrg(out, adf, "aht")
                # defaults
                out["connect_rate"] = out.get("connect_rate", 0.0)
                out["rpc_rate"] = out.get("rpc_rate", 0.0)
                out["rpc"] = out.get("rpc", 0.0)
                out["aht"] = pd.to_numeric(out.get("aht"), errors="coerce").fillna(float(s.get("ob_aht_sec", s.get("target_aht", 240)) or 240.0))
                # limit to representative day
                out = out.dropna(subset=["date"]).copy(); out = out[out["date"].eq(ref_day)]
                return out

            F = _expected(volF, connF, rpcF, rpcrF, ahtF)
            A = _expected(volA if isinstance(volA, pd.DataFrame) and not volA.empty else volF,
                          connA if isinstance(connA, pd.DataFrame) and not connA.empty else connF,
                          rpcA if isinstance(rpcA, pd.DataFrame) and not rpcA.empty else rpcF,
                          rpcrA if isinstance(rpcrA, pd.DataFrame) and not rpcrA.empty else rpcrF,
                          ahtA if isinstance(ahtA, pd.DataFrame) and not ahtA.empty else ahtF)

            def _calls(row):
                opc = float(row.get("opc", 0.0) or 0.0)
                rpc_ct = float(row.get("rpc", 0.0) or 0.0)
                conn = row.get("connect_rate", None)
                rpcr = row.get("rpc_rate", None)
                try:
                    conn = float(conn)
                except Exception:
                    conn = None
                try:
                    rpcr = float(rpcr)
                except Exception:
                    rpcr = None
                if rpc_ct and rpc_ct > 0: return rpc_ct
                if (conn is not None) and (rpcr is not None): return opc * conn * rpcr
                if conn is not None: return opc * conn
                return opc

            def _agents(df) -> dict:
                if not isinstance(df, pd.DataFrame) or df.empty:
                    return {}
                labs = None
                if "interval" in df.columns and df["interval"].notna().any():
                    df["interval"] = df["interval"].astype(str)
                    ivm = df["interval"].map(lambda ssv: _ivl_minutes_from_str(ssv, int(float(s.get("ob_interval_minutes", s.get("interval_minutes", 30)) or 30)))).fillna(30).astype(int)
                    labs = df["interval"].astype(str).str.slice(0,5)
                    target_sl = float(s.get("ob_target_sl", s.get("target_sl", 0.8)) or 0.8)
                    T_sec = float(s.get("ob_sl_seconds", s.get("sl_seconds", 20)) or 20.0)
                    occ_cap = s.get("occupancy_cap_ob", s.get("util_ob", s.get("occupancy_cap_voice", 0.85)))
                    occ_cap = float(occ_cap or 0.85)
                    agents = []
                    for i, r in df.iterrows():
                        calls = float(_calls(r))
                        aht = float(r.get("aht", 0.0) or 0.0)
                        N, *_ = min_agents(calls, aht, int(ivm.iloc[i]), target_sl, T_sec, occ_cap)
                        agents.append(float(N))
                    return dict(zip(labs, agents))
                # Uniform fallback if no interval labels
                ivm = int(float(s.get("ob_interval_minutes", s.get("interval_minutes", 30)) or 30))
                coverage_min = float(s.get("ob_coverage_minutes", s.get("hours_per_fte", 8.0) * 60.0) or 480.0)
                n = max(1, int(round(coverage_min / ivm)))
                labs = interval_cols_for_day(ref_day, ivm, start_hhmm=start_hhmm)[1]
                total_calls = float(df.apply(_calls, axis=1).sum())
                per = total_calls / float(n)
                target_sl = float(s.get("ob_target_sl", s.get("target_sl", 0.8)) or 0.8)
                T_sec = float(s.get("ob_sl_seconds", s.get("sl_seconds", 20)) or 20.0)
                occ_cap = s.get("occupancy_cap_ob", s.get("util_ob", s.get("occupancy_cap_voice", 0.85)))
                occ_cap = float(occ_cap or 0.85)
                aht = float(df.get("aht").dropna().iloc[0] if "aht" in df.columns and df["aht"].notna().any() else (s.get("ob_aht_sec", s.get("target_aht", 240)) or 240.0))
                out = {}
                for lab in labs:
                    N, *_ = min_agents(per, aht, ivm, target_sl, T_sec, occ_cap)
                    out[lab] = float(N)
                return out

            mF = _agents(F)
            mA = _agents(A if isinstance(A, pd.DataFrame) and not A.empty else F)
            for k, v in mF.items():
                if k in upper_all.columns:
                    upper_all.loc[upper_all["metric"].eq("FTE Required @ Forecast Volume"), k] = float(v)
            for k, v in mA.items():
                if k in upper_all.columns:
                    upper_all.loc[upper_all["metric"].eq("FTE Required @ Actual Volume"), k] = float(v)
            # Fill FW rows from native interval series if present
            def _fill_fw_from_ob(df, row_name_vol, row_name_aht):
                if not isinstance(df, pd.DataFrame) or df.empty:
                    return
                d = df.copy(); d["date"] = pd.to_datetime(d.get("date"), errors="coerce").dt.date
                d = d[d["date"].eq(ref_day)]
                if "interval" in d.columns and d["interval"].notna().any():
                    labs = d["interval"].astype(str).str.slice(0,5)
                    vol = pd.to_numeric(d.get("opc") if "opc" in d.columns else d.get("calls"), errors="coerce").fillna(0.0).tolist()
                    aht = pd.to_numeric(d.get("aht"), errors="coerce").fillna(0.0).tolist()
                    mapping_vol = dict(zip(labs, vol)); mapping_aht = dict(zip(labs, aht))
                    m = fw_i["metric"].astype(str).str.strip()
                    if row_name_vol in m.values:
                        for slot in ivl_ids:
                            if slot in mapping_vol and slot in fw_i.columns:
                                fw_i.loc[m == row_name_vol, slot] = float(mapping_vol[slot])
                    if row_name_aht in m.values:
                        for slot in ivl_ids:
                            if slot in mapping_aht and slot in fw_i.columns:
                                fw_i.loc[m == row_name_aht, slot] = float(mapping_aht[slot])
            _fill_fw_from_ob(F, "Forecast", "Forecast AHT/SUT")
            _fill_fw_from_ob(A if isinstance(A, pd.DataFrame) and not A.empty else F, "Actual Volume", "Actual AHT/SUT")
            # Interval-level capacity and service level (Outbound)
            def _parse_hhmm_to_min(hhmm: str) -> int:
                try:
                    h, m = hhmm.split(":", 1)
                    return int(h) * 60 + int(m)
                except Exception:
                    return 0
            cov_start_min = _parse_hhmm_to_min(start_hhmm)
            ivl_sec = max(60, int(ivl_min) * 60)
            T_sec = int(float(s.get("ob_sl_seconds", s.get("sl_seconds", 20)) or 20))
            target_sl_pct = float(s.get("ob_target_sl", s.get("target_sl", 0.8)) or 0.8) * 100.0
            occ_cap = float(s.get("occupancy_cap_ob", s.get("util_ob", s.get("occupancy_cap_voice", 0.85))) or 0.85)
            def _erlang_c(A: float, N: int) -> float:
                if N <= 0: return 1.0
                if A <= 0: return 0.0
                if A >= N: return 1.0
                term = 1.0; ssum = term
                for k in range(1, N):
                    term *= A / k
                    ssum += term
                term *= A / N
                last = term * (N / (N - A))
                denom = ssum + last
                if denom <= 0: return 1.0
                p0 = 1.0 / denom
                return last * p0
            def _erlang_sl(calls: float, aht: float, agents: float) -> float:
                if aht <= 0 or ivl_sec <= 0 or agents <= 0: return 0.0
                if calls <= 0: return 1.0
                A = (calls * aht) / ivl_sec
                pw = _erlang_c(A, int(math.floor(agents)))
                return max(0.0, min(1.0, 1.0 - pw * math.exp(-max(0.0, (agents - A)) * (T_sec / max(1.0, aht)))))
            def _erlang_calls_capacity(agents: float, aht: float) -> float:
                if agents <= 0 or aht <= 0: return 0.0
                target = target_sl_pct / 100.0
                def sl_for(x: int) -> float:
                    return _erlang_sl(x, aht, agents)
                hi = max(1, int((agents * ivl_sec) / aht))
                if sl_for(hi) < target:
                    cap_hi = hi
                else:
                    lo = 0
                    cap_hi = hi
                    while sl_for(cap_hi) >= target and cap_hi < 10000000:
                        lo = cap_hi
                        cap_hi *= 2
                    while lo < cap_hi:
                        mid = (lo + cap_hi + 1) // 2
                        if sl_for(mid) >= target:
                            lo = mid
                        else:
                            cap_hi = mid - 1
                    cap_hi = lo
                occ_erlangs = occ_cap * agents
                occ_calls_cap = (occ_erlangs * ivl_sec) / max(1.0, aht)
                return float(min(cap_hi, occ_calls_cap))
            # staffing counts by slot
            def _staff_by_slot_for_day() -> dict[str, float]:
                try:
                    rl = load_roster_long()
                except Exception:
                    return {}
                if not isinstance(rl, pd.DataFrame) or rl.empty:
                    return {}
                df = rl.copy()
                def _col(opts):
                    for c in opts:
                        if c in df.columns:
                            return c
                    return None
                c_ba  = _col(["Business Area","business area","vertical"]) 
                c_sba = _col(["Sub Business Area","sub business area","sub_ba"]) 
                c_lob = _col(["LOB","lob","Channel","channel"]) 
                c_site= _col(["Site","site","Location","location","Country","country"]) 
                BA  = p.get("vertical"); SBA = p.get("sub_ba"); LOB = (p.get("channel") or p.get("lob") or "").split(",")[0].strip()
                SITE= (p.get("site") or p.get("location") or p.get("country") or "").strip()
                def _match(series, val):
                    if not val or not isinstance(series, pd.Series):
                        return pd.Series([True]*len(series))
                    s = series.astype(str).str.strip().str.lower()
                    return s.eq(str(val).strip().lower())
                msk = pd.Series([True]*len(df))
                if c_ba:  msk &= _match(df[c_ba], BA)
                if c_sba and (SBA not in (None, "")): msk &= _match(df[c_sba], SBA)
                if c_lob: msk &= _match(df[c_lob], LOB)
                if c_site and (SITE not in (None, "")): msk &= _match(df[c_site], SITE)
                df = df[msk]
                if "is_leave" in df.columns:
                    df = df[~df["is_leave"].astype(bool)]
                if "date" not in df.columns or "entry" not in df.columns:
                    return {}
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
                df = df[df["date"].eq(ref_day)]
                if df.empty:
                    return {}
                slots = {lab: 0.0 for lab in ivl_ids}
                for _, rr in df.iterrows():
                    try:
                        sft = str(rr.get("entry", "")).strip()
                        m = re.match(r"^(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})$", sft)
                        if not m:
                            continue
                        sh, sm, eh, em = map(int, m.groups())
                        sh = min(23, max(0, sh)); eh = min(24, max(0, eh))
                        start_min = sh*60 + sm
                        end_min   = eh*60 + em
                        if end_min <= start_min:
                            end_min += 24*60
                        for idx2, lab2 in enumerate(ivl_ids):
                            slot_abs = cov_start_min + idx2*ivl_min
                            slot_rel = slot_abs
                            if slot_rel < start_min:
                                slot_rel += 24*60
                            if start_min <= slot_rel < end_min:
                                slots[lab2] = slots.get(lab2, 0.0) + 1.0
                    except Exception:
                        continue
                return slots
            staff_counts = _staff_by_slot_for_day()
            # Map expected calls per slot from F
            def _calls_map(df: pd.DataFrame) -> dict[str, float]:
                if not isinstance(df, pd.DataFrame) or df.empty:
                    return {}
                d2 = df.copy(); d2["date"] = pd.to_datetime(d2.get("date"), errors="coerce").dt.date
                d2 = d2[d2["date"].eq(ref_day)]
                if "interval" not in d2.columns:
                    return {}
                labs2 = d2["interval"].astype(str).str.slice(0,5)
                def _calls_row(row):
                    opc = float(row.get("opc", 0.0) or 0.0)
                    rpc_ct = float(row.get("rpc", 0.0) or 0.0)
                    conn = row.get("connect_rate", None)
                    rpcr = row.get("rpc_rate", None)
                    try: conn = float(conn)
                    except Exception: conn = None
                    try: rpcr = float(rpcr)
                    except Exception: rpcr = None
                    if rpc_ct and rpc_ct > 0: return rpc_ct
                    if (conn is not None) and (rpcr is not None): return opc * conn * rpcr
                    if conn is not None: return opc * conn
                    return opc
                calls = d2.apply(_calls_row, axis=1).astype(float).tolist()
                return dict(zip(labs2, calls))
            calls_map = _calls_map(F)
            aht_map = {}
            if isinstance(F, pd.DataFrame) and not F.empty and "aht" in F.columns:
                d3 = F.copy(); d3 = d3[d3["date"].eq(ref_day)]
                labs3 = d3["interval"].astype(str).str.slice(0,5)
                aht_map = dict(zip(labs3, pd.to_numeric(d3.get("aht"), errors="coerce").fillna(0.0).tolist()))
            if "Projected Handling Capacity (#)" not in upper_all["metric"].astype(str).values:
                upper_all = pd.concat([upper_all, pd.DataFrame({"metric":["Projected Handling Capacity (#)"]})], ignore_index=True)
                for iv in ivl_ids:
                    if iv not in upper_all.columns:
                        upper_all[iv] = 0.0
            if "Projected Service Level" not in upper_all["metric"].astype(str).values:
                upper_all = pd.concat([upper_all, pd.DataFrame({"metric":["Projected Service Level"]})], ignore_index=True)
                for iv in ivl_ids:
                    if iv not in upper_all.columns:
                        upper_all[iv] = 0.0
            mser = upper_all["metric"].astype(str)
            for lab in ivl_ids:
                agents = float(staff_counts.get(lab, 0.0) or 0.0)
                aht = float(aht_map.get(lab, aht_map.get(ivl_ids[0], 240.0)) or 240.0)
                calls = float(calls_map.get(lab, 0.0))
                cap_calls = _erlang_calls_capacity(agents, aht)
                sl_pct = 100.0 * _erlang_sl(calls, aht, agents)
                upper_all.loc[mser == "Projected Handling Capacity (#)", lab] = cap_calls
                upper_all.loc[mser == "Projected Service Level", lab] = sl_pct
        else:
            pass
    except Exception:
        pass

    upper_tbl = _make_upper_table(_round_one_decimal(upper_all), ivl_cols)

    return (
        upper_tbl,
        fw_i.to_dict("records"),
        hc_i.to_dict("records"),
        att_i.to_dict("records"),
        shr_i.to_dict("records"),
        trn_i.to_dict("records"),
        rat_i.to_dict("records"),
        seat_i.to_dict("records"),
        bva_i.to_dict("records"),
        nh_i.to_dict("records"),
        roster_rec,
        bulk_files_rec,
        notes_rec,
    )
