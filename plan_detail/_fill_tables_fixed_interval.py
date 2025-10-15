from __future__ import annotations
import pandas as pd
import numpy as np
import datetime as dt
from typing import List, Tuple
from dash import dash_table

from plan_store import get_plan
from cap_store import resolve_settings
from ._common import _week_span, _scope_key, _assemble_voice, _load_ts_with_fallback
from ._calc import _fill_tables_fixed
from ._grain_cols import interval_cols_for_day
from capacity_core import voice_requirements_interval, min_agents, _ivl_minutes_from_str


def _broadcast_daily_to_intervals(df: pd.DataFrame, interval_ids: List[str]) -> pd.DataFrame:
    """Convert a daily-matrix DataFrame (metric + date cols) to an interval-matrix
    for a single representative day by uniformly distributing additive rows across intervals.
    Percent/AHT/SUT-like rows are replicated per interval.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["metric"] + list(interval_ids))

    # Identify daily columns (ISO dates)
    day_cols = [c for c in df.columns if c != "metric"]
    if not day_cols:
        return pd.DataFrame(columns=["metric"] + list(interval_ids))

    # Use the first day as representative
    rep_day = day_cols[0]

    out = pd.DataFrame({"metric": df["metric"].astype(str)})
    for ivl in interval_ids:
        out[ivl] = 0.0

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

    n = max(1, len(interval_ids))
    for i, row in df.iterrows():
        name = str(row.get("metric", ""))
        try:
            val = float(pd.to_numeric(row.get(rep_day), errors="coerce"))
        except Exception:
            val = 0.0
        if is_percent_like(name):
            for ivl in interval_ids:
                out.at[i, ivl] = val
        else:
            per = float(val) / float(n)
            for ivl in interval_ids:
                out.at[i, ivl] = per

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


def _fill_tables_fixed_interval(ptype, pid, _fw_cols_unused, _tick, whatif=None, ivl_min: int = 30):
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

    # Weekly -> daily (single representative day = first week Monday)
    monday = pd.to_datetime(weeks[0]).date() if weeks else dt.date.today()
    rep_day = monday.isoformat()
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

    # Daily -> intervals
    ivl_cols, ivl_ids = interval_cols_for_day(monday, ivl_min=ivl_min)
    fw_i   = _round_one_decimal(_broadcast_daily_to_intervals(fw_d,   ivl_ids))
    hc_i   = _round_one_decimal(_broadcast_daily_to_intervals(hc_d,   ivl_ids))
    att_i  = _round_one_decimal(_broadcast_daily_to_intervals(att_d,  ivl_ids))
    shr_i  = _round_one_decimal(_broadcast_daily_to_intervals(shr_d,  ivl_ids))
    trn_i  = _round_one_decimal(_broadcast_daily_to_intervals(trn_d,  ivl_ids))
    rat_i  = _round_one_decimal(_broadcast_daily_to_intervals(rat_d,  ivl_ids))
    seat_i = _round_one_decimal(_broadcast_daily_to_intervals(seat_d, ivl_ids))
    bva_i  = _round_one_decimal(_broadcast_daily_to_intervals(bva_d,  ivl_ids))
    nh_i   = _round_one_decimal(_broadcast_daily_to_intervals(nh_d,   ivl_ids))

    # Upper (interval): compute agents per interval for the representative day per channel; fallback to broadcast
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
                    iv = iv[iv["date"].eq(monday)]
                    lab = iv.get("interval")
                    if lab is None: return {}
                    s = lab.astype(str).str.slice(0,5)
                    return dict(zip(s, pd.to_numeric(iv.get("agents_req"), errors="coerce").fillna(0.0)))
                except Exception:
                    return {}
            mF = _agents(vF)
            mA = _agents(vA if isinstance(vA, pd.DataFrame) and not vA.empty else vF)
            upper_i = pd.DataFrame({"metric": ["FTE Required @ Forecast Volume", "FTE Required @ Actual Volume"]})
            for slot in ivl_ids:
                upper_i[slot] = 0.0
            for k, v in mF.items():
                if k in upper_i.columns:
                    upper_i.loc[upper_i["metric"].eq("FTE Required @ Forecast Volume"), k] = float(v)
            for k, v in mA.items():
                if k in upper_i.columns:
                    upper_i.loc[upper_i["metric"].eq("FTE Required @ Actual Volume"), k] = float(v)
            upper_tbl = _make_upper_table(_round_one_decimal(upper_i), ivl_cols)
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
                    df = df[df["date"].eq(monday)]
                    # If no interval column, distribute uniformly across coverage window
                    if not c_ivl or c_ivl not in df.columns or df[c_ivl].isna().all():
                        coverage_min = float(s.get("chat_coverage_minutes", s.get("hours_per_fte", 8.0) * 60.0) or 480.0)
                        ivm = int(float(s.get("chat_interval_minutes", s.get("interval_minutes", 30)) or 30))
                        n = max(1, int(round(coverage_min / ivm)))
                        slot_labels = interval_cols_for_day(monday, ivm)[1]
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
            upper_i = pd.DataFrame({"metric": ["FTE Required @ Forecast Volume", "FTE Required @ Actual Volume"]})
            for slot in ivl_ids:
                upper_i[slot] = 0.0
            for k, v in mF.items():
                if k in upper_i.columns:
                    upper_i.loc[upper_i["metric"].eq("FTE Required @ Forecast Volume"), k] = float(v)
            for k, v in mA.items():
                if k in upper_i.columns:
                    upper_i.loc[upper_i["metric"].eq("FTE Required @ Actual Volume"), k] = float(v)
            upper_tbl = _make_upper_table(_round_one_decimal(upper_i), ivl_cols)
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
                out = out.dropna(subset=["date"]).copy(); out = out[out["date"].eq(monday)]
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
                labs = interval_cols_for_day(monday, ivm)[1]
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
            upper_i = pd.DataFrame({"metric": ["FTE Required @ Forecast Volume", "FTE Required @ Actual Volume"]})
            for slot in ivl_ids:
                upper_i[slot] = 0.0
            for k, v in mF.items():
                if k in upper_i.columns:
                    upper_i.loc[upper_i["metric"].eq("FTE Required @ Forecast Volume"), k] = float(v)
            for k, v in mA.items():
                if k in upper_i.columns:
                    upper_i.loc[upper_i["metric"].eq("FTE Required @ Actual Volume"), k] = float(v)
            upper_tbl = _make_upper_table(_round_one_decimal(upper_i), ivl_cols)
        else:
            upper_tbl = _make_upper_table(_round_one_decimal(_broadcast_daily_to_intervals(upper_d, ivl_ids)), ivl_cols)
    except Exception:
        upper_tbl = _make_upper_table(_round_one_decimal(_broadcast_daily_to_intervals(upper_d, ivl_ids)), ivl_cols)

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
