from __future__ import annotations
import json
import datetime as dt
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import datetime as dt
import pandas as pd
import getpass
import os

# Reuse the same DB file/connection as the rest of the app
from cap_db import _conn
from cap_db import delete_datasets_by_prefix

# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────────────────────────────────────────
def _init():
    with _conn() as cx:
        cx.execute("""
        CREATE TABLE IF NOT EXISTS capacity_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            org TEXT,
            business_entity TEXT,
            vertical TEXT,            -- Business Area
            sub_ba TEXT,
            channel TEXT,             -- CSV (normalized)
            location TEXT,            -- NEW: location/country/city
            site TEXT,                -- campus/site
            plan_name TEXT NOT NULL,
            plan_type TEXT,
            start_week TEXT,          -- YYYY-MM-DD (Monday)
            end_week TEXT,            -- YYYY-MM-DD
            ft_weekly_hours REAL,
            pt_weekly_hours REAL,
            tags TEXT,                -- JSON list or ""
            is_current INTEGER DEFAULT 0,
            status TEXT DEFAULT 'draft',  -- 'current' | 'history' | 'draft'
            hierarchy_json TEXT,      -- optional BA/SubBA/Channels/Site bundle
            owner TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """)
        cx.commit()

_init()

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def _norm_text(s) -> str:
    return ("" if s is None else str(s)).strip().lower()

def _norm_channel_csv(x) -> str:
    """
    Return a normalized, sorted CSV for a channel field that may be a list or CSV.
    All lower-case, spaces trimmed, sorted and deduped.
    """
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        parts = [str(v).strip().lower() for v in x if str(v).strip()]
    else:
        parts = [p.strip().lower() for p in str(x).split(",") if p.strip()]
    parts = sorted(set(parts))
    return ", ".join(parts)

# plan_store.py
def touch_plan(pid: int, user: str):
    with _conn() as cx:
        cx.execute(
            "UPDATE capacity_plans SET updated_at = ?, updated_by = ? WHERE id = ?",
            (pd.Timestamp.utcnow().isoformat(), user, pid)
        )
        cx.commit()

def _user():
    return os.environ.get("HOSTNAME") or os.environ.get("USERNAME") or getpass.getuser() or "system"

# ---- Auto-lock plans by month -------------------------------------------------
def _first_of_month(d: dt.date) -> dt.date:
    return dt.date(d.year, d.month, 1)

def auto_lock_previous_month_plans() -> int:
    """
    Automatically lock plans whose end_week falls strictly before the first day of the previous month.
    Examples with current month M:
      - On Sep 1: cutoff = Aug 1 → plans ending before Aug 1 (i.e., July and earlier) are locked.
      - On Oct 1: cutoff = Sep 1 → plans ending before Sep 1 (i.e., August and earlier) are locked.
    Returns count of plans updated.
    """
    today = dt.date.today()
    first_curr = _first_of_month(today)
    # first day of previous month
    prev_month_first = (first_curr - dt.timedelta(days=1)).replace(day=1)
    cutoff = prev_month_first.isoformat()
    with _conn() as cx:
        # Identify current plans with end_week < cutoff
        rows = cx.execute(
            """
            SELECT id FROM capacity_plans
             WHERE is_current=1
               AND COALESCE(end_week,'') <> ''
               AND end_week < ?
            """,
            (cutoff,)
        ).fetchall()
        ids = [r["id"] if isinstance(r, dict) else r[0] for r in rows]
        if not ids:
            return 0
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        placeholders = ",".join("?" for _ in ids)
        cx.execute(
            f"UPDATE capacity_plans SET is_current=0, status='history', updated_at=? WHERE id IN ({placeholders})",
            [ts] + ids,
        )
        cx.commit()
        return len(ids)
# ──────────────────────────────────────────────────────────────────────────────
# Create / Update
# ──────────────────────────────────────────────────────────────────────────────
def create_plan(payload: dict) -> int:
    with _conn() as cx:
        vertical   = (payload.get("vertical") or "").strip()
        sub_ba     = (payload.get("sub_ba") or "").strip()
        name       = (payload.get("plan_name") or "").strip()
        location   = (payload.get("location") or "").strip()
        site       = (payload.get("site") or "").strip()
        chan_norm  = _norm_channel_csv(payload.get("channel"))
        is_current = 1 if payload.get("is_current") else 0
        status     = payload.get("status") or ("current" if is_current else "draft")
        cols = [r["name"] for r in cx.execute("PRAGMA table_info(capacity_plans)").fetchall()]
        colset = set(cols)
        not_deleted_clause = ""
        if "is_deleted" in colset:
            not_deleted_clause = " AND COALESCE(is_deleted,0)=0"
        elif "deleted_at" in colset:
            not_deleted_clause = " AND deleted_at IS NULL"
        

        # --- Rule #1: Exact-duplicate guard ----------------------------------
        # Match BA+SBA+PlanName+Location+Site (case-insensitive) and then confirm
        # the normalized Channel-set matches.
        dup_sql = """
            SELECT id, channel, location, site, plan_name
              FROM capacity_plans
             WHERE LOWER(vertical) = LOWER(?)
               AND COALESCE(sub_ba,'') = COALESCE(?, '')
               AND LOWER(TRIM(plan_name)) = LOWER(?)
               AND LOWER(COALESCE(TRIM(location),'')) = LOWER(COALESCE(?, ''))
               AND LOWER(COALESCE(TRIM(site),''))     = LOWER(COALESCE(?, ''))
        """ + not_deleted_clause
        dup_rows = cx.execute(dup_sql, (vertical, sub_ba, name, location, site)).fetchall()

        for r in dup_rows:
            if _norm_channel_csv(r["channel"]) == chan_norm:
                # same Channel-set + same Location/Site + same Name → duplicate
                raise ValueError("Duplicate: that plan already exists for this Business Area & Sub Business Area with the same channels, location/site and name.")

        # --- Rule #2: Demote other current plans in the *same scope* ----------
        # Scope for demotion is: BA + SubBA + Channel-set + Location + Site.
        if is_current:
            # Fetch all CURRENT plans in BA/SBA/Location/Site; compare channels in Python.
            cand_sql = """
                SELECT id, channel
                  FROM capacity_plans
                 WHERE LOWER(vertical) = LOWER(?)
                   AND COALESCE(sub_ba,'') = COALESCE(?, '')
                   AND LOWER(COALESCE(TRIM(location),'')) = LOWER(COALESCE(?, ''))
                   AND LOWER(COALESCE(TRIM(site),''))     = LOWER(COALESCE(?, ''))
                   AND is_current = 1
        """ + not_deleted_clause
            cand = cx.execute(cand_sql, (vertical, sub_ba, location, site)).fetchall()

            to_demote = [r["id"] for r in cand if _norm_channel_csv(r["channel"]) == chan_norm]
            if to_demote:
                placeholders = ",".join("?" for _ in to_demote)
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                cx.execute(
                    f"UPDATE capacity_plans "
                    f"   SET is_current=0, status='history', updated_at=? "
                    f" WHERE id IN ({placeholders})",
                    [ts] + to_demote
                )

        # --- Insert new plan ---------------------------------------------------
        p = payload.copy()

        # normalize/serialize
        p["channel"] = chan_norm
        if isinstance(p.get("tags"), (list, dict)):
            p["tags"] = json.dumps(p["tags"])
        elif p.get("tags") is None:
            p["tags"] = ""

        p["is_current"] = is_current
        p["status"] = status

        # timestamps
        now  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        if "created_at" in cols and "created_at" not in p:
            p["created_at"] = now
        if "updated_at" in cols and "updated_at" not in p:
            p["updated_at"] = now
        now = pd.Timestamp.utcnow().isoformat()
        user = _user()

        p.setdefault("created_by", user)
        p.setdefault("updated_by", user) 

        fields_all = [
            "org","business_entity","vertical","sub_ba","channel","location","site",
            "plan_name","plan_type","start_week","end_week",
            "ft_weekly_hours","pt_weekly_hours","tags","is_current","status","hierarchy_json",
            "created_at","updated_at","created_by","updated_by"
        ]
        fields = [f for f in fields_all if f in cols]
        for f in fields:
            p.setdefault(f, None)
        sql = f"INSERT INTO capacity_plans ({', '.join(fields)}) VALUES ({', '.join(':'+f for f in fields)})"

        cur = cx.execute(sql, p)
        pid = cur.lastrowid
        cx.commit()
        return pid

def extend_plan_weeks(plan_id: int, add_weeks: int) -> None:
    """Extend a plan's end_week in the core store and refresh plan metadata."""
    try:
        pid = int(plan_id)
        delta = int(add_weeks)
    except Exception as exc:
        raise ValueError("Invalid plan id or weeks") from exc
    if delta <= 0:
        return
    with _conn() as cx:
        row = cx.execute("SELECT end_week FROM capacity_plans WHERE id=?", (pid,)).fetchone()
        if not row:
            raise ValueError(f"Plan {pid} not found")
        raw_end = row["end_week"] if (isinstance(row, dict) or hasattr(row, '__getitem__')) else row[0]
        try:
            if raw_end:
                end_date = dt.datetime.fromisoformat(str(raw_end)).date()
            else:
                raise ValueError
        except Exception:
            try:
                end_date = pd.to_datetime(raw_end).date() if raw_end else dt.date.today()
            except Exception:
                end_date = dt.date.today()
        new_end = end_date + dt.timedelta(weeks=delta)
        ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
        cx.execute("UPDATE capacity_plans SET end_week=?, updated_at=? WHERE id=?", (new_end.isoformat(), ts, pid))
        cx.commit()
    try:
        from plan_detail._common import extend_plan_weeks as _extend_meta
        _extend_meta(pid, delta)
    except Exception:
        # Metadata update failure shouldn't block DB change
        pass


def delete_plan(plan_id: int) -> None:
    """Hard-delete a plan row and associated per-plan datasets."""
    pid = int(plan_id)
    with _conn() as cx:
        cx.execute("DELETE FROM capacity_plans WHERE id=?", (pid,))
        cx.commit()
    try:
        delete_datasets_by_prefix(f"plan_{pid}_")
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Status helpers
# ──────────────────────────────────────────────────────────────────────────────
def set_plan_status(plan_id: int, status: str):
    assert status in ("current", "history", "draft")
    # ts = _now_iso()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") 
    with _conn() as cx:
        if status == "current":
            row = cx.execute(
                "SELECT vertical, sub_ba, channel, location, site FROM capacity_plans WHERE id=?",
                (plan_id,)
            ).fetchone()
            if row:
                chan_norm = _norm_channel_csv(row["channel"])
                # demote any other current plan with SAME BA/SBA/Channel-set/Location/Site
                cand = cx.execute(
                    """
                    SELECT id, channel
                      FROM capacity_plans
                     WHERE LOWER(vertical)=LOWER(?)
                       AND COALESCE(sub_ba,'')=COALESCE(?, '')
                       AND LOWER(COALESCE(TRIM(location),''))=LOWER(COALESCE(?, ''))
                       AND LOWER(COALESCE(TRIM(site),''))    =LOWER(COALESCE(?, ''))
                       AND is_current=1
                       AND id <> ?
                    """,
                    (row["vertical"], row["sub_ba"], row["location"], row["site"], plan_id)
                ).fetchall()
                to_demote = [r["id"] for r in cand if _norm_channel_csv(r["channel"]) == chan_norm]
                if to_demote:
                    placeholders = ",".join("?" for _ in to_demote)
                    cx.execute(
                        f"UPDATE capacity_plans "
                        f"   SET is_current=0, status='history', updated_at=? "
                        f" WHERE id IN ({placeholders})",
                        [ts] + to_demote
                    )
        cx.execute(
            "UPDATE capacity_plans SET status=?, is_current=?, updated_at=? WHERE id=?",
            (status, 1 if status == "current" else 0, ts, plan_id)
        )
        cx.commit()

# ──────────────────────────────────────────────────────────────────────────────
# Reads / Deletes
# ──────────────────────────────────────────────────────────────────────────────

def list_business_areas(status_filter: Optional[str] = "current") -> List[str]:
    q = "SELECT DISTINCT vertical FROM capacity_plans WHERE 1=1"
    args: list = []
    if status_filter == "current":
        q += " AND (COALESCE(is_current,0)=1 OR status='current')"
    elif status_filter == "history":
        q += " AND (COALESCE(is_current,0)=0 OR COALESCE(status,'') IN ('history','draft'))"
    elif status_filter:
        q += " AND status=?"; args.append(status_filter)
    with _conn() as cx:
        cols = {r["name"] for r in cx.execute("PRAGMA table_info(capacity_plans)")}
        if "is_deleted" in cols:
            q += " AND COALESCE(is_deleted,0)=0"
        elif "deleted_at" in cols:
            q += " AND deleted_at IS NULL"
        rows = cx.execute(q, args).fetchall()
        return sorted([r["vertical"] for r in rows if r["vertical"]])


def list_plans(vertical: Optional[str] = None,
               status_filter: Optional[str] = None,
               include_deleted: bool = False) -> List[Dict]:
    sql = "SELECT * FROM capacity_plans WHERE 1=1"
    params: list = []
    if vertical:
        sql += " AND vertical=?"; params.append(vertical)
    if status_filter == "current":
        sql += " AND (COALESCE(is_current,0)=1 OR status='current')"
    elif status_filter == "history":
        sql += " AND (COALESCE(is_current,0)=0 OR COALESCE(status,'') IN ('history','draft'))"
    elif status_filter:
        sql += " AND status=?"; params.append(status_filter)

    with _conn() as cx:
        cols = {r["name"] for r in cx.execute("PRAGMA table_info(capacity_plans)")}

        if not include_deleted:
            if "is_deleted" in cols:
                sql += " AND COALESCE(is_deleted,0)=0"
            elif "deleted_at" in cols:
                sql += " AND deleted_at IS NULL"

        rows = cx.execute(sql + " ORDER BY created_at DESC", params).fetchall()
        return [dict(r) for r in rows]


def mark_history(plan_id: int):
    set_plan_status(plan_id, "history")

def get_plan(plan_id: int) -> Optional[Dict[str, Any]]:
    with _conn() as cx:
        row = cx.execute("SELECT * FROM capacity_plans WHERE id=?", (plan_id,)).fetchone()
        return dict(row) if row else None

def delete_plan(plan_id: int, hard_if_missing: bool = True) -> None:
    with _conn() as cx:
        cols = {r["name"] for r in cx.execute("PRAGMA table_info(capacity_plans)")}
        try:
            if "is_deleted" in cols:
                cx.execute("UPDATE capacity_plans SET is_deleted=1, updated_at=datetime('now') WHERE id=?", (plan_id,))
            elif "deleted_at" in cols:
                cx.execute("UPDATE capacity_plans SET deleted_at=datetime('now') WHERE id=?", (plan_id,))
            elif hard_if_missing:
                cx.execute("DELETE FROM capacity_plans WHERE id=?", (plan_id,))
        finally:
            cx.commit()
