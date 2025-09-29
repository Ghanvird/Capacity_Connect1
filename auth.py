from __future__ import annotations
from cap_db import _conn

def _init_roles():
    with _conn() as cx:
        cx.execute("""
        CREATE TABLE IF NOT EXISTS user_roles (
            username TEXT PRIMARY KEY,
            role TEXT CHECK(role in ('admin','planner','viewer'))
        )
        """)
        cx.commit()

_init_roles()

def _normalize_username(username: str | None) -> str:
    if username is None:
        return ''
    return str(username).strip()

def _canonical_username(username: str | None) -> str:
    return _normalize_username(username).lower()

def get_user_role(username: str | None) -> str:
    user = _normalize_username(username)
    if not user:
        return 'viewer'
    canon = _canonical_username(user)
    with _conn() as cx:
        row = cx.execute("SELECT role FROM user_roles WHERE lower(username)=?", (canon,)).fetchone()
        if row and row["role"] in ("admin","planner","viewer"):
            return row["role"]
    return 'viewer'

def set_user_role(username: str, role: str) -> None:
    assert role in ("admin","planner","viewer")
    user = _normalize_username(username)
    if not user:
        return
    canon = _canonical_username(user)
    with _conn() as cx:
        cur = cx.execute("UPDATE user_roles SET role=?, username=? WHERE lower(username)=?", (role, canon, canon))
        if cur.rowcount == 0:
            cx.execute("INSERT INTO user_roles(username,role) VALUES(?,?)", (canon, role))
        cx.commit()

def can_delete_plans(role: str) -> bool:
    return role == 'admin'

def can_save_settings(role: str) -> bool:
    return role in ('admin','planner')

