# file: plan_detail/__init__.py
from __future__ import annotations
from ._ui import layout_for_plan, plan_detail_validation_layout
# from ._callbacks_roster import register_plan_detail_roster
from ._callbacks_core import register_plan_detail_core

def register_plan_detail(app):
    # Register callbacks in logical groups
    # register_plan_detail_roster(app)
    register_plan_detail_core(app)
