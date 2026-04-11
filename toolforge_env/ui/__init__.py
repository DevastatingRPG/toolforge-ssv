# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
UI package for the ToolForge Gradio interface.

Exposes three tab builders:
    - build_demo_tab    : Tab 1 — Demo Mode (pre-scripted agent simulation)
    - build_byoa_tab    : Tab 2 — Bring Your Own Agent (live LLM connection)
    - build_hvl_tab     : Tab 3 — Human vs LLM (interactive game)
"""

from toolforge_env.ui.demo_tab import build_demo_tab
from toolforge_env.ui.byoa_tab import build_byoa_tab
from toolforge_env.ui.hvl_tab  import build_hvl_tab

__all__ = [
    "build_demo_tab",
    "build_byoa_tab",
    "build_hvl_tab",
]
