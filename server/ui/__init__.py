# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
UI package for the ToolForge Gradio interface.

Import pattern (runs from toolforge_env/ dir):
    from server.ui import build_demo_tab, build_byoa_tab, build_hvl_tab
"""

from server.ui.demo_tab import build_demo_tab
from server.ui.byoa_tab import build_byoa_tab
from server.ui.hvl_tab  import build_hvl_tab

__all__ = [
    "build_demo_tab",
    "build_byoa_tab",
    "build_hvl_tab",
]
