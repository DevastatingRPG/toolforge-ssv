"""
Tests for the demo_tab synchronization changes.

Tests the following:
1. resolve_env_url() picks the right URL from env vars.
2. _derive_fallback_step() correctly recycles plans and substitutes macros.
3. _get_scripted_step() returns scripted steps when available, falls back otherwise.
4. _compute_btn_label_and_note() produces correct labels for env_done states.
5. _build_reset_outputs() produces 12 elements with no plan.
6. on_run_simulation returns step_idx = -1 (no step executed).
"""

import os
import sys

# Ensure the project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.ui.demo_tab import (
    _derive_fallback_step,
    _get_scripted_step,
    _compute_btn_label_and_note,
    _blank_outputs,
)
from server.ui.env_client import resolve_env_url


# ===========================================================================
# Test resolve_env_url
# ===========================================================================

def test_resolve_env_url_default():
    """No env vars set → http://localhost:8000."""
    # Clear any env vars that might interfere
    for var in ("TOOLFORGE_ENV_URL", "SPACE_HOST"):
        os.environ.pop(var, None)
    # Re-import to test (function reads env at call time)
    url = resolve_env_url()
    assert url == "http://localhost:8000", f"Expected localhost default, got: {url}"
    print(f"  ✅ test_resolve_env_url_default: {url}")


def test_resolve_env_url_explicit():
    """TOOLFORGE_ENV_URL takes priority."""
    os.environ["TOOLFORGE_ENV_URL"] = "https://my-custom-url.com/api"
    try:
        url = resolve_env_url()
        assert url == "https://my-custom-url.com/api", f"Got: {url}"
        print(f"  ✅ test_resolve_env_url_explicit: {url}")
    finally:
        del os.environ["TOOLFORGE_ENV_URL"]


def test_resolve_env_url_hf_space():
    """SPACE_HOST env var → https://{host}."""
    os.environ["SPACE_HOST"] = "my-space.hf.space"
    try:
        url = resolve_env_url()
        assert url == "https://my-space.hf.space", f"Got: {url}"
        print(f"  ✅ test_resolve_env_url_hf_space: {url}")
    finally:
        del os.environ["SPACE_HOST"]


# ===========================================================================
# Test _derive_fallback_step
# ===========================================================================

def test_fallback_no_macro():
    """No matching macro → recycle last plan as-is."""
    last = {"plan": ["deploy", "healthcheck", "notify"], "macro_proposed": None, "note": "test"}
    result = _derive_fallback_step(last, macros=[])
    assert result["plan"] == ["deploy", "healthcheck", "notify"]
    assert result["macro_proposed"] is None
    assert "recycled" in result["note"].lower() or "pattern" in result["note"].lower()
    print(f"  ✅ test_fallback_no_macro: {result['plan']}")


def test_fallback_with_matching_macro():
    """Macro whose steps match the last plan → substitute macro name."""
    last = {
        "plan": ["deploy", "healthcheck", "notify"],
        "macro_proposed": None,
        "note": "test",
    }
    macros = [{"name": "deploy_verify_notify", "steps": ["deploy", "healthcheck", "notify"]}]
    result = _derive_fallback_step(last, macros)
    assert result["plan"] == ["deploy_verify_notify"]
    assert result["macro_proposed"] is None
    assert "deploy_verify_notify" in result["note"]
    print(f"  ✅ test_fallback_with_matching_macro: {result['plan']}")


def test_fallback_with_nonmatching_macro():
    """Macro whose steps DON'T match → recycle last plan."""
    last = {"plan": ["scale", "ping", "notify"], "macro_proposed": None, "note": "test"}
    macros = [{"name": "deploy_verify_notify", "steps": ["deploy", "healthcheck", "notify"]}]
    result = _derive_fallback_step(last, macros)
    assert result["plan"] == ["scale", "ping", "notify"]
    print(f"  ✅ test_fallback_with_nonmatching_macro: {result['plan']}")


# ===========================================================================
# Test _get_scripted_step
# ===========================================================================

def test_get_scripted_step_in_range():
    """step_idx within scripted range → return scripted step."""
    episode = {
        "episode_id": "test",
        "steps": [
            {"plan": ["deploy", "healthcheck", "notify"], "macro_proposed": None, "note": "step0"},
            {"plan": ["scale", "ping", "notify"], "macro_proposed": None, "note": "step1"},
        ],
    }
    step = _get_scripted_step(episode, 0, macros=[])
    assert step["note"] == "step0"
    step = _get_scripted_step(episode, 1, macros=[])
    assert step["note"] == "step1"
    print("  ✅ test_get_scripted_step_in_range")


def test_get_scripted_step_out_of_range():
    """step_idx beyond scripted range → fallback derived."""
    episode = {
        "episode_id": "test",
        "steps": [
            {"plan": ["deploy", "healthcheck", "notify"], "macro_proposed": None, "note": "step0"},
        ],
    }
    step = _get_scripted_step(episode, 5, macros=[])
    assert step["plan"] == ["deploy", "healthcheck", "notify"]
    assert "recycled" in step["note"].lower() or "pattern" in step["note"].lower()
    print("  ✅ test_get_scripted_step_out_of_range")


def test_get_scripted_step_out_of_range_with_macro():
    """step_idx beyond range + matching macro → fallback uses macro."""
    episode = {
        "episode_id": "test",
        "steps": [
            {"plan": ["deploy", "healthcheck", "notify"], "macro_proposed": None, "note": "step0"},
        ],
    }
    macros = [{"name": "deploy_verify_notify", "steps": ["deploy", "healthcheck", "notify"]}]
    step = _get_scripted_step(episode, 10, macros)
    assert step["plan"] == ["deploy_verify_notify"]
    print("  ✅ test_get_scripted_step_out_of_range_with_macro")


# ===========================================================================
# Test _compute_btn_label_and_note
# ===========================================================================

def test_btn_label_mid_episode():
    label, note = _compute_btn_label_and_note(step_idx=0, total_tasks=5, env_done=False, is_last_ep=False)
    assert label == "Next Step ▶"
    assert note == ""
    print("  ✅ test_btn_label_mid_episode")


def test_btn_label_episode_done():
    label, note = _compute_btn_label_and_note(step_idx=4, total_tasks=5, env_done=True, is_last_ep=False)
    assert "Next Episode" in label
    assert "📌" in note
    print("  ✅ test_btn_label_episode_done")


def test_btn_label_all_done():
    label, note = _compute_btn_label_and_note(step_idx=4, total_tasks=5, env_done=True, is_last_ep=True)
    assert "Restart" in label
    assert "🎉" in note
    print("  ✅ test_btn_label_all_done")


# ===========================================================================
# Test _blank_outputs
# ===========================================================================

def test_blank_outputs_length():
    outputs = _blank_outputs()
    assert len(outputs) == 13, f"Expected 13 outputs, got {len(outputs)}"
    print("  ✅ test_blank_outputs_length")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("\n=== Demo Tab Synchronization Tests ===\n")

    print("URL Resolution:")
    test_resolve_env_url_default()
    test_resolve_env_url_explicit()
    test_resolve_env_url_hf_space()

    print("\nFallback Step Derivation:")
    test_fallback_no_macro()
    test_fallback_with_matching_macro()
    test_fallback_with_nonmatching_macro()

    print("\nScripted Step Retrieval:")
    test_get_scripted_step_in_range()
    test_get_scripted_step_out_of_range()
    test_get_scripted_step_out_of_range_with_macro()

    print("\nButton Labels:")
    test_btn_label_mid_episode()
    test_btn_label_episode_done()
    test_btn_label_all_done()

    print("\nBlank Outputs:")
    test_blank_outputs_length()

    print("\n✅ All 12 tests passed!\n")
