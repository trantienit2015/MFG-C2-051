# PB-7: HITL Interrupt Propagation - required in every template.
#
# MFG-C2-051 does not set `hitl.enabled: true` in config/agent.yaml (no interrupt()
# call anywhere in src/nodes/ or src/graph/), so the boundary is N/A here and the
# test auto-skips. A template that turns on HITL must replace this stub with a real
# GraphInterrupt-propagation test (PB-7 boundary spec).

import os
import re

import pytest


def _hitl_enabled() -> bool:
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "agent.yaml")
    if not os.path.exists(config_path):
        return False
    with open(config_path, encoding="utf-8") as f:
        content = f.read()
    return bool(re.search(r"hitl:\s*\n(?:\s+.*\n)*?\s+enabled:\s*true", content))


@pytest.mark.skipif(not _hitl_enabled(), reason="hitl.enabled not set — PB-7 N/A for this template")
def test_hitl_interrupt_propagation():
    """Placeholder — only meaningful for templates with hitl.enabled: true."""
    pytest.skip("HITL not enabled for this template")
