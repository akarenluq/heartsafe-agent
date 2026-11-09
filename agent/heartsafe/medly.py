"""Backward-compatible names. The adapter now lives in `monitoring.py`; Medly stays the main reference platform."""
from .monitoring import LEVELS, REQUIRED, PlatformAdapter, platform_levels, rule_levels  # noqa: F401

MedlyAdapter = PlatformAdapter
medly_levels = platform_levels
