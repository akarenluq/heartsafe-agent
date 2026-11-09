"""HeartSafe: equity-aware trust and coordination layer on top of a remote-monitoring platform feed (modelled on Medly)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_config(path=None):
    with open(path or ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    if "medly_rules" in cfg:  # older configs
        cfg.setdefault("alert_rules", cfg.pop("medly_rules"))
    return cfg


def age_band(age):
    return "<60" if age < 60 else ("60-74" if age < 75 else "75+")
