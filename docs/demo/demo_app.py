"""Run the real dashboard (agent/app.py) on a copy of agent/data, so recording a demo changes nothing.

From the agent/ folder:
    streamlit run ../docs/demo/demo_app.py --server.port 8503 -- <copy of the data folder>
"""
import sys
from pathlib import Path

AGENT = Path(__file__).resolve().parents[2] / "agent"
sys.path.insert(0, str(AGENT))
import heartsafe  # noqa: E402

heartsafe.DATA = Path(sys.argv[1]).resolve()
APP = AGENT / "app.py"
exec(compile(APP.read_text(encoding="utf-8"), str(APP), "exec"), {"__file__": str(APP), "__name__": "__main__"})
