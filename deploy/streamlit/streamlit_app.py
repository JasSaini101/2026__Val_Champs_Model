"""Hosted dashboard for Streamlit Community Cloud (entrypoint: deploy/streamlit/streamlit_app.py).

Runs the regular dashboard against the odds the hourly update job commits to GitHub, so the
hosted app needs no API, database or model: only the light packages in requirements.txt here.
"""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault(
    "VALCHAMPS_DATA_URL",
    "https://raw.githubusercontent.com/JasSaini101/2026__Val_Champs_Model/main/odds",
)
runpy.run_path(str(ROOT / "src" / "valchamps" / "dashboard" / "app.py"), run_name="__main__")
