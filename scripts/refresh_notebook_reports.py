"""Refresh data/report display cells using a fresh kernel; retain recorded live demo outputs."""
import json
from pathlib import Path

from execute_notebook import execute
import build_live_notebook as builder

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "UrbanCart_Submission.ipynb"
notebook = json.loads(path.read_text(encoding="utf-8"))
new = {c["id"]: c for c in builder.base.cells}
refresh = {"urbancart-13", "urbancart-21", "urbancart-25"}
for c in notebook["cells"]:
    source = new[c["id"]]
    if c["cell_type"] == "markdown" or c["id"] in refresh:
        c["source"] = source["source"]
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
execute(path, selected_ids=refresh, bootstrap='''
import json
import pandas as pd
from IPython.display import display
from urbancart.config import ROOT, RAW, PROCESSED, Settings
from urbancart.data import read_source, CATALOG
settings = Settings()
catalog = read_source(RAW / CATALOG)
profile = json.loads((PROCESSED / "profile.json").read_text(encoding="utf-8"))
def bars(series, title, color="#A6D6CA"):
    display(series.sort_values(ascending=False).to_frame("Value").style.set_caption(title)
            .bar(subset=["Value"], color=color, vmin=0).format(precision=1))
''')
