import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from banking_agent import run_calibration_sweep

report = run_calibration_sweep()
out_path = Path(__file__).resolve().parent / 'calibration_report.json'
out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(out_path)
print(json.dumps(report, indent=2))
