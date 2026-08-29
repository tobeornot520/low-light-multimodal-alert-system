from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

raise SystemExit(import_module("lowlight_alert.cli").main())
