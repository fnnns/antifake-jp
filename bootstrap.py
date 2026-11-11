"""システム全体の Python を変更せず、このフォルダー専用のライブラリを探す。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
local_packages = ROOT / ".runtime"
if local_packages.is_dir() and sys.prefix == sys.base_prefix:
    sys.path.insert(0, str(local_packages))
