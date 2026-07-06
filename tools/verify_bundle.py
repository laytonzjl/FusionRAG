from __future__ import annotations

"""Static verification for the optimized RAG core bundle."""

import compileall
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ok = compileall.compile_dir(str(ROOT / "rag_core"), quiet=1)
if not ok:
    raise SystemExit("编译失败：请检查上方报错。")
print("OK: rag_core 静态编译通过。")
