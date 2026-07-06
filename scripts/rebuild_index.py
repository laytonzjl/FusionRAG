from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_api_config, load_rag_config
from rag_core import build_engine


def main() -> int:
    """从 data/uploads 中的原始上传文件重建当前知识库索引。"""

    api_config = load_api_config()
    rag_config = load_rag_config()
    engine = build_engine(api_config=api_config, rag_config=rag_config)
    compatibility = engine.index_compatibility()

    print("当前索引兼容性：")
    print(json.dumps(compatibility, ensure_ascii=False, indent=2))
    print("\n开始重建当前知识库，请保持终端窗口打开...")

    result = engine.rebuild_from_uploads()

    print("\n重建结果：")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"completed", "empty"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
