from __future__ import annotations

import logging
from pathlib import Path

from config import settings


def setup_logging() -> None:
    """只配置一次基础日志，便于排查文件解析和 API 问题。"""

    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(settings.log_dir) / "app.log", encoding="utf-8"),
        ],
    )

