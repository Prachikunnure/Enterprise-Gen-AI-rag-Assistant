"""
Shared utilities: logging setup and small helpers.
"""

import logging
import sys
from typing import Optional


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure application logging (no secrets in logs)."""
    logger = logging.getLogger("enterprise_rag")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def truncate_for_log(text: str, max_len: int = 120) -> str:
    """Shorten text for safe log lines."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def validate_question(question: str, min_len: int = 3, max_len: int = 4000) -> Optional[str]:
    """
    Return an error message if the question is invalid, else None.
    """
    q = (question or "").strip()
    if not q:
        return "Please enter a question."
    if len(q) < min_len:
        return "Your question is too short. Please provide more detail."
    if len(q) > max_len:
        return "Your question is too long. Please shorten it."
    return None
