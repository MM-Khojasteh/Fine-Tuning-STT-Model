"""Logging configuration for the fine-tuning project."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    log_format: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """Setup logging configuration.

    Args:
        log_file: Path to log file (if None, no file logging)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string
        console_output: Whether to output logs to console

    Returns:
        Configured logger instance
    """
    # Default format includes timestamp, level, and message
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Parse log level
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Get root logger for the project
    logger = logging.getLogger("wav2vec2_finetuning")
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get logger instance.

    Args:
        name: Logger name (if None, returns root project logger)

    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"wav2vec2_finetuning.{name}")
    return logging.getLogger("wav2vec2_finetuning")
