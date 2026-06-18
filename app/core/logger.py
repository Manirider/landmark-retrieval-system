import sys

from loguru import logger

from app.core.config import get_settings


def setup_logger() -> None:
    settings = get_settings()

    logger.remove()

    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=settings.log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    logger.info("Logger initialized | level={}", settings.log_level)


__all__ = ["logger", "setup_logger"]
