"""
Simplified logging configuration for the Alligator entity linking system.
"""

import logging
import os
import sys
from enum import IntEnum
from typing import Optional


class LogLevel(IntEnum):
    """Log levels that can be set by users via environment variables."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def _get_env_log_level() -> int:
    """Get log level from environment variable."""
    level_str = os.environ.get("ALLIGATOR_MIN_LOG_LEVEL", "INFO").upper()
    try:
        return getattr(LogLevel, level_str).value
    except AttributeError:
        return LogLevel.INFO.value


def _is_logging_disabled() -> bool:
    """Check if logging is disabled via environment variable."""
    return os.environ.get("ALLIGATOR_DISABLE_LOGGING", "0") == "1"


def setup_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True,
) -> logging.Logger:
    """
    Set up logging configuration for the Alligator system.

    Args:
        level: Logging level override (if None, uses environment variable)
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in log messages

    Returns:
        Root logger for Alligator
    """

    # Check if logging should be disabled
    if _is_logging_disabled():
        logging.disable(logging.CRITICAL)
        return logging.getLogger("alligator")
    else:
        logging.disable(logging.NOTSET)  # Ensure logging is enabled

    # Determine log level for Alligator
    if level is not None:
        alligator_level = getattr(logging, level.upper(), logging.INFO)
    else:
        alligator_level = _get_env_log_level()

    # Default format string
    if format_string is None:
        if include_timestamp:
            format_string = "[%(asctime)s - %(name)s - %(levelname)s] %(message)s"
        else:
            format_string = "[%(name)s - %(levelname)s] %(message)s"

    # Configure root logger to WARNING (only errors from third-party libs)
    logging.basicConfig(
        level=logging.WARNING,  # Root logger at WARNING level
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Override any existing configuration
    )

    # Set Alligator logger to the desired level
    alligator_logger = logging.getLogger("alligator")
    alligator_logger.setLevel(alligator_level)

    # Suppress ALL third-party loggers automatically
    _suppress_all_third_party_loggers()

    return alligator_logger


def _suppress_all_third_party_loggers():
    """Automatically suppress ALL third-party loggers to WARNING level."""
    # Get all existing loggers from the logging manager
    for logger_name in logging.Logger.manager.loggerDict:
        if not logger_name.startswith("alligator"):
            # Set all non-alligator loggers to WARNING to reduce noise
            logging.getLogger(logger_name).setLevel(logging.CRITICAL)

    # Also set a custom filter to catch new loggers created after this
    class ThirdPartyLoggerFilter(logging.Filter):
        def filter(self, record):
            # Allow all alligator logs through
            if record.name.startswith("alligator"):
                return True
            return False

    # Add the filter to the root logger's handlers
    for handler in logging.getLogger().handlers:
        handler.addFilter(ThirdPartyLoggerFilter())


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific component.

    Args:
        name: Name of the component (e.g., 'coordinator', 'data_manager')

    Returns:
        Logger instance for the component
    """
    logger = logging.getLogger(f"alligator.{name}")

    # Apply current environment settings
    refresh_logging()

    return logger


def refresh_logging():
    """
    Refresh logging configuration based on current environment variables.
    Call this after changing environment variables to apply changes.
    """
    if _is_logging_disabled():
        logging.disable(logging.CRITICAL)
    else:
        logging.disable(logging.NOTSET)
        # Update log level for Alligator loggers only
        new_level = _get_env_log_level()
        logging.getLogger("alligator").setLevel(new_level)
        # Automatically suppress all third-party loggers
        _suppress_all_third_party_loggers()


def disable_logging():
    """Completely disable all logging."""
    os.environ["ALLIGATOR_DISABLE_LOGGING"] = "1"
    logging.disable(logging.CRITICAL)


def enable_logging():
    """Re-enable logging."""
    os.environ.pop("ALLIGATOR_DISABLE_LOGGING", None)
    logging.disable(logging.NOTSET)


# Initialize logging on import
logger = setup_logging()
