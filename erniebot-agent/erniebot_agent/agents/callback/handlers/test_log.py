import logging
import os
from typing import Any, Optional


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[95m",  # Purple
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[91m" + "\033[1m",  # Bold Red
        "RESET": "\033[0m",
    }

    def format(self, record):
        log_message = super(ColorFormatter, self).format(record)
        log_message = self.COLORS.get(record.levelname, "") + log_message + self.COLORS["RESET"]
        return log_message


def set_up_logging(verbosity: Optional[str] = None, standard_format: Optional[bool] = True) -> None:
    if not verbosity:
        verbosity = os.environ.get("EB_LOGGING_LEVEL", None)

    if verbosity:
        numeric_level = getattr(logging, verbosity.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {verbosity}")
    else:
        numeric_level = getattr(logging, "INFO", None)

    logger = logging.getLogger(__name__)

    if standard_format:
        logger.setLevel(numeric_level)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColorFormatter("%(levelname)s - %(message)s"))
        logger.addHandler(console_handler)
        logger.propagate = False

    print(logger)
    logger.warning("Hello")


# set_up_logging('INFO')
# logger = logging.getLogger(__name__)
# logger.warning("Hello")
