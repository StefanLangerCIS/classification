import logging
import sys

app_logger: logging.Logger = logging.getLogger("app_logger")

if not app_logger.hasHandlers():
    stdout_handler_name = "stdout_handler"
    app_logger.setLevel(logging.DEBUG)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_formatter = logging.Formatter("%(asctime)-16s %(levelname)-10s %(message)s")
    stdout_handler.setFormatter(stdout_formatter)
    stdout_handler.name = stdout_handler_name
    app_logger.addHandler(stdout_handler)


def add_file_logger(logging_path: str):
    # Check whether stdout handler was already added
    app_logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(logging_path, encoding="utf-8", mode="a")
    formatter = logging.Formatter("%(asctime)-16s %(levelname)-10s %(message)s")
    handler.setFormatter(formatter)
    app_logger.addHandler(handler)
