import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class DRLogger:
    def __init__(
        self,
        name: str = "DRGrading",
        log_dir: str = "./logs",
        level: int = logging.INFO,
        console_output: bool = True,
        file_output: bool = True,
        json_format: bool = True,
    ):
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers.clear()
        formatter = (
            JsonFormatter()
            if json_format
            else logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )

        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        if file_output:
            main_log_file = log_path / f"{name}_{timestamp}.log"
            file_handler = logging.FileHandler(main_log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            error_log_file = log_path / f"{name}_errors_{timestamp}.log"
            error_handler = logging.FileHandler(error_log_file)
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            logger.addHandler(error_handler)

            self.main_log_file = str(main_log_file)
            self.error_log_file = str(error_log_file)

        self.logger = logger

    def debug(self, message: str, **kwargs: Any):
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs: Any):
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs: Any):
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs: Any):
        self.logger.error(message, extra=kwargs)

    def critical(self, message: str, **kwargs: Any):
        self.logger.critical(message, extra=kwargs)


class JsonFormatter(logging.Formatter):
    RESERVED = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in self.RESERVED:
                log_entry[key] = value

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def get_logger(
    name: str = "DRGrading",
    log_dir: str = "./logs",
    level: str = "INFO",
    console_output: bool = True,
    file_output: bool = True,
    json_format: bool = True,
) -> DRLogger:
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    return DRLogger(
        name=name,
        log_dir=log_dir,
        level=level_map.get(level.upper(), logging.INFO),
        console_output=console_output,
        file_output=file_output,
        json_format=json_format,
    )
