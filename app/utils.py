import json
import logging
import os
import re
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

from fastapi import Request

from .config import load_config


def ensure_directories() -> None:
	cfg = load_config()
	for path in [cfg.data_dir, cfg.logs_dir]:
		os.makedirs(path, exist_ok=True)


def _build_logger(name: str, filename: str) -> logging.Logger:
	ensure_directories()
	cfg = load_config()
	log_path = os.path.join(cfg.logs_dir, filename)
	logger = logging.getLogger(name)
	if logger.handlers:
		return logger
	logger.setLevel(logging.INFO)
	handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
	formatter = logging.Formatter("%(message)s")
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	return logger


def get_logger(name: str = "app") -> logging.Logger:
	return _build_logger(name, "requests.log")


def get_error_logger() -> logging.Logger:
	return _build_logger("errors", "errors.log")


def get_client_ip(request: Request) -> str:
	x_forwarded_for = request.headers.get("x-forwarded-for")
	if x_forwarded_for:
		return x_forwarded_for.split(",")[0].strip()
	return request.client.host if request.client else "unknown"


def redact_text(text: str, max_len: int = 200) -> str:
	cleaned = re.sub(r"\s+", " ", text).strip()
	return cleaned[:max_len]


def json_log(record: dict) -> str:
	return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def log_request(request: Request, api_key: str, text: str, score: float, is_spam: bool) -> None:
	logger = get_logger("requests")
	entry = {
		"timestamp": datetime.utcnow().isoformat() + "Z",
		"ip": get_client_ip(request),
		"api_key": api_key[-4:],
		"path": request.url.path,
		"text": redact_text(text),
		"score": round(float(score), 6),
		"is_spam": bool(is_spam),
	}
	logger.info(json_log(entry))


def log_error(request: Optional[Request], request_id: str, status_code: int, message: str) -> None:
	logger = get_error_logger()
	entry = {
		"timestamp": datetime.utcnow().isoformat() + "Z",
		"request_id": request_id,
		"path": request.url.path if request else None,
		"status_code": status_code,
		"message": message,
	}
	logger.error(json_log(entry))


def extract_domains(text: str) -> list[str]:
	pattern = re.compile(r"(?i)https?://(?:www\.)?([a-z0-9\-]+(?:\.[a-z0-9\-]+)+)")
	return pattern.findall(text)


def extract_ips(text: str) -> list[str]:
	pattern = re.compile(r"\b(?:(?:2[0-5]{2}|1?\d?\d)\.){3}(?:2[0-5]{2}|1?\d?\d)\b")
	return pattern.findall(text)


def normalize_label(label: str) -> str:
	label_lower = label.strip().lower()
	if label_lower in {"spam", "1", "true", "yes"}:
		return "spam"
	return "ham" 