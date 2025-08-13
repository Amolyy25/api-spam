import os
import sqlite3
from datetime import datetime
from typing import Dict, Optional, Tuple
import hashlib
import uuid

from fastapi import Header, HTTPException, status, Depends, Request

from .config import load_config
from .utils import log_error


USAGE_DB_FILENAME = "usage.db"


def _get_db_path() -> str:
	cfg = load_config()
	os.makedirs(cfg.data_dir, exist_ok=True)
	return os.path.join(cfg.data_dir, USAGE_DB_FILENAME)


def _hash_key(raw_key: str) -> str:
	return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _migrate_schema(conn: sqlite3.Connection) -> None:
	cur = conn.execute("PRAGMA table_info(usage)")
	cols = [row[1] for row in cur.fetchall()]
	if cols and "date" in cols and "period" not in cols:
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS usage_new (
				api_key TEXT NOT NULL,
				period TEXT NOT NULL,
				count INTEGER NOT NULL DEFAULT 0,
				PRIMARY KEY(api_key, period)
			)
			"""
		)
		conn.execute(
			"""
			INSERT INTO usage_new(api_key, period, count)
			SELECT api_key, substr(date, 1, 7) AS period, SUM(count)
			FROM usage
			GROUP BY api_key, substr(date, 1, 7)
			"""
		)
		conn.execute("DROP TABLE usage")
		conn.execute("ALTER TABLE usage_new RENAME TO usage")
		conn.commit()


def init_usage_store() -> None:
	path = _get_db_path()
	conn = sqlite3.connect(path)
	try:
		# Core tables
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS usage (
				api_key TEXT NOT NULL,
				period TEXT NOT NULL,  -- YYYY-MM
				count INTEGER NOT NULL DEFAULT 0,
				PRIMARY KEY(api_key, period)
			)
			"""
		)
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS rate_limit (
				api_key TEXT NOT NULL,
				window TEXT NOT NULL, -- YYYYMMDDHHMM
				count INTEGER NOT NULL DEFAULT 0,
				PRIMARY KEY(api_key, window)
			)
			"""
		)
		# Auth tables
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS users (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				email TEXT NOT NULL,
				created_at TEXT NOT NULL
			)
			"""
		)
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS api_keys (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				user_id INTEGER NOT NULL,
				key_hash TEXT NOT NULL,
				plan TEXT NOT NULL,
				active INTEGER NOT NULL DEFAULT 1,
				created_at TEXT NOT NULL,
				FOREIGN KEY(user_id) REFERENCES users(id)
			)
			"""
		)
		_migrate_schema(conn)
		conn.commit()
	finally:
		conn.close()


def seed_test_keys() -> None:
	# Seed users and keys based on config.yaml for dev/testing
	cfg = load_config()
	path = _get_db_path()
	conn = sqlite3.connect(path)
	try:
		conn.execute("BEGIN")
		for raw_key, plan in cfg.api_keys.items():
			key_hash = _hash_key(raw_key)
			# Insert user placeholder
			conn.execute(
				"INSERT OR IGNORE INTO users(id, email, created_at) VALUES(?, ?, ?)",
				(1, "test@example.com", datetime.utcnow().isoformat() + "Z"),
			)
			# Insert api key if missing
			conn.execute(
				"""
				INSERT OR IGNORE INTO api_keys(user_id, key_hash, plan, active, created_at)
				VALUES(?, ?, ?, 1, ?)
				""",
				(1, key_hash, plan, datetime.utcnow().isoformat() + "Z"),
			)
		conn.commit()
	finally:
		conn.close()


def _current_period() -> str:
	# Monthly period YYYY-MM
	return datetime.utcnow().strftime("%Y-%m")


def _current_minute_window() -> str:
	# Minute window YYYYMMDDHHMM
	return datetime.utcnow().strftime("%Y%m%d%H%M")


def get_usage_count(api_key: str, period: Optional[str] = None) -> int:
	if not period:
		period = _current_period()
	init_usage_store()
	conn = sqlite3.connect(_get_db_path())
	try:
		cur = conn.execute(
			"SELECT count FROM usage WHERE api_key = ? AND period = ?",
			(api_key, period),
		)
		row = cur.fetchone()
		return int(row[0]) if row else 0
	finally:
		conn.close()


def increment_usage(api_key: str, inc: int = 1) -> int:
	period = _current_period()
	init_usage_store()
	conn = sqlite3.connect(_get_db_path())
	try:
		conn.execute(
			"INSERT OR IGNORE INTO usage(api_key, period, count) VALUES(?, ?, 0)",
			(api_key, period),
		)
		conn.execute(
			"UPDATE usage SET count = count + ? WHERE api_key = ? AND period = ?",
			(inc, api_key, period),
		)
		conn.commit()
		cur = conn.execute(
			"SELECT count FROM usage WHERE api_key = ? AND period = ?",
			(api_key, period),
		)
		row = cur.fetchone()
		return int(row[0]) if row else 0
	finally:
		conn.close()


def get_rate_count(api_key: str, window: Optional[str] = None) -> int:
	if not window:
		window = _current_minute_window()
	init_usage_store()
	conn = sqlite3.connect(_get_db_path())
	try:
		cur = conn.execute(
			"SELECT count FROM rate_limit WHERE api_key = ? AND window = ?",
			(api_key, window),
		)
		row = cur.fetchone()
		return int(row[0]) if row else 0
	finally:
		conn.close()


def increment_rate(api_key: str, inc: int = 1) -> int:
	window = _current_minute_window()
	init_usage_store()
	conn = sqlite3.connect(_get_db_path())
	try:
		conn.execute(
			"INSERT OR IGNORE INTO rate_limit(api_key, window, count) VALUES(?, ?, 0)",
			(api_key, window),
		)
		conn.execute(
			"UPDATE rate_limit SET count = count + ? WHERE api_key = ? AND window = ?",
			(inc, api_key, window),
		)
		conn.commit()
		cur = conn.execute(
			"SELECT count FROM rate_limit WHERE api_key = ? AND window = ?",
			(api_key, window),
		)
		row = cur.fetchone()
		return int(row[0]) if row else 0
	finally:
		conn.close()


def validate_api_key_db(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> Tuple[str, str]:
	if not x_api_key:
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
	path = _get_db_path()
	conn = sqlite3.connect(path)
	try:
		key_hash = _hash_key(x_api_key)
		cur = conn.execute(
			"SELECT plan, active FROM api_keys WHERE key_hash = ?",
			(key_hash,),
		)
		row = cur.fetchone()
		if not row:
			# Dev fallback: if key exists in config mapping, auto-seed into DB
			cfg = load_config()
			if x_api_key in cfg.api_keys:
				plan = cfg.api_keys[x_api_key]
				# ensure a default user
				conn.execute(
					"INSERT OR IGNORE INTO users(id, email, created_at) VALUES(?, ?, ?)",
					(1, "test@example.com", datetime.utcnow().isoformat() + "Z"),
				)
				conn.execute(
					"INSERT OR IGNORE INTO api_keys(user_id, key_hash, plan, active, created_at) VALUES(?, ?, ?, 1, ?)",
					(1, key_hash, plan, datetime.utcnow().isoformat() + "Z"),
				)
				conn.commit()
				row = (plan, 1)
			else:
				raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
		plan, active = row
		if not int(active):
			raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key disabled")
		return x_api_key, plan
	finally:
		conn.close()


def enforce_quota_and_rate(request: Request, keys: Tuple[str, str] = Depends(validate_api_key_db)) -> Tuple[str, str]:
	"""
	Dependency that enforces both monthly quota and per-minute rate limits.

	Limits are defined in config.yaml:
	- quotas: monthly requests per plan (e.g., free=100, pro5k=5000, ...)
	- rate_limit_per_minute: per-minute requests per plan (e.g., free=5, pro5k=120, ...)

	If either limit is exceeded, raises 429 Too Many Requests and writes an entry to
	the error log with minimal API key disclosure (last 4 chars only).
	"""
	api_key, plan = keys
	cfg = load_config()
	request_id = str(uuid.uuid4())
	# Monthly quota
	limit_month = int(cfg.quotas.get(plan, 0))
	used_month = get_usage_count(api_key)
	if used_month >= limit_month:
		log_error(
			request,
			request_id,
			status.HTTP_429_TOO_MANY_REQUESTS,
			f"Monthly quota exceeded (used={used_month}, limit={limit_month}) for plan={plan} key=****{api_key[-4:]}",
		)
		raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Monthly quota exceeded")
	# Per-minute rate limit
	rate_limits = cfg.rate_limit_per_minute
	limit_min = int(rate_limits.get(plan, 60))
	used_min = get_rate_count(api_key)
	if used_min >= limit_min:
		log_error(
			request,
			request_id,
			status.HTTP_429_TOO_MANY_REQUESTS,
			f"Rate limit exceeded (used={used_min}, limit={limit_min}) for plan={plan} key=****{api_key[-4:]}",
		)
		raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
	return api_key, plan


# Ensure DB is initialized at import time as well
init_usage_store() 