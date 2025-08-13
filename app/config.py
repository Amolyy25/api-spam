import os
import yaml
from typing import Any, Dict

"""
Configuration loader for the API.

Relevant keys in config.yaml:
- api_keys: mapping of dev/test API keys to plans (used for seeding local DB only)
- quotas: monthly quotas per plan (requests per month)
- rate_limit_per_minute: per-minute limits per plan (requests per minute)
- pricing: suggested monthly prices per plan (for marketplaces like RapidAPI)
- cache: in-memory cache configuration
- model, regex_rules, blacklist, gemini: detection and integration settings

Example plans:
- free: 100 req/month, 5 req/min, price 0€/mois
- pro5k: 5000 req/month, 120 req/min, price 10€/mois
- pro15k: 15000 req/month, 180 req/min, price 25€/mois
- pro50k: 50000 req/month, 240 req/min, price 50€/mois
"""

# Load .env file if it exists
try:
	from dotenv import load_dotenv
	load_dotenv()
except ImportError:
	pass


class Config:
	def __init__(self, data: Dict[str, Any]):
		self._data = data

	@property
	def api_keys(self) -> Dict[str, str]:
		"""Mapping of legacy/dev API keys to plans (used only for seeding and local dev)."""
		return self._data.get("api_keys", {})

	@property
	def quotas(self) -> Dict[str, int]:
		"""Monthly quotas per plan (requests per month)."""
		return self._data.get("quotas", {"free": 100, "pro5k": 5000, "pro15k": 15000, "pro50k": 50000})

	@property
	def rate_limit_per_minute(self) -> Dict[str, int]:
		"""Per-minute rate limits per plan (requests per minute)."""
		return self._data.get("rate_limit_per_minute", {"free": 5, "pro5k": 120, "pro15k": 180, "pro50k": 240})

	@property
	def pricing(self) -> Dict[str, str]:
		"""Suggested pricing per plan (for marketplaces like RapidAPI)."""
		return self._data.get("pricing", {"free": "0€/mois", "pro5k": "10€/mois", "pro15k": "25€/mois", "pro50k": "50€/mois"})

	@property
	def cache(self) -> Dict[str, Any]:
		cfg = self._data.get("cache", {"ttl_seconds": 600, "max_size": 10000})
		# Environment overrides
		ttl_env = os.getenv("CACHE_TTL_SECONDS")
		max_env = os.getenv("CACHE_MAX_SIZE")
		if ttl_env:
			try:
				cfg = dict(cfg)
				cfg["ttl_seconds"] = int(ttl_env)
			except ValueError:
				pass
		if max_env:
			try:
				cfg = dict(cfg)
				cfg["max_size"] = int(max_env)
			except ValueError:
				pass
		return cfg

	@property
	def model(self) -> Dict[str, Any]:
		return self._data.get(
			"model",
			{"threshold": 0.5, "ml_weight": 0.6, "rules_weight": 0.4},
		)

	@property
	def regex_rules(self) -> Dict[str, Dict[str, Any]]:
		return self._data.get("regex_rules", {})

	@property
	def blacklist(self) -> Dict[str, Any]:
		return self._data.get("blacklist", {"domains": [], "ips": []})

	@property
	def gemini(self) -> Dict[str, Any]:
		# Merge YAML with environment overrides
		cfg = self._data.get(
			"gemini",
			{"enabled": False, "model": "gemini-1.5-pro", "weight": 0.5, "timeout": 8},
		)
		# Check .env first, then system environment
		api_key = os.getenv("GEMINI_API_KEY")
		if api_key and api_key != "your_gemini_api_key_here":
			cfg = dict(cfg)
			cfg["api_key"] = api_key
		return cfg

	@property
	def data_dir(self) -> str:
		return self._data.get("data_dir", os.environ.get("DATA_DIR", "data"))

	@property
	def logs_dir(self) -> str:
		return self._data.get("logs_dir", os.environ.get("LOGS_DIR", "logs"))


_config_instance: Config | None = None


def load_config() -> Config:
	global _config_instance
	if _config_instance is not None:
		return _config_instance

	config_path = os.environ.get("CONFIG_PATH", os.path.join(os.getcwd(), "config.yaml"))
	if not os.path.isfile(config_path):
		# Provide sensible defaults if no YAML present
		data = {
			"api_keys": {"test_free_key": "free", "test_pro_key": "pro5k"},
			"quotas": {"free": 100, "pro5k": 5000, "pro15k": 15000, "pro50k": 50000},
			"rate_limit_per_minute": {"free": 5, "pro5k": 120, "pro15k": 180, "pro50k": 240},
			"pricing": {"free": "0€/mois", "pro5k": "10€/mois", "pro15k": "25€/mois", "pro50k": "50€/mois"},
			"cache": {"ttl_seconds": 600, "max_size": 10000},
			"model": {"threshold": 0.5, "ml_weight": 0.6, "rules_weight": 0.4},
			"regex_rules": {},
			"blacklist": {"domains": [], "ips": []},
			"gemini": {"enabled": False, "model": "gemini-1.5-pro", "weight": 0.5, "timeout": 8},
		}
		_config_instance = Config(data)
		return _config_instance

	with open(config_path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}

	_config_instance = Config(data)
	return _config_instance 