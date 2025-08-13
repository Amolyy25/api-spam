import os
import re
import threading
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from cachetools import TTLCache

from .config import load_config
from .utils import extract_domains, extract_ips

try:
	import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
	genai = None  # type: ignore


MODEL_FILENAME = "model.joblib"
DATASET_FILENAME = "dataset.csv"


@dataclass
class DetectionResult:
	is_spam: bool
	score: float
	ml_proba: float
	rules_score: float
	gemini_proba: Optional[float] = None
	gemini_model_used: Optional[str] = None
	gemini_error: Optional[str] = None
	gemini_raw: Optional[str] = None


class GeminiClient:
	def __init__(self, api_key: str, model_name: str, timeout: int = 8) -> None:
		self._api_key = api_key
		self._model_name = model_name
		self._timeout = timeout
		self._model = None
		self._candidates: List[str] = [model_name, "gemini-1.5-flash", "gemini-1.5-pro"]
		if genai is not None:
			genai.configure(api_key=api_key)
			# defer actual model creation until first call to allow fallback

	def _build_prompt(self, text: str) -> str:
		return f"""You are a production-grade spam detection system for emails, messages, and chat content.

TASK:
Analyze the provided input and decide if it is spam.

CRITERIA (non-exhaustive, apply in combination):
1. Phishing attempts (fake login pages, credential harvesting)
2. Crypto scams (fake investments, giveaways, pump-and-dump)
3. Adult/explicit content and sexual solicitation
4. Lottery or prize scams
5. Deceptive or obfuscated links (shorteners, misspelled domains)
6. Excessive punctuation or symbol repetition
7. Keyword stuffing or unnatural repetition
8. Generic marketing blasts sent without personalization
9. Urgency + threat patterns ("act now", "account will be closed")

OUTPUT FORMAT (STRICT):
- Output MUST be a single JSON object with exactly two fields:
  - "is_spam": boolean (true or false)
  - "confidence": float between 0.0 and 1.0 (use decimal point, not comma)
- No extra fields, no explanations, no surrounding text.
- Confidence must reflect certainty, not probability of spam category.

INPUT:
{text}

OUTPUT:"""

	def _ensure_model(self, name: str):
		if genai is None:
			return None
		return genai.GenerativeModel(name)

	def score_text(self, text: str) -> Tuple[Optional[float], Optional[str], Optional[str], Optional[str]]:
		# returns: (proba, raw_text, model_used, error)
		if genai is None:
			return None, None, None, "sdk_not_installed"
		last_error: Optional[str] = None
		for name in self._candidates:
			try:
				model = self._ensure_model(name)
				if model is None:
					last_error = "model_unavailable"
					continue
				resp = model.generate_content(
					self._build_prompt(text),
					request_options={"timeout": self._timeout},
				)
				raw = getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else None)
				if not raw:
					last_error = "empty_response"
					continue
				try:
					data = json.loads(raw)
				except json.JSONDecodeError:
					start = raw.find("{")
					end = raw.rfind("}")
					if start != -1 and end != -1 and end > start:
						data = json.loads(raw[start : end + 1])
					else:
						last_error = "invalid_json"
						continue
				is_spam = bool(data.get("is_spam"))
				confidence = float(data.get("confidence", 0.0))
				confidence = max(0.0, min(1.0, confidence))
				proba = confidence if is_spam else (1.0 - confidence)
				return proba, raw, name, None
			except Exception as e:  # pragma: no cover
				last_error = type(e).__name__
				continue
		return None, None, None, last_error or "unknown_error"


class SpamDetectorService:
	def __init__(self) -> None:
		self._cfg = load_config()
		self._model: Pipeline | None = None
		self._regex_rules: List[Tuple[re.Pattern, float]] = []
		self._gemini: Optional[GeminiClient] = None
		self._lock = threading.RLock()
		cache_cfg = self._cfg.cache
		self._cache: TTLCache[str, DetectionResult] = TTLCache(maxsize=int(cache_cfg.get("max_size", 10000)), ttl=int(cache_cfg.get("ttl_seconds", 600)))
		self._load_regex_rules()
		self._ensure_data_dir()
		self._setup_gemini()
		self._load_or_train()

	def _ensure_data_dir(self) -> None:
		os.makedirs(self._cfg.data_dir, exist_ok=True)

	def _model_path(self) -> str:
		return os.path.join(self._cfg.data_dir, MODEL_FILENAME)

	def _dataset_path(self) -> str:
		path = os.path.join("data", DATASET_FILENAME)
		if os.path.isfile(path):
			return path
		return os.path.join(self._cfg.data_dir, DATASET_FILENAME)

	def _load_regex_rules(self) -> None:
		self._regex_rules.clear()
		for rule_name, rule in self._cfg.regex_rules.items():
			pattern = rule.get("pattern", "")
			weight = float(rule.get("weight", 0.0))
			if not pattern or weight <= 0:
				continue
			try:
				compiled = re.compile(pattern)
				self._regex_rules.append((compiled, weight))
			except re.error:
				continue

	def _setup_gemini(self) -> None:
		gcfg = self._cfg.gemini
		if not gcfg.get("enabled"):
			logging.info("Gemini disabled in config")
			return
		api_key = gcfg.get("api_key") or os.getenv("GEMINI_API_KEY")
		model_name = str(gcfg.get("model", "gemini-1.5-pro"))
		timeout = int(gcfg.get("timeout", 8))
		if api_key and api_key != "your_gemini_api_key_here":
			if genai is not None:
				self._gemini = GeminiClient(api_key=api_key, model_name=model_name, timeout=timeout)
				logging.info(f"Gemini configured with model {model_name}")
			else:
				logging.error("Gemini SDK not available")
		else:
			logging.warning("Gemini enabled but no valid API key found")

	def _load_or_train(self) -> None:
		with self._lock:
			path = self._model_path()
			if os.path.isfile(path):
				self._model = joblib.load(path)
				return
			self._train_internal()

	def _train_internal(self) -> None:
		dataset_path = self._dataset_path()
		if not os.path.isfile(dataset_path):
			raise FileNotFoundError(f"Dataset not found at {dataset_path}")
		df = pd.read_csv(dataset_path)
		texts = df["text"].astype(str).tolist()
		labels = df["label"].astype(str).str.lower().map(lambda x: 1 if x in {"spam", "1", "true", "yes"} else 0).values
		pipeline: Pipeline = Pipeline(
			[
				("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
				("clf", MultinomialNB()),
			]
		)
		pipeline.fit(texts, labels)
		self._model = pipeline
		joblib.dump(self._model, self._model_path())

	def predict_proba_spam(self, text: str) -> float:
		with self._lock:
			if self._model is None:
				self._load_or_train()
			assert self._model is not None
			proba = self._model.predict_proba([text])[0][1]
			return float(proba)

	def rules_score(self, text: str) -> float:
		score = 0.0
		for pattern, weight in self._regex_rules:
			if pattern.search(text):
				score += weight
		blacklist = self._cfg.blacklist
		domains = set(extract_domains(text))
		ips = set(extract_ips(text))
		if any(d in blacklist.get("domains", []) for d in domains):
			score += 0.4
		if any(ip in blacklist.get("ips", []) for ip in ips):
			score += 0.4
		return float(min(score, 1.0))

	def _cache_key(self, text: str) -> str:
		return text.strip().lower()

	def detect(self, text: str) -> DetectionResult:
		key = self._cache_key(text)
		if key in self._cache:
			return self._cache[key]

		ml_p = self.predict_proba_spam(text)
		rules_p = self.rules_score(text)
		weights = self._cfg.model
		ml_w = float(weights.get("ml_weight", 0.6))
		rules_w = float(weights.get("rules_weight", 0.4))
		threshold = float(weights.get("threshold", 0.5))

		gcfg = self._cfg.gemini
		g_w = 0.0
		g_p: Optional[float] = None
		g_raw: Optional[str] = None
		g_model_used: Optional[str] = None
		g_error: Optional[str] = None
		if gcfg.get("enabled"):
			g_w = float(gcfg.get("weight", 0.5))
			g_p, g_raw, g_model_used, g_error = self._gemini.score_text(text) if self._gemini else (None, None, None, "gemini_not_configured")
			if g_p is None:
				g_w = 0.0

		w_sum = ml_w + rules_w + g_w
		if w_sum <= 0:
			w_sum = 1.0
		ml_w_n = ml_w / w_sum
		rules_w_n = rules_w / w_sum
		g_w_n = g_w / w_sum

		final = ml_w_n * ml_p + rules_w_n * rules_p + (g_w_n * g_p if g_p is not None else 0.0)
		result = DetectionResult(
			is_spam=final >= threshold,
			score=float(final),
			ml_proba=ml_p,
			rules_score=rules_p,
			gemini_proba=(g_p if g_p is not None else None),
			gemini_model_used=g_model_used,
			gemini_error=g_error,
			gemini_raw=(g_raw[:500] if g_raw else None),
		)
		self._cache[key] = result
		return result

	def add_training_examples(self, examples: List[Dict[str, str]]) -> int:
		dataset_path = self._dataset_path()
		os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
		df = pd.DataFrame(examples)
		if not {"text", "label"}.issubset(df.columns):
			raise ValueError("Each example must have 'text' and 'label'")
		with self._lock:
			if os.path.isfile(dataset_path):
				df.to_csv(dataset_path, mode="a", header=False, index=False)
			else:
				df.to_csv(dataset_path, mode="w", header=True, index=False)
			self._train_internal()
			self._cache.clear()
			return len(df)


_service_singleton: SpamDetectorService | None = None


def get_service() -> SpamDetectorService:
	global _service_singleton
	if _service_singleton is None:
		_service_singleton = SpamDetectorService()
	return _service_singleton 