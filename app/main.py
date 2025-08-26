from typing import Any, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .auth import enforce_quota_and_rate, increment_usage, init_usage_store, get_usage_count, increment_rate, validate_api_key_db, seed_test_keys
from .config import load_config
from .service import DetectionResult, get_service
from .utils import log_request


class CheckRequest(BaseModel):
	text: str = Field(..., min_length=1, max_length=10000)


class CheckDebugRequest(BaseModel):
	text: str = Field(..., min_length=1, max_length=10000)
	detail: bool = True


class StrictCheckResponse(BaseModel):
	is_spam: bool
	confidence: float


class ApiResponse(BaseModel):
	status: str
	data: Dict[str, Any]


class TrainExample(BaseModel):
	text: str
	label: str


class StatusResponse(BaseModel):
	status: str
	version: str


app = FastAPI(title="Spam Detection API", version="1.2.0")


@app.on_event("startup")
async def startup_event() -> None:
	"""Initialize databases, seed dev keys, and warm the model service."""
	init_usage_store()
	seed_test_keys()
	get_service()

# Serve the static website under /site
app.mount("/site", StaticFiles(directory="website", html=True), name="site")


@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    """Redirect the root to the static landing page."""
    return RedirectResponse(url="/site/index.html", status_code=307)


@app.get("/index.html", include_in_schema=False)
async def legacy_index() -> RedirectResponse:
	return RedirectResponse(url="/site/index.html", status_code=307)


@app.get("/docs.html", include_in_schema=False)
async def legacy_docs() -> RedirectResponse:
	return RedirectResponse(url="/site/docs.html", status_code=307)


@app.get("/playground.html", include_in_schema=False)
async def legacy_playground() -> RedirectResponse:
	return RedirectResponse(url="/site/playground.html", status_code=307)


@app.get("/contact.html", include_in_schema=False)
async def legacy_contact() -> RedirectResponse:
	return RedirectResponse(url="/site/contact.html", status_code=307)


@app.get("/website/{path:path}", include_in_schema=False)
async def legacy_website(path: str) -> RedirectResponse:
	# Support old absolute links like /website/index.html
	return RedirectResponse(url=f"/site/{path}", status_code=307)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
	"""Return sanitized JSON error responses without stack traces."""
	return JSONResponse(status_code=exc.status_code, content={"status": "error", "message": exc.detail})


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
	"""Public liveness probe: returns 200 if the app is running."""
	return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> Dict[str, str]:
	"""Public readiness probe: checks DB and model cache availability."""
	init_usage_store()
	get_service()
	return {"status": "ready"}


@app.get("/status", response_model=StatusResponse)
async def status_endpoint() -> StatusResponse:
	"""Public status with version info."""
	return StatusResponse(status="ok", version="1.2.0")


@app.get("/usage", response_model=ApiResponse)
async def usage_endpoint(keys: Tuple[str, str] = Depends(validate_api_key_db)) -> ApiResponse:
	"""Return monthly usage and quota limit for the calling API key."""
	api_key, plan = keys
	cfg = load_config()
	limit = int(cfg.quotas.get(plan, 0))
	used = get_usage_count(api_key)
	return ApiResponse(status="success", data={"plan": plan, "used_this_month": used, "monthly_limit": limit})


@app.post("/check", response_model=StrictCheckResponse)
async def check_endpoint(payload: CheckRequest, request: Request, keys: Tuple[str, str] = Depends(enforce_quota_and_rate)) -> StrictCheckResponse:
	"""Analyze text and return minimal JSON: {"is_spam": bool, "confidence": float}."""
	api_key, plan = keys
	service = get_service()
	result: DetectionResult = service.detect(payload.text)
	increment_usage(api_key)
	increment_rate(api_key)
	log_request(request, api_key, payload.text, result.score, result.is_spam)
	return StrictCheckResponse(is_spam=result.is_spam, confidence=round(result.score, 6))


class CheckDebugResponse(BaseModel):
	status: str
	data: Dict[str, Any]


@app.post("/check_debug", response_model=CheckDebugResponse)
async def check_debug_endpoint(payload: CheckDebugRequest, request: Request, keys: Tuple[str, str] = Depends(enforce_quota_and_rate)) -> CheckDebugResponse:
	"""Analyze text and return detailed diagnostics for debugging and tuning."""
	api_key, plan = keys
	service = get_service()
	result: DetectionResult = service.detect(payload.text)
	increment_usage(api_key)
	increment_rate(api_key)
	log_request(request, api_key, payload.text, result.score, result.is_spam)
	data: Dict[str, Any] = {
		"is_spam": result.is_spam,
		"confidence": round(result.score, 6),
		"ml_proba": round(result.ml_proba, 6),
		"rules_score": round(result.rules_score, 6),
		"gemini_proba": (round(result.gemini_proba, 6) if result.gemini_proba is not None else None),
		"gemini_model_used": result.gemini_model_used,
		"gemini_error": result.gemini_error,
	}
	return CheckDebugResponse(status="success", data=data)


class TrainRequest(BaseModel):
	examples: List[TrainExample]


@app.post("/train", response_model=ApiResponse)
async def train_endpoint(body: TrainRequest, keys: Tuple[str, str] = Depends(validate_api_key_db)) -> ApiResponse:
	"""Append training examples and retrain the underlying ML model."""
	service = get_service()
	added = service.add_training_examples([e.model_dump() for e in body.examples])
	return ApiResponse(status="success", data={"added": added})


try:
	from mangum import Mangum

	handler = Mangum(app)
except Exception:  # pragma: no cover
	handler = None


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False) 