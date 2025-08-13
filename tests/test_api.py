import os
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)
API_KEY = "test_free_key"
HEADERS = {"X-API-Key": API_KEY}


def test_status_ok():
	resp = client.get("/status")
	assert resp.status_code == 200
	data = resp.json()
	assert data["status"] == "ok"


def test_missing_api_key_rejected():
	resp = client.get("/usage")
	assert resp.status_code == 401


def test_usage_with_key():
	resp = client.get("/usage", headers=HEADERS)
	assert resp.status_code == 200
	payload = resp.json()
	assert payload["status"] == "success"
	assert "used_this_month" in payload["data"]
	assert "monthly_limit" in payload["data"]


def test_check_increments_usage_and_rate():
	before = client.get("/usage", headers=HEADERS).json()["data"]["used_this_month"]
	resp = client.post("/check", headers=HEADERS, json={"text": "Win a FREE iPhone now!"})
	assert resp.status_code == 200
	body = resp.json()
	assert set(body.keys()) == {"is_spam", "confidence"}
	after = client.get("/usage", headers=HEADERS).json()["data"]["used_this_month"]
	assert after == before + 1


def test_check_debug_returns_details():
	resp = client.post(
		"/check_debug",
		headers=HEADERS,
		json={"text": "Your account has been compromised. Verify at http://tinyurl.com/secure"},
	)
	assert resp.status_code == 200
	payload = resp.json()
	assert payload["status"] == "success"
	data = payload["data"]
	for key in ["is_spam", "confidence", "ml_proba", "rules_score"]:
		assert key in data 