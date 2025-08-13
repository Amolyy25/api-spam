# Spam Detection API

API REST de détection de spam hybride (ML + règles + Gemini optionnel) construite avec FastAPI. Auth par clé API, quotas mensuels, rate limiting, cache TTL et logs production-ready.

## Prérequis
- Python 3.11+
- pip

## Installation
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Configuration
- Fichier `config.yaml` pour les plans et règles
- Variables d’environnement clés:
  - `GEMINI_API_KEY`: clé Gemini (optionnel)
  - `DATA_DIR` (défaut: `data`)
  - `LOGS_DIR` (défaut: `logs`)
  - `CACHE_TTL_SECONDS` (défaut: 600)
  - `CACHE_MAX_SIZE` (défaut: 10000)

Gemini (optionnel): dans `config.yaml` → `gemini.enabled: true`, `gemini.model`, `gemini.weight`.

## Lancer en local
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Swagger UI: `http://localhost:8000/docs`

## Endpoints
- GET `/healthz` (public): liveness
- GET `/readyz` (public): readiness
- GET `/status` (public): version de l’API
- GET `/usage` (auth): consommation/mois pour la clé
- POST `/check` (auth): détection stricte (JSON minimal)
- POST `/check_debug` (auth): détection + détails (diagnostic)
- POST `/train` (auth): ajoute des exemples et réentraîne

## Exemples clients
Assurez-vous d’utiliser l’en-tête `X-API-Key`. Des clés de test sont auto-seed en dev (ex: `test_free_key`). En production, générez vos clés et stockez-les en DB.

### curl
```bash
# Status (public)
curl -s http://localhost:8000/status

# Usage (auth)
curl -s -H "X-API-Key: test_free_key" http://localhost:8000/usage

# Check
curl -s -X POST http://localhost:8000/check \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test_free_key" \
  -d '{"text":"Win a FREE iPhone now! Click here: http://spamdomain.com/win"}'

# Check (debug)
curl -s -X POST http://localhost:8000/check_debug \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test_free_key" \
  -d '{"text":"Your account has been compromised. Verify at http://tinyurl.com/secure"}'
```

### Python (requests)
```python
import requests

BASE = "http://localhost:8000"
HEADERS = {"X-API-Key": "test_free_key", "Content-Type": "application/json"}

# Usage
r = requests.get(f"{BASE}/usage", headers=HEADERS)
print(r.status_code, r.json())

# Check
payload = {"text": "Limited time offer!!! Visit https://bit.ly/deal now"}
r = requests.post(f"{BASE}/check", headers=HEADERS, json=payload)
print(r.status_code, r.json())
```

### JavaScript (fetch)
```javascript
const BASE = "http://localhost:8000";
const headers = {"X-API-Key": "test_free_key", "Content-Type": "application/json"};

async function usage() {
  const res = await fetch(`${BASE}/usage`, { headers });
  const data = await res.json();
  console.log(res.status, data);
}

async function check(text) {
  const res = await fetch(`${BASE}/check`, {
    method: "POST",
    headers,
    body: JSON.stringify({ text }),
  });
  const data = await res.json();
  console.log(res.status, data);
}

usage();
check("URGENT: Your account will be closed! Click here: http://bit.ly/verify");
```

## Format de réponse
- `/check` (strict): `{"is_spam": boolean, "confidence": float}`
- `/check_debug`: ajoute `ml_proba`, `rules_score`, `gemini_proba`, etc.

## Codes d’erreur
- 401 Unauthorized: clé API absente ou invalide (ou désactivée)
- 422 Unprocessable Entity: entrée invalide (ex: corps JSON malformé)
- 429 Too Many Requests: quota mensuel dépassé ou rate limit par minute atteint
- 500 Internal Server Error: erreur interne (réponse JSON sans stack trace)

## Clés API: test vs production
- Dev: `config.yaml` peut contenir des clés de test (ex: `test_free_key`), auto-seedées en DB au démarrage.
- Prod: générez vos clés, stockées dans la table `api_keys` (hash SHA-256 en DB); ne mettez pas de clés sensibles dans `config.yaml`.

## Monitoring
- `/healthz` et `/readyz` disponibles.
- Optionnel: ajoutez `/metrics` (Prometheus) via `prometheus-fastapi-instrumentator` si trafic élevé.

## Déploiement
### Render/Railway/Heroku
- Build: `pip install -r requirements.txt`
- Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Variables d’environnement:
  - `DATA_DIR` (ex: `/var/data`)
  - `LOGS_DIR` (ex: `/var/log/app`)
  - `CACHE_TTL_SECONDS` (ex: `600`)
  - `CACHE_MAX_SIZE` (ex: `10000`)
  - `GEMINI_API_KEY` (si Gemini activé)
  - `CONFIG_PATH` si vous fournissez un YAML custom

Persistance (SQLite): Assurez-vous que `DATA_DIR` est un volume persistant sur la plateforme choisie.

## Tests de charge / adversarial
- Rejouez des messages borderline: phishing discret, crypto, liens masqués, ponctuation excessive, répétition de mots.
- Montez progressivement le RPS (k6, Locust). Vérifiez 429 attendus et stabilité.
- Surveillez `logs/requests.log` et `logs/errors.log`.

## Structure
```
app/
  __init__.py
  main.py
  auth.py
  config.py
  service.py
  utils.py
config.yaml
requirements.txt
data/dataset.csv
```

## Licence
MIT 