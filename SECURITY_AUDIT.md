# Security Issues & Recommendations for VividMedi

## Critical Issues (Fix Immediately)

### 1. Input Validation Missing
**Location:** `/api/generate` and `/api/consult` endpoints
**Issue:** No validation on query/text length
**Fix:**
```python
MAX_QUERY_LENGTH = 10000
if len(query) > MAX_QUERY_LENGTH:
    return jsonify({"error": f"Query too long (max {MAX_QUERY_LENGTH} chars)"}), 400
if len(query) < 3:
    return jsonify({"error": "Query too short (min 3 chars)"}), 400
```

### 2. File Upload Size Limit Missing
**Location:** `/api/transcribe` endpoint
**Issue:** No max file size check
**Fix:**
```python
MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50MB
if len(f.read()) > MAX_AUDIO_SIZE:
    return jsonify({"error": "File too large"}), 413
```

### 3. Stripe Webhook Edge Cases
**Location:** `/api/stripe/webhook`
**Issue:** Null checks missing before accessing dict keys
**Fixes Needed:**
- Check `obj.get("customer")` is not None before querying DB
- Validate `subscription_id` exists before storing
- Add try/except for DB operations in webhook

### 4. Frontend Token Validation
**Location:** `templates/index.html`, line ~23 (`handleCredentialResponse`)
**Issue:** Token not validated before storing
**Fix:**
```javascript
if (!data.ok || !data.token || !data.user) {
    alert('Invalid response from server');
    return;
}
```

### 5. Session Cookie Security
**Location:** `app.py`, lines 77-80
**Issue:** `SESSION_COOKIE_SECURE` only True if HTTPS
**Fix:** Force secure in production
```python
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Strict",  # Change from Lax
    SESSION_COOKIE_SECURE=True,  # Always True in production
    PERMANENT_SESSION_LIFETIME=60 * 60 * 24,
)
```

## High Priority Issues

### 6. No Rate Limiting Per IP
**Current:** Rate limiting only per user/guest
**Add:**
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)
```

### 7. DeepSeek API Timeout Handling
**Current:** 70 second timeout with no retry logic
**Add:** Exponential backoff + max retries
```python
MAX_RETRIES = 2
for attempt in range(MAX_RETRIES):
    try:
        resp = http.post(DEEPSEEK_URL, ..., timeout=45)
        break
    except requests.Timeout:
        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
        else:
            raise
```

### 8. XSS in Sidebar Item Labels
**Location:** `templates/index.html`, line ~418
**Issue:** User input directly inserted into DOM
**Fix:**
```javascript
const label = clinicalInput.value.substring(0, 50) || 'Answer';
// Sanitize before inserting
const sanitized = document.createElement('span');
sanitized.textContent = label;
```

### 9. Missing CORS Headers
**Current:** Flask-CORS imported but not used
**Add:**
```python
from flask_cors import CORS
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://www.vividmedi.com"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
```

### 10. Database Connection Pool Missing
**Current:** New connection created per request
**Issue:** Could exhaust connections under load
**Fix:** Use connection pool
```python
from queue import Queue
DB_POOL = Queue(maxsize=10)
```

## Medium Priority

### 11. Whisper Model Lazy Loading
**Current:** Model loads on first transcription request
**Better:** Warmup model on startup
```python
if __name__ == "__main__":
    if not os.getenv("SKIP_MODEL_WARMUP"):
        get_whisper_model()  # Load immediately
```

### 12. Missing Content-Security-Policy Header
**Add:**
```python
@app.after_request
def set_security_headers(resp):
    resp.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' accounts.google.com; style-src 'self' 'unsafe-inline';"
    resp.headers['X-Content-Type-Options'] = 'nosniff'
    resp.headers['X-Frame-Options'] = 'DENY'
    return resp
```

### 13. Missing .env Example File
**Create:** `.env.example` with all required vars
```
GOOGLE_CLIENT_ID=xxx
DEEPSEEK_API_KEY=xxx
STRIPE_SECRET_KEY=xxx
STRIPE_WEBHOOK_SECRET=xxx
STRIPE_PRICE_ID_PRO=price_xxx
CREATOR_EMAIL=you@example.com
PRO_EMAILS=friend@example.com
APP_SECRET_KEY=generate-random-key
```

### 14. Conversation History Leakage
**Issue:** User's previous queries visible in context (privacy risk)
**Consider:** Add privacy mode or limit context depth

### 15. No Error Logging
**Current:** Prints to stdout only
**Better:** Structured logging with timestamps
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## Deployment Recommendations

1. **Add health check endpoint** ✓ (Already have `/health`)
2. **Set `SECURE_SSL_REDIRECT=True` in production**
3. **Use environment-based config** ✓ (Already doing this)
4. **Rotate `APP_SECRET_KEY` every 90 days**
5. **Monitor DeepSeek API usage** (Add quota tracking)
6. **Set up alerts for failed Stripe webhooks**
7. **Monitor database size** (No expiration policy on conversation_history)

## Low Priority / Nice-to-Have

- Add request logging middleware
- Implement request ID tracing
- Add performance metrics to /health endpoint
- Cache Google public keys for faster token verification
- Add feature flags for gradual rollouts
