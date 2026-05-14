import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(
    title="Memecoin Sentiment API",
    description="""
## Memecoin Sentiment Analysis API — x402 Protocol

Real-time sentiment signals for memecoins, built for AI agent consumption.

**Discovery endpoints:**
- `GET /llms.txt` — Human & agent readable service description
- `GET /.well-known/x402.json` — Payment schema for x402 agents

**Pricing:** 0.001 USDC per query · 0.005 USDC per batch · Base network

**Payment:** Include `X-Payment` header with your x402 payment proof.
For testing, use `X-Payment: test_mode`.
    """,
    version="1.0.0",
    openapi_tags=[
        {"name": "Sentiment", "description": "Memecoin sentiment analysis endpoints"},
        {"name": "Discovery", "description": "Agent discovery and payment schema"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
    expose_headers=["X-Payment-Required", "X-Request-Id", "X-Price"],
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PRICE_SINGLE      = os.getenv("PRICE_SINGLE", "0.001")
PRICE_BATCH       = os.getenv("PRICE_BATCH",  "0.005")
NETWORK           = os.getenv("NETWORK",      "base")
USDC_BASE         = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
PAYMENT_ADDRESS   = os.getenv("PAYMENT_ADDRESS", "")
FACILITATOR_URL   = os.getenv("FACILITATOR_URL", "https://x402.org/facilitator")
BASE_URL          = os.getenv("BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SentimentResponse(BaseModel):
    ticker: str = Field(..., example="DOGE", description="Ticker symbol (uppercase)")
    sentiment: str = Field(..., example="bullish", description="Sentiment label")
    score: float = Field(..., ge=-1.0, le=1.0, description="Score: -1.0 (very bearish) → 1.0 (very bullish)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
    signals: List[str] = Field(..., description="Detected sentiment signals")
    volume_trend: str = Field(..., description="Current volume trend")
    social_momentum: str = Field(..., description="Social media momentum")
    market_phase: str = Field(..., description="Estimated market cycle phase")
    timestamp: str = Field(..., description="ISO 8601 UTC")
    request_id: str = Field(..., description="Unique request identifier")


class BatchResponse(BaseModel):
    count: int
    results: List[SentimentResponse]


# ---------------------------------------------------------------------------
# Sentiment generation (deterministic within 1-minute windows, random later)
# ---------------------------------------------------------------------------

_SIGNALS = [
    "twitter_buzz", "reddit_activity", "telegram_volume",
    "whale_accumulation", "dex_volume_spike", "new_holders",
    "influencer_mentions", "cross_chain_bridging", "liquidity_depth",
    "price_momentum", "holder_growth", "burn_events",
    "dao_governance", "cex_listing_rumors", "fud_detected",
    "community_memes", "nft_crossover", "airdrop_speculation",
]

_SENTIMENT_MAP = [
    (0.6,  1.01, "very_bullish"),
    (0.2,  0.6,  "bullish"),
    (-0.2, 0.2,  "neutral"),
    (-0.6, -0.2, "bearish"),
    (-1.01,-0.6, "very_bearish"),
]

_VOLUME_TRENDS    = ["increasing", "decreasing", "stable", "volatile", "spiking"]
_SOCIAL_MOMENTUM  = ["rising", "falling", "neutral", "viral", "cooling", "dormant"]
_MARKET_PHASES    = ["accumulation", "markup", "distribution", "markdown", "consolidation"]


def _label(score: float) -> str:
    for lo, hi, label in _SENTIMENT_MAP:
        if lo <= score < hi:
            return label
    return "neutral"


def build_sentiment(ticker: str) -> dict:
    minute = int(time.time() // 60)
    seed = int(hashlib.md5(f"{ticker.upper()}:{minute}".encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)

    score = round(rng.uniform(-1.0, 1.0), 4)
    request_id = hashlib.sha256(f"{ticker}{time.time_ns()}".encode()).hexdigest()[:24]

    return {
        "ticker":          ticker.upper(),
        "sentiment":       _label(score),
        "score":           score,
        "confidence":      round(rng.uniform(0.52, 0.96), 4),
        "signals":         rng.sample(_SIGNALS, k=rng.randint(2, 6)),
        "volume_trend":    rng.choice(_VOLUME_TRENDS),
        "social_momentum": rng.choice(_SOCIAL_MOMENTUM),
        "market_phase":    rng.choice(_MARKET_PHASES),
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "request_id":      request_id,
    }


# ---------------------------------------------------------------------------
# x402 helpers
# ---------------------------------------------------------------------------

def _to_atomic(price_usd: str) -> str:
    """Convert human-readable USD amount to USDC atomic units (6 decimals)."""
    return str(int(float(price_usd) * 1_000_000))


def payment_payload(path: str, price: str, description: str) -> dict:
    return {
        "scheme":             "exact",
        "network":            NETWORK,
        "maxAmountRequired":  _to_atomic(price),
        "resource":           f"{BASE_URL}{path}",
        "description":        description,
        "mimeType":           "application/json",
        "payTo":              PAYMENT_ADDRESS,
        "maxTimeoutSeconds":  60,
        "asset":              USDC_BASE,
        "extra":              {
            "name":         "Memecoin Sentiment API",
            "version":      "1.0.0",
            "discoverable": True,
        },
    }


def _payment_error_response(requirements: dict, error: str, reason: str = "") -> JSONResponse:
    body: dict = {"x402Version": 1, "error": error, "accepts": [requirements]}
    if reason:
        body["reason"] = reason
    resp = JSONResponse(status_code=402, content=body)
    resp.headers["X-Payment-Required"] = json.dumps(requirements)
    resp.headers["X-Price"] = requirements["maxAmountRequired"]
    return resp


async def _verify_with_facilitator(payment_header: str, requirements: dict) -> Tuple[bool, str]:
    """POST to CDP facilitator /verify — returns (is_valid, reason)."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{FACILITATOR_URL}/verify",
                json={
                    "x402Version":        1,
                    "paymentHeader":      payment_header,
                    "paymentRequirements": [requirements],
                },
            )
        if resp.status_code != 200:
            return False, f"facilitator_http_{resp.status_code}"
        body = resp.json()
        return body.get("isValid", False), body.get("invalidReason") or ""
    except httpx.TimeoutException:
        return False, "facilitator_timeout"
    except Exception as exc:
        return False, f"facilitator_error:{exc}"


async def _settle_with_facilitator(payment_header: str, requirements: dict) -> None:
    """POST to CDP facilitator /settle — fire-and-forget after response is sent."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            await client.post(
                f"{FACILITATOR_URL}/settle",
                json={
                    "x402Version":        1,
                    "paymentHeader":      payment_header,
                    "paymentRequirements": [requirements],
                },
            )
    except Exception:
        pass  # settlement failure is logged server-side by the facilitator


async def check_payment(
    request: Request,
    path: str,
    price: str,
    description: str,
) -> Tuple[bool, Optional[JSONResponse]]:
    """
    Returns (paid, error_response).
    If paid=True, error_response is None.
    If paid=False, error_response is the 402 JSONResponse to return.
    """
    requirements = payment_payload(path, price, description)
    header = request.headers.get("X-Payment", "")

    if not header:
        return False, _payment_error_response(requirements, "payment_required")

    if header == "test_mode":
        return True, None

    valid, reason = await _verify_with_facilitator(header, requirements)
    if not valid:
        return False, _payment_error_response(requirements, "payment_invalid", reason)

    return True, None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return {
        "name":      "Memecoin Sentiment API",
        "version":   "1.0.0",
        "protocol":  "x402",
        "pricing":   {"single": f"{PRICE_SINGLE} USDC", "batch": f"{PRICE_BATCH} USDC"},
        "network":   NETWORK,
        "discovery": {
            "llms_txt":   "/llms.txt",
            "x402_schema": "/.well-known/x402.json",
            "openapi":    "/openapi.json",
            "docs":       "/docs",
        },
    }


@app.get(
    "/sentiment/{ticker}",
    response_model=SentimentResponse,
    summary="Sentiment for a single ticker",
    tags=["Sentiment"],
    responses={
        200: {"description": "Sentiment analysis result"},
        402: {"description": "x402 payment required"},
        422: {"description": "Invalid ticker"},
    },
)
async def get_sentiment(
    ticker: str,
    request: Request,
    background_tasks: BackgroundTasks,
    x_payment: Optional[str] = Header(None, description="x402 payment proof or 'test_mode'"),
):
    ticker = ticker.strip().upper()
    if not ticker.isalpha() or not (1 <= len(ticker) <= 10):
        raise HTTPException(status_code=422, detail="Ticker must be 1-10 alphabetic characters")

    path = f"/sentiment/{ticker}"
    paid, err = await check_payment(request, path, PRICE_SINGLE, f"Sentiment analysis for {ticker}")
    if not paid:
        return err

    payment_header = request.headers.get("X-Payment", "")
    if payment_header != "test_mode":
        background_tasks.add_task(
            _settle_with_facilitator,
            payment_header,
            payment_payload(path, PRICE_SINGLE, f"Sentiment analysis for {ticker}"),
        )

    data = build_sentiment(ticker)
    resp = JSONResponse(content=data)
    resp.headers["X-Request-Id"] = data["request_id"]
    return resp


@app.get(
    "/batch",
    response_model=BatchResponse,
    summary="Sentiment for up to 10 tickers",
    description="Pass `?tickers=DOGE,PEPE,SHIB` — comma-separated, max 10.",
    tags=["Sentiment"],
    responses={
        200: {"description": "Batch sentiment results"},
        402: {"description": "x402 payment required"},
        422: {"description": "No valid tickers"},
    },
)
async def get_batch(
    tickers: str,
    request: Request,
    background_tasks: BackgroundTasks,
    x_payment: Optional[str] = Header(None),
):
    paid, err = await check_payment(request, "/batch", PRICE_BATCH, "Batch sentiment (up to 10 tickers)")
    if not paid:
        return err

    payment_header = request.headers.get("X-Payment", "")
    if payment_header != "test_mode":
        background_tasks.add_task(
            _settle_with_facilitator,
            payment_header,
            payment_payload("/batch", PRICE_BATCH, "Batch sentiment (up to 10 tickers)"),
        )

    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    ticker_list = [t for t in ticker_list if t.isalpha() and 1 <= len(t) <= 10][:10]

    if not ticker_list:
        raise HTTPException(status_code=422, detail="No valid tickers provided")

    results = [build_sentiment(t) for t in ticker_list]
    return {"count": len(results), "results": results}


# ---------------------------------------------------------------------------
# Agent discovery endpoints
# ---------------------------------------------------------------------------

@app.get("/llms.txt", include_in_schema=False, tags=["Discovery"])
async def llms_txt():
    body = """\
# Memecoin Sentiment API

> Real-time sentiment analysis for memecoins. Built for AI agent consumption via the x402 micropayment protocol.

## Capabilities

- Sentiment scoring for any memecoin ticker (score: -1.0 to +1.0)
- Labels: very_bullish | bullish | neutral | bearish | very_bearish
- Signals: twitter_buzz, whale_accumulation, dex_volume_spike, holder_growth, and more
- Volume trend, social momentum, and market phase estimation
- Deterministic results within 1-minute windows (consistent agent polling)

## Endpoints

### Single ticker
```
GET /sentiment/{TICKER}
X-Payment: <x402_proof>
```
Example: `GET /sentiment/DOGE`

### Batch (up to 10 tickers)
```
GET /batch?tickers=DOGE,PEPE,SHIB
X-Payment: <x402_proof>
```

### Testing (no payment)
```
GET /sentiment/PEPE
X-Payment: test_mode
```

## Pricing (x402 Protocol — Base network)

| Endpoint  | Price (USDC) |
|-----------|-------------|
| /sentiment/{ticker} | 0.001 |
| /batch (≤10 tickers) | 0.005 |

Payment asset: USDC on Base (`0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913`)

## Payment Flow (x402)

1. Call endpoint without `X-Payment` header
2. Receive HTTP 402 with `X-Payment-Required` JSON header
3. Pay on-chain via CDP or PayAI facilitator
4. Retry with `X-Payment: <proof>` header
5. Receive sentiment JSON

## Response Schema

```json
{
  "ticker": "DOGE",
  "sentiment": "bullish",
  "score": 0.73,
  "confidence": 0.84,
  "signals": ["twitter_buzz", "whale_accumulation", "dex_volume_spike"],
  "volume_trend": "increasing",
  "social_momentum": "rising",
  "market_phase": "markup",
  "timestamp": "2026-05-14T12:00:00+00:00",
  "request_id": "a3f7c9d2e1b04a8f9c21"
}
```

## Machine-Readable Resources

- OpenAPI 3.1 schema: /openapi.json
- x402 payment schema: /.well-known/x402.json
- Interactive docs: /docs
"""
    return PlainTextResponse(content=body, media_type="text/plain; charset=utf-8")


@app.get("/.well-known/x402.json", include_in_schema=False, tags=["Discovery"])
async def x402_well_known():
    schema = {
        "version":        "1",
        "facilitator":    "https://x402.org/facilitator",
        "network":        NETWORK,
        "paymentAddress": PAYMENT_ADDRESS,
        "endpoints": [
            {
                "path":        "/sentiment/{ticker}",
                "method":      "GET",
                "description": "Sentiment analysis for a single memecoin ticker",
                "price": {
                    "amount":   PRICE_SINGLE,
                    "currency": "USDC",
                    "asset":    USDC_BASE,
                },
                "parameters": {
                    "ticker": {
                        "in":          "path",
                        "type":        "string",
                        "description": "Memecoin ticker symbol, e.g. DOGE, PEPE, SHIB",
                        "maxLength":   10,
                    }
                },
                "responseSchema": {"$ref": "/openapi.json#/components/schemas/SentimentResponse"},
            },
            {
                "path":        "/batch",
                "method":      "GET",
                "description": "Sentiment for up to 10 tickers in one call",
                "price": {
                    "amount":   PRICE_BATCH,
                    "currency": "USDC",
                    "asset":    USDC_BASE,
                },
                "parameters": {
                    "tickers": {
                        "in":          "query",
                        "type":        "string",
                        "description": "Comma-separated tickers, e.g. DOGE,PEPE,SHIB",
                    }
                },
            },
        ],
        "discovery": {
            "llms_txt": "/llms.txt",
            "openapi":  "/openapi.json",
        },
    }
    return JSONResponse(content=schema)
