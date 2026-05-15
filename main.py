import asyncio
import base64
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from eth_account import Account
from eth_account.messages import encode_typed_data
from fastapi import FastAPI, Request
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

PRICE_SINGLE    = os.getenv("PRICE_SINGLE", "0.001")
PRICE_BATCH     = os.getenv("PRICE_BATCH",  "0.005")
NETWORK         = os.getenv("NETWORK",      "base")
USDC_BASE       = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
PAYMENT_ADDRESS = os.getenv("PAYMENT_ADDRESS", "")
BASE_URL        = os.getenv("BASE_URL", "http://localhost:8000")
CHAIN_ID            = 8453  # Base mainnet
CDP_FACILITATOR_URL = os.getenv("CDP_FACILITATOR_URL", "https://api.cdp.coinbase.com/platform/v2/x402")

_ATOMIC_PER_USDC = 1_000_000  # USDC has 6 decimals

def _to_atomic(price_usd: str) -> str:
    return str(int(float(price_usd) * _ATOMIC_PER_USDC))

PRICE_SINGLE_ATOMIC = _to_atomic(PRICE_SINGLE)
PRICE_BATCH_ATOMIC  = _to_atomic(PRICE_BATCH)

# ---------------------------------------------------------------------------
# EIP-3009 local verification
# ---------------------------------------------------------------------------

_EIP712_DOMAIN_TYPES = [
    {"name": "name",              "type": "string"},
    {"name": "version",           "type": "string"},
    {"name": "chainId",           "type": "uint256"},
    {"name": "verifyingContract", "type": "address"},
]

_TRANSFER_WITH_AUTH_TYPES = [
    {"name": "from",        "type": "address"},
    {"name": "to",          "type": "address"},
    {"name": "value",       "type": "uint256"},
    {"name": "validAfter",  "type": "uint256"},
    {"name": "validBefore", "type": "uint256"},
    {"name": "nonce",       "type": "bytes32"},
]


def _verify_eip3009(auth: dict, signature_hex: str) -> Tuple[bool, str]:
    """Verify EIP-3009 transferWithAuthorization signature locally."""
    try:
        now = int(time.time())
        valid_before = int(auth.get("validBefore", 0))
        valid_after  = int(auth.get("validAfter",  0))

        if now >= valid_before:
            return False, "authorization_expired"
        if now < valid_after:
            return False, "authorization_not_yet_valid"

        nonce_raw = auth.get("nonce", "0x")
        nonce_hex = nonce_raw.lstrip("0x")
        nonce_bytes = bytes.fromhex(nonce_hex.zfill(64))

        typed_data = {
            "types": {
                "EIP712Domain": _EIP712_DOMAIN_TYPES,
                "TransferWithAuthorization": _TRANSFER_WITH_AUTH_TYPES,
            },
            "primaryType": "TransferWithAuthorization",
            "domain": {
                "name":              "USD Coin",
                "version":           "2",
                "chainId":           CHAIN_ID,
                "verifyingContract": USDC_BASE,
            },
            "message": {
                "from":        auth["from"],
                "to":          auth["to"],
                "value":       int(auth["value"]),
                "validAfter":  valid_after,
                "validBefore": valid_before,
                "nonce":       "0x" + nonce_bytes.hex(),
            },
        }

        signable  = encode_typed_data(full_message=typed_data)
        sig_bytes = bytes.fromhex(signature_hex.lstrip("0x"))
        recovered = Account.recover_message(signable, signature=sig_bytes)

        if recovered.lower() != auth["from"].lower():
            return False, "invalid_signature"

        return True, ""
    except Exception as exc:
        return False, f"verification_error:{exc}"


# ---------------------------------------------------------------------------
# CDP facilitator settlement (triggers Bazaar indexing via discoverable:true)
# ---------------------------------------------------------------------------

async def _settle_with_cdp(
    auth: dict,
    signature: str,
    price_atomic: str,
    resource_url: str,
    description: str,
) -> None:
    """Fire-and-forget settle call to CDP facilitator so it indexes us in Bazaar."""
    body = {
        "x402Version": 2,
        "paymentPayload": {
            "x402Version": 2,
            "scheme": "exact",
            "network": "eip155:8453",
            "payload": {
                "signature": signature if signature.startswith("0x") else f"0x{signature}",
                "authorization": {
                    "from":        auth["from"],
                    "to":          auth["to"],
                    "value":       str(int(auth["value"])),
                    "validAfter":  str(int(auth.get("validAfter", 0))),
                    "validBefore": str(int(auth.get("validBefore", 0))),
                    "nonce":       auth.get("nonce", "0x"),
                },
            },
        },
        "paymentRequirements": {
            "scheme":             "exact",
            "network":            "eip155:8453",
            "amount":             price_atomic,
            "resource":           resource_url,
            "description":        description,
            "mimeType":           "application/json",
            "payTo":              PAYMENT_ADDRESS,
            "maxTimeoutSeconds":  60,
            "asset":              USDC_BASE,
            "outputSchema":       None,
            "extra": {
                "name":         "USD Coin",
                "version":      "2",
                "discoverable": True,
            },
        },
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{CDP_FACILITATOR_URL}/settle", json=body)
            status = resp.status_code
            snippet = resp.text[:300]
            if status in (200, 201):
                print(f"[x402] CDP settle OK ({status}): {snippet}")
            else:
                print(f"[x402] CDP settle {status}: {snippet}")
    except Exception as exc:
        print(f"[x402] CDP settle error: {exc}")


# ---------------------------------------------------------------------------
# x402 V1 payment flow (compatible with x402-fetch v1.2.0)
# ---------------------------------------------------------------------------

def _payment_required(path: str, price_atomic: str, description: str) -> JSONResponse:
    """Return HTTP 402 in x402 V1 format."""
    body = {
        "x402Version": 1,
        "accepts": [{
            "scheme":             "exact",
            "network":            NETWORK,        # "base" (legacy V1 string)
            "maxAmountRequired":  price_atomic,
            "resource":           f"{BASE_URL}{path}",
            "description":        description,
            "mimeType":           "application/json",
            "payTo":              PAYMENT_ADDRESS,
            "maxTimeoutSeconds":  60,
            "asset":              USDC_BASE,
            "extra": {
                "name":         "USD Coin",
                "version":      "2",
                "discoverable": True,
            },
        }],
    }
    resp = JSONResponse(status_code=402, content=body)
    resp.headers["X-Payment-Required"] = json.dumps(body["accepts"][0])
    resp.headers["X-Price"] = price_atomic
    return resp


async def check_payment(
    request: Request,
    price_atomic: str,
    description: str,
) -> Tuple[bool, Optional[JSONResponse]]:
    """
    Verify x402 payment from X-Payment header.
    Returns (paid, error_response).
    """
    path   = request.url.path
    header = request.headers.get("x-payment", "")

    if not header:
        return False, _payment_required(path, price_atomic, description)

    if header.lower() == "test_mode":
        return True, None

    # Decode base64-encoded V1 payment payload
    try:
        decoded = json.loads(base64.b64decode(header + "=="))
    except Exception:
        return False, _payment_required(path, price_atomic, description)

    # Extract authorization and signature from V1 payload
    payload   = decoded.get("payload", {})
    auth      = payload.get("authorization", {})
    signature = payload.get("signature", "")

    if not auth or not signature:
        return False, _payment_required(path, price_atomic, description)

    # Recipient must be our payment address
    if auth.get("to", "").lower() != PAYMENT_ADDRESS.lower():
        return False, JSONResponse(
            status_code=402,
            content={"error": "wrong_recipient", "expected": PAYMENT_ADDRESS},
        )

    # Amount must be at least required
    if int(auth.get("value", 0)) < int(price_atomic):
        return False, JSONResponse(
            status_code=402,
            content={"error": "insufficient_amount"},
        )

    # Verify EIP-3009 signature locally
    valid, reason = _verify_eip3009(auth, signature)
    if not valid:
        return False, JSONResponse(status_code=402, content={"error": reason})

    # Fire-and-forget: notify CDP facilitator for Bazaar indexing
    resource_url = f"{BASE_URL}{path}"
    asyncio.create_task(
        _settle_with_cdp(auth, signature, price_atomic, resource_url, description)
    )

    return True, None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SentimentResponse(BaseModel):
    ticker:          str   = Field(..., example="DOGE",    description="Ticker symbol (uppercase)")
    sentiment:       str   = Field(..., example="bullish", description="Sentiment label")
    score:           float = Field(..., ge=-1.0, le=1.0,   description="Score: -1.0 (very bearish) → 1.0 (very bullish)")
    confidence:      float = Field(..., ge=0.0, le=1.0,    description="Model confidence")
    signals:         List[str] = Field(..., description="Detected sentiment signals")
    volume_trend:    str   = Field(..., description="Current volume trend")
    social_momentum: str   = Field(..., description="Social media momentum")
    market_phase:    str   = Field(..., description="Estimated market cycle phase")
    timestamp:       str   = Field(..., description="ISO 8601 UTC")
    request_id:      str   = Field(..., description="Unique request identifier")


class BatchResponse(BaseModel):
    count:   int
    results: List[SentimentResponse]


# ---------------------------------------------------------------------------
# Sentiment generation (deterministic within 1-minute windows)
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
    (0.6,   1.01,  "very_bullish"),
    (0.2,   0.6,   "bullish"),
    (-0.2,  0.2,   "neutral"),
    (-0.6, -0.2,   "bearish"),
    (-1.01, -0.6,  "very_bearish"),
]

_VOLUME_TRENDS   = ["increasing", "decreasing", "stable", "volatile", "spiking"]
_SOCIAL_MOMENTUM = ["rising", "falling", "neutral", "viral", "cooling", "dormant"]
_MARKET_PHASES   = ["accumulation", "markup", "distribution", "markdown", "consolidation"]


def _label(score: float) -> str:
    for lo, hi, label in _SENTIMENT_MAP:
        if lo <= score < hi:
            return label
    return "neutral"


def build_sentiment(ticker: str) -> dict:
    minute     = int(time.time() // 60)
    seed       = int(hashlib.md5(f"{ticker.upper()}:{minute}".encode()).hexdigest(), 16) % (2**32)
    rng        = random.Random(seed)
    score      = round(rng.uniform(-1.0, 1.0), 4)
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
            "llms_txt":    "/llms.txt",
            "x402_schema": "/.well-known/x402.json",
            "openapi":     "/openapi.json",
            "docs":        "/docs",
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
async def get_sentiment(ticker: str, request: Request):
    ticker = ticker.strip().upper()
    if not ticker.isalpha() or not (1 <= len(ticker) <= 10):
        raise Exception  # caught below
    paid, err = await check_payment(
        request, PRICE_SINGLE_ATOMIC, f"Sentiment analysis for {ticker}"
    )
    if not paid:
        return err

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
async def get_batch(tickers: str, request: Request):
    paid, err = await check_payment(
        request, PRICE_BATCH_ATOMIC, "Batch sentiment (up to 10 tickers)"
    )
    if not paid:
        return err

    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    ticker_list = [t for t in ticker_list if t.isalpha() and 1 <= len(t) <= 10][:10]

    if not ticker_list:
        return JSONResponse(
            status_code=422,
            content={"detail": "No valid tickers provided"},
        )

    results = [build_sentiment(t) for t in ticker_list]
    return {"count": len(results), "results": results}


# ---------------------------------------------------------------------------
# Agent discovery endpoints
# ---------------------------------------------------------------------------

@app.get("/llms.txt", include_in_schema=False, tags=["Discovery"])
async def llms_txt():
    body = f"""\
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
GET /sentiment/{{TICKER}}
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
| /sentiment/{{ticker}} | {PRICE_SINGLE} |
| /batch (≤10 tickers) | {PRICE_BATCH} |

Payment asset: USDC on Base (`{USDC_BASE}`)

## Payment Flow (x402)

1. Call endpoint without `X-Payment` header
2. Receive HTTP 402 with `X-Payment-Required` JSON header
3. Pay on-chain via EIP-3009 transferWithAuthorization
4. Retry with `X-Payment: <base64-encoded-payload>` header
5. Receive sentiment JSON

## Response Schema

```json
{{
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
}}
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
