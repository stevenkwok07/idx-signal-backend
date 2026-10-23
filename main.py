import os
import logging
from functools import lru_cache
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("idx-signal-api")

TICKERS = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK",
    "GOTO.JK", "BYAN.JK", "UNVR.JK", "HMSP.JK", "ICBP.JK",
    "KLBF.JK", "SIDO.JK", "EXCL.JK", "ISAT.JK", "MNCN.JK",
    "ADRO.JK", "PTBA.JK", "INCO.JK", "ANTM.JK", "MDKA.JK",
    "SMGR.JK", "INKP.JK", "TKIM.JK", "JPFA.JK", "CPIN.JK",
    "ACES.JK", "MAPI.JK", "LPPF.JK", "ERAA.JK", "HEAL.JK",
]

app = FastAPI(title="IDX Signal API", version="1.0.0",
              description="API analisis teknikal saham IDX untuk frontend Netlify.")

# Untuk prototipe publik. Untuk produksi, batasi allow_origins ke domain Netlify milikmu.
origins_env = os.getenv("CORS_ORIGINS", "*")
origins = ["*"] if origins_env.strip() == "*" else [x.strip() for x in origins_env.split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

def clean_num(value):
    try:
        value = float(value)
        return round(value, 4) if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None

@lru_cache(maxsize=64)
def fetch_history(ticker: str):
    """Cache hasil selama proses berjalan supaya permintaan berulang lebih ringan."""
    df = yf.Ticker(ticker).history(period="6mo", auto_adjust=True)
    if df is None or df.empty or len(df) < 20:
        raise ValueError(f"Data historis tidak cukup untuk {ticker}")
    df = df.dropna(subset=["Close", "High", "Low"])
    if len(df) < 20:
        raise ValueError(f"Data valid tidak cukup untuk {ticker}")
    return df

def analyze_ticker(ticker: str):
    df = fetch_history(ticker)
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].fillna(0).astype(float)
    price = float(close.iloc[-1])

    ma5 = float(close.rolling(5).mean().iloc[-1])
    ma10 = float(close.rolling(10).mean().iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(min(50, len(close))).mean().iloc[-1])

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = float((100 - 100 / (1 + rs)).iloc[-1])
    if not np.isfinite(rsi):
        rsi = 50.0

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    low_min = low.rolling(14).min()
    high_max = high.rolling(14).max()
    stoch = 100 * (close - low_min) / (high_max - low_min + 1e-9)

    score = 0
    max_score = 10
    details = []
    for condition, yes, no in [
        (price > ma5, "Harga di atas MA5", "Harga di bawah MA5"),
        (ma5 > ma10, "MA5 di atas MA10", "MA5 di bawah MA10"),
        (price > ma20, "Harga di atas MA20", "Harga di bawah MA20"),
        (price > ma50, "Harga di atas MA50", "Harga di bawah MA50"),
    ]:
        score += int(condition)
        details.append(yes if condition else no)

    if 40 <= rsi <= 65:
        score += 2
        details.append(f"RSI {rsi:.1f}: rentang momentum moderat")
    elif rsi < 40 or rsi <= 70:
        score += 1
        details.append(f"RSI {rsi:.1f}: perlu konfirmasi")
    else:
        details.append(f"RSI {rsi:.1f}: momentum tinggi, waspadai risiko")

    macd_now, macd_sig_now, hist_now = float(macd.iloc[-1]), float(macd_signal.iloc[-1]), float(macd_hist.iloc[-1])
    score += int(macd_now > macd_sig_now) + int(hist_now > 0)
    details.append("MACD di atas signal" if macd_now > macd_sig_now else "MACD di bawah signal")
    details.append("Histogram MACD positif" if hist_now > 0 else "Histogram MACD negatif")

    upper = float(bb_upper.iloc[-1]) if pd.notna(bb_upper.iloc[-1]) else price
    middle = float(bb_mid.iloc[-1]) if pd.notna(bb_mid.iloc[-1]) else price
    lower = float(bb_lower.iloc[-1]) if pd.notna(bb_lower.iloc[-1]) else price
    bb_pos = (price - lower) / (upper - lower + 1e-9) * 100
    score += 2 if 20 <= bb_pos <= 70 else (1 if bb_pos <= 85 else 0)

    vol_ma5 = float(volume.rolling(5).mean().iloc[-1])
    vol_ma20 = float(volume.rolling(min(20, len(volume))).mean().iloc[-1])
    if vol_ma5 > vol_ma20:
        score += 1
        details.append("Volume rata-rata 5 hari di atas 20 hari")
    else:
        details.append("Volume belum mengonfirmasi kenaikan")

    score = min(score, max_score + 1)
    score_pct = round(100 * score / (max_score + 1), 1)
    momentum_5 = (price / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 else 0
    momentum_20 = (price / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0
    trend = "BULLISH" if price > ma20 and macd_now > macd_sig_now else ("BEARISH" if price < ma20 and macd_now < macd_sig_now else "SIDEWAYS")
    recommendation = "Pantau momentum" if trend == "BULLISH" else ("Waspadai tekanan turun" if trend == "BEARISH" else "Tunggu konfirmasi")

    history = []
    for date, row in df.tail(60).iterrows():
        history.append({
            "date": date.strftime("%Y-%m-%d"),
            "close": clean_num(row["Close"]),
            "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else 0
        })

    return {
        "ticker": ticker,
        "code": ticker.replace(".JK", ""),
        "name": ticker.replace(".JK", ""),
        "currency": "IDR",
        "price": clean_num(price),
        "previous_close": clean_num(close.iloc[-2]),
        "change_percent": clean_num((price / float(close.iloc[-2]) - 1) * 100) if len(close) > 1 else None,
        "rsi": clean_num(rsi),
        "macd": clean_num(macd_now),
        "macd_signal": clean_num(macd_sig_now),
        "macd_histogram": clean_num(hist_now),
        "ma5": clean_num(ma5), "ma10": clean_num(ma10),
        "ma20": clean_num(ma20), "ma50": clean_num(ma50),
        "bollinger_upper": clean_num(upper), "bollinger_middle": clean_num(middle),
        "bollinger_lower": clean_num(lower), "stochastic": clean_num(stoch.iloc[-1]),
        "support": clean_num(low.tail(20).min()),
        "resistance": clean_num(high.tail(20).max()),
        "momentum_5d_percent": clean_num(momentum_5),
        "momentum_20d_percent": clean_num(momentum_20),
        "score": score, "score_percent": score_pct,
        "trend": trend, "summary": recommendation, "details": details,
        "history": history,
        "data_as_of": df.index[-1].strftime("%Y-%m-%d"),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "Yahoo Finance via yfinance",
    }

@app.get("/")
def root():
    return {"name": "IDX Signal API", "status": "ok", "docs": "/docs",
            "endpoints": ["/api/health", "/api/scan", "/api/trend", "/api/stock/BBCA"]}

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "idx-signal-api"}

@app.get("/api/stock/{code}")
def stock(code: str):
    ticker = code.upper().strip()
    if not ticker.endswith(".JK"):
        ticker += ".JK"
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail="Kode saham belum ada di daftar ticker yang diizinkan.")
    try:
        return analyze_ticker(ticker)
    except Exception as exc:
        log.exception("Gagal menganalisis %s", ticker)
        raise HTTPException(status_code=502, detail=f"Data saham tidak tersedia saat ini untuk {ticker}. Coba lagi nanti.") from exc

@app.get("/api/scan")
def scan(limit: int = Query(default=30, ge=1, le=30)):
    results, errors = [], []
    for ticker in TICKERS:
        try:
            item = analyze_ticker(ticker)
            results.append(item)
        except Exception as exc:
            log.warning("Skip %s: %s", ticker, exc)
            errors.append(ticker)
    results.sort(key=lambda x: (x["score_percent"], x["momentum_5d_percent"] or -999), reverse=True)
    return {"count": len(results[:limit]), "unavailable": errors, "results": results[:limit],
            "data_as_of": max((x["data_as_of"] for x in results), default=None),
            "note": "Harga historis dari penyedia data; bisa terlambat dan dapat gagal sementara."}

@app.get("/api/trend")
def trend():
    bullish, bearish, sideways, unavailable = [], [], [], []
    for ticker in TICKERS:
        try:
            item = analyze_ticker(ticker)
            if item["trend"] == "BULLISH":
                bullish.append(item)
            elif item["trend"] == "BEARISH":
                bearish.append(item)
            else:
                sideways.append(item)
        except Exception as exc:
            log.warning("Skip %s: %s", ticker, exc)
            unavailable.append(ticker)
    bullish.sort(key=lambda x: x["score_percent"], reverse=True)
    bearish.sort(key=lambda x: x["score_percent"])
    return {"bullish": bullish[:10], "bearish": bearish[:10],
            "sideways": sideways[:10], "unavailable": unavailable}

@app.post("/api/cache/clear")
def clear_cache():
    fetch_history.cache_clear()
    return {"status": "cleared"}
