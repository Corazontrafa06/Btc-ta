#!/usr/bin/env python3
"""
BTC Technical Analysis Bot
==========================
Multi-timeframe BTCUSDT, alert hanya kalau gerbang lolos, lalu jejak hasilnya.

Gerbang alert
- Daily tidak melawan dan H4 searah. 15m tidak masuk skor.
- R:R dihitung dari tengah zona, bukan dari harga sekarang.
- SL > 2.5% ditandai swing (min R:R 1.8). SL > 6% dibuang.
- Jarak ke level lawan minimal 0.8 ATR.
- Trigger butuh candle (engulf / pin / sweep) + volume.
- Funding ekstrem + OI naik + harga mandek mematikan entry aggressive.
- Dinding order book di depan target memotong TP atau membatalkan ide.

State: FORMING → TRIGGERED → TP1 → TP2 / BE, atau SL / INVALID / EXPIRED.
/stats membaca hasil yang tersimpan. Cooldown ada di database, bukan memory.

Env: BOT_TOKEN, DB_PATH (opsional)
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import os
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import aiohttp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from telegram import Update
from telegram.constants import ParseMode
from telegram.error import Forbidden
from telegram.ext import Application, CommandHandler, ContextTypes

# ====================== CONFIG ======================
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
DB_PATH = os.getenv("DB_PATH", "btc_ta.db")
SYMBOL = "BTCUSDT"

SPOT_KLINE_URLS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
    "https://fapi.binance.com/fapi/v1/klines",
]
SPOT_DEPTH_URLS = [
    "https://fapi.binance.com/fapi/v1/depth",
    "https://data-api.binance.vision/api/v3/depth",
    "https://api.binance.com/api/v3/depth",
]
FAPI_PREMIUM = [
    "https://fapi.binance.com/fapi/v1/premiumIndex",
    "https://fapi1.binance.com/fapi/v1/premiumIndex",
]
FAPI_OI = [
    "https://fapi.binance.com/fapi/v1/openInterest",
    "https://fapi1.binance.com/fapi/v1/openInterest",
]
FAPI_OI_HIST = [
    "https://fapi.binance.com/futures/data/openInterestHist",
    "https://fapi1.binance.com/futures/data/openInterestHist",
]

TIMEFRAMES = {
    "15m": {"interval": "15m", "limit": 300, "label": "15m"},
    "1h": {"interval": "1h", "limit": 320, "label": "H1"},
    "4h": {"interval": "4h", "limit": 320, "label": "H4"},
    "1d": {"interval": "1d", "limit": 400, "label": "Daily"},
    "1w": {"interval": "1w", "limit": 300, "label": "Weekly"},
}
TF_WEIGHT = {"Weekly": 3, "Daily": 3, "H4": 2, "H1": 1, "15m": 0}
TF_ORDER = ["Weekly", "Daily", "H4", "H1", "15m"]

MIN_RR_INTRADAY = 1.6
MIN_RR_SWING = 1.8
MAX_SL_INTRADAY_PCT = 2.2
MAX_SL_PCT = 5.5
ROOM_ATR = 0.9
WALL_USD = 8_000_000
FUNDING_WARM = 0.0003
FUNDING_CROWDED = 0.0005
FUNDING_EXTREME = 0.001
OI_UP_PCT = 1.5
SCAN_MINUTES = int(os.getenv("SCAN_MINUTES", "12"))
MONITOR_MINUTES = int(os.getenv("MONITOR_MINUTES", "3"))
REENTRY_HOURS = 6
FORMING_MAX_ATR = 1.2          # hanya FORMING kalau zona ≤ 1.2 ATR
TRIGGER_MAX_ATR = 0.35         # harga harus sangat dekat / sudah di zona saat trigger

REQUEST_TIMEOUT = 15
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BTC-TA-Bot/2.0)",
    "Accept": "application/json",
}
# ====================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("BTC_TA")

_db_lock = asyncio.Lock()


# -------------------- Database --------------------

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with closing(_connect()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS subscribers (
                chat_id     INTEGER PRIMARY KEY,
                subscribed  INTEGER DEFAULT 1,
                created_at  TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS setups (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint   TEXT,
                side          TEXT NOT NULL,
                style         TEXT,
                state         TEXT NOT NULL,
                kind          TEXT,
                entry_low     REAL,
                entry_high    REAL,
                sl            REAL,
                tp1           REAL,
                tp2           REAL,
                rr            REAL,
                atr           REAL,
                price         REAL,
                score         INTEGER,
                note          TEXT,
                result        TEXT,
                r_multiple    REAL,
                created_at    TEXT,
                updated_at    TEXT,
                triggered_at  TEXT,
                closed_at     TEXT,
                last_notified TEXT
            )
            """
        )
        conn.commit()


def add_subscriber(chat_id: int) -> None:
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO subscribers (chat_id, subscribed) VALUES (?, 1)",
            (chat_id,),
        )
        conn.commit()


def remove_subscriber(chat_id: int) -> None:
    with closing(_connect()) as conn:
        conn.execute(
            "UPDATE subscribers SET subscribed = 0 WHERE chat_id = ?",
            (chat_id,),
        )
        conn.commit()


def get_subscribers() -> List[int]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT chat_id FROM subscribers WHERE subscribed = 1"
        ).fetchall()
    return [int(r["chat_id"]) for r in rows]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime] = None) -> str:
    return (dt or _now()).strftime("%Y-%m-%d %H:%M:%S")


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def get_active_setup() -> Optional[sqlite3.Row]:
    with closing(_connect()) as conn:
        row = conn.execute(
            """
            SELECT * FROM setups
            WHERE closed_at IS NULL
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    return row


def recent_fingerprint(fingerprint: str, hours: int = REENTRY_HOURS) -> bool:
    with closing(_connect()) as conn:
        row = conn.execute(
            """
            SELECT closed_at FROM setups
            WHERE fingerprint = ? AND closed_at IS NOT NULL
            ORDER BY id DESC LIMIT 1
            """,
            (fingerprint,),
        ).fetchone()
    if not row:
        return False
    closed = _parse_iso(row["closed_at"])
    if not closed:
        return False
    return (_now() - closed) < timedelta(hours=hours)


def insert_setup(plan: Dict[str, Any], state: str) -> int:
    now = _iso()
    triggered = now if state == "TRIGGERED" else None
    with closing(_connect()) as conn:
        cur = conn.execute(
            """
            INSERT INTO setups (
                fingerprint, side, style, state, kind,
                entry_low, entry_high, sl, tp1, tp2, rr, atr, price, score, note,
                created_at, updated_at, triggered_at, last_notified
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan["fingerprint"], plan["side"], plan["style"], state, plan["kind"],
                plan["entry_low"], plan["entry_high"], plan["sl"], plan["tp1"], plan["tp2"],
                plan["rr"], plan["atr"], plan["price"], plan["score"], plan.get("note", ""),
                now, now, triggered, state,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def update_setup(setup_id: int, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = _iso()
    cols = ", ".join(f"{k} = ?" for k in fields)
    with closing(_connect()) as conn:
        conn.execute(
            f"UPDATE setups SET {cols} WHERE id = ?",
            (*fields.values(), setup_id),
        )
        conn.commit()


def close_setup(
    setup_id: int,
    state: str,
    result: str,
    r_multiple: Optional[float],
    note: str = "",
) -> None:
    now = _iso()
    with closing(_connect()) as conn:
        conn.execute(
            """
            UPDATE setups
            SET state = ?, result = ?, r_multiple = ?, note = ?,
                closed_at = ?, updated_at = ?, last_notified = ?
            WHERE id = ?
            """,
            (state, result, r_multiple, note, now, now, state, setup_id),
        )
        conn.commit()


def recent_results(limit: int = 10) -> List[sqlite3.Row]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT * FROM setups
            WHERE closed_at IS NOT NULL
            ORDER BY closed_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return list(rows)


def all_closed() -> List[sqlite3.Row]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT result, r_multiple FROM setups
            WHERE closed_at IS NOT NULL
            """
        ).fetchall()
    return list(rows)


# -------------------- HTTP --------------------

async def fetch_json(
    session: aiohttp.ClientSession,
    urls: Sequence[str],
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    for url in urls:
        try:
            async with session.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.warning("HTTP %s %s", resp.status, url)
        except Exception as exc:
            logger.warning("Fetch %s: %s", url, exc)
    return None


def _klines_to_df(raw: Any) -> Optional[pd.DataFrame]:
    if not isinstance(raw, list) or len(raw) < 50:
        return None
    df = pd.DataFrame(
        raw,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    for col in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df if len(df) >= 50 else None


async def fetch_klines(
    session: aiohttp.ClientSession, interval: str, limit: int
) -> Optional[pd.DataFrame]:
    raw = await fetch_json(
        session,
        SPOT_KLINE_URLS,
        {"symbol": SYMBOL, "interval": interval, "limit": limit},
    )
    return _klines_to_df(raw)


# -------------------- Indicators --------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def bollinger(series: pd.Series, period: int = 20, nstd: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + nstd * std
    lower = mid - nstd * std
    width = (upper - lower) / mid.replace(0, np.nan)
    percent_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, width, percent_b


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def swing_points(df: pd.DataFrame, order: int = 5) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    swing_highs: List[Tuple[int, float]] = []
    swing_lows: List[Tuple[int, float]] = []
    n = len(df)
    for i in range(order, n - order):
        if highs[i] == np.max(highs[i - order : i + order + 1]):
            swing_highs.append((i, float(highs[i])))
        if lows[i] == np.min(lows[i - order : i + order + 1]):
            swing_lows.append((i, float(lows[i])))
    return swing_highs, swing_lows


def cluster_levels(levels: Sequence[float], last: float, threshold_pct: float = 0.004, max_levels: int = 6) -> List[float]:
    if not levels:
        return []
    ordered = sorted(levels)
    groups: List[List[float]] = [[ordered[0]]]
    for level in ordered[1:]:
        if abs(level - groups[-1][-1]) / groups[-1][-1] <= threshold_pct:
            groups[-1].append(level)
        else:
            groups.append([level])
    merged = [float(np.median(group)) for group in groups]
    return sorted(merged, key=lambda level: abs(level - last))[:max_levels]


def find_swing_levels(df: pd.DataFrame, order: int = 5, max_levels: int = 6):
    closed = df.iloc[:-1] if len(df) > order * 2 + 5 else df
    highs, lows = swing_points(closed, order=order)
    last = float(df["close"].iloc[-1])
    return (
        cluster_levels([price for _, price in highs], last, max_levels=max_levels),
        cluster_levels([price for _, price in lows], last, max_levels=max_levels),
    )


def _closed(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[:-1] if len(df) > 2 else df


def bullish_engulf(df: pd.DataFrame) -> bool:
    candles = _closed(df)
    if len(candles) < 2:
        return False
    prev, row = candles.iloc[-2], candles.iloc[-1]
    return bool(
        prev.close < prev.open
        and row.close > row.open
        and row.close >= prev.open
        and row.open <= prev.close
    )


def bearish_engulf(df: pd.DataFrame) -> bool:
    candles = _closed(df)
    if len(candles) < 2:
        return False
    prev, row = candles.iloc[-2], candles.iloc[-1]
    return bool(
        prev.close > prev.open
        and row.close < row.open
        and row.close <= prev.open
        and row.open >= prev.close
    )


def _pin(row: pd.Series, side: str) -> bool:
    rng = float(row.high - row.low)
    if rng <= 0:
        return False
    body = abs(float(row.close - row.open))
    lower = float(min(row.open, row.close) - row.low)
    upper = float(row.high - max(row.open, row.close))
    if side == "LONG":
        return lower >= 0.55 * rng and body <= 0.35 * rng and upper <= 0.22 * rng and float(row.close) >= float(row.low) + 0.60 * rng
    return upper >= 0.55 * rng and body <= 0.35 * rng and lower <= 0.22 * rng and float(row.close) <= float(row.high) - 0.60 * rng


def bullish_pin(df: pd.DataFrame) -> bool:
    candles = _closed(df)
    return len(candles) > 0 and _pin(candles.iloc[-1], "LONG")


def bearish_pin(df: pd.DataFrame) -> bool:
    candles = _closed(df)
    return len(candles) > 0 and _pin(candles.iloc[-1], "SHORT")


def volume_stats(df: pd.DataFrame) -> Tuple[float, float]:
    """Return (volume vs 20-avg, taker-buy ratio) of the last closed candle."""
    candles = _closed(df)
    if len(candles) < 22:
        return 0.0, 0.0
    last = candles.iloc[-1]
    avg = float(candles["volume"].iloc[-21:-1].mean())
    vol = float(last.volume)
    ratio = vol / avg if avg > 0 else 0.0
    taker = float(last.taker_buy_base) / vol if vol > 0 else 0.0
    return ratio, taker


def volume_confirms(df: pd.DataFrame, side: str, strict_taker: bool = True) -> bool:
    ratio, taker = volume_stats(df)
    if ratio < 1.0:
        return False
    if not strict_taker:
        if side == "LONG":
            return taker >= 0.40
        return taker <= 0.60
    if side == "LONG":
        return taker >= 0.52
    return taker <= 0.48


def detect_sweep(df: pd.DataFrame, label: str, order: int = 4) -> Optional[Dict[str, Any]]:
    closed = _closed(df)
    if len(closed) < 40:
        return None
    base = closed.iloc[:-3]
    if len(base) < 30:
        return None
    highs, lows = swing_points(base, order=order)
    last_rows = list(closed.iloc[-3:].itertuples(index=False))
    for row in reversed(last_rows):
        if lows:
            level = lows[-1][1]
            pierced = level - float(row.low)
            if pierced > 0 and float(row.close) > level and pierced >= 0.15 * max(float(row.high - row.low), 1e-9):
                if (float(row.close) - level) >= 0.20 * pierced and abs(float(row.close) - level) / level < 0.03:
                    return {"side": "LONG", "level": level, "wick": float(row.low), "tf": label}
        if highs:
            level = highs[-1][1]
            pierced = float(row.high) - level
            if pierced > 0 and float(row.close) < level and pierced >= 0.15 * max(float(row.high - row.low), 1e-9):
                if (level - float(row.close)) >= 0.20 * pierced and abs(float(row.close) - level) / level < 0.03:
                    return {"side": "SHORT", "level": level, "wick": float(row.high), "tf": label}
    return None


def is_stalling(df: pd.DataFrame, atr_val: float) -> bool:
    candles = _closed(df).tail(6)
    if len(candles) < 4 or atr_val <= 0:
        return False
    return float(candles.high.max() - candles.low.min()) < 1.6 * atr_val


# -------------------- Order book / derivatives --------------------

def cluster_book(levels: Sequence[Tuple[float, float]], pct: float = 0.0015) -> List[Dict[str, float]]:
    """Group size by a fixed anchor so adjacent ticks don't merge into one fake wall."""
    if not levels:
        return []
    ordered = sorted(levels, key=lambda item: item[0])
    groups: List[List[Tuple[float, float]]] = []
    current: List[Tuple[float, float]] = []
    anchor = 0.0
    for price, qty in ordered:
        if not current or abs(price - anchor) / anchor > pct:
            if current:
                groups.append(current)
            current = [(price, qty)]
            anchor = price
        else:
            current.append((price, qty))
    if current:
        groups.append(current)
    clusters = []
    for group in groups:
        qty = sum(q for _, q in group)
        if qty <= 0:
            continue
        notional = sum(p * q for p, q in group)
        clusters.append({"price": notional / qty, "notional": notional, "qty": qty})
    return clusters


def describe_book(
    bids: Sequence[Tuple[float, float]],
    asks: Sequence[Tuple[float, float]],
    mid: float,
    source: str,
) -> Dict[str, Any]:
    min_dist = mid * 0.0025
    bid_levels = [(p, q) for p, q in bids if mid * 0.97 <= p <= mid - min_dist]
    ask_levels = [(p, q) for p, q in asks if mid + min_dist <= p <= mid * 1.03]
    bid_clusters = cluster_book(bid_levels, pct=0.0015)
    ask_clusters = cluster_book(ask_levels, pct=0.0015)

    def pick(clusters: List[Dict[str, float]], above: bool) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        if not clusters:
            return None, None
        biggest = max(clusters, key=lambda item: item["notional"])
        if len(clusters) >= 5:
            median = float(np.median([item["notional"] for item in clusters]))
            floor = max(WALL_USD, median * 4)
        else:
            floor = WALL_USD
        walls = [item for item in clusters if item["notional"] >= floor]
        if not walls:
            return None, biggest
        nearest = min(walls, key=lambda item: item["price"] if above else -item["price"])
        nearest = dict(nearest)
        nearest["floor"] = floor
        return nearest, biggest

    ask_wall, ask_big = pick(ask_clusters, above=True)
    bid_wall, bid_big = pick(bid_clusters, above=False)
    return {
        "source": source,
        "ask_wall": ask_wall,
        "bid_wall": bid_wall,
        "ask_big": ask_big,
        "bid_big": bid_big,
    }


def _binance_levels(raw: Dict[str, Any]) -> Optional[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
    if not isinstance(raw, dict) or "bids" not in raw or "asks" not in raw:
        return None
    try:
        bids = [(float(p), float(q)) for p, q in raw["bids"]]
        asks = [(float(p), float(q)) for p, q in raw["asks"]]
    except (TypeError, ValueError):
        return None
    return bids, asks


async def load_book(session: aiohttp.ClientSession, mid: float) -> Dict[str, Any]:
    raw = await fetch_json(session, SPOT_DEPTH_URLS[:1], {"symbol": SYMBOL, "limit": 1000})
    parsed = _binance_levels(raw) if isinstance(raw, dict) else None
    source = "Binance perp"
    if not parsed:
        raw = await fetch_json(session, SPOT_DEPTH_URLS[1:], {"symbol": SYMBOL, "limit": 5000})
        parsed = _binance_levels(raw) if isinstance(raw, dict) else None
        source = "Binance spot"
    if parsed:
        return describe_book(parsed[0], parsed[1], mid, source)

    okx = await fetch_json(
        session,
        ["https://www.okx.com/api/v5/market/books"],
        {"instId": "BTC-USDT-SWAP", "sz": "400"},
    )
    try:
        book = okx["data"][0]
        ct = 0.01  # BTC-USDT-SWAP contract size
        bids = [(float(level[0]), float(level[1]) * ct) for level in book["bids"]]
        asks = [(float(level[0]), float(level[1]) * ct) for level in book["asks"]]
        return describe_book(bids, asks, mid, "OKX perp")
    except (TypeError, ValueError, KeyError, IndexError):
        return {"source": "", "ask_wall": None, "bid_wall": None, "ask_big": None, "bid_big": None}


def _oi_change(points: Sequence[Tuple[int, float]]) -> Optional[float]:
    if len(points) < 2:
        return None
    ordered = sorted(points, key=lambda item: item[0])
    latest_ts, latest = ordered[-1]
    target = latest_ts - 6 * 60 * 60 * 1000
    prior = min(ordered[:-1], key=lambda item: abs(item[0] - target))
    if prior[1] <= 0:
        return None
    return (latest - prior[1]) / prior[1] * 100


async def load_flow(session: aiohttp.ClientSession) -> Dict[str, Any]:
    empty = {"ok": False, "source": "", "funding": None, "oi_change_pct": None, "next_funding": None}
    premium = await fetch_json(session, FAPI_PREMIUM, {"symbol": SYMBOL})
    if isinstance(premium, dict) and "lastFundingRate" in premium:
        hist = await fetch_json(
            session,
            FAPI_OI_HIST,
            {"symbol": SYMBOL, "period": "1h", "limit": 10},
        )
        points: List[Tuple[int, float]] = []
        if isinstance(hist, list):
            for row in hist:
                try:
                    points.append((int(row["timestamp"]), float(row["sumOpenInterestValue"])))
                except (KeyError, TypeError, ValueError):
                    continue
        nxt = premium.get("nextFundingTime")
        return {
            "ok": True,
            "source": "Binance",
            "funding": float(premium["lastFundingRate"]),
            "oi_change_pct": _oi_change(points),
            "next_funding": int(nxt) if nxt else None,
        }

    funding = await fetch_json(
        session,
        ["https://www.okx.com/api/v5/public/funding-rate"],
        {"instId": "BTC-USDT-SWAP"},
    )
    hist = await fetch_json(
        session,
        ["https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-history"],
        {"instId": "BTC-USDT-SWAP", "period": "1H"},
    )
    try:
        row = funding["data"][0]
        points = [(int(item[0]), float(item[3])) for item in hist["data"]]
        return {
            "ok": True,
            "source": "OKX",
            "funding": float(row["fundingRate"]),
            "oi_change_pct": _oi_change(points),
            "next_funding": int(row.get("fundingTime") or 0) or None,
        }
    except (TypeError, ValueError, KeyError, IndexError):
        return empty


# -------------------- Analysis --------------------

def analyze_tf(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    close = df["close"]
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else last
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema_len = 200 if len(close) >= 220 else max(20, len(close) // 2)
    ema200 = ema(close, ema_len)
    e20, e50, e200 = float(ema20.iloc[-1]), float(ema50.iloc[-1]), float(ema200.iloc[-1])

    if e20 > e50 > e200 and last > e20:
        bias, bias_score = "STRONG BULLISH", 2
    elif e20 > e50 and last > e50:
        bias, bias_score = "BULLISH", 1
    elif e20 < e50 < e200 and last < e20:
        bias, bias_score = "STRONG BEARISH", -2
    elif e20 < e50 and last < e50:
        bias, bias_score = "BEARISH", -1
    else:
        bias, bias_score = "NEUTRAL / RANGE", 0

    rsi_val = _finite(rsi(close).iloc[-1], 50.0)
    if rsi_val >= 70:
        rsi_status = "Overbought"
    elif rsi_val <= 30:
        rsi_status = "Oversold"
    elif rsi_val >= 55:
        rsi_status = "Bullish zone"
    elif rsi_val <= 45:
        rsi_status = "Bearish zone"
    else:
        rsi_status = "Neutral"

    _, _, hist = macd(close)
    hist_now = _finite(hist.iloc[-1])
    hist_prev = _finite(hist.iloc[-2]) if len(hist) > 1 else hist_now
    if hist_prev <= 0 < hist_now:
        macd_cross = "Bullish cross"
    elif hist_prev >= 0 > hist_now:
        macd_cross = "Bearish cross"
    else:
        macd_cross = "No cross"

    atr_val = _finite(atr(df).iloc[-1], last * 0.01)
    order = 4 if label in ("15m", "H1") else 5
    res_levels, sup_levels = find_swing_levels(df, order=order)
    resistances = sorted(level for level in res_levels if level > last)[:4]
    supports = sorted((level for level in sup_levels if level < last), reverse=True)[:4]

    mid, upper, lower, width, percent_b = bollinger(close)
    width_now = _finite(width.iloc[-2] if len(width) > 2 else width.iloc[-1])
    look = width.dropna().iloc[-120:]
    squeeze = bool(len(look) >= 40 and width_now <= float(look.quantile(0.20)))
    vol_ratio, taker_ratio = volume_stats(df)
    sweep = detect_sweep(df, label, order=order)

    return {
        "label": label,
        "last": last,
        "change_pct": ((last - prev) / prev * 100) if prev else 0.0,
        "bias": bias,
        "bias_score": bias_score,
        "ema20": e20,
        "ema50": e50,
        "ema200": e200,
        "rsi": rsi_val,
        "rsi_status": rsi_status,
        "macd_hist": hist_now,
        "macd_cross": macd_cross,
        "atr": atr_val,
        "resistances": resistances,
        "supports": supports,
        "nearest_res": resistances[0] if resistances else None,
        "nearest_sup": supports[0] if supports else None,
        "bb_upper": _finite(upper.iloc[-1], last),
        "bb_mid": _finite(mid.iloc[-1], last),
        "bb_lower": _finite(lower.iloc[-1], last),
        "bb_percent_b": _finite(percent_b.iloc[-1], 0.5),
        "bb_squeeze": squeeze,
        "vol_ratio": vol_ratio,
        "taker_ratio": taker_ratio,
        "sweep": sweep,
        "stalling": is_stalling(df, atr_val),
    }


def weighted_score(analyses: Dict[str, Dict[str, Any]]) -> Tuple[int, int]:
    score = 0
    maximum = 0
    for label, weight in TF_WEIGHT.items():
        if label not in analyses or weight == 0:
            continue
        score += int(analyses[label]["bias_score"]) * weight
        maximum += 2 * weight
    return score, maximum


def htf_direction(analyses: Dict[str, Dict[str, Any]]) -> str:
    """H4 wajib searah, Daily tidak boleh melawan. H1 dipakai sebagai konfirmasi tambahan di build_decision."""
    daily = int(analyses.get("Daily", {}).get("bias_score", 0))
    h4 = int(analyses.get("H4", {}).get("bias_score", 0))
    h1 = int(analyses.get("H1", {}).get("bias_score", 0))
    # H4 harus jelas, Daily tidak boleh berlawanan
    if h4 >= 1 and daily >= 0:
        # Bonus kalau H1 juga mendukung
        if h1 >= 0:
            return "BULLISH"
        # H1 sedikit melawan masih boleh, tapi skor akan lebih rendah
        return "BULLISH"
    if h4 <= -1 and daily <= 0:
        if h1 <= 0:
            return "BEARISH"
        return "BEARISH"
    return "NEUTRAL"


def _fingerprint(side: str, entry_mid: float, sl: float) -> str:
    def bucket(price: float) -> int:
        step = max(price * 0.0015, 25.0)
        return int(round(price / step))

    return f"{side}:{bucket(entry_mid)}:{bucket(sl)}"


def zone_distance(price: float, idea: Dict[str, Any]) -> float:
    if idea["entry_low"] <= price <= idea["entry_high"]:
        return 0.0
    if price > idea["entry_high"]:
        return price - idea["entry_high"]
    return idea["entry_low"] - price


def _place_targets(
    side: str,
    mid: float,
    risk: float,
    min_rr: float,
    levels: Sequence[float],
    wall: Optional[Dict[str, float]],
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    min_tp_dist = min_rr * risk
    note = None
    if side == "LONG":
        if wall and mid < wall["price"] <= mid + min_tp_dist:
            return None, None, "ask wall menutup R:R minimum"
        structural = [level for level in levels if level >= mid + min_tp_dist]
        if wall:
            structural = [level for level in structural if level < wall["price"]]
        if structural:
            tp1 = structural[0]
            pulled = tp1 * (1 - 0.0004)
            if pulled >= mid + min_tp_dist:
                tp1 = pulled
            further = [level for level in structural[1:] if level > tp1 * 1.001]
            tp2 = further[0] if further else mid + 2.6 * risk
        else:
            tp1 = mid + max(min_rr, 1.6) * risk
            tp2 = mid + 2.6 * risk
            if not levels:
                note = "TP terukur dari R, belum ada resistance di atas zona"
        if wall and tp2 > wall["price"]:
            tp2 = wall["price"] * (1 - 0.0005)
            note = "TP2 dipotong di depan ask wall"
        if tp2 <= tp1:
            tp2 = max(tp1, mid + 2.2 * risk)
            if wall:
                tp2 = min(tp2, wall["price"] * (1 - 0.0004))
        return tp1, tp2, note

    if wall and mid - min_tp_dist <= wall["price"] < mid:
        return None, None, "bid wall menutup R:R minimum"
    structural = sorted((level for level in levels if level <= mid - min_tp_dist), reverse=True)
    if wall:
        structural = [level for level in structural if level > wall["price"]]
    if structural:
        tp1 = structural[0]
        pulled = tp1 * (1 + 0.0004)
        if pulled <= mid - min_tp_dist:
            tp1 = pulled
        further = [level for level in structural[1:] if level < tp1 * 0.999]
        tp2 = further[0] if further else mid - 2.6 * risk
    else:
        tp1 = mid - max(min_rr, 1.6) * risk
        tp2 = mid - 2.6 * risk
        if not levels:
            note = "TP terukur dari R, belum ada support di bawah zona"
    if wall and tp2 < wall["price"]:
        tp2 = wall["price"] * (1 + 0.0005)
        note = "TP2 dipotong di depan bid wall"
    if tp2 >= tp1:
        tp2 = min(tp1, mid - 2.2 * risk)
        if wall:
            tp2 = max(tp2, wall["price"] * (1 + 0.0004))
    return tp1, tp2, note


def make_idea(
    *,
    side: str,
    style: str,
    entry_low: float,
    entry_high: float,
    sl: float,
    atr_val: float,
    price: float,
    score: int,
    resistances: Sequence[float],
    supports: Sequence[float],
    book: Dict[str, Any],
    note: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    entry_low, entry_high = min(entry_low, entry_high), max(entry_low, entry_high)
    mid = (entry_low + entry_high) / 2
    if side == "LONG" and not sl < entry_low:
        return None, "SL tidak di bawah zona entry"
    if side == "SHORT" and not sl > entry_high:
        return None, "SL tidak di atas zona entry"
    risk = abs(mid - sl)
    if risk <= 0:
        return None, "jarak SL nol"
    sl_pct = risk / mid * 100
    if sl_pct > MAX_SL_PCT:
        return None, f"SL terlalu lebar ({sl_pct:.1f}%)"
    kind = "swing" if sl_pct > MAX_SL_INTRADAY_PCT else "intraday"
    min_rr = MIN_RR_SWING if kind == "swing" else MIN_RR_INTRADAY
    wall = book.get("ask_wall") if side == "LONG" else book.get("bid_wall")
    levels = resistances if side == "LONG" else supports
    if side == "LONG":
        room_level = next((level for level in sorted(resistances) if level > entry_high), None)
        room = (room_level - mid) / atr_val if room_level and atr_val else 99
    else:
        room_level = next((level for level in sorted(supports, reverse=True) if level < entry_low), None)
        room = (mid - room_level) / atr_val if room_level and atr_val else 99
    if room < ROOM_ATR:
        return None, f"ruang ke level lawan cuma {room:.1f} ATR"

    tp1, tp2, tp_note = _place_targets(side, mid, risk, min_rr, levels, wall)
    if tp1 is None or tp2 is None:
        return None, tp_note or "target tidak valid"
    rr = abs(tp1 - mid) / risk
    if rr + 1e-9 < min_rr:
        return None, f"R:R 1:{rr:.1f} di bawah minimum 1:{min_rr:.1f}"
    if tp_note:
        note = f"{note} {tp_note}".strip()
    idea = {
        "side": side,
        "style": style,
        "kind": kind,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "entry_mid": mid,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr": round(rr, 2),
        "sl_pct": sl_pct,
        "atr": atr_val,
        "price": price,
        "score": score,
        "note": note.strip(),
        "fingerprint": _fingerprint(side, mid, sl),
        "wall_note": tp_note or "",
    }
    return idea, None


def zone_tagged(df: pd.DataFrame, idea: Dict[str, Any]) -> bool:
    candles = df.tail(4)
    low, high = idea["entry_low"], idea["entry_high"]
    for row in candles.itertuples(index=False):
        if float(row.low) <= high and float(row.high) >= low:
            return True
    return False


def confirmation(dfs: Dict[str, pd.DataFrame], side: str, sweep: Optional[Dict[str, Any]]) -> Tuple[bool, str, bool]:
    """Prioritas H1 → H4. Return (confirmed, note, taker_is_strict)."""
    # Sweep tetap dihormati (paling kuat)
    if sweep and sweep.get("side") == side:
        return True, f"sweep {sweep['tf']} di {sweep['level']:,.0f}", False

    # 1. Prioritas utama: H1
    df_h1 = dfs.get("H1")
    if df_h1 is not None:
        if side == "LONG":
            if bullish_engulf(df_h1):
                return True, "bullish engulf H1", True
            if bullish_pin(df_h1):
                return True, "pin bullish H1", False
        else:
            if bearish_engulf(df_h1):
                return True, "bearish engulf H1", True
            if bearish_pin(df_h1):
                return True, "pin bearish H1", False

    # 2. Fallback H4 (lebih jarang dipakai)
    df_h4 = dfs.get("H4")
    if df_h4 is not None:
        if side == "LONG":
            if bullish_engulf(df_h4):
                return True, "bullish engulf H4", True
            if bullish_pin(df_h4):
                return True, "pin bullish H4", False
        else:
            if bearish_engulf(df_h4):
                return True, "bearish engulf H4", True
            if bearish_pin(df_h4):
                return True, "pin bearish H4", False

    return False, "", True


def _funding_blocks(side: str, flow: Dict[str, Any], stalling: bool) -> List[str]:
    if not flow.get("ok") or flow.get("funding") is None:
        return []
    funding = float(flow["funding"])
    oi = flow.get("oi_change_pct")
    oi_up = oi is not None and oi >= OI_UP_PCT
    oi_down = oi is not None and oi <= -OI_UP_PCT
    blocks = []
    if side == "LONG":
        crowded = funding >= FUNDING_CROWDED
        extreme = funding >= FUNDING_EXTREME
        if extreme or (crowded and (oi_up or oi is None) and (stalling or oi is None)):
            blocks.append(
                f"funding long crowded {funding * 100:.4f}%"
                + (f", OI {oi:+.1f}%" if oi is not None else ", OI tidak tersedia")
                + (" , harga H4 mandek" if stalling else "")
            )
    else:
        crowded = funding <= -FUNDING_WARM
        extreme = funding <= -FUNDING_EXTREME
        if extreme or (crowded and (oi_down or oi is None) and (stalling or oi is None)):
            blocks.append(
                f"funding short crowded {funding * 100:.4f}%"
                + (f", OI {oi:+.1f}%" if oi is not None else "")
                + (" , harga H4 mandek" if stalling else "")
            )
    return blocks


def _aggressive_blocks(
    side: str,
    analyses: Dict[str, Dict[str, Any]],
    dfs: Dict[str, pd.DataFrame],
    flow: Dict[str, Any],
) -> List[str]:
    h4 = analyses["H4"]
    h1 = dfs.get("H1")
    blocks: List[str] = []
    if side == "LONG":
        if h4["rsi"] >= 75:
            blocks.append(f"RSI H4 {h4['rsi']:.0f} terlalu tinggi untuk kejar")
        if h4["bb_percent_b"] > 1:
            blocks.append("harga di atas Bollinger atas")
        if h4["macd_hist"] < 0:
            blocks.append("MACD H4 masih negatif")
        if h1 is not None and float(_closed(h1).iloc[-1].close) <= float(_closed(h1).iloc[-1].open):
            blocks.append("candle H1 terakhir belum hijau")
    else:
        if h4["rsi"] <= 25:
            blocks.append(f"RSI H4 {h4['rsi']:.0f} terlalu rendah untuk kejar")
        if h4["bb_percent_b"] < 0:
            blocks.append("harga di bawah Bollinger bawah")
        if h4["macd_hist"] > 0:
            blocks.append("MACD H4 masih positif")
        if h1 is not None and float(_closed(h1).iloc[-1].close) >= float(_closed(h1).iloc[-1].open):
            blocks.append("candle H1 terakhir belum merah")
    weekly = int(analyses.get("Weekly", {}).get("bias_score", 0))
    if side == "LONG" and weekly <= -2:
        blocks.append("Weekly strong bearish")
    if side == "SHORT" and weekly >= 2:
        blocks.append("Weekly strong bullish")
    if h4["bb_squeeze"] and abs(int(h4["bias_score"])) < 2:
        blocks.append("Bollinger squeeze, jangan kejar")
    if h1 is not None and not volume_confirms(h1, side, strict_taker=True):
        ratio, taker = volume_stats(h1)
        blocks.append(f"volume/taker H1 belum mendukung ({ratio:.1f}x, taker {taker * 100:.0f}%)")
    blocks.extend(_funding_blocks(side, flow, bool(h4.get("stalling"))))
    return blocks


def build_decision(bundle: Dict[str, Any]) -> Dict[str, Any]:
    analyses: Dict[str, Dict[str, Any]] = bundle["analyses"]
    dfs: Dict[str, pd.DataFrame] = bundle["dfs"]
    price = float(bundle["price"])
    book = bundle["book"]
    flow = bundle["flow"]
    score, score_max = weighted_score(analyses)
    direction = htf_direction(analyses)
    h4 = analyses["H4"]
    daily = analyses.get("Daily", {})
    atr_val = float(h4["atr"] or price * 0.01)
    resistances = sorted(set((h4.get("resistances") or []) + (daily.get("resistances") or [])))
    supports = sorted(set((h4.get("supports") or []) + (daily.get("supports") or [])), reverse=True)
    sweep = None
    for label in ("H4", "H1"):
        found = analyses.get(label, {}).get("sweep")
        if found and found.get("side") == direction:
            sweep = found
            break
        if found and direction == "NEUTRAL":
            sweep = found

    passes: List[str] = []
    blocks: List[str] = []
    if direction == "BULLISH":
        passes.append("HTF: H4 bullish dan Daily tidak bearish")
    elif direction == "BEARISH":
        passes.append("HTF: H4 bearish dan Daily tidak bullish")
    else:
        blocks.append("HTF tidak sepakat (butuh H4 searah dan Daily tidak melawan)")

    weekly = int(analyses.get("Weekly", {}).get("bias_score", 0))
    if direction == "BULLISH" and weekly < 0:
        passes.append("Weekly masih melawan — aggressive dimatikan kalau weekly strong")
    elif direction == "BEARISH" and weekly > 0:
        passes.append("Weekly masih melawan — aggressive dimatikan kalau weekly strong")
    elif direction != "NEUTRAL" and weekly == 0:
        passes.append("Weekly netral, tidak memblokir")
    elif direction != "NEUTRAL":
        passes.append("Weekly searah")

    cons_idea = None
    cons_reason = "menunggu HTF sepakat"
    agg_idea = None
    agg_reason = "menunggu HTF sepakat"
    h1 = analyses.get("H1", {})

    if direction in ("BULLISH", "BEARISH"):
        side = "LONG" if direction == "BULLISH" else "SHORT"
        atr_h1 = float(h1.get("atr") or atr_val * 0.55)

        # ========== ZONA ENTRY: prioritaskan H1 + H4 confluence ==========
        entry_low = entry_high = sl = 0.0
        anchor_note = ""

        if side == "LONG":
            # 1. Sweep (paling kuat)
            if sweep and sweep["side"] == "LONG":
                entry_low = float(sweep["level"])
                entry_high = entry_low + atr_h1 * 0.35
                sl = float(sweep["wick"]) - atr_h1 * 0.25
                anchor_note = f"reclaim sweep {sweep['tf']}"
            else:
                # 2. H1 nearest support + H4 support (confluence)
                h1_sup = h1.get("nearest_sup")
                h4_sup = h4.get("nearest_sup")
                candidates = []
                if h1_sup:
                    candidates.append(("H1", float(h1_sup)))
                if h4_sup:
                    candidates.append(("H4", float(h4_sup)))
                # Pilih yang paling dekat di bawah harga, tapi tidak terlalu jauh
                valid = [(tf, lv) for tf, lv in candidates if lv < price and (price - lv) / atr_val < 2.8]
                if valid:
                    # Prefer level yang lebih dekat + ada confluence
                    valid.sort(key=lambda x: abs(price - x[1]))
                    best_tf, best_lv = valid[0]
                    # Cek confluence: kalau H1 & H4 dekat, pakai zona lebih lebar sedikit
                    if len(valid) >= 2 and abs(valid[0][1] - valid[1][1]) / atr_val < 0.45:
                        lo = min(valid[0][1], valid[1][1])
                        hi = max(valid[0][1], valid[1][1])
                        entry_low = lo
                        entry_high = hi + atr_h1 * 0.20
                        sl = lo - atr_h1 * 0.75
                        anchor_note = f"zona confluence H1+H4 support"
                    else:
                        entry_low = best_lv
                        entry_high = best_lv + atr_h1 * 0.40
                        sl = best_lv - atr_h1 * 0.85
                        anchor_note = f"zona support {best_tf}"
                else:
                    cons_reason = "belum ada support H1/H4 yang layak"

            if entry_low > 0:
                cons_idea, cons_reason = make_idea(
                    side=side, style="conservative", entry_low=entry_low, entry_high=entry_high,
                    sl=sl, atr_val=atr_val, price=price, score=score,
                    resistances=resistances, supports=supports, book=book, note=anchor_note,
                )

        else:  # SHORT
            if sweep and sweep["side"] == "SHORT":
                entry_high = float(sweep["level"])
                entry_low = entry_high - atr_h1 * 0.35
                sl = float(sweep["wick"]) + atr_h1 * 0.25
                anchor_note = f"reclaim sweep {sweep['tf']}"
            else:
                h1_res = h1.get("nearest_res")
                h4_res = h4.get("nearest_res")
                candidates = []
                if h1_res:
                    candidates.append(("H1", float(h1_res)))
                if h4_res:
                    candidates.append(("H4", float(h4_res)))
                valid = [(tf, lv) for tf, lv in candidates if lv > price and (lv - price) / atr_val < 2.8]
                if valid:
                    valid.sort(key=lambda x: abs(x[1] - price))
                    best_tf, best_lv = valid[0]
                    if len(valid) >= 2 and abs(valid[0][1] - valid[1][1]) / atr_val < 0.45:
                        lo = min(valid[0][1], valid[1][1])
                        hi = max(valid[0][1], valid[1][1])
                        entry_high = hi
                        entry_low = lo - atr_h1 * 0.20
                        sl = hi + atr_h1 * 0.75
                        anchor_note = f"zona confluence H1+H4 resistance"
                    else:
                        entry_high = best_lv
                        entry_low = best_lv - atr_h1 * 0.40
                        sl = best_lv + atr_h1 * 0.85
                        anchor_note = f"zona resistance {best_tf}"
                else:
                    cons_reason = "belum ada resistance H1/H4 yang layak"

            if entry_high > 0:
                cons_idea, cons_reason = make_idea(
                    side=side, style="conservative", entry_low=entry_low, entry_high=entry_high,
                    sl=sl, atr_val=atr_val, price=price, score=score,
                    resistances=resistances, supports=supports, book=book, note=anchor_note,
                )

        # Aggressive hanya kalau momentum sangat kuat + tidak ada zona pullback dekat
        agg_blocks = _aggressive_blocks(side, analyses, dfs, flow)
        h1_bias = int(h1.get("bias_score", 0))
        strong_momentum = (side == "LONG" and h1_bias >= 1) or (side == "SHORT" and h1_bias <= -1)
        if strong_momentum and not agg_blocks:
            if side == "LONG":
                agg_sl = price - atr_val * 1.05
                agg_idea, agg_reason = make_idea(
                    side=side, style="aggressive", entry_low=price, entry_high=price, sl=agg_sl,
                    atr_val=atr_val, price=price, score=score, resistances=resistances,
                    supports=supports, book=book, note="entry market H1 momentum kuat",
                )
            else:
                agg_sl = price + atr_val * 1.05
                agg_idea, agg_reason = make_idea(
                    side=side, style="aggressive", entry_low=price, entry_high=price, sl=agg_sl,
                    atr_val=atr_val, price=price, score=score, resistances=resistances,
                    supports=supports, book=book, note="entry market H1 momentum kuat",
                )
        else:
            if agg_blocks:
                agg_reason = "; ".join(agg_blocks)
            else:
                agg_reason = "momentum H1 belum cukup kuat untuk aggressive"

    # ========== PHASE LOGIC (lebih ketat) ==========
    if cons_idea:
        dist = zone_distance(price, cons_idea)
        dist_atr = dist / atr_val if atr_val else 99
        cons_idea["distance_atr"] = dist_atr

        # Hanya anggap "tagged" kalau H1 benar-benar menyentuh zona
        tagged = zone_tagged(dfs.get("H1"), cons_idea) if "H1" in dfs else False
        if not tagged and "H4" in dfs:
            tagged = zone_tagged(dfs["H4"], cons_idea)

        ok, how, strict = confirmation(
            dfs, cons_idea["side"],
            sweep if sweep and sweep["side"] == cons_idea["side"] else None
        )

        # Volume wajib dari H1 (atau H4 kalau confirmation dari H4)
        vol_ok = False
        if ok:
            prefer_tf = "H1" if "H1" in how else ("H4" if "H4" in how else "H1")
            if prefer_tf in dfs:
                vol_ok = volume_confirms(dfs[prefer_tf], cons_idea["side"], strict_taker=strict)
            # Fallback longgar sedikit kalau volume H1 borderline
            if not vol_ok and "H1" in dfs:
                ratio, taker = volume_stats(dfs["H1"])
                if ratio >= 0.85:
                    if cons_idea["side"] == "LONG" and taker >= 0.48:
                        vol_ok = True
                    elif cons_idea["side"] == "SHORT" and taker <= 0.52:
                        vol_ok = True

        # Phase decision
        if tagged and ok and vol_ok and dist_atr <= TRIGGER_MAX_ATR:
            cons_idea["phase"] = "TRIGGERED"
            cons_idea["note"] = f"{cons_idea['note']}; konfirmasi {how}"
        elif tagged and ok:
            cons_idea["phase"] = "FORMING"
            cons_idea["note"] = f"{cons_idea['note']}; {how}, volume/jarak belum ideal"
        elif tagged:
            cons_idea["phase"] = "FORMING"
            cons_idea["note"] = f"{cons_idea['note']}; zona tersentuh, tunggu candle H1 konfirmasi"
        else:
            cons_idea["phase"] = "FORMING"
            cons_idea["note"] = f"{cons_idea['note']}; menunggu harga masuk zona"

        # Alertable jauh lebih ketat
        cons_idea["alertable"] = (
            cons_idea["phase"] == "TRIGGERED"
            or (cons_idea["phase"] == "FORMING" and dist_atr <= FORMING_MAX_ATR and tagged)
        )
        if not cons_idea["alertable"]:
            passes.append(f"conservative siap tapi zona masih {dist_atr:.1f} ATR / belum ideal — tidak di-push")

    if agg_idea:
        agg_idea["phase"] = "TRIGGERED"
        agg_idea["distance_atr"] = 0.0
        agg_idea["alertable"] = True

    selected = None
    if cons_idea and cons_idea.get("alertable"):
        selected = cons_idea
        passes.append(f"alert memakai conservative {cons_idea['phase'].lower()}")
    elif agg_idea and agg_idea.get("alertable"):
        # Aggressive hanya kalau zona conservative jauh (> 2.5 ATR) atau tidak ada
        far = not cons_idea or cons_idea.get("distance_atr", 99) > 2.5
        if cons_idea and not far:
            blocks.append("aggressive lolos, tapi zona conservative masih cukup dekat — tidak dikejar")
        else:
            selected = agg_idea
            passes.append("alert memakai aggressive (momentum H1 kuat, pullback terlalu jauh)")
    elif cons_idea or agg_idea:
        blocks.append("ada ide, belum cukup dekat atau belum lolos filter alert")
    else:
        if cons_reason and direction != "NEUTRAL":
            blocks.append(f"conservative: {cons_reason}")
        if agg_reason and direction != "NEUTRAL":
            blocks.append(f"aggressive: {agg_reason}")

    return {
        "price": price,
        "score": score,
        "score_max": score_max,
        "direction": direction,
        "passes": passes,
        "blocks": blocks,
        "conservative": cons_idea,
        "conservative_reason": cons_reason,
        "aggressive": agg_idea,
        "aggressive_reason": agg_reason,
        "selected": selected,
        "sweep": sweep,
        "atr": atr_val,
    }


# -------------------- Trade path --------------------

def walk_trade(
    candles: Sequence[Tuple[float, float]],
    side: str,
    entry_mid: float,
    sl: float,
    tp1: float,
    tp2: float,
) -> Dict[str, Any]:
    """Replay highs/lows after trigger. Same-candle SL and target counts as SL."""
    risk = abs(entry_mid - sl)
    if risk <= 0:
        return {"state": "INVALID", "result": "INVALID", "r": None}
    sl_live = sl
    hit_tp1 = False
    for high, low in candles:
        if side == "LONG":
            target = tp2 if hit_tp1 else tp1
            hit_sl = low <= sl_live
            hit_tp = high >= target
            if hit_sl and hit_tp:
                if hit_tp1:
                    return {"state": "TP1", "result": "TP1", "r": abs(tp1 - entry_mid) / risk, "detail": "TP1 lalu runner ke BE"}
                return {"state": "SL", "result": "SL", "r": -1.0}
            if hit_sl:
                if hit_tp1:
                    return {"state": "TP1", "result": "TP1", "r": abs(tp1 - entry_mid) / risk, "detail": "TP1 lalu runner ke BE"}
                return {"state": "SL", "result": "SL", "r": -1.0}
            if not hit_tp1 and high >= tp1:
                hit_tp1 = True
                sl_live = entry_mid
                if high >= tp2:
                    return {"state": "TP2", "result": "TP2", "r": abs(tp2 - entry_mid) / risk}
            elif hit_tp1 and high >= tp2:
                return {"state": "TP2", "result": "TP2", "r": abs(tp2 - entry_mid) / risk}
        else:
            target = tp2 if hit_tp1 else tp1
            hit_sl = high >= sl_live
            hit_tp = low <= target
            if hit_sl and hit_tp:
                if hit_tp1:
                    return {"state": "TP1", "result": "TP1", "r": abs(entry_mid - tp1) / risk, "detail": "TP1 lalu runner ke BE"}
                return {"state": "SL", "result": "SL", "r": -1.0}
            if hit_sl:
                if hit_tp1:
                    return {"state": "TP1", "result": "TP1", "r": abs(entry_mid - tp1) / risk, "detail": "TP1 lalu runner ke BE"}
                return {"state": "SL", "result": "SL", "r": -1.0}
            if not hit_tp1 and low <= tp1:
                hit_tp1 = True
                sl_live = entry_mid
                if low <= tp2:
                    return {"state": "TP2", "result": "TP2", "r": abs(entry_mid - tp2) / risk}
            elif hit_tp1 and low <= tp2:
                return {"state": "TP2", "result": "TP2", "r": abs(entry_mid - tp2) / risk}
    if hit_tp1:
        r = abs(tp1 - entry_mid) / risk
        return {"state": "TP1", "result": None, "r": r}
    return {"state": "TRIGGERED", "result": None, "r": None}


def _candles_since(df: pd.DataFrame, since: datetime) -> List[Tuple[float, float]]:
    if "open_time" not in df.columns:
        rows = _closed(df)
    else:
        rows = df[df["open_time"] >= pd.Timestamp(since)]
        if len(rows) and rows.iloc[-1].open_time == df.iloc[-1].open_time:
            rows = rows.iloc[:-1]
    return [(float(row.high), float(row.low)) for row in rows.itertuples(index=False)]


# -------------------- Formatting / chart --------------------

def fmt_price(price: float) -> str:
    return f"{price:,.0f}"


def fmt_usd(notional: float) -> str:
    if notional >= 1_000_000:
        return f"${notional / 1_000_000:.1f}jt"
    if notional >= 1_000:
        return f"${notional / 1_000:.0f}rb"
    return f"${notional:,.0f}"


def bias_emoji(bias: str) -> str:
    if "STRONG BULLISH" in bias:
        return "🟢🟢"
    if "BULLISH" in bias:
        return "🟢"
    if "STRONG BEARISH" in bias:
        return "🔴🔴"
    if "BEARISH" in bias:
        return "🔴"
    return "⚪"


def _idea_lines(title: str, idea: Optional[Dict[str, Any]], reason: Optional[str]) -> List[str]:
    if not idea:
        return [f"{title}: {reason or 'tidak ada'}"]
    zone = fmt_price(idea["entry_mid"]) if idea["style"] == "aggressive" else f"{fmt_price(idea['entry_low'])} – {fmt_price(idea['entry_high'])}"
    if idea["style"] == "aggressive":
        zone = f"Market / {fmt_price(idea['price'])}"
    lines = [
        f"{title} {idea['side']} · {idea['kind']}",
        f"   Entry : {zone}",
        f"   SL    : {fmt_price(idea['sl'])} ({idea['sl_pct']:.1f}%)",
        f"   TP1   : {fmt_price(idea['tp1'])}  |  TP2: {fmt_price(idea['tp2'])}",
        f"   R:R   : 1 : {idea['rr']:.1f}",
        f"   Note  : {idea['note']}",
    ]
    return lines


def _flow_line(flow: Dict[str, Any]) -> str:
    if not flow.get("ok") or flow.get("funding") is None:
        return "Funding / OI: tidak tersedia"
    oi = flow.get("oi_change_pct")
    oi_txt = f"{oi:+.1f}% / 6j" if oi is not None else "n/a"
    return f"Funding {float(flow['funding']) * 100:.4f}% · OI {oi_txt} · sumber {flow['source']}"


def _wall_line(book: Dict[str, Any], price: float) -> str:
    if not book.get("source"):
        return "Dinding: buku order tidak tersedia"

    def one(label: str, wall: Optional[Dict[str, float]], big: Optional[Dict[str, float]]) -> str:
        item = wall or big
        if not item:
            return f"{label} -"
        tag = "WALL" if wall else "terbesar"
        pct = (item["price"] - price) / price * 100
        return f"{label} {tag} {fmt_usd(item['notional'])} @ {fmt_price(item['price'])} ({pct:+.1f}%)"

    return f"{one('Ask', book.get('ask_wall'), book.get('ask_big'))} | {one('Bid', book.get('bid_wall'), book.get('bid_big'))} · {book['source']}"


def format_report(bundle: Dict[str, Any], decision: Dict[str, Any], active: Optional[sqlite3.Row] = None) -> str:
    analyses = bundle["analyses"]
    price = decision["price"]
    now = _now().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "🟠 <b>BTCUSDT TECHNICAL ANALYSIS</b>",
        f"<i>{now}</i>",
        f"Harga: <b>${fmt_price(price)}</b>",
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        "<b>📊 MULTI-TIMEFRAME BIAS</b>",
    ]
    for label in TF_ORDER:
        if label not in analyses:
            continue
        row = analyses[label]
        weight = TF_WEIGHT[label]
        extra = "  · timing saja" if weight == 0 else f"  · bobot {weight}"
        lines.append(
            f"{bias_emoji(row['bias'])} <b>{label}</b>: {row['bias']}  |  RSI {row['rsi']:.0f} ({row['rsi_status']}){extra}"
        )
    lines.append("")
    lines.append(
        f"🎯 <b>Overall:</b> <b>{decision['direction']}</b> "
        f"(skor {decision['score']:+d}/{decision['score_max']})"
    )
    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("<b>📌 KEY LEVELS (H4 + Daily)</b>")
    h4 = analyses.get("H4", {})
    daily = analyses.get("Daily", {})
    res_set = sorted(set((h4.get("resistances") or []) + (daily.get("resistances") or [])))
    sup_set = sorted(set((h4.get("supports") or []) + (daily.get("supports") or [])), reverse=True)
    if res_set:
        lines.append("Resistance:")
        for level in res_set[:4]:
            lines.append(f"  • ${fmt_price(level)}  ({(level - price) / price * 100:+.1f}%)")
    if sup_set:
        lines.append("Support:")
        for level in sup_set[:4]:
            lines.append(f"  • ${fmt_price(level)}  ({(level - price) / price * 100:+.1f}%)")

    if h4:
        lines += [
            "",
            "━━━━━━━━━━━━━━━━━━━━",
            "<b>📈 H4 SNAPSHOT</b>",
            f"EMA20 / 50 / 200: {fmt_price(h4['ema20'])} / {fmt_price(h4['ema50'])} / {fmt_price(h4['ema200'])}",
            f"RSI: {h4['rsi']:.1f} — {h4['rsi_status']}",
            f"MACD Hist: {h4['macd_hist']:.1f} ({h4['macd_cross']})",
            f"ATR(14): ${fmt_price(h4['atr'])}",
            f"BB: {fmt_price(h4['bb_lower'])} / {fmt_price(h4['bb_mid'])} / {fmt_price(h4['bb_upper'])}  ·  %B {h4['bb_percent_b']:.2f}"
            + ("  ·  SQUEEZE" if h4["bb_squeeze"] else ""),
            f"Volume: {h4['vol_ratio']:.1f}x avg  ·  taker buy {h4['taker_ratio'] * 100:.0f}%",
        ]
        sweep = h4.get("sweep")
        if sweep:
            lines.append(f"Sweep: {sweep['side']} {sweep['tf']} @ {fmt_price(sweep['level'])}")
        else:
            lines.append("Sweep: tidak ada di 3 candle H4 terakhir")

    lines += [
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        "<b>🧪 GERBANG</b>",
        _flow_line(bundle["flow"]),
        _wall_line(bundle["book"], price),
    ]
    for item in decision["passes"]:
        lines.append(f"✓ {item}")
    for item in decision["blocks"]:
        lines.append(f"✗ {item}")

    lines += ["", "━━━━━━━━━━━━━━━━━━━━", "<b>💡 ENTRY IDEAS</b>"]
    lines.extend(_idea_lines("🛡 <b>Conservative</b>", decision["conservative"], decision["conservative_reason"]))
    lines.append("")
    lines.extend(_idea_lines("⚡ <b>Aggressive</b>", decision["aggressive"], decision["aggressive_reason"]))

    if active is not None:
        lines += [
            "",
            f"📍 Setup aktif #{active['id']}: {active['state']} {active['side']} {active['style']}",
            f"   {fmt_price(active['entry_low'])}–{fmt_price(active['entry_high'])} · SL {fmt_price(active['sl'])} · TP1 {fmt_price(active['tp1'])}",
        ]
    elif decision["selected"]:
        lines.append("")
        lines.append(f"📍 Yang akan di-alert: {decision['selected']['style']} {decision['selected']['phase']}")
    else:
        lines.append("")
        lines.append("📍 Tidak ada alert. /ta tetap bisa dipanggil kapan saja.")

    lines += ["", "━━━━━━━━━━━━━━━━━━━━", "<i>⚠️ Bukan saran finansial. Selalu gunakan risk management.</i>"]
    text = "\n".join(lines)
    if len(text) > 4000:
        text = text[:3900] + "\n\n… (terpotong)"
    return text


def format_alert(state: str, plan: Dict[str, Any], extra: str = "") -> str:
    titles = {
        "FORMING": "SETUP MULAI TERBENTUK",
        "TRIGGERED": "ENTRY AKTIF",
        "TP1": "TP1 KENA",
        "TP2": "TP2 KENA",
        "SL": "STOP KENA",
        "INVALID": "SETUP BATAL",
        "EXPIRED": "SETUP KEDALUWARSA",
    }
    zone = (
        f"Market / {fmt_price(plan['price'])}"
        if plan.get("style") == "aggressive"
        else f"{fmt_price(plan['entry_low'])} – {fmt_price(plan['entry_high'])}"
    )
    lines = [
        f"🚨 <b>{titles.get(state, state)}</b> — {plan['side']} {plan.get('style', '')}",
        f"Zona  : {zone}",
        f"SL    : {fmt_price(plan['sl'])}",
        f"TP1   : {fmt_price(plan['tp1'])}  |  TP2 {fmt_price(plan['tp2'])}",
        f"R:R   : 1 : {float(plan['rr']):.1f} · {plan.get('kind', '')}",
    ]
    if plan.get("note"):
        lines.append(plan["note"])
    if state == "TP1":
        lines.append("Geser SL ke breakeven. Runner boleh ditujukan ke TP2.")
    if state == "SL":
        lines.append("Level gagal. Tidak ada re-entry di zona yang sama selama beberapa jam.")
    if extra:
        lines.append(extra)
    lines.append("<i>Bukan saran finansial.</i>")
    return "\n".join(lines)


def format_tf(row: Dict[str, Any]) -> str:
    lines = [
        f"🟠 <b>BTCUSDT — {row['label']}</b>",
        f"Harga: <b>${fmt_price(row['last'])}</b>  ({row['change_pct']:+.2f}%)",
        "",
        f"Bias: {bias_emoji(row['bias'])} <b>{row['bias']}</b>",
        "",
        f"EMA20:  ${fmt_price(row['ema20'])}",
        f"EMA50:  ${fmt_price(row['ema50'])}",
        f"EMA200: ${fmt_price(row['ema200'])}",
        "",
        f"RSI(14): {row['rsi']:.1f} — {row['rsi_status']}",
        f"MACD Hist: {row['macd_hist']:.2f} ({row['macd_cross']})",
        f"ATR(14): ${fmt_price(row['atr'])}",
        f"BB %B: {row['bb_percent_b']:.2f}" + (" · squeeze" if row["bb_squeeze"] else ""),
        f"Volume: {row['vol_ratio']:.1f}x · taker {row['taker_ratio'] * 100:.0f}%",
        "",
        "Resistance terdekat:",
    ]
    for level in row["resistances"][:3] or []:
        lines.append(f"  • ${fmt_price(level)}")
    lines.append("Support terdekat:")
    for level in row["supports"][:3] or []:
        lines.append(f"  • ${fmt_price(level)}")
    if row.get("sweep"):
        sweep = row["sweep"]
        lines.append(f"Sweep {sweep['side']} @ ${fmt_price(sweep['level'])}")
    return "\n".join(lines)


def format_stats() -> str:
    rows = all_closed()
    wins = [r for r in rows if r["result"] in ("TP1", "TP2")]
    losses = [r for r in rows if r["result"] == "SL"]
    scratched = [r for r in rows if r["result"] == "BE"]
    ignored = [r for r in rows if r["result"] in ("INVALID", "EXPIRED")]
    decided = len(wins) + len(losses)
    rate = (len(wins) / decided * 100) if decided else 0.0
    rs = [float(r["r_multiple"]) for r in rows if r["result"] in ("TP1", "TP2", "SL", "BE") and r["r_multiple"] is not None]
    avg_r = sum(rs) / len(rs) if rs else 0.0
    lines = [
        "<b>TRACK RECORD</b>",
        f"Ditutup: {len(rows)} · Menang: {len(wins)} · Kalah: {len(losses)} · Impas: {len(scratched)}",
        f"Batal/kedaluwarsa (tidak masuk win rate): {len(ignored)}",
        f"Win rate: {rate:.1f}% dari {decided} setup yang kena TP atau SL",
        f"Rata-rata R: {avg_r:+.2f}",
        "",
        "<b>10 terakhir</b>",
    ]
    last = recent_results(10)
    if not last:
        lines.append("Belum ada setup tertutup.")
    for row in last:
        rtxt = f"{float(row['r_multiple']):+.1f}R" if row["r_multiple"] is not None else "-"
        when = (row["closed_at"] or "")[:16]
        lines.append(f"{when}  {row['side']} {row['style']}  {row['result']}  {rtxt}")
    lines.append("")
    lines.append("<i>Menang = TP1 atau TP2 lebih dulu dari SL. Candle yang menyentuh keduanya dihitung SL.</i>")
    return "\n".join(lines)


def render_chart(df: pd.DataFrame, analysis: Dict[str, Any], plan: Optional[Dict[str, Any]]) -> bytes:
    view = df.tail(120).reset_index(drop=True)
    n = len(view)
    full_close = df["close"]
    e20 = ema(full_close, 20).tail(n).to_numpy()
    e50 = ema(full_close, 50).tail(n).to_numpy()
    e200 = ema(full_close, 200 if len(full_close) >= 220 else max(20, len(full_close) // 2)).tail(n).to_numpy()

    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(12.6, 7.3), dpi=120, sharex=True,
        gridspec_kw={"height_ratios": [4.2, 1]},
    )
    fig.patch.set_facecolor("#121418")
    for axis in (ax, axv):
        axis.set_facecolor("#121418")
        axis.tick_params(colors="#b7b3ac")
        for spine in axis.spines.values():
            spine.set_color("#2c313a")
    ax.grid(True, color="#2c313a", linewidth=0.6)
    width = 0.62
    for i, row in view.iterrows():
        up = row.close >= row.open
        color = "#3DDC97" if up else "#FF5D73"
        ax.plot([i, i], [row.low, row.high], color=color, linewidth=1.0)
        body_low = min(row.open, row.close)
        body_h = max(abs(row.close - row.open), (row.high - row.low) * 0.004, 1)
        ax.add_patch(Rectangle((i - width / 2, body_low), width, body_h, facecolor=color, edgecolor=color, linewidth=0))
        axv.bar(i, row.volume, width=width, color=color, alpha=0.85)
    ax.plot(range(n), e20, color="#e8c36a", linewidth=1.2, label="EMA20")
    ax.plot(range(n), e50, color="#7aa2f7", linewidth=1.1, label="EMA50")
    ax.plot(range(n), e200, color="#d38bb5", linewidth=1.1, label="EMA200")
    for level in (analysis.get("resistances") or [])[:3]:
        ax.axhline(level, color="#8d93a1", linewidth=0.6, linestyle=":", alpha=0.7)
    for level in (analysis.get("supports") or [])[:3]:
        ax.axhline(level, color="#8d93a1", linewidth=0.6, linestyle=":", alpha=0.7)
    if plan:
        ax.axhspan(plan["entry_low"], plan["entry_high"], color="#e8c36a", alpha=0.16, label="Entry")
        ax.axhline(plan["sl"], color="#FF5D73", linewidth=1.1, linestyle="--", label="SL")
        ax.axhline(plan["tp1"], color="#3DDC97", linewidth=1.1, linestyle="--", label="TP1")
        ax.axhline(plan["tp2"], color="#3DDC97", linewidth=1.0, linestyle=":", label="TP2")
    ax.set_xlim(-1, n)
    ax.set_title(
        f"BTCUSDT H4   ${fmt_price(analysis['last'])}   {analysis['bias']}",
        color="#f2efe9", loc="left", fontsize=13, pad=10,
    )
    legend = ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=8)
    for text in legend.get_texts():
        text.set_color("#d9d6d0")
    axv.set_ylabel("Vol", color="#b7b3ac")
    ax.ticklabel_format(style="plain", axis="y")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    buf.name = "btc_h4.png"
    return buf.getvalue()


# -------------------- Market bundle --------------------

async def collect_market(session: aiohttp.ClientSession) -> Dict[str, Any]:
    tasks = [
        fetch_klines(session, conf["interval"], conf["limit"])
        for conf in TIMEFRAMES.values()
    ]
    frames = await asyncio.gather(*tasks)
    dfs: Dict[str, pd.DataFrame] = {}
    analyses: Dict[str, Dict[str, Any]] = {}
    for key, df in zip(TIMEFRAMES.keys(), frames):
        label = TIMEFRAMES[key]["label"]
        if df is not None and len(df) > 50:
            dfs[label] = df
            analyses[label] = analyze_tf(df, label)
    if "H4" not in analyses or "H1" not in analyses:
        raise RuntimeError("data H4/H1 tidak lengkap")
    price = float(analyses["H1"]["last"])
    book, flow = await asyncio.gather(load_book(session, price), load_flow(session))
    return {"dfs": dfs, "analyses": analyses, "price": price, "book": book, "flow": flow}


def _plan_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "side": row["side"],
        "style": row["style"],
        "kind": row["kind"],
        "entry_low": row["entry_low"],
        "entry_high": row["entry_high"],
        "entry_mid": (row["entry_low"] + row["entry_high"]) / 2,
        "sl": row["sl"],
        "tp1": row["tp1"],
        "tp2": row["tp2"],
        "rr": row["rr"],
        "atr": row["atr"],
        "price": row["price"],
        "score": row["score"],
        "note": row["note"] or "",
        "fingerprint": row["fingerprint"],
    }


def _same_zone(row: sqlite3.Row, plan: Dict[str, Any]) -> bool:
    if row["side"] != plan["side"] or row["style"] != plan["style"]:
        return False
    mid = (row["entry_low"] + row["entry_high"]) / 2
    return abs(mid - plan["entry_mid"]) / max(plan["entry_mid"], 1) <= 0.004 and abs(row["sl"] - plan["sl"]) / max(plan["sl"], 1) <= 0.004


# -------------------- Broadcast / jobs --------------------

async def broadcast(bot, text: str, photo: Optional[bytes] = None) -> None:
    subs = get_subscribers()
    if not subs:
        logger.info("Tidak ada subscriber, alert tidak dikirim")
        return
    for chat_id in subs:
        try:
            if photo:
                payload = io.BytesIO(photo)
                payload.name = "btc_h4.png"
                caption = text if len(text) <= 1000 else text[:980] + "…"
                await bot.send_photo(chat_id=chat_id, photo=payload, caption=caption, parse_mode=ParseMode.HTML)
                if len(text) > 1000:
                    await bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            else:
                await bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            await asyncio.sleep(0.05)
        except Forbidden:
            logger.info("Chat %s memblokir bot, unsubscribe", chat_id)
            remove_subscriber(chat_id)
        except Exception as exc:
            logger.warning("Gagal kirim ke %s: %s", chat_id, exc)


async def scan_setups(context: ContextTypes.DEFAULT_TYPE) -> None:
    async with _db_lock:
        await _scan_locked(context)


async def _push(context: ContextTypes.DEFAULT_TYPE, bundle: Dict[str, Any], state: str, plan: Dict[str, Any], extra: str = "") -> None:
    photo = None
    if state in ("FORMING", "TRIGGERED") and "H4" in bundle.get("dfs", {}):
        try:
            photo = render_chart(bundle["dfs"]["H4"], bundle["analyses"]["H4"], plan)
        except Exception as exc:
            logger.warning("Chart alert gagal: %s", exc)
    await broadcast(context.bot, format_alert(state, plan, extra), photo)


async def _scan_locked(context: ContextTypes.DEFAULT_TYPE) -> None:
    session: aiohttp.ClientSession = context.application.bot_data["session"]
    try:
        bundle = await collect_market(session)
    except Exception as exc:
        logger.warning("Scan gagal ambil market: %s", exc)
        return
    decision = build_decision(bundle)
    context.application.bot_data["last_bundle"] = bundle
    context.application.bot_data["last_decision"] = decision
    logger.info(
        "Scan %s skor %+d/%d selected=%s blocks=%s",
        decision["direction"], decision["score"], decision["score_max"],
        decision["selected"]["style"] if decision["selected"] else "-",
        "; ".join(decision["blocks"]) or "-",
    )
    active = get_active_setup()
    selected = decision["selected"]

    if active is not None and selected is not None and _same_zone(active, selected):
        if active["state"] == "FORMING" and selected["phase"] == "TRIGGERED":
            update_setup(
                active["id"],
                state="TRIGGERED",
                triggered_at=_iso(),
                last_notified="TRIGGERED",
                note=selected.get("note", ""),
            )
            await _push(context, bundle, "TRIGGERED", selected)
        return

    # Trade yang sudah aktif ditutup oleh harga (SL / TP), bukan oleh flicker bias.
    if active is not None and active["state"] in ("TRIGGERED", "TP1"):
        return

    if active is not None and active["state"] == "FORMING":
        plan = _plan_from_row(active)
        against = (
            decision["direction"] == "NEUTRAL"
            or _bias_against(active, decision)
            or (selected is not None and selected["side"] != active["side"])
        )
        moved = selected is not None and not _same_zone(active, selected)
        created = _parse_iso(active["created_at"])
        age = (_now() - created) if created else timedelta(0)
        dist = zone_distance(decision["price"], plan)
        atr_val = float(active["atr"] or 1)
        ran_away = False
        if active["side"] == "LONG" and decision["price"] >= plan["tp1"] and decision["price"] > plan["entry_high"] + 2.5 * atr_val:
            ran_away = True
        if active["side"] == "SHORT" and decision["price"] <= plan["tp1"] and decision["price"] < plan["entry_low"] - 2.5 * atr_val:
            ran_away = True
        # FORMING lebih cepat expire supaya tidak numpuk setup jelek
        stale = age > timedelta(hours=18) or ran_away or dist > 3.2 * atr_val
        if against or moved or stale:
            if against:
                close_setup(active["id"], "INVALID", "INVALID", None, "bias HTF berubah")
                await broadcast(context.bot, format_alert("INVALID", plan, "Arah HTF tidak lagi mendukung."))
            else:
                why = "Zona berpindah." if moved else "Harga tidak mengisi zona dalam waktu wajar."
                close_setup(active["id"], "EXPIRED", "EXPIRED", None, why)
                await broadcast(context.bot, format_alert("EXPIRED", plan, why))
            active = None
        else:
            return

    if active is not None or selected is None:
        return
    if recent_fingerprint(selected["fingerprint"]):
        logger.info("Fingerprint %s masih dalam jeda re-entry", selected["fingerprint"])
        return
    insert_setup(selected, selected["phase"])
    await _push(context, bundle, selected["phase"], selected)


def _bias_against(active: sqlite3.Row, decision: Dict[str, Any]) -> bool:
    if active["side"] == "LONG" and decision["direction"] == "BEARISH":
        return True
    if active["side"] == "SHORT" and decision["direction"] == "BULLISH":
        return True
    return False


async def monitor_setups(context: ContextTypes.DEFAULT_TYPE) -> None:
    async with _db_lock:
        await _monitor_locked(context)


async def _monitor_locked(context: ContextTypes.DEFAULT_TYPE) -> None:
    active = get_active_setup()
    if active is None:
        return
    session: aiohttp.ClientSession = context.application.bot_data["session"]
    plan = _plan_from_row(active)

    if active["state"] == "FORMING":
        df = await fetch_klines(session, "15m", 20)
        if df is None or len(df) < 3:
            return
        last_close = float(_closed(df).iloc[-1].close)
        broken = (active["side"] == "LONG" and last_close < active["sl"]) or (
            active["side"] == "SHORT" and last_close > active["sl"]
        )
        if broken:
            close_setup(active["id"], "INVALID", "INVALID", None, "level jebol sebelum entry")
            await broadcast(context.bot, format_alert("INVALID", plan, "Close sudah melewati SL sebelum zona terisi."))
        return

    if active["state"] not in ("TRIGGERED", "TP1"):
        return
    df = await fetch_klines(session, "15m", 200)
    if df is None:
        return
    since = _parse_iso(active["triggered_at"]) or _parse_iso(active["created_at"]) or _now()
    candles = _candles_since(df, since)
    outcome = walk_trade(candles, active["side"], plan["entry_mid"], active["sl"], active["tp1"], active["tp2"])
    new_state = outcome["state"]
    if new_state == active["state"]:
        return
    if outcome["result"]:
        note = outcome.get("detail") or active["note"] or ""
        close_setup(active["id"], outcome["result"], outcome["result"], outcome["r"], note)
        alert_state = "TP1" if outcome["result"] == "TP1" else outcome["result"]
        extra = outcome.get("detail") or ""
        if outcome["r"] is not None:
            extra = f"{extra}  Hasil {outcome['r']:+.1f}R".strip()
        await broadcast(context.bot, format_alert(alert_state, plan, extra))
    elif new_state == "TP1":
        update_setup(active["id"], state="TP1", last_notified="TP1", r_multiple=outcome["r"])
        await broadcast(
            context.bot,
            format_alert("TP1", plan, "SL virtual digeser ke entry. Setup tetap dipantau sampai TP2 atau BE."),
        )


# -------------------- Handlers --------------------

WELCOME = """
🟠 <b>BTC Technical Analysis Bot</b>

Analisis <b>BTCUSDT</b> multi-timeframe fokusus <b>H1 + H4</b>.
Alert hanya keluar saat setup berkualitas.

<b>Perintah</b>
/ta — laporan lengkap + chart H4
/h1 /h4 /daily /weekly /15m — detail satu TF
/sr — support & resistance
/setup — setup yang sedang dipantau
/stats — win rate dan 10 hasil terakhir
/subscribe — alert FORMING, ENTRY, TP, SL, batal
/unsubscribe

<b>Gerbang alert (versi ketat)</b>
• H4 wajib searah, Daily tidak melawan
• Zona entry dari confluence H1 + H4
• TRIGGERED hanya kalau candle H1 + volume konfirmasi
• FORMING hanya jika zona ≤ 1.2 ATR
• Cooldown zona 6 jam

<i>Bukan saran finansial.</i>
"""


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.HTML)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.HTML)


async def _bundle_now(context: ContextTypes.DEFAULT_TYPE):
    session: aiohttp.ClientSession = context.application.bot_data["session"]
    bundle = await collect_market(session)
    decision = build_decision(bundle)
    context.application.bot_data["last_bundle"] = bundle
    context.application.bot_data["last_decision"] = decision
    return bundle, decision


async def cmd_ta(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    status = await update.message.reply_text("⏳ Menganalisis multi-timeframe BTC…")
    try:
        bundle, decision = await _bundle_now(context)
    except Exception as exc:
        logger.warning("TA gagal: %s", exc)
        await status.edit_text("❌ Gagal mengambil data market saat ini. Coba lagi sebentar.")
        return
    text = format_report(bundle, decision, get_active_setup())
    await status.edit_text(text, parse_mode=ParseMode.HTML)
    try:
        plan = decision["selected"] or decision["conservative"] or decision["aggressive"]
        photo = render_chart(bundle["dfs"]["H4"], bundle["analyses"]["H4"], plan)
        payload = io.BytesIO(photo)
        payload.name = "btc_h4.png"
        await update.message.reply_photo(photo=payload)
    except Exception as exc:
        logger.warning("Chart /ta gagal: %s", exc)


async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_ta(update, context)


async def cmd_tf(update: Update, context: ContextTypes.DEFAULT_TYPE, tf_key: str) -> None:
    conf = TIMEFRAMES.get(tf_key)
    status = await update.message.reply_text(f"⏳ Mengambil data {tf_key}…")
    if not conf:
        await status.edit_text("Timeframe tidak dikenal.")
        return
    session: aiohttp.ClientSession = context.application.bot_data["session"]
    df = await fetch_klines(session, conf["interval"], conf["limit"])
    if df is None or len(df) < 50:
        await status.edit_text(f"Gagal mengambil data {conf['label']}.")
        return
    await status.edit_text(format_tf(analyze_tf(df, conf["label"])), parse_mode=ParseMode.HTML)


async def cmd_h1(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_tf(update, context, "1h")


async def cmd_h4(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_tf(update, context, "4h")


async def cmd_daily(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_tf(update, context, "1d")


async def cmd_weekly(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_tf(update, context, "1w")


async def cmd_15m(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_tf(update, context, "15m")


async def cmd_sr(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    status = await update.message.reply_text("⏳ Menghitung Support & Resistance…")
    session: aiohttp.ClientSession = context.application.bot_data["session"]
    df_h4 = await fetch_klines(session, "4h", 320)
    df_d1 = await fetch_klines(session, "1d", 400)
    if df_h4 is None or df_d1 is None:
        await status.edit_text("Gagal mengambil data.")
        return
    a_h4 = analyze_tf(df_h4, "H4")
    a_d1 = analyze_tf(df_d1, "Daily")
    price = a_h4["last"]
    lines = [
        "📌 <b>BTCUSDT — Support & Resistance</b>",
        f"Harga saat ini: <b>${fmt_price(price)}</b>",
        "",
        "<b>H4</b>",
        "Resistance:",
    ]
    for level in a_h4["resistances"][:4]:
        lines.append(f"  • ${fmt_price(level)}  ({(level - price) / price * 100:+.1f}%)")
    lines.append("Support:")
    for level in a_h4["supports"][:4]:
        lines.append(f"  • ${fmt_price(level)}  ({(level - price) / price * 100:+.1f}%)")
    lines += ["", "<b>Daily</b>", "Resistance:"]
    for level in a_d1["resistances"][:4]:
        lines.append(f"  • ${fmt_price(level)}  ({(level - price) / price * 100:+.1f}%)")
    lines.append("Support:")
    for level in a_d1["supports"][:4]:
        lines.append(f"  • ${fmt_price(level)}  ({(level - price) / price * 100:+.1f}%)")
    await status.edit_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def cmd_setup(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    active = get_active_setup()
    if active is None:
        await update.message.reply_text("Tidak ada setup aktif. Alert berikutnya akan muncul di sini setelah /subscribe.")
        return
    plan = _plan_from_row(active)
    text = format_alert(active["state"], plan, f"Sejak {active['created_at']} UTC")
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(format_stats(), parse_mode=ParseMode.HTML)


async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    add_subscriber(update.effective_chat.id)
    await update.message.reply_text(
        "✅ <b>Berhasil subscribe.</b>\n\n"
        "Kamu hanya dapat pesan ketika state berubah:\n"
        "• FORMING — zona H1/H4 sudah layak (≤ 1.2 ATR)\n"
        "• ENTRY AKTIF — zona tersentuh + candle H1 + volume konfirmasi\n"
        "• TP1 / TP2 / STOP / BATAL\n\n"
        "Setup lebih selektif. Tidak ada spam zona yang sama.\n"
        "/unsubscribe untuk berhenti. /stats untuk jejak hasil.",
        parse_mode=ParseMode.HTML,
    )


async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    remove_subscriber(update.effective_chat.id)
    await update.message.reply_text("✅ Berhasil berhenti berlangganan alert otomatis.")


# -------------------- Main --------------------

async def post_init(app: Application) -> None:
    app.bot_data["session"] = aiohttp.ClientSession()
    if app.job_queue is None:
        logger.error("Job queue tidak tersedia. Install python-telegram-bot[job-queue].")
        return
    app.job_queue.run_repeating(scan_setups, interval=SCAN_MINUTES * 60, first=20, name="scan_setups")
    app.job_queue.run_repeating(monitor_setups, interval=MONITOR_MINUTES * 60, first=40, name="monitor_setups")
    logger.info("Jobs aktif: scan %dm, monitor %dm", SCAN_MINUTES, MONITOR_MINUTES)


async def post_shutdown(app: Application) -> None:
    session: Optional[aiohttp.ClientSession] = app.bot_data.get("session")
    if session:
        await session.close()


def main() -> None:
    if not BOT_TOKEN:
        print("Set BOT_TOKEN dulu.")
        return
    init_db()
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("ta", cmd_ta))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("h1", cmd_h1))
    app.add_handler(CommandHandler("h4", cmd_h4))
    app.add_handler(CommandHandler("daily", cmd_daily))
    app.add_handler(CommandHandler("weekly", cmd_weekly))
    app.add_handler(CommandHandler("15m", cmd_15m))
    app.add_handler(CommandHandler("sr", cmd_sr))
    app.add_handler(CommandHandler("setup", cmd_setup))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("subscribe", cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    logger.info("Starting BTC TA bot")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


# -------------------- Self test --------------------

def _frame_from_rows(rows: List[Tuple[float, float, float, float, float]]) -> pd.DataFrame:
    start = 1_700_000_000_000
    data = []
    for i, (o, h, l, c, v) in enumerate(rows):
        data.append({
            "open_time": pd.to_datetime(start + i * 3_600_000, unit="ms", utc=True),
            "open": o, "high": h, "low": l, "close": c, "volume": v,
            "taker_buy_base": v * 0.6,
        })
    return pd.DataFrame(data)


def _selftest() -> None:
    assert TF_WEIGHT["15m"] == 0
    analyses = {
        "Weekly": {"bias_score": 0},
        "Daily": {"bias_score": 0},
        "H4": {"bias_score": 0},
        "H1": {"bias_score": 1},
        "15m": {"bias_score": 2},
    }
    score, maximum = weighted_score(analyses)
    assert score == 1 and maximum == 18, (score, maximum)
    assert htf_direction(analyses) == "NEUTRAL"
    analyses["Daily"]["bias_score"] = 1
    analyses["H4"]["bias_score"] = 2
    assert htf_direction(analyses) == "BULLISH"
    analyses["Daily"]["bias_score"] = -1
    assert htf_direction(analyses) == "NEUTRAL"

    # SL before TP, same candle, must be a loss.
    lost = walk_trade([(102, 99)], "LONG", 100, 99.5, 103, 106)
    assert lost["result"] == "SL" and lost["r"] == -1
    opened = walk_trade([(103.2, 100.2)], "LONG", 100, 99, 103, 106)
    assert opened["state"] == "TP1" and opened["result"] is None
    be = walk_trade([(103.2, 100.2), (103.4, 100)], "LONG", 100, 99, 103, 106)
    assert be["result"] == "TP1" and be["r"] > 1
    both = walk_trade([(106.5, 103.1)], "LONG", 100, 99, 103, 106)
    assert both["result"] == "TP2"
    short = walk_trade([(100.2, 93)], "SHORT", 100, 101, 97, 94)
    assert short["result"] == "TP2", short
    short_sl = walk_trade([(101.2, 96.5)], "SHORT", 100, 101, 97, 94)
    assert short_sl["result"] == "SL"

    # Engulf + pin
    up = [110, 112, 109, 111, 100]
    down = [111, 112, 108, 108.5, 100]
    engulf = [108.2, 113, 108, 112.5, 140]
    pin = [110, 111, 104, 110.4, 130]
    base = [(100 + (i % 5), 101 + (i % 5), 99 + (i % 5), 100.5 + (i % 5), 80) for i in range(30)]
    forming = (112.6, 113, 112.2, 112.8, 70)
    df = _frame_from_rows(base + [tuple(up), tuple(down), tuple(engulf), forming])
    assert bullish_engulf(df)
    df_pin = _frame_from_rows(base + [tuple(pin), forming])
    assert bullish_pin(df_pin)

    # Sweep: flat base, a swing low, then a wick reclaim on the last closed candle.
    rows = []
    for i in range(48):
        rows.append((100, 101, 99.6, 100.2, 80))
    rows[20] = (100, 100.4, 97.5, 99.8, 90)  # swing low inside the series
    rows.append((100, 100.5, 97.2, 100.1, 150))  # pierce 97.5 and close back above
    rows.append((100.1, 100.6, 99.9, 100.3, 80))  # forming candle, ignored
    swept = detect_sweep(_frame_from_rows(rows), "H4", order=3)
    assert swept and swept["side"] == "LONG", swept

    book = describe_book(
        bids=[(99.9, 1), (99.5, 90_000)],  # ~$9jt di 99.5
        asks=[(100.2, 0.5), (100.4, 0.4)],
        mid=100,
        source="test",
    )
    assert book["bid_wall"] is not None
    assert book["ask_wall"] is None
    inside = describe_book(
        bids=[(99.95, 200_000)],
        asks=[(100.05, 200_000)],
        mid=100,
        source="test",
    )
    assert inside["bid_wall"] is None and inside["ask_wall"] is None

    idea, reason = make_idea(
        side="LONG", style="conservative", entry_low=100, entry_high=100.3, sl=99.2,
        atr_val=0.8, price=101, score=8,
        resistances=[103.5, 106], supports=[100],
        book={"ask_wall": None, "bid_wall": None}, note="test",
    )
    assert idea and idea["rr"] >= MIN_RR_INTRADAY, reason
    assert abs(((idea["tp1"] - idea["entry_mid"]) / (idea["entry_mid"] - idea["sl"])) - idea["rr"]) < 0.05
    blocked, why = make_idea(
        side="LONG", style="conservative", entry_low=100, entry_high=100.2, sl=99.8,
        atr_val=0.8, price=100.5, score=4,
        resistances=[100.4], supports=[100],
        book={"ask_wall": None, "bid_wall": None}, note="test",
    )
    assert blocked is None and why

    wide, why_wide = make_idea(
        side="LONG", style="conservative", entry_low=100, entry_high=101, sl=90,
        atr_val=2, price=102, score=4, resistances=[120], supports=[100],
        book={"ask_wall": None, "bid_wall": None}, note="test",
    )
    assert wide is None and "lebar" in (why_wide or "")

    wall_block, why_wall = make_idea(
        side="LONG", style="aggressive", entry_low=100, entry_high=100, sl=98.9,
        atr_val=1, price=100, score=8, resistances=[110], supports=[96],
        book={"ask_wall": {"price": 100.4, "notional": 20_000_000}, "bid_wall": None},
        note="test",
    )
    assert wall_block is None and why_wall

    fp1 = _fingerprint("LONG", 100000, 99000)
    fp2 = _fingerprint("LONG", 100050, 99040)
    assert fp1 == fp2
    assert _fingerprint("SHORT", 100000, 99000) != fp1

    init_db()
    with closing(_connect()) as conn:
        conn.execute("DELETE FROM setups")
        conn.commit()
    sample = dict(idea)
    sample["fingerprint"] = "TEST-LONG"
    sample["phase"] = "FORMING"
    setup_id = insert_setup(sample, "FORMING")
    assert get_active_setup()["state"] == "FORMING"
    update_setup(setup_id, state="TRIGGERED", triggered_at=_iso(), last_notified="TRIGGERED")
    assert get_active_setup()["state"] == "TRIGGERED"
    close_setup(setup_id, "TP1", "TP1", 1.7, "test")
    assert get_active_setup() is None
    assert recent_fingerprint("TEST-LONG")
    assert "Menang: 1" in format_stats()
    print("selftest ok")


async def _once() -> None:
    init_db()
    async with aiohttp.ClientSession() as session:
        bundle = await collect_market(session)
    decision = build_decision(bundle)
    text = format_report(bundle, decision, get_active_setup())
    print(text)
    plan = decision["selected"] or decision["conservative"] or decision["aggressive"]
    png = render_chart(bundle["dfs"]["H4"], bundle["analyses"]["H4"], plan)
    out = "/workspace/artifacts/btc_h4.png"
    with open(out, "wb") as handle:
        handle.write(png)
    print(f"\nchart {out} {len(png)} bytes")
    print("selected", decision["selected"]["style"] if decision["selected"] else None, decision["selected"]["phase"] if decision["selected"] else "")
    print("cons", bool(decision["conservative"]), decision["conservative_reason"])
    print("agg", bool(decision["aggressive"]), decision["aggressive_reason"])


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif "--once" in sys.argv:
        asyncio.run(_once())
    else:
        main()
