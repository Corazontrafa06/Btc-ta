#!/usr/bin/env python3
"""
BTC Technical Analysis Bot — Professional Multi-Timeframe + Auto Alert
======================================================================
Analisis teknikal lengkap khusus BTCUSDT.

Fitur:
- Multi-timeframe: 15m • 1H • 4H • Daily • Weekly
- Support & Resistance (swing + clustering)
- EMA 20/50/200, RSI, MACD, ATR, Bollinger
- Bias arah per TF + confluence
- Dual Entry Suggestion: Conservative & Aggressive
- Stop Loss & Take Profit berbasis struktur + ATR
- AUTO ALERT: hanya kirim saat ada setup bagus (confluence jelas)
- /subscribe  → dapat notifikasi otomatis
- /unsubscribe

Data: Binance public klines (tanpa API key)
Deploy: Railway / VPS
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# ====================== CONFIG ======================
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
DB_PATH = os.getenv("DB_PATH", "btc_ta.db")
SYMBOL = "BTCUSDT"

BINANCE_ENDPOINTS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]

TIMEFRAMES = {
    "15m": {"interval": "15m", "limit": 200, "label": "15m"},
    "1h":  {"interval": "1h",  "limit": 200, "label": "H1"},
    "4h":  {"interval": "4h",  "limit": 200, "label": "H4"},
    "1d":  {"interval": "1d",  "limit": 200, "label": "Daily"},
    "1w":  {"interval": "1w",  "limit": 100, "label": "Weekly"},
}

# Auto-alert settings
CHECK_INTERVAL_MINUTES = 45          # cek setiap 45 menit
MIN_ABS_SCORE = 3                    # minimal |score| supaya dianggap setup bagus
COOLDOWN_MINUTES = 180               # jangan spam alert yang sama dalam 3 jam

REQUEST_TIMEOUT = 15
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BTC-TA-Bot/1.1)",
    "Accept": "application/json",
}
# ====================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("BTC_TA")

# State untuk cooldown
_last_alert_bias: Optional[str] = None
_last_alert_time: Optional[datetime] = None


# -------------------- Database --------------------

def init_db() -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS subscribers (
                chat_id     INTEGER PRIMARY KEY,
                subscribed  INTEGER DEFAULT 1,
                created_at  TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def add_subscriber(chat_id: int) -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO subscribers (chat_id, subscribed) VALUES (?, 1)",
            (chat_id,),
        )
        conn.commit()


def remove_subscriber(chat_id: int) -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            "UPDATE subscribers SET subscribed = 0 WHERE chat_id = ?",
            (chat_id,),
        )
        conn.commit()


def get_subscribers() -> List[int]:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute(
            "SELECT chat_id FROM subscribers WHERE subscribed = 1"
        ).fetchall()
    return [r[0] for r in rows]


# -------------------- Data --------------------

async def fetch_klines(session: aiohttp.ClientSession, interval: str, limit: int = 200) -> Optional[pd.DataFrame]:
    params = {"symbol": SYMBOL, "interval": interval, "limit": limit}
    raw = None

    for base_url in BINANCE_ENDPOINTS:
        try:
            async with session.get(
                base_url,
                params=params,
                headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    raw = await resp.json()
                    break
                else:
                    logger.warning("Binance %s %s → HTTP %s", base_url, interval, resp.status)
        except Exception as e:
            logger.warning("Fetch error %s %s: %s", base_url, interval, e)
            continue

    if not raw:
        logger.error("All endpoints failed for interval %s", interval)
        return None

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


# -------------------- Indicators --------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def find_swing_levels(df: pd.DataFrame, order: int = 5, max_levels: int = 6):
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    swing_highs, swing_lows = [], []

    for i in range(order, n - order):
        if highs[i] == max(highs[i - order: i + order + 1]):
            swing_highs.append(highs[i])
        if lows[i] == min(lows[i - order: i + order + 1]):
            swing_lows.append(lows[i])

    def cluster(levels: List[float], threshold_pct: float = 0.004) -> List[float]:
        if not levels:
            return []
        levels = sorted(levels)
        clusters = [[levels[0]]]
        for lv in levels[1:]:
            if abs(lv - clusters[-1][-1]) / clusters[-1][-1] <= threshold_pct:
                clusters[-1].append(lv)
            else:
                clusters.append([lv])
        result = [float(np.median(c)) for c in clusters]
        last = df["close"].iloc[-1]
        return sorted(result, key=lambda x: abs(x - last))[:max_levels]

    return cluster(swing_highs), cluster(swing_lows)


# -------------------- Analysis Engine --------------------

def analyze_tf(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    close = df["close"]
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else last

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema200 = ema(close, 200) if len(close) >= 200 else ema(close, min(100, len(close)//2))

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

    rsi_val = float(rsi(close).iloc[-1])
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

    macd_line, signal_line, hist = macd(close)
    hist_now = float(hist.iloc[-1])
    hist_prev = float(hist.iloc[-2]) if len(hist) > 1 else hist_now
    macd_cross = "Bullish cross" if hist_prev <= 0 < hist_now else ("Bearish cross" if hist_prev >= 0 > hist_now else "No cross")

    atr_val = float(atr(df).iloc[-1])
    res_levels, sup_levels = find_swing_levels(df, order=4 if label in ["15m", "H1"] else 5)

    nearest_res = min([r for r in res_levels if r > last], default=None, key=lambda x: x - last) if any(r > last for r in res_levels) else None
    nearest_sup = max([s for s in sup_levels if s < last], default=None, key=lambda x: last - x) if any(s < last for s in sup_levels) else None

    return {
        "label": label,
        "last": last,
        "change_pct": ((last - prev) / prev * 100) if prev else 0,
        "bias": bias,
        "bias_score": bias_score,
        "ema20": e20, "ema50": e50, "ema200": e200,
        "rsi": rsi_val, "rsi_status": rsi_status,
        "macd_hist": hist_now, "macd_cross": macd_cross,
        "atr": atr_val,
        "resistances": sorted([r for r in res_levels if r > last])[:3],
        "supports": sorted([s for s in sup_levels if s < last], reverse=True)[:3],
        "nearest_res": nearest_res,
        "nearest_sup": nearest_sup,
    }


def build_entry_ideas(analyses: Dict[str, Dict], current_price: float) -> Dict[str, Any]:
    h4 = analyses.get("H4", {})
    d1 = analyses.get("Daily", {})
    atr_h4 = h4.get("atr") or current_price * 0.01
    nearest_sup = h4.get("nearest_sup") or d1.get("nearest_sup")
    nearest_res = h4.get("nearest_res") or d1.get("nearest_res")

    overall_score = sum(a.get("bias_score", 0) for a in analyses.values())
    if overall_score >= 3:
        main_bias = "BULLISH"
    elif overall_score <= -3:
        main_bias = "BEARISH"
    else:
        main_bias = "NEUTRAL"

    ideas = {"bias": main_bias, "score": overall_score}

    # Conservative
    if main_bias == "BULLISH" and nearest_sup:
        entry_low = nearest_sup
        entry_high = nearest_sup + atr_h4 * 0.3
        sl = nearest_sup - atr_h4 * 0.8
        tp1 = current_price + atr_h4 * 1.2
        tp2 = nearest_res if nearest_res else current_price + atr_h4 * 2.5
        ideas["conservative"] = {
            "side": "LONG",
            "entry": f"{entry_low:,.0f} – {entry_high:,.0f}",
            "sl": f"{sl:,.0f}",
            "tp1": f"{tp1:,.0f}",
            "tp2": f"{tp2:,.0f}",
            "rr": round((tp1 - current_price) / max(current_price - sl, 1), 1),
            "note": "Tunggu pullback ke support + konfirmasi candle bullish",
        }
    elif main_bias == "BEARISH" and nearest_res:
        entry_high = nearest_res
        entry_low = nearest_res - atr_h4 * 0.3
        sl = nearest_res + atr_h4 * 0.8
        tp1 = current_price - atr_h4 * 1.2
        tp2 = nearest_sup if nearest_sup else current_price - atr_h4 * 2.5
        ideas["conservative"] = {
            "side": "SHORT",
            "entry": f"{entry_low:,.0f} – {entry_high:,.0f}",
            "sl": f"{sl:,.0f}",
            "tp1": f"{tp1:,.0f}",
            "tp2": f"{tp2:,.0f}",
            "rr": round((current_price - tp1) / max(sl - current_price, 1), 1),
            "note": "Tunggu rejection di resistance + konfirmasi",
        }
    else:
        ideas["conservative"] = {"side": "WAIT", "note": "Kondisi range / kurang confluence."}

    # Aggressive
    if main_bias == "BULLISH":
        sl = current_price - atr_h4 * 1.1
        tp1 = current_price + atr_h4 * 1.5
        tp2 = current_price + atr_h4 * 2.8
        ideas["aggressive"] = {
            "side": "LONG",
            "entry": f"Market / {current_price:,.0f}",
            "sl": f"{sl:,.0f}",
            "tp1": f"{tp1:,.0f}",
            "tp2": f"{tp2:,.0f}",
            "rr": 1.4,
            "note": "Entry langsung mengikuti momentum (risiko lebih tinggi)",
        }
    elif main_bias == "BEARISH":
        sl = current_price + atr_h4 * 1.1
        tp1 = current_price - atr_h4 * 1.5
        tp2 = current_price - atr_h4 * 2.8
        ideas["aggressive"] = {
            "side": "SHORT",
            "entry": f"Market / {current_price:,.0f}",
            "sl": f"{sl:,.0f}",
            "tp1": f"{tp1:,.0f}",
            "tp2": f"{tp2:,.0f}",
            "rr": 1.4,
            "note": "Entry langsung mengikuti momentum (risiko lebih tinggi)",
        }
    else:
        ideas["aggressive"] = {"side": "WAIT", "note": "Tidak ada edge jelas untuk entry agresif."}

    return ideas


# -------------------- Formatting --------------------

def fmt_price(p: float) -> str:
    return f"{p:,.0f}"


def bias_emoji(bias: str) -> str:
    if "STRONG BULLISH" in bias: return "🟢🟢"
    if "BULLISH" in bias: return "🟢"
    if "STRONG BEARISH" in bias: return "🔴🔴"
    if "BEARISH" in bias: return "🔴"
    return "⚪"


async def build_full_ta(session: aiohttp.ClientSession) -> Tuple[str, Dict]:
    """Return (text, ideas_dict)"""
    tasks = {key: fetch_klines(session, conf["interval"], conf["limit"]) for key, conf in TIMEFRAMES.items()}
    results = await asyncio.gather(*tasks.values())
    dfs = dict(zip(tasks.keys(), results))

    analyses = {}
    for key, df in dfs.items():
        if df is not None and len(df) > 50:
            label = TIMEFRAMES[key]["label"]
            analyses[label] = analyze_tf(df, label)

    if not analyses:
        return "❌ Gagal mengambil data market saat ini. Coba lagi sebentar.", {}

    current = None
    for tf in ["H1", "15m", "H4", "Daily"]:
        if tf in analyses:
            current = analyses[tf]["last"]
            break
    if current is None:
        current = list(analyses.values())[0]["last"]

    ideas = build_entry_ideas(analyses, current)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"🟠 <b>BTCUSDT TECHNICAL ANALYSIS</b>",
        f"<i>{now}</i>",
        f"Harga: <b>${fmt_price(current)}</b>",
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        "<b>📊 MULTI-TIMEFRAME BIAS</b>",
    ]

    for tf in ["Weekly", "Daily", "H4", "H1", "15m"]:
        if tf in analyses:
            a = analyses[tf]
            lines.append(f"{bias_emoji(a['bias'])} <b>{tf}</b>: {a['bias']}  |  RSI {a['rsi']:.0f} ({a['rsi_status']})")

    lines.append("")
    lines.append(f"🎯 <b>Overall Bias:</b> <b>{ideas['bias']}</b> (score {ideas['score']:+d})")
    lines.append("")

    # Key Levels
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("<b>📌 KEY LEVELS (H4 + Daily)</b>")
    h4 = analyses.get("H4", {})
    d1 = analyses.get("Daily", {})
    res_set = sorted(set((h4.get("resistances") or []) + (d1.get("resistances") or [])))
    sup_set = sorted(set((h4.get("supports") or []) + (d1.get("supports") or [])), reverse=True)

    if res_set:
        lines.append("Resistance:")
        for r in res_set[:4]:
            lines.append(f"  • ${fmt_price(r)}  (+{(r-current)/current*100:.1f}%)")
    if sup_set:
        lines.append("Support:")
        for s in sup_set[:4]:
            lines.append(f"  • ${fmt_price(s)}  (-{(current-s)/current*100:.1f}%)")

    if h4:
        lines.append("")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("<b>📈 H4 SNAPSHOT</b>")
        lines.append(f"EMA20 / 50 / 200: {fmt_price(h4['ema20'])} / {fmt_price(h4['ema50'])} / {fmt_price(h4['ema200'])}")
        lines.append(f"RSI: {h4['rsi']:.1f} — {h4['rsi_status']}")
        lines.append(f"MACD Hist: {h4['macd_hist']:.1f} ({h4['macd_cross']})")
        lines.append(f"ATR(14): ${fmt_price(h4['atr'])}")

    # Entry Ideas (selalu tampilkan keduanya)
    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("<b>💡 ENTRY IDEAS</b>")

    cons = ideas.get("conservative", {})
    if cons.get("side") == "WAIT":
        lines.append(f"🛡 <b>Conservative:</b> {cons.get('note')}")
    else:
        lines.append(f"🛡 <b>Conservative {cons['side']}</b>")
        lines.append(f"   Entry : {cons.get('entry')}")
        lines.append(f"   SL    : {cons.get('sl')}")
        lines.append(f"   TP1   : {cons.get('tp1')}  |  TP2: {cons.get('tp2')}")
        if cons.get("rr"):
            lines.append(f"   R:R   : 1 : {cons['rr']}")
        lines.append(f"   Note  : {cons.get('note')}")

    lines.append("")
    agg = ideas.get("aggressive", {})
    if agg.get("side") == "WAIT":
        lines.append(f"⚡ <b>Aggressive:</b> {agg.get('note')}")
    else:
        lines.append(f"⚡ <b>Aggressive {agg['side']}</b>")
        lines.append(f"   Entry : {agg.get('entry')}")
        lines.append(f"   SL    : {agg.get('sl')}")
        lines.append(f"   TP1   : {agg.get('tp1')}  |  TP2: {agg.get('tp2')}")
        if agg.get("rr"):
            lines.append(f"   R:R   : 1 : {agg['rr']}")
        lines.append(f"   Note  : {agg.get('note')}")

    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("<i>⚠️ Bukan saran finansial. Selalu gunakan risk management.</i>")

    return "\n".join(lines), ideas


async def build_tf_detail(session: aiohttp.ClientSession, tf_key: str) -> str:
    conf = TIMEFRAMES.get(tf_key)
    if not conf:
        return "Timeframe tidak dikenal."
    df = await fetch_klines(session, conf["interval"], conf["limit"])
    if df is None or len(df) < 50:
        return f"Gagal mengambil data {conf['label']}."
    a = analyze_tf(df, conf["label"])
    lines = [
        f"🟠 <b>BTCUSDT — {a['label']}</b>",
        f"Harga: <b>${fmt_price(a['last'])}</b>  ({a['change_pct']:+.2f}%)",
        "",
        f"Bias: {bias_emoji(a['bias'])} <b>{a['bias']}</b>",
        "",
        f"EMA20:  ${fmt_price(a['ema20'])}",
        f"EMA50:  ${fmt_price(a['ema50'])}",
        f"EMA200: ${fmt_price(a['ema200'])}",
        "",
        f"RSI(14): {a['rsi']:.1f} — {a['rsi_status']}",
        f"MACD Hist: {a['macd_hist']:.2f} ({a['macd_cross']})",
        f"ATR(14): ${fmt_price(a['atr'])}",
        "",
        "Resistance terdekat:",
    ]
    for r in a["resistances"][:3]:
        lines.append(f"  • ${fmt_price(r)}")
    lines.append("Support terdekat:")
    for s in a["supports"][:3]:
        lines.append(f"  • ${fmt_price(s)}")
    return "\n".join(lines)


# -------------------- Auto Alert Job --------------------

async def check_and_alert(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Cek market setiap interval. Kirim hanya jika setup bagus."""
    global _last_alert_bias, _last_alert_time

    subs = get_subscribers()
    if not subs:
        return

    session: aiohttp.ClientSession = context.application.bot_data["session"]
    text, ideas = await build_full_ta(session)

    if not ideas:
        return

    score = ideas.get("score", 0)
    bias = ideas.get("bias", "NEUTRAL")

    # Hanya kirim jika confluence cukup kuat
    if abs(score) < MIN_ABS_SCORE:
        logger.info("Score %s terlalu lemah, skip alert", score)
        return

    # Cooldown: jangan spam bias yang sama
    now = datetime.now(timezone.utc)
    if _last_alert_bias == bias and _last_alert_time:
        minutes_since = (now - _last_alert_time).total_seconds() / 60
        if minutes_since < COOLDOWN_MINUTES:
            logger.info("Masih cooldown untuk bias %s (%.0f mnt lagi)", bias, COOLDOWN_MINUTES - minutes_since)
            return

    # Ada setup bagus → kirim
    header = f"🚨 <b>SETUP BAGUS TERDETEKSI</b> — {bias} (score {score:+d})\n\n"
    full_text = header + text

    if len(full_text) > 4000:
        full_text = full_text[:3900] + "\n\n… (terpotong)"

    for chat_id in subs:
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=full_text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
            await asyncio.sleep(0.05)
        except Exception as e:
            logger.warning("Gagal kirim alert ke %s: %s", chat_id, e)

    _last_alert_bias = bias
    _last_alert_time = now
    logger.info("Auto-alert terkirim: %s (score %s) ke %d subscriber", bias, score, len(subs))


# -------------------- Handlers --------------------

WELCOME = """
🟠 <b>BTC Technical Analysis Bot</b>

Analisis teknikal profesional khusus <b>BTCUSDT</b>.

<b>Perintah:</b>
/ta          — Full Multi-Timeframe + Entry Ideas
/h1 /h4 /daily /weekly /15m — Detail per TF
/sr          — Focus Support & Resistance
/signal      — Sama seperti /ta
/subscribe   — <b>Dapat alert otomatis saat setup bagus</b>
/unsubscribe — Berhenti menerima alert
/help

<b>Auto Alert:</b>
Bot akan otomatis kirim notifikasi <b>hanya ketika</b> ada setup bagus (confluence tinggi).
Isi alert: bias + level + saran entry <b>Conservative & Aggressive</b>.

Data real-time dari Binance public market data.
"""


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.HTML)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.HTML)


async def cmd_ta(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    status = await update.message.reply_text("⏳ Menganalisis multi-timeframe BTC…")
    session: aiohttp.ClientSession = context.application.bot_data["session"]
    text, _ = await build_full_ta(session)
    if len(text) > 4000:
        text = text[:3900] + "\n\n… (terpotong)"
    await status.edit_text(text, parse_mode=ParseMode.HTML)


async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_ta(update, context)


async def cmd_tf(update: Update, context: ContextTypes.DEFAULT_TYPE, tf_key: str) -> None:
    status = await update.message.reply_text(f"⏳ Mengambil data {tf_key}…")
    session: aiohttp.ClientSession = context.application.bot_data["session"]
    text = await build_tf_detail(session, tf_key)
    await status.edit_text(text, parse_mode=ParseMode.HTML)


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
    df_h4 = await fetch_klines(session, "4h", 200)
    df_d1 = await fetch_klines(session, "1d", 200)
    if df_h4 is None or df_d1 is None:
        await status.edit_text("Gagal mengambil data.")
        return
    a_h4 = analyze_tf(df_h4, "H4")
    a_d1 = analyze_tf(df_d1, "Daily")
    price = a_h4["last"]
    lines = [
        f"📌 <b>BTCUSDT — Support & Resistance</b>",
        f"Harga saat ini: <b>${fmt_price(price)}</b>",
        "",
        "<b>H4 Levels</b>",
        "Resistance:",
    ]
    for r in a_h4["resistances"][:4]:
        lines.append(f"  • ${fmt_price(r)}  (+{(r-price)/price*100:.1f}%)")
    lines.append("Support:")
    for s in a_h4["supports"][:4]:
        lines.append(f"  • ${fmt_price(s)}  (-{(price-s)/price*100:.1f}%)")
    lines.append("")
    lines.append("<b>Daily Levels</b>")
    lines.append("Resistance:")
    for r in a_d1["resistances"][:4]:
        lines.append(f"  • ${fmt_price(r)}  (+{(r-price)/price*100:.1f}%)")
    lines.append("Support:")
    for s in a_d1["supports"][:4]:
        lines.append(f"  • ${fmt_price(s)}  (-{(price-s)/price*100:.1f}%)")
    await status.edit_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    add_subscriber(update.effective_chat.id)
    await update.message.reply_text(
        "✅ <b>Berhasil subscribe!</b>\n\n"
        "Kamu akan menerima notifikasi <b>otomatis</b> hanya ketika bot mendeteksi <b>setup bagus</b> "
        "(confluence tinggi).\n\n"
        "Isi notifikasi:\n"
        "• Overall bias + score\n"
        "• Key Support / Resistance\n"
        "• Saran Entry <b>Conservative</b> & <b>Aggressive</b>\n"
        "• SL / TP\n\n"
        "Ketik /unsubscribe untuk berhenti.",
        parse_mode=ParseMode.HTML,
    )


async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    remove_subscriber(update.effective_chat.id)
    await update.message.reply_text("✅ Berhasil berhenti berlangganan alert otomatis.")


# -------------------- Main --------------------

async def post_init(app: Application) -> None:
    session = aiohttp.ClientSession()
    app.bot_data["session"] = session

    # Job: cek market secara berkala
    app.job_queue.run_repeating(
        check_and_alert,
        interval=CHECK_INTERVAL_MINUTES * 60,
        first=30,  # mulai 30 detik setelah bot start
        name="btc_setup_checker",
    )
    logger.info("Auto-alert job started (setiap %d menit, min score ±%d)", CHECK_INTERVAL_MINUTES, MIN_ABS_SCORE)


async def post_shutdown(app: Application) -> None:
    session: aiohttp.ClientSession = app.bot_data.get("session")
    if session:
        await session.close()


def main() -> None:
    if not BOT_TOKEN:
        print("❌ Set BOT_TOKEN environment variable dulu!")
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
    app.add_handler(CommandHandler("subscribe", cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))

    logger.info("Starting BTC Technical Analysis Bot (with Auto Alert)…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
