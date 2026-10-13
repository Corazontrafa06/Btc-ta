#!/usr/bin/env python3
"""
BTC Technical Analysis Bot — Professional Multi-Timeframe
=========================================================
Analisis teknikal lengkap khusus BTCUSDT.

Fitur:
- Multi-timeframe: 15m • 1H • 4H • Daily • Weekly
- Support & Resistance (swing + clustering)
- EMA 20/50/200, RSI, MACD, ATR, Bollinger
- Bias arah per TF + confluence
- Dual Entry Suggestion: Conservative & Aggressive
- Stop Loss & Take Profit berbasis struktur + ATR
- Format profesional siap baca di Telegram

Data: Binance public klines (tanpa API key)
Deploy: Railway / VPS
"""

from __future__ import annotations

import asyncio
import logging
import os
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
SYMBOL = "BTCUSDT"

# Gunakan data-api.binance.vision (lebih stabil & jarang di-block)
# Fallback ke api.binance.com jika perlu
BINANCE_ENDPOINTS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]

# Timeframes yang dianalisis
TIMEFRAMES = {
    "15m": {"interval": "15m", "limit": 200, "label": "15m"},
    "1h":  {"interval": "1h",  "limit": 200, "label": "H1"},
    "4h":  {"interval": "4h",  "limit": 200, "label": "H4"},
    "1d":  {"interval": "1d",  "limit": 200, "label": "Daily"},
    "1w":  {"interval": "1w",  "limit": 100, "label": "Weekly"},
}

REQUEST_TIMEOUT = 15
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BTC-TA-Bot/1.0)",
    "Accept": "application/json",
}
# ====================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("BTC_TA")


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


# -------------------- Indicators (pure) --------------------

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


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
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


def bollinger(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def find_swing_levels(df: pd.DataFrame, order: int = 5, max_levels: int = 6) -> Tuple[List[float], List[float]]:
    """Deteksi swing high/low lalu cluster menjadi level S/R."""
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    swing_highs = []
    swing_lows = []

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
        # Ambil median tiap cluster, prioritaskan yang sering disentuh
        result = []
        for c in clusters:
            result.append(float(np.median(c)))
        # Urutkan dari terdekat ke harga terakhir
        last = df["close"].iloc[-1]
        result = sorted(result, key=lambda x: abs(x - last))
        return result[:max_levels]

    res_levels = cluster(swing_highs)
    sup_levels = cluster(swing_lows)
    return res_levels, sup_levels


# -------------------- Analysis Engine --------------------

def analyze_tf(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    """Analisis lengkap satu timeframe."""
    close = df["close"]
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else last

    # EMAs
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema200 = ema(close, 200) if len(close) >= 200 else ema(close, min(100, len(close)//2))

    e20, e50, e200 = float(ema20.iloc[-1]), float(ema50.iloc[-1]), float(ema200.iloc[-1])

    # Trend bias
    if e20 > e50 > e200 and last > e20:
        bias = "STRONG BULLISH"
        bias_score = 2
    elif e20 > e50 and last > e50:
        bias = "BULLISH"
        bias_score = 1
    elif e20 < e50 < e200 and last < e20:
        bias = "STRONG BEARISH"
        bias_score = -2
    elif e20 < e50 and last < e50:
        bias = "BEARISH"
        bias_score = -1
    else:
        bias = "NEUTRAL / RANGE"
        bias_score = 0

    # RSI
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

    # MACD
    macd_line, signal_line, hist = macd(close)
    macd_now = float(macd_line.iloc[-1])
    signal_now = float(signal_line.iloc[-1])
    hist_now = float(hist.iloc[-1])
    hist_prev = float(hist.iloc[-2]) if len(hist) > 1 else hist_now
    macd_cross = "Bullish cross" if hist_prev <= 0 < hist_now else ("Bearish cross" if hist_prev >= 0 > hist_now else "No cross")

    # ATR
    atr_val = float(atr(df).iloc[-1])

    # Bollinger
    bb_upper, bb_mid, bb_lower = bollinger(close)
    bb_u, bb_m, bb_l = float(bb_upper.iloc[-1]), float(bb_mid.iloc[-1]), float(bb_lower.iloc[-1])

    # Support / Resistance
    res_levels, sup_levels = find_swing_levels(df, order=4 if label in ["15m", "H1"] else 5)

    # Nearest levels
    nearest_res = min([r for r in res_levels if r > last], default=None, key=lambda x: x - last) if any(r > last for r in res_levels) else (res_levels[0] if res_levels else None)
    nearest_sup = max([s for s in sup_levels if s < last], default=None, key=lambda x: last - x) if any(s < last for s in sup_levels) else (sup_levels[0] if sup_levels else None)

    return {
        "label": label,
        "last": last,
        "change_pct": ((last - prev) / prev * 100) if prev else 0,
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
        "bb_upper": bb_u,
        "bb_mid": bb_m,
        "bb_lower": bb_l,
        "resistances": sorted([r for r in res_levels if r > last])[:3],
        "supports": sorted([s for s in sup_levels if s < last], reverse=True)[:3],
        "nearest_res": nearest_res,
        "nearest_sup": nearest_sup,
    }


def build_entry_ideas(analyses: Dict[str, Dict], current_price: float) -> Dict[str, Any]:
    """Buat dua skenario entry: Conservative & Aggressive."""
    h4 = analyses.get("H4", {})
    d1 = analyses.get("Daily", {})
    h1 = analyses.get("H1", {})

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

    # --- Conservative ---
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
            "rr": round((tp1 - current_price) / (current_price - sl), 1) if current_price > sl else None,
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
            "rr": round((current_price - tp1) / (sl - current_price), 1) if sl > current_price else None,
            "note": "Tunggu rejection di resistance + konfirmasi",
        }
    else:
        ideas["conservative"] = {
            "side": "WAIT",
            "note": "Kondisi range / kurang confluence. Lebih baik menunggu breakout jelas.",
        }

    # --- Aggressive ---
    if main_bias == "BULLISH":
        entry = current_price
        sl = current_price - atr_h4 * 1.1
        tp1 = current_price + atr_h4 * 1.5
        tp2 = current_price + atr_h4 * 2.8
        ideas["aggressive"] = {
            "side": "LONG",
            "entry": f"Market / {entry:,.0f}",
            "sl": f"{sl:,.0f}",
            "tp1": f"{tp1:,.0f}",
            "tp2": f"{tp2:,.0f}",
            "rr": round(1.5 / 1.1, 1),
            "note": "Entry langsung mengikuti momentum (risiko lebih tinggi)",
        }
    elif main_bias == "BEARISH":
        entry = current_price
        sl = current_price + atr_h4 * 1.1
        tp1 = current_price - atr_h4 * 1.5
        tp2 = current_price - atr_h4 * 2.8
        ideas["aggressive"] = {
            "side": "SHORT",
            "entry": f"Market / {entry:,.0f}",
            "sl": f"{sl:,.0f}",
            "tp1": f"{tp1:,.0f}",
            "tp2": f"{tp2:,.0f}",
            "rr": round(1.5 / 1.1, 1),
            "note": "Entry langsung mengikuti momentum (risiko lebih tinggi)",
        }
    else:
        ideas["aggressive"] = {
            "side": "WAIT",
            "note": "Tidak ada edge jelas untuk entry agresif saat ini.",
        }

    return ideas


# -------------------- Formatting --------------------

def fmt_price(p: float) -> str:
    return f"{p:,.0f}"


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


async def build_full_ta(session: aiohttp.ClientSession) -> str:
    """Bangun laporan TA lengkap multi-timeframe."""
    tasks = {
        key: fetch_klines(session, conf["interval"], conf["limit"])
        for key, conf in TIMEFRAMES.items()
    }
    results = await asyncio.gather(*tasks.values())
    dfs = dict(zip(tasks.keys(), results))

    analyses = {}
    for key, df in dfs.items():
        if df is not None and len(df) > 50:
            label = TIMEFRAMES[key]["label"]
            analyses[label] = analyze_tf(df, label)

    if not analyses:
        return "❌ Gagal mengambil data market saat ini. Coba lagi sebentar."

    # Ambil harga terkini dari TF terendah yang ada
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

    order = ["Weekly", "Daily", "H4", "H1", "15m"]
    for tf in order:
        if tf in analyses:
            a = analyses[tf]
            lines.append(
                f"{bias_emoji(a['bias'])} <b>{tf}</b>: {a['bias']}  |  RSI {a['rsi']:.0f} ({a['rsi_status']})"
            )

    lines.append("")
    lines.append(f"🎯 <b>Overall Bias:</b> <b>{ideas['bias']}</b> (score {ideas['score']:+d})")
    lines.append("")

    # Key Levels dari H4 + Daily
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("<b>📌 KEY LEVELS (H4 + Daily)</b>")

    h4 = analyses.get("H4", {})
    d1 = analyses.get("Daily", {})

    res_set = sorted(set((h4.get("resistances") or []) + (d1.get("resistances") or [])))
    sup_set = sorted(set((h4.get("supports") or []) + (d1.get("supports") or [])), reverse=True)

    if res_set:
        lines.append("Resistance:")
        for r in res_set[:4]:
            dist = (r - current) / current * 100
            lines.append(f"  • ${fmt_price(r)}  (+{dist:.1f}%)")
    if sup_set:
        lines.append("Support:")
        for s in sup_set[:4]:
            dist = (current - s) / current * 100
            lines.append(f"  • ${fmt_price(s)}  (-{dist:.1f}%)")

    # Indicators snapshot H4
    if h4:
        lines.append("")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("<b>📈 H4 SNAPSHOT</b>")
        lines.append(f"EMA20 / 50 / 200: {fmt_price(h4['ema20'])} / {fmt_price(h4['ema50'])} / {fmt_price(h4['ema200'])}")
        lines.append(f"RSI: {h4['rsi']:.1f} — {h4['rsi_status']}")
        lines.append(f"MACD Hist: {h4['macd_hist']:.1f} ({h4['macd_cross']})")
        lines.append(f"ATR(14): ${fmt_price(h4['atr'])}")

    # Entry Ideas
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
    lines.append("<i>⚠️ Bukan saran finansial. Selalu gunakan risk management. Data: Binance public market data.</i>")

    return "\n".join(lines)


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


# -------------------- Handlers --------------------

WELCOME = """
🟠 <b>BTC Technical Analysis Bot</b>

Analisis teknikal profesional khusus <b>BTCUSDT</b>.

<b>Perintah:</b>
/ta          — Full Multi-Timeframe Analysis + Entry Ideas
/h1          — Detail H1
/h4          — Detail H4
/daily       — Detail Daily
/weekly      — Detail Weekly
/15m         — Detail 15 menit
/sr          — Focus Support & Resistance
/signal      — Quick bias + entry

Semua analisis mencakup:
• Bias arah per timeframe
• Support / Resistance
• EMA • RSI • MACD • ATR
• Saran Entry Conservative & Aggressive
• SL / TP berbasis struktur

Data real-time dari Binance public market data.
"""


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.HTML)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.HTML)


async def cmd_ta(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    status = await update.message.reply_text("⏳ Menganalisis multi-timeframe BTC…")
    session: aiohttp.ClientSession = context.application.bot_data["session"]
    text = await build_full_ta(session)
    # Telegram limit
    if len(text) > 4000:
        text = text[:3900] + "\n\n… (terpotong)"
    await status.edit_text(text, parse_mode=ParseMode.HTML)


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
    """Focus Support & Resistance."""
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

    lines.append("")
    lines.append("<i>Level dihitung dari swing high/low + clustering.</i>")
    await status.edit_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Quick signal version."""
    status = await update.message.reply_text("⏳ Generate signal…")
    session: aiohttp.ClientSession = context.application.bot_data["session"]
    text = await build_full_ta(session)
    # Ambil bagian penting saja
    if "ENTRY IDEAS" in text:
        # Kirim full tetap, karena sudah cukup ringkas
        pass
    await status.edit_text(text, parse_mode=ParseMode.HTML)


# -------------------- Main --------------------

async def post_init(app: Application) -> None:
    session = aiohttp.ClientSession()
    app.bot_data["session"] = session
    logger.info("BTC TA Bot ready")


async def post_shutdown(app: Application) -> None:
    session: aiohttp.ClientSession = app.bot_data.get("session")
    if session:
        await session.close()


def main() -> None:
    if not BOT_TOKEN:
        print("❌ Set BOT_TOKEN environment variable dulu!")
        print("   export BOT_TOKEN='your_telegram_bot_token'")
        return

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

    logger.info("Starting BTC Technical Analysis Bot…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
