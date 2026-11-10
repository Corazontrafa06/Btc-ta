"""
Panel strategi BTC — dipakai bersama oleh bot (status live) dan backtest.py (uji historis).

Semua fitur dihitung dari candle CLOSED per timeframe, lalu dipetakan ke grid waktu memakai
waktu close candle (open_time + durasi). Jadi logika live == logika backtest, tanpa lookahead.
Parameter sengaja sedikit dan standar (tidak dioptimasi).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

H1 = pd.Timedelta(hours=1)
DUR = {"5m": pd.Timedelta(minutes=5), "15m": pd.Timedelta(minutes=15), "1h": H1,
       "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def tf_features(df: pd.DataFrame, dur: pd.Timedelta, p: str) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]
    cols: Dict[str, Any] = {"avail": df["open_time"] + dur, f"{p}_time": df["open_time"]}
    for n in (7, 9, 20, 21, 25, 50, 99, 200):
        cols[f"{p}_e{n}"] = _ema(c, n)
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    rng = (h - l).replace(0, np.nan)
    cols.update({
        f"{p}_close": c, f"{p}_open": o, f"{p}_high": h, f"{p}_low": l,
        f"{p}_atr": _atr(df), f"{p}_rsi": _rsi(c),
        f"{p}_bbu": mid + 2 * sd, f"{p}_bbl": mid - 2 * sd, f"{p}_bbm": mid,
        f"{p}_hh20p": h.shift(1).rolling(20).max(), f"{p}_ll20p": l.shift(1).rolling(20).min(),
        f"{p}_hh24p": h.shift(1).rolling(24).max(), f"{p}_ll24p": l.shift(1).rolling(24).min(),
        f"{p}_vol_ratio": v / v.shift(1).rolling(20).mean(),
        f"{p}_lower_wick": (np.minimum(o, c) - l) / rng, f"{p}_upper_wick": (h - np.maximum(o, c)) / rng,
    })
    out = pd.DataFrame(cols)
    out[f"{p}_above"] = (out[f"{p}_e7"] > out[f"{p}_e25"]) & (out[f"{p}_e7"] > out[f"{p}_e99"])
    out[f"{p}_below"] = (out[f"{p}_e7"] < out[f"{p}_e25"]) & (out[f"{p}_e7"] < out[f"{p}_e99"])
    return out


def resample_3h(df_1h: pd.DataFrame) -> pd.DataFrame:
    f = df_1h.copy()
    f["b"] = f["open_time"].dt.floor("3h")
    g = f.groupby("b").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                           close=("close", "last"), volume=("volume", "sum"), n=("close", "size")).reset_index()
    g = g.rename(columns={"b": "open_time"})
    if len(g) and g["n"].iloc[0] < 3:
        g = g.iloc[1:]
    return g.drop(columns="n").reset_index(drop=True)


def _merge(grid: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    return pd.merge_asof(grid, feat.sort_values("avail"), left_on="t", right_on="avail", direction="backward").drop(columns="avail")


def swing_grid(k1h: pd.DataFrame, k4h: pd.DataFrame, k1d: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    """Grid per jam (t = waktu close H1 terakhir). Kolom prefix h1_/h3_/h4_/d_."""
    g = pd.DataFrame({"t": times})
    for p, df, dur in (("h1", k1h, H1), ("h3", resample_3h(k1h), pd.Timedelta(hours=3)),
                       ("h4", k4h, DUR["4h"]), ("d", k1d, DUR["1d"])):
        g = _merge(g, tf_features(df, dur, p))
    g = g.copy()
    for col in ("h1_close", "h1_rsi", "h1_e20", "h1_e50", "h1_bbl", "h1_bbu", "h4_rsi"):
        g[f"{col}_prev"] = g[col].shift(1)
    g["price"] = g["h1_close"]
    g["h4_new"] = g["h4_time"] != g["h4_time"].shift(1)
    g["stack_above"] = g.h1_above & g.h3_above & g.h4_above
    g["stack_below"] = g.h1_below & g.h3_below & g.h4_below
    g["stack_above_new"] = g.stack_above & ~g.stack_above.shift(1, fill_value=False)
    g["stack_below_new"] = g.stack_below & ~g.stack_below.shift(1, fill_value=False)
    g["h4_up"] = (g.h4_e50 > g.h4_e200) & (g.h4_close > g.h4_e200)
    g["h4_down"] = (g.h4_e50 < g.h4_e200) & (g.h4_close < g.h4_e200)
    g["d_up"] = g.d_close > g.d_e50
    g["d_down"] = g.d_close < g.d_e50
    g["h4_range"] = (g.h4_e50 - g.h4_e200).abs() / g.h4_atr < 1.5
    return g


def scalp_grid(base: pd.DataFrame, base_key: str, k1h: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    g = pd.DataFrame({"t": times})
    g = _merge(g, tf_features(base, DUR[base_key], "s"))
    g = _merge(g, tf_features(k1h, H1, "h1"))
    g = g.copy()
    g["price"] = g["s_close"]
    g["d_up"] = g["h1_close"] > g["h1_e200"]
    return g


@dataclass(frozen=True)
class Strategy:
    key: str
    name: str
    side: str
    grid: str                 # "swing" | "scalp15" | "scalp5"
    mask: Callable[[pd.DataFrame], pd.Series]
    desc: str
    sl_atr: float = 1.0
    atr_col: str = "h4_atr"
    tp_r: float = 2.0
    hold_h: int = 48


def _m(f):
    return f


STRATEGIES: List[Strategy] = [
    Strategy("trend", "Trend Following", "LONG", "swing",
             lambda g: g.h4_up & (g.h1_e20 > g.h1_e50) & (g.h1_e20_prev <= g.h1_e50_prev),
             "EMA20 H1 cross ke atas EMA50, H4 uptrend (EMA50>EMA200, close>EMA200)"),
    Strategy("trend", "Trend Following", "SHORT", "swing",
             lambda g: g.h4_down & (g.h1_e20 < g.h1_e50) & (g.h1_e20_prev >= g.h1_e50_prev),
             "EMA20 H1 cross ke bawah EMA50, H4 downtrend"),
    Strategy("breakout", "Breakout", "LONG", "swing",
             lambda g: g.h4_new & (g.h4_close > g.h4_hh20p) & (g.h4_close > g.h4_e200),
             "Close H4 tembus high 20 candle, di atas EMA200 H4"),
    Strategy("breakout", "Breakout", "SHORT", "swing",
             lambda g: g.h4_new & (g.h4_close < g.h4_ll20p) & (g.h4_close < g.h4_e200),
             "Close H4 tembus low 20 candle, di bawah EMA200 H4"),
    Strategy("pullback", "Pullback", "LONG", "swing",
             lambda g: g.h4_up & g.d_up & (g.h1_low <= g.h1_e50) & (g.h1_close > g.h1_e20) & (g.h1_close > g.h1_open),
             "H4+Daily naik, H1 menyentuh EMA50 lalu close hijau di atas EMA20"),
    Strategy("pullback", "Pullback", "SHORT", "swing",
             lambda g: g.h4_down & g.d_down & (g.h1_high >= g.h1_e50) & (g.h1_close < g.h1_e20) & (g.h1_close < g.h1_open),
             "H4+Daily turun, H1 menyentuh EMA50 lalu close merah di bawah EMA20"),
    Strategy("reversal", "Reversal", "LONG", "swing",
             lambda g: g.h4_new & (g.h4_rsi > 30) & (g.h4_rsi_prev <= 30),
             "RSI H4 keluar dari oversold (naik lewat 30)"),
    Strategy("reversal", "Reversal", "SHORT", "swing",
             lambda g: g.h4_new & (g.h4_rsi < 70) & (g.h4_rsi_prev >= 70),
             "RSI H4 keluar dari overbought (turun lewat 70)"),
    Strategy("range", "Range", "LONG", "swing",
             lambda g: g.h4_range & (g.h1_close_prev < g.h1_bbl_prev) & (g.h1_close > g.h1_bbl),
             "H4 sideways, close H1 kembali masuk dari bawah Bollinger"),
    Strategy("range", "Range", "SHORT", "swing",
             lambda g: g.h4_range & (g.h1_close_prev > g.h1_bbu_prev) & (g.h1_close < g.h1_bbu),
             "H4 sideways, close H1 kembali masuk dari atas Bollinger"),
    Strategy("momentum", "Momentum", "LONG", "swing",
             lambda g: (g.h1_close > g.h1_hh20p) & (g.h1_rsi > 60) & (g.h1_vol_ratio >= 1.5),
             "Close H1 tembus high 20 candle, RSI>60, volume ≥1.5x"),
    Strategy("momentum", "Momentum", "SHORT", "swing",
             lambda g: (g.h1_close < g.h1_ll20p) & (g.h1_rsi < 40) & (g.h1_vol_ratio >= 1.5),
             "Close H1 tembus low 20 candle, RSI<40, volume ≥1.5x"),
    Strategy("price_action", "Price Action (wick liquidity)", "LONG", "swing",
             lambda g: (g.h1_low < g.h1_ll24p) & (g.h1_close > g.h1_ll24p) & (g.h1_lower_wick >= 0.5),
             "Wick H1 sapu low 24 jam lalu close balik, ekor bawah ≥50% candle"),
    Strategy("price_action", "Price Action (wick liquidity)", "SHORT", "swing",
             lambda g: (g.h1_high > g.h1_hh24p) & (g.h1_close < g.h1_hh24p) & (g.h1_upper_wick >= 0.5),
             "Wick H1 sapu high 24 jam lalu close balik, ekor atas ≥50% candle"),
    Strategy("ema_stack", "EMA Stack 7/25/99 (aturan kamu)", "SHORT", "swing",
             lambda g: g.stack_above_new, "Kuning (EMA7) baru di atas EMA25 & EMA99 di H1+H3+H4"),
    Strategy("ema_stack", "EMA Stack 7/25/99 (mirror)", "LONG", "swing",
             lambda g: g.stack_below_new, "Kuning (EMA7) baru di bawah EMA25 & EMA99 di H1+H3+H4"),
    Strategy("scalp15", "Scalping Momentum 15m", "LONG", "scalp15",
             lambda g: (g.s_close > g.s_hh20p) & (g.s_e9 > g.s_e21) & (g.h1_close > g.h1_e50),
             "Close 15m tembus high 20 candle, EMA9>EMA21, H1 di atas EMA50", 1.0, "s_atr", 1.5, 4),
    Strategy("scalp15", "Scalping Momentum 15m", "SHORT", "scalp15",
             lambda g: (g.s_close < g.s_ll20p) & (g.s_e9 < g.s_e21) & (g.h1_close < g.h1_e50),
             "Close 15m tembus low 20 candle, EMA9<EMA21, H1 di bawah EMA50", 1.0, "s_atr", 1.5, 4),
    Strategy("scalp5", "Scalping Momentum 5m", "LONG", "scalp5",
             lambda g: (g.s_close > g.s_hh20p) & (g.s_e9 > g.s_e21) & (g.h1_close > g.h1_e50),
             "Close 5m tembus high 20 candle, EMA9>EMA21, H1 di atas EMA50", 1.0, "s_atr", 1.5, 2),
    Strategy("scalp5", "Scalping Momentum 5m", "SHORT", "scalp5",
             lambda g: (g.s_close < g.s_ll20p) & (g.s_e9 < g.s_e21) & (g.h1_close < g.h1_e50),
             "Close 5m tembus low 20 candle, EMA9<EMA21, H1 di bawah EMA50", 1.0, "s_atr", 1.5, 2),
]

FAMILY_ORDER = ["trend", "breakout", "pullback", "reversal", "range", "momentum", "price_action", "ema_stack", "scalp15", "scalp5"]


def stat_key(s: Strategy) -> str:
    return f"{s.key}:{s.side}"


def plan_for(s: Strategy, row: pd.Series) -> Optional[Dict[str, float]]:
    px = float(row["price"])
    atr_v = float(row[s.atr_col])
    if not (atr_v > 0) or not np.isfinite(px):
        return None
    risk = s.sl_atr * atr_v
    sgn = 1 if s.side == "LONG" else -1
    return {"entry": px, "sl": px - sgn * risk, "tp": px + sgn * s.tp_r * risk, "rr": s.tp_r,
            "risk_pct": risk / px * 100, "hold_h": s.hold_h}


def live_status(grids: Dict[str, pd.DataFrame], lookback_rows: Dict[str, int] = None) -> List[Dict[str, Any]]:
    """Status setiap strategi di baris terakhir grid (candle closed terakhir) + sinyal terakhir dalam lookback."""
    lookback_rows = lookback_rows or {"swing": 24, "scalp15": 16, "scalp5": 24}
    out = []
    for s in STRATEGIES:
        g = grids.get(s.grid)
        if g is None or len(g) < 3:
            out.append({"strategy": s, "ok": False})
            continue
        mask = s.mask(g).fillna(False).astype(bool)
        now = bool(mask.iloc[-1])
        recent = mask.iloc[-lookback_rows[s.grid]:]
        last_t = g["t"][recent[recent].index[-1]] if recent.any() else None
        row = g.iloc[-1]
        out.append({"strategy": s, "ok": True, "signal": now, "t": row["t"], "last_signal_t": last_t,
                    "plan": plan_for(s, row) if now else None, "price": float(row["price"])})
    return out
