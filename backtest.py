#!/usr/bin/env python3
"""
Backtest BTC TA bot — replay logika sinyal ASLI (btc_ta_bot.build_decision) di data historis.

Tahap 1 (decisions): tiap close H1, bangun bundle persis seperti collect_market:
  candle closed + 1 candle "forming" sintetis (agregasi H1 yang sudah closed di bucket berjalan,
  atau candle datar kalau bucket baru mulai). Tidak ada lookahead.
  Funding = funding rate historis terakhir; OI historis tidak tersedia (None); order book kosong.
Tahap 2 (simulate): state machine mirip _scan_locked/_monitor_locked
  (FORMING → TRIGGERED → TP1 → TP2/BE, INVALID/EXPIRED, satu setup aktif, cooldown fingerprint 6 jam),
  exit dijalankan di candle 15m. Fill di harga close saat TRIGGERED (bukan entry_mid), fee 0.08% round-trip.
  Model hasil realistis: 50% ditutup di TP1, SL sisa digeser ke BE, 50% runner ke TP2 / BE.

Pakai:
  python fetch_history.py            # sekali, ambil data ke backtest_data/
  python backtest.py decisions       # hitung keputusan (cache backtest_data/decisions.pkl)
  python backtest.py report          # baseline + analisis loss + varian + walk-forward + chart
"""
from __future__ import annotations

import math
import os
import pickle
import sys
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("DB_PATH", "/tmp/btc_ta_backtest.db")
import btc_ta_bot as B  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_data")
START = pd.Timestamp(os.getenv("BT_START", "2025-08-01"), tz="UTC")
END = pd.Timestamp(os.getenv("BT_END", "2026-11-09"), tz="UTC")
SPLIT = pd.Timestamp(os.getenv("BT_SPLIT", "2026-05-01"), tz="UTC")
FEE_RT = float(os.getenv("BT_FEE_RT", "0.0008"))

TF_SPECS = {  # key: (label, limit, bucket freq)
    "15m": ("15m", 300, "15min"),
    "1h": ("H1", 1000, "1h"),
    "4h": ("H4", 320, "4h"),
    "1d": ("Daily", 400, "1D"),
    "1w": ("Weekly", 300, "W"),
}
TD = {"15m": pd.Timedelta(minutes=15), "1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4),
      "1d": pd.Timedelta(days=1), "1w": pd.Timedelta(days=7)}

_G: Dict[str, Any] = {}


def _load() -> Dict[str, pd.DataFrame]:
    out = {k: pd.read_pickle(os.path.join(DATA, f"{k}.pkl")) for k in TF_SPECS}
    out["funding"] = pd.read_pickle(os.path.join(DATA, "funding.pkl"))
    return out


def _bucket_start(t: pd.Timestamp, key: str) -> pd.Timestamp:
    if key == "1w":  # Binance weekly buka Senin 00:00 UTC
        d = t.normalize()
        return d - pd.Timedelta(days=d.weekday())
    return t.floor(TF_SPECS[key][2])


def _frame_at(key: str, t: pd.Timestamp) -> pd.DataFrame:
    df = _G["data"][key]
    ot = _G["ot"][key]
    limit = TF_SPECS[key][1]
    n_closed = int(np.searchsorted(ot, (t - TD[key]).to_datetime64(), side="right"))
    closed = df.iloc[max(0, n_closed - (limit - 1)):n_closed]
    bstart = _bucket_start(t, key)
    h1 = _G["data"]["1h"]
    if bstart < t and key in ("4h", "1d", "1w"):
        a = int(np.searchsorted(_G["ot"]["1h"], bstart.to_datetime64(), side="left"))
        b = int(np.searchsorted(_G["ot"]["1h"], t.to_datetime64(), side="left"))
        part = h1.iloc[a:b]
        forming = {"open_time": bstart, "open": float(part.open.iloc[0]), "high": float(part.high.max()),
                   "low": float(part.low.min()), "close": float(part.close.iloc[-1]),
                   "volume": float(part.volume.sum()), "taker_buy_base": float(part.taker_buy_base.sum())}
    else:
        last = float(closed.close.iloc[-1])
        forming = {"open_time": bstart, "open": last, "high": last, "low": last, "close": last,
                   "volume": 0.0, "taker_buy_base": 0.0}
    return pd.concat([closed, pd.DataFrame([forming])], ignore_index=True)


def bundle_at(t: pd.Timestamp) -> Dict[str, Any]:
    dfs: Dict[str, pd.DataFrame] = {}
    analyses: Dict[str, Dict[str, Any]] = {}
    h1_full = None
    for key, (label, _, _) in TF_SPECS.items():
        df = _frame_at(key, t)
        if label == "H1":
            h1_full = df
            df = df.tail(B.H1_ANALYSIS_BARS).reset_index(drop=True)
        dfs[label] = df
        analyses[label] = B.analyze_tf(df, label)
    h3 = B.resample_h3(h1_full)
    if h3 is not None:
        dfs["H3"] = h3
        analyses["H3"] = B.analyze_tf(h3, "H3")
    stack_dfs = dict(dfs)
    stack_dfs["H1"] = h1_full
    stack = B.ema_stack(stack_dfs)
    fund = _G["data"]["funding"]
    idx = int(np.searchsorted(fund.time.values, t.to_datetime64(), side="right")) - 1
    flow = {"ok": idx >= 0, "source": "hist", "funding": float(fund.rate.iloc[idx]) if idx >= 0 else None,
            "oi_change_pct": None, "next_funding": None}
    price = float(analyses["H1"]["last"])
    book = {"ask_wall": None, "bid_wall": None, "source": "none"}
    return {"dfs": dfs, "analyses": analyses, "price": price, "book": book, "flow": flow, "ema_stack": stack}


def _features(t: pd.Timestamp, bundle: Dict[str, Any], dec: Dict[str, Any]) -> Dict[str, Any]:
    a = bundle["analyses"]
    h4df = bundle["dfs"]["H4"]
    atr_s = B.atr(B._closed(h4df))
    atr_reg = float(atr_s.iloc[-1] / atr_s.tail(100).median()) if len(atr_s) > 100 else 1.0
    st = bundle["ema_stack"]
    f = {
        "t": t, "price": bundle["price"], "direction": dec["direction"], "score": dec["score"],
        "hour": t.hour, "weekday": t.weekday(),
        "funding": bundle["flow"]["funding"],
        "atr_h4": a["H4"]["atr"], "atr_h4_pct": a["H4"]["atr"] / bundle["price"] * 100, "atr_regime": atr_reg,
        "h4_macd": a["H4"]["macd_hist"], "h4_bbpb": a["H4"]["bb_percent_b"], "h4_squeeze": a["H4"]["bb_squeeze"],
        "h4_stalling": a["H4"]["stalling"], "h1_vol": a["H1"]["vol_ratio"], "h1_taker": a["H1"]["taker_ratio"],
        "h4_vs_ema200_atr": (bundle["price"] - a["H4"]["ema200"]) / a["H4"]["atr"],
        "stack_short": st["short"], "stack_long": st["long"], "stack_above": st["above_count"],
        "stack_below": st["below_count"],
    }
    for lab in ("Weekly", "Daily", "H4", "H3", "H1", "15m"):
        if lab in a:
            f[f"bias_{lab}"] = a[lab]["bias_score"]
            f[f"rsi_{lab}"] = a[lab]["rsi"]
    for lab in ("H1", "H3", "H4"):
        row = st["per_tf"].get(lab)
        f[f"stack_{lab}"] = row["state"] if row else None
    return f


def _slim(idea: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not idea:
        return None
    keep = ("side", "style", "kind", "entry_low", "entry_high", "entry_mid", "sl", "tp1", "tp2", "rr",
            "sl_pct", "atr", "phase", "alertable", "fingerprint", "note", "distance_atr")
    return {k: idea.get(k) for k in keep}


def _init(data):
    _G["data"] = data
    _G["ot"] = {k: data[k].open_time.values for k in TF_SPECS}


def _decide(t: pd.Timestamp) -> Dict[str, Any]:
    bundle = bundle_at(t)
    dec = B.build_decision(bundle)
    return {
        "t": t, "price": dec["price"], "direction": dec["direction"],
        "selected": _slim(dec["selected"]), "conservative": _slim(dec["conservative"]),
        "aggressive": _slim(dec["aggressive"]), "blocks": dec["blocks"], "passes": dec["passes"],
        "f": _features(t, bundle, dec),
    }


def compute_decisions(tag: str = "current") -> List[Dict[str, Any]]:
    data = _load()
    times = list(pd.date_range(START, END, freq="1h"))
    with Pool(os.cpu_count() or 2, initializer=_init, initargs=(data,)) as pool:
        out = pool.map(_decide, times, chunksize=64)
    with open(os.path.join(DATA, f"decisions_{tag}.pkl"), "wb") as fh:
        pickle.dump(out, fh)
    return out


# ------------------------------------------------------------------ simulation

def _same_zone(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if a["side"] != b["side"] or a["style"] != b["style"]:
        return False
    return abs(a["entry_mid"] - b["entry_mid"]) / max(b["entry_mid"], 1) <= 0.012 and abs(a["sl"] - b["sl"]) / max(b["sl"], 1) <= 0.012


def _exit_walk(m15: pd.DataFrame, i0: int, side: str, fill: float, sl: float, tp1: float, tp2: float,
               max_bars: Optional[int] = None, be_at_r: Optional[float] = None):
    """Mulai dari candle 15m index i0 (candle pertama setelah fill). Return (r_realistic, r_botstyle, outcome, exit_i)."""
    risk = abs(fill - sl)
    hi = m15.high.values
    lo = m15.low.values
    cl = m15.close.values
    hit1 = False
    sl_live = sl
    n = len(m15)
    last = n - 1 if max_bars is None else min(n - 1, i0 + max_bars)
    r1 = abs(tp1 - fill) / risk
    r2 = abs(tp2 - fill) / risk
    be_px = None
    if be_at_r:
        be_px = fill + be_at_r * risk if side == "LONG" else fill - be_at_r * risk
    be_armed = False
    for i in range(i0, last + 1):
        h, l = hi[i], lo[i]
        if be_armed and not hit1:
            # SL sudah di BE (+fee kecil diabaikan) sebelum TP1
            if (side == "LONG" and l <= fill) or (side == "SHORT" and h >= fill):
                return 0.0, 0.0, "BE", i
        if side == "LONG":
            stop = l <= sl_live
            tgt = h >= (tp2 if hit1 else tp1)
        else:
            stop = h >= sl_live
            tgt = l <= (tp2 if hit1 else tp1)
        if stop:  # SL duluan kalau satu candle kena keduanya (konservatif, sama seperti walk_trade)
            if hit1:
                return 0.5 * r1, r1, "TP1_BE", i
            return -1.0, -1.0, "SL", i
        if tgt:
            if not hit1:
                hit1 = True
                sl_live = fill
                far = (h >= tp2) if side == "LONG" else (l <= tp2)
                if far:
                    return 0.5 * r1 + 0.5 * r2, r2, "TP2", i
            else:
                return 0.5 * r1 + 0.5 * r2, r2, "TP2", i
        if be_px is not None and not be_armed and not hit1:
            if (side == "LONG" and h >= be_px) or (side == "SHORT" and l <= be_px):
                be_armed = True  # berlaku mulai candle berikutnya (tanpa lookahead intrabar)
    px = cl[last]
    open_r = ((px - fill) if side == "LONG" else (fill - px)) / risk
    if hit1:
        return 0.5 * r1 + 0.5 * open_r, r1, "OPEN_TP1", last
    return open_r, open_r, "OPEN", last


def simulate(decisions: List[Dict[str, Any]], m15: pd.DataFrame, accept=None, modify=None,
             max_hold_bars: Optional[int] = None, be_at_r: Optional[float] = None,
             fill_mode: str = "trigger", min_fill_rr: Optional[float] = None) -> pd.DataFrame:
    """fill_mode: "trigger" = isi di harga saat TRIGGERED (perilaku bot sekarang);
                  "limit"   = limit order di entry_mid begitu setup dibuat (FORMING/TRIGGERED), isi saat tersentuh."""
    """accept(idea, feat) -> bool  (filter tambahan sebelum setup dibuat / di-trigger)
       modify(idea, feat) -> idea  (ubah SL/TP)"""
    ot15 = m15.open_time.values
    trades: List[Dict[str, Any]] = []
    active: Optional[Dict[str, Any]] = None
    busy_until: Optional[pd.Timestamp] = None
    last_closed_fp: Dict[str, pd.Timestamp] = {}

    def prep(idea, f):
        if idea is None:
            return None
        if accept is not None and not accept(idea, f):
            return None
        return modify(dict(idea), f) if modify is not None else idea

    def fill(setup, d, px=None, i0=None):
        nonlocal busy_until
        t = d["t"]
        if i0 is None:
            i0 = int(np.searchsorted(ot15, t.to_datetime64(), side="left"))
        else:
            t = pd.Timestamp(ot15[i0]).tz_localize("UTC")
        px = d["price"] if px is None else px
        side = setup["side"]
        if (side == "LONG" and not (setup["sl"] < px < setup["tp1"])) or (side == "SHORT" and not (setup["tp1"] < px < setup["sl"])):
            return
        if min_fill_rr and abs(setup["tp1"] - px) / abs(px - setup["sl"]) < min_fill_rr:
            return  # R:R dari harga fill nyata terlalu kecil → skip (setup dianggap selesai)
        r, rbot, outcome, ei = _exit_walk(m15, i0, side, px, setup["sl"], setup["tp1"], setup["tp2"], max_hold_bars, be_at_r)
        risk = abs(px - setup["sl"])
        fee_r = FEE_RT * px / risk
        exit_t = pd.Timestamp(ot15[ei]).tz_localize("UTC") + pd.Timedelta(minutes=15)
        trades.append({**{k: v for k, v in setup.items() if k not in ("f",)}, **{f"f_{k}": v for k, v in setup["f"].items()},
                       "entry_t": t, "fill": px, "exit_t": exit_t, "outcome": outcome,
                       "r_gross": r, "r": r - fee_r, "r_bot": rbot, "fee_r": fee_r,
                       "risk_pct": risk / px * 100,
                       "rr_tp1": abs(setup["tp1"] - px) / risk, "rr_tp2": abs(setup["tp2"] - px) / risk,
                       "hold_h": (exit_t - t).total_seconds() / 3600})
        last_closed_fp[setup["fingerprint"]] = exit_t
        busy_until = exit_t

    for d in decisions:
        t = d["t"]
        if busy_until is not None and t < busy_until:
            continue  # trade aktif (TRIGGERED/TP1) memblokir setup baru
        f = d["f"]
        sel = prep(d["selected"], f)
        # FORMING invalid by 15m close beyond SL (monitor) — dicek kasar per jam
        if active is not None:
            a = active
            i_prev = int(np.searchsorted(ot15, a["_last_t"].to_datetime64(), side="left"))
            i_now = int(np.searchsorted(ot15, t.to_datetime64(), side="left"))
            closes = m15.close.values[i_prev:i_now]
            broken = (a["side"] == "LONG" and (closes < a["sl"]).any()) or (a["side"] == "SHORT" and (closes > a["sl"]).any())
            a["_last_t"] = t
            if fill_mode == "limit" and not broken:
                lows = m15.low.values[i_prev:i_now]
                highs = m15.high.values[i_prev:i_now]
                touch = np.nonzero(lows <= a["entry_mid"])[0] if a["side"] == "LONG" else np.nonzero(highs >= a["entry_mid"])[0]
                if len(touch):
                    j = i_prev + int(touch[0])
                    active = None
                    fill(a, d, px=a["entry_mid"], i0=j)
                    if busy_until is not None and t < busy_until:
                        continue
                    a = None
            if a is not None and broken:
                last_closed_fp[a["fingerprint"]] = t
                active = None
        if active is not None and sel is not None and _same_zone(active, sel):
            if fill_mode == "limit":
                for k in ("entry_low", "entry_high", "entry_mid", "sl", "tp1", "tp2", "rr"):
                    active[k] = sel[k]
            elif sel["phase"] == "TRIGGERED":
                setup = {**active, **{k: sel[k] for k in ("entry_low", "entry_high", "entry_mid", "sl", "tp1", "tp2", "rr")}}
                active = None
                fill(setup, d)
            continue
        if active is not None:
            age = t - active["_created"]
            against = d["direction"] == "NEUTRAL" or (active["side"] == "LONG" and d["direction"] == "BEARISH") \
                or (active["side"] == "SHORT" and d["direction"] == "BULLISH") or (sel is not None and sel["side"] != active["side"])
            moved = sel is not None
            atr_v = active["atr"] or 1
            px = d["price"]
            dist = 0 if active["entry_low"] <= px <= active["entry_high"] else min(abs(px - active["entry_low"]), abs(px - active["entry_high"]))
            stale = age > pd.Timedelta(hours=36) or dist > 4 * atr_v
            if against or (moved and age > pd.Timedelta(hours=2)) or stale:
                last_closed_fp[active["fingerprint"]] = t
                active = None
            else:
                continue
        if active is None and sel is not None:
            fp = sel["fingerprint"]
            if fp in last_closed_fp and t - last_closed_fp[fp] < pd.Timedelta(hours=B.REENTRY_HOURS):
                continue
            setup = {**sel, "f": f, "_created": t, "_last_t": t}
            if fill_mode == "limit":
                px = d["price"]
                at_or_better = (sel["side"] == "LONG" and px <= sel["entry_mid"]) or (sel["side"] == "SHORT" and px >= sel["entry_mid"])
                if at_or_better:
                    fill(setup, d)
                else:
                    active = setup
            elif sel["phase"] == "TRIGGERED":
                fill(setup, d)
            else:
                active = setup
    return pd.DataFrame(trades)


# ------------------------------------------------------------------ metrics

def metrics(tr: pd.DataFrame) -> Dict[str, Any]:
    if tr is None or len(tr) == 0:
        return {"n": 0, "wr": float("nan"), "avg_r": float("nan"), "pf": float("nan"), "maxdd": 0.0, "tot_r": 0.0}
    r = tr["r"].values
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq).max())
    return {"n": len(r), "wr": float((r > 0).mean() * 100), "avg_r": float(r.mean()),
            "pf": float(wins / losses) if losses > 0 else float("inf"), "maxdd": dd, "tot_r": float(r.sum())}


def fmt_m(m: Dict[str, Any]) -> str:
    return (f"n={m['n']:>3}  WR={m['wr']:5.1f}%  avgR={m['avg_r']:+.3f}  PF={m['pf']:.2f}  "
            f"totR={m['tot_r']:+6.1f}  maxDD={m['maxdd']:.1f}R")


def split(tr: pd.DataFrame):
    if len(tr) == 0:
        return tr, tr
    return tr[tr.entry_t < SPLIT], tr[tr.entry_t >= SPLIT]


def load_decisions(tag: str = "current"):
    with open(os.path.join(DATA, f"decisions_{tag}.pkl"), "rb") as fh:
        return pickle.load(fh)


FOLDS = [("2025-08-01", "2026-01-01"), ("2026-01-01", "2026-06-01"), ("2026-06-01", "2026-11-10")]


def summary_row(name: str, tr: pd.DataFrame) -> str:
    ins, oos = split(tr)
    mi, mo = metrics(ins), metrics(oos)
    folds = []
    for a, b in FOLDS:
        part = tr[(tr.entry_t >= pd.Timestamp(a, tz="UTC")) & (tr.entry_t < pd.Timestamp(b, tz="UTC"))] if len(tr) else tr
        folds.append(f"{metrics(part)['avg_r']:+.2f}")
    return (f"| {name} | {mi['n']} | {mi['wr']:.0f}% | {mi['avg_r']:+.2f} | {mi['pf']:.2f} | {mi['maxdd']:.1f} | "
            f"{mo['n']} | {mo['wr']:.0f}% | {mo['avg_r']:+.2f} | {mo['pf']:.2f} | {mo['maxdd']:.1f} | {' / '.join(folds)} |")


def equity_chart(curves: Dict[str, pd.DataFrame], out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=120)
    for name, tr in curves.items():
        if len(tr) == 0:
            continue
        tr = tr.sort_values("exit_t")
        ax.step(tr.exit_t, tr.r.cumsum(), where="post", label=f"{name} ({len(tr)} trade, {tr.r.sum():+.1f}R)")
    ax.axvline(SPLIT, color="grey", linestyle="--", linewidth=1)
    ax.text(SPLIT, ax.get_ylim()[1], "  out-of-sample →", va="top", color="grey")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("BTC TA bot — equity (R, setelah fee 0.08% round-trip)")
    ax.set_ylabel("Kumulatif R")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)


def ema_stack_rule(decisions: List[Dict[str, Any]], m15: pd.DataFrame, side: str = "SHORT", tp_r: float = 2.0,
                   sl_atr: float = 1.0, hold_bars: int = 192, every_hour: bool = False) -> pd.DataFrame:
    """Aturan user berdiri sendiri: saat EMA stack SHORT BARU aktif → trade sederhana (SL k·ATR H4, TP n·R, maks 48 jam).
    every_hour=True = benchmark tanpa sinyal (trade yang sama di jam mana saja)."""
    ot = m15.open_time.values
    rows, busy, prev = [], None, False
    for d in decisions:
        on = bool(d["f"]["stack_short"])
        fire = True if every_hour else (on and not prev)
        prev = on
        if not fire or (busy is not None and d["t"] < busy):
            continue
        px, risk = d["price"], sl_atr * d["f"]["atr_h4"]
        s = 1 if side == "LONG" else -1
        i0 = int(np.searchsorted(ot, d["t"].to_datetime64()))
        r, _, out, ei = _exit_walk(m15, i0, side, px, px - s * risk, px + s * tp_r * risk, px + s * tp_r * risk, hold_bars)
        busy = pd.Timestamp(ot[ei]).tz_localize("UTC")
        rows.append({"entry_t": d["t"], "exit_t": busy, "r": r - FEE_RT * px / risk, "outcome": out})
    return pd.DataFrame(rows)


def report() -> None:
    m15 = pd.read_pickle(os.path.join(DATA, "15m.pkl"))
    base_dec = load_decisions("baseline")
    new_dec = load_decisions("current")
    base = simulate(base_dec, m15)
    new = simulate(new_dec, m15)
    ema = ema_stack_rule(new_dec, m15)
    bench = ema_stack_rule(new_dec, m15, every_hour=True)
    hdr = ("| Versi | IS n | IS WR | IS avgR | IS PF | IS maxDD | OOS n | OOS WR | OOS avgR | OOS PF | OOS maxDD | "
           "avgR per fold |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    print(f"IS = {START.date()} → {SPLIT.date()}, OOS = {SPLIT.date()} → {END.date()}. Fee {FEE_RT*100:.2f}% round-trip.")
    print(hdr)
    print(summary_row("Baseline (kode lama)", base))
    print(summary_row("Improved (kode baru)", new))
    print(summary_row("EMA stack SHORT sendiri", ema))
    print(summary_row("Benchmark SHORT acak", bench))
    base.to_pickle(os.path.join(DATA, "trades_baseline.pkl"))
    new.to_pickle(os.path.join(DATA, "trades_current.pkl"))
    out = os.getenv("BT_CHART", "/workspace/artifacts/backtest_equity.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    equity_chart({"Baseline": base, "Improved": new, "EMA stack SHORT (sendiri)": ema}, out)
    print("chart", out)


# ================================================================== strategy lab (1 tahun)
# Semua indikator dihitung per TF dari candle CLOSED, lalu dipetakan ke grid jam memakai waktu close
# candle (open_time + durasi) → tidak ada lookahead. Entry di harga close H1 terakhir, exit dicek di 15m.
LAB_START = pd.Timestamp(os.getenv("LAB_START", "2025-11-10"), tz="UTC")
LAB_END = pd.Timestamp(os.getenv("LAB_END", "2026-11-09"), tz="UTC")
LAB_SPLIT = pd.Timestamp(os.getenv("LAB_SPLIT", "2026-05-10"), tz="UTC")
LAB_FOLDS = [("2025-11-10", "2026-03-10"), ("2026-03-10", "2026-07-10"), ("2026-07-10", "2026-11-10")]


def _tf_features(df: pd.DataFrame, dur: pd.Timedelta, prefix: str) -> pd.DataFrame:
    c = df["close"]
    out = pd.DataFrame({"avail": df["open_time"] + dur})
    for n in (7, 20, 25, 50, 99, 200):
        out[f"{prefix}_e{n}"] = B.ema(c, n).values
    out[f"{prefix}_close"] = c.values
    out[f"{prefix}_open"] = df["open"].values
    out[f"{prefix}_high"] = df["high"].values
    out[f"{prefix}_low"] = df["low"].values
    out[f"{prefix}_atr"] = B.atr(df).values
    out[f"{prefix}_rsi"] = B.rsi(c).values
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    out[f"{prefix}_bbu"] = (mid + 2 * sd).values
    out[f"{prefix}_bbl"] = (mid - 2 * sd).values
    out[f"{prefix}_bbm"] = mid.values
    out[f"{prefix}_hh20"] = df["high"].rolling(20).max().values
    out[f"{prefix}_ll20"] = df["low"].rolling(20).min().values
    out[f"{prefix}_hh24p"] = df["high"].shift(1).rolling(24).max().values  # 24 candle sebelumnya
    out[f"{prefix}_ll24p"] = df["low"].shift(1).rolling(24).min().values
    e7, e25, e99 = out[f"{prefix}_e7"], out[f"{prefix}_e25"], out[f"{prefix}_e99"]
    out[f"{prefix}_above"] = (e7 > e25) & (e7 > e99)
    out[f"{prefix}_below"] = (e7 < e25) & (e7 < e99)
    return out


def lab_grid() -> pd.DataFrame:
    data = _load()
    h1 = data["1h"]
    grid = pd.DataFrame({"t": pd.date_range(LAB_START, LAB_END, freq="1h")})
    frames = {
        "h1": (h1, pd.Timedelta(hours=1)),
        "h3": (B.resample_h3(h1), pd.Timedelta(hours=3)),
        "h4": (data["4h"], pd.Timedelta(hours=4)),
        "d": (data["1d"], pd.Timedelta(days=1)),
    }
    for p, (df, dur) in frames.items():
        feat = _tf_features(df.reset_index(drop=True), dur, p).sort_values("avail")
        grid = pd.merge_asof(grid, feat, left_on="t", right_on="avail", direction="backward").drop(columns="avail")
    grid = grid.copy()  # de-fragment setelah banyak merge
    # nilai H1 sebelumnya (untuk deteksi cross)
    for col in ("h1_close", "h1_rsi", "h1_e20", "h1_e50", "h1_bbl", "h1_bbu", "h1_low", "h1_high"):
        grid[f"{col}_prev"] = grid[col].shift(1)
    grid["price"] = grid["h1_close"]
    grid["stack_above"] = grid.h1_above & grid.h3_above & grid.h4_above
    grid["stack_below"] = grid.h1_below & grid.h3_below & grid.h4_below
    grid["stack_above_new"] = grid.stack_above & ~grid.stack_above.shift(1, fill_value=False)
    grid["stack_below_new"] = grid.stack_below & ~grid.stack_below.shift(1, fill_value=False)
    grid["h4_up"] = (grid.h4_e50 > grid.h4_e200) & (grid.h4_close > grid.h4_e200)
    grid["h4_down"] = (grid.h4_e50 < grid.h4_e200) & (grid.h4_close < grid.h4_e200)
    grid["d_up"] = grid.d_close > grid.d_e50
    grid["d_down"] = grid.d_close < grid.d_e50
    grid["h4_range"] = (grid.h4_e50 - grid.h4_e200).abs() / grid.h4_atr < 1.5
    return grid


def _cross_up(g, a, b):
    return (g[a] > g[b]) & (g[f"{a}_prev"] <= g[f"{b}_prev"]) if f"{b}_prev" in g else (g[a] > g[b]) & (g[f"{a}_prev"] <= g[b])


def lab_signals(g: pd.DataFrame) -> Dict[str, Tuple[str, pd.Series]]:
    """name -> (side, mask). Parameter sengaja sedikit dan standar (tidak dioptimasi)."""
    s: Dict[str, Tuple[str, pd.Series]] = {}
    # 1. Aturan user dan mirror-nya
    s["EMA stack user: kuning di atas → SHORT"] = ("SHORT", g.stack_above_new)
    s["EMA stack mirror: kuning di bawah → LONG"] = ("LONG", g.stack_below_new)
    # 2. Bacaan tren normal
    s["EMA stack normal: kuning di atas → LONG"] = ("LONG", g.stack_above_new)
    s["EMA stack normal: kuning di bawah → SHORT"] = ("SHORT", g.stack_below_new)
    # 3. Tren MTF + pullback ke EMA50 H1 (H4 & Daily searah), close H1 kembali di atas/bawah EMA20
    s["Trend pullback EMA (H4+D) LONG"] = ("LONG", g.h4_up & g.d_up & (g.h1_low <= g.h1_e50) & (g.h1_close > g.h1_e20) & (g.h1_close > g.h1_open))
    s["Trend pullback EMA (H4+D) SHORT"] = ("SHORT", g.h4_down & g.d_down & (g.h1_high >= g.h1_e50) & (g.h1_close < g.h1_e20) & (g.h1_close < g.h1_open))
    # 4. RSI pullback dalam tren: RSI H1 naik lagi lewat 40 (LONG) / turun lewat 60 (SHORT)
    s["RSI pullback in trend LONG"] = ("LONG", g.h4_up & (g.h1_rsi > 40) & (g.h1_rsi_prev <= 40))
    s["RSI pullback in trend SHORT"] = ("SHORT", g.h4_down & (g.h1_rsi < 60) & (g.h1_rsi_prev >= 60))
    # 5. EMA20/50 H1 cross searah tren H4
    s["EMA cross H1 + H4 filter LONG"] = ("LONG", g.h4_up & (g.h1_e20 > g.h1_e50) & (g.h1_e20_prev <= g.h1_e50_prev))
    s["EMA cross H1 + H4 filter SHORT"] = ("SHORT", g.h4_down & (g.h1_e20 < g.h1_e50) & (g.h1_e20_prev >= g.h1_e50_prev))
    # 6. Bollinger mean-reversion saat H4 range: close H1 kembali ke dalam band
    s["BB mean-reversion (range) LONG"] = ("LONG", g.h4_range & (g.h1_close_prev < g.h1_bbl_prev) & (g.h1_close > g.h1_bbl))
    s["BB mean-reversion (range) SHORT"] = ("SHORT", g.h4_range & (g.h1_close_prev > g.h1_bbu_prev) & (g.h1_close < g.h1_bbu))
    # 7. Breakout H4 (high/low 20 candle) lalu retest dalam 12 jam, close H1 menolak level
    lvl_up = g.h4_hh20.shift(1).where(g.h4_close > g.h4_hh20.shift(1)).ffill(limit=12)
    lvl_dn = g.h4_ll20.shift(1).where(g.h4_close < g.h4_ll20.shift(1)).ffill(limit=12)
    s["Breakout + retest H4 LONG"] = ("LONG", (g.h4_close > g.h4_e200) & lvl_up.notna() & (g.h1_low <= lvl_up) & (g.h1_close > lvl_up))
    s["Breakout + retest H4 SHORT"] = ("SHORT", (g.h4_close < g.h4_e200) & lvl_dn.notna() & (g.h1_high >= lvl_dn) & (g.h1_close < lvl_dn))
    # 8. Sweep likuiditas H1 (tembus low/high 24 jam lalu close balik), dengan filter H4
    s["Sweep reversal + H4 filter LONG"] = ("LONG", g.h4_up & (g.h1_low < g.h1_ll24p) & (g.h1_close > g.h1_ll24p))
    s["Sweep reversal + H4 filter SHORT"] = ("SHORT", g.h4_down & (g.h1_high > g.h1_hh24p) & (g.h1_close < g.h1_hh24p))
    s["Sweep reversal tanpa filter LONG"] = ("LONG", (g.h1_low < g.h1_ll24p) & (g.h1_close > g.h1_ll24p))
    s["Sweep reversal tanpa filter SHORT"] = ("SHORT", (g.h1_high > g.h1_hh24p) & (g.h1_close < g.h1_hh24p))
    # Benchmark
    s["Benchmark acak LONG"] = ("LONG", pd.Series(True, index=g.index))
    s["Benchmark acak SHORT"] = ("SHORT", pd.Series(True, index=g.index))
    return s


def lab_trades(g: pd.DataFrame, m15: pd.DataFrame, side: str, mask: pd.Series, tp_r: float = 2.0,
               sl_atr: float = 1.0, hold_bars: int = 192) -> pd.DataFrame:
    ot = m15.open_time.values
    rows, busy = [], None
    sub = g[mask.fillna(False).astype(bool)]
    for r in sub.itertuples():
        if busy is not None and r.t < busy:
            continue
        px, atr_v = float(r.price), float(r.h4_atr)
        if not (atr_v > 0) or math.isnan(px):
            continue
        risk = sl_atr * atr_v
        s = 1 if side == "LONG" else -1
        i0 = int(np.searchsorted(ot, r.t.to_datetime64()))
        if i0 >= len(ot):
            break
        tp = px + s * tp_r * risk
        rr, _, out, ei = _exit_walk(m15, i0, side, px, px - s * risk, tp, tp, hold_bars)
        busy = pd.Timestamp(ot[ei]).tz_localize("UTC") + pd.Timedelta(minutes=15)
        rows.append({"entry_t": r.t, "exit_t": busy, "side": side, "r": rr - FEE_RT * px / risk, "outcome": out,
                     "d_up": bool(r.d_up)})
    return pd.DataFrame(rows)


def lab_metrics(tr: pd.DataFrame) -> Dict[str, Any]:
    ins = tr[tr.entry_t < LAB_SPLIT] if len(tr) else tr
    oos = tr[tr.entry_t >= LAB_SPLIT] if len(tr) else tr
    folds = []
    for a, b in LAB_FOLDS:
        part = tr[(tr.entry_t >= pd.Timestamp(a, tz="UTC")) & (tr.entry_t < pd.Timestamp(b, tz="UTC"))] if len(tr) else tr
        folds.append(metrics(part)["avg_r"])
    up = tr[tr.d_up] if len(tr) else tr
    return {"all": metrics(tr), "is": metrics(ins), "oos": metrics(oos), "folds": folds, "up": metrics(up)}


def bot_trades_1y() -> pd.DataFrame:
    """Bot sendiri (kode sekarang) di jendela yang sama, dari cache decisions_current."""
    m15 = pd.read_pickle(os.path.join(DATA, "15m.pkl"))
    dec = [d for d in load_decisions("current") if LAB_START <= d["t"] <= LAB_END]
    tr = simulate(dec, m15)
    if len(tr):
        tr["d_up"] = tr["f_bias_Daily"] > 0
    return tr


def strategies_report() -> None:
    m15 = pd.read_pickle(os.path.join(DATA, "15m.pkl"))
    g = lab_grid()
    sigs = lab_signals(g)
    results, curves = [], {}

    def add(name, side, tr, plan):
        m = lab_metrics(tr)
        results.append({"name": name, "side": side, "plan": plan, **{f"oos_{k}": v for k, v in m["oos"].items()},
                        **{f"is_{k}": v for k, v in m["is"].items()}, "all_n": m["all"]["n"], "all_tot": m["all"]["tot_r"],
                        "all_dd": m["all"]["maxdd"], "folds": m["folds"], "up_n": m["up"]["n"], "up_avg": m["up"]["avg_r"]})
        curves[f"{name} [{plan}]"] = tr

    for name, (side, mask) in sigs.items():
        add(name, side, lab_trades(g, m15, side, mask), "SL1ATR TP2R 48j")
    # variasi SL/TP untuk aturan user + mirror
    for name in ("EMA stack user: kuning di atas → SHORT", "EMA stack mirror: kuning di bawah → LONG"):
        side, mask = sigs[name]
        for sl in (0.75, 1.0, 1.5):
            for tp in (1.5, 2.0, 3.0):
                if sl == 1.0 and tp == 2.0:
                    continue
                add(name, side, lab_trades(g, m15, side, mask, tp_r=tp, sl_atr=sl), f"SL{sl:g}ATR TP{tp:g}R 48j")
    bot = bot_trades_1y()
    for side in ("LONG", "SHORT"):
        add(f"Bot sekarang ({side})", side, bot[bot.side == side] if len(bot) else bot, "zona bot")
    df = pd.DataFrame(results)
    df.to_pickle(os.path.join(DATA, "lab_results.pkl"))
    with open(os.path.join(DATA, "lab_curves.pkl"), "wb") as fh:
        pickle.dump(curves, fh)
    p0, p1 = float(g.price.iloc[0]), float(g.price.iloc[-1])
    print(f"Window {LAB_START.date()} → {LAB_END.date()} (IS < {LAB_SPLIT.date()} ≤ OOS). BTC {p0:,.0f} → {p1:,.0f} "
          f"({(p1 / p0 - 1) * 100:+.0f}%). Jam Daily bullish (close > EMA50 D): {g.d_up.mean() * 100:.0f}%")
    cols = ["name", "side", "plan", "is_n", "is_avg_r", "oos_n", "oos_wr", "oos_avg_r", "oos_pf", "oos_maxdd", "oos_tot_r", "up_n", "up_avg"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 48)
    for side in ("LONG", "SHORT"):
        part = df[df.side == side].sort_values("oos_avg_r", ascending=False)
        print(f"\n=== {side} (urut OOS avgR) ===")
        print(part[cols].round(2).to_string(index=False))
        print("folds:", {r.name + " [" + r.plan + "]": [round(x, 2) for x in r.folds] for r in part.head(5).itertuples()})


def lab_chart(names: List[str], out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(os.path.join(DATA, "lab_curves.pkl"), "rb") as fh:
        curves = pickle.load(fh)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
    for name in names:
        tr = curves[name]
        if len(tr) == 0:
            continue
        tr = tr.sort_values("exit_t")
        ax.step(tr.exit_t, tr.r.cumsum(), where="post", label=f"{name}: {len(tr)} trade, {tr.r.sum():+.1f}R",
                linestyle="--" if "Benchmark" in name else "-")
    ax.axvline(LAB_SPLIT, color="grey", linestyle="--", linewidth=1)
    ax.text(LAB_SPLIT, ax.get_ylim()[1], "  out-of-sample →", va="top", color="grey")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("BTCUSDT 1 tahun — equity strategi terbaik (R setelah fee 0.08%)")
    ax.set_ylabel("Kumulatif R")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out)


# ================================================================== panel strategi (strategies.py)
import json as _json  # noqa: E402

import strategies as S  # noqa: E402

PANEL_PERIODS = {
    "holdout": ("2025-08-01", "2025-11-10"),
    "is": ("2025-11-10", "2026-05-10"),
    "oos": ("2026-05-10", "2026-11-10"),
    "y1": ("2025-11-10", "2026-11-10"),
}
STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy_stats.json")


def panel_grids() -> Dict[str, pd.DataFrame]:
    data = _load()
    m5 = pd.read_pickle(os.path.join(DATA, "5m.pkl"))
    a, b = pd.Timestamp("2025-08-01", tz="UTC"), pd.Timestamp("2026-11-09 23:00", tz="UTC")
    return {
        "swing": S.swing_grid(data["1h"], data["4h"], data["1d"], pd.date_range(a, b, freq="1h")),
        "scalp15": S.scalp_grid(data["15m"], "15m", data["1h"], pd.date_range(a, b, freq="15min")),
        "scalp5": S.scalp_grid(m5, "5m", data["1h"], pd.date_range(a, b, freq="5min")),
    }


def strat_trades(g: pd.DataFrame, s: "S.Strategy", exit_df: pd.DataFrame, bar_minutes: int, fee_rt: float) -> pd.DataFrame:
    ot = exit_df.open_time.values
    hold = int(s.hold_h * 60 / bar_minutes)
    rows, busy = [], None
    sub = g[s.mask(g).fillna(False).astype(bool)]
    for r in sub.itertuples(index=False):
        t = r.t
        if busy is not None and t < busy:
            continue
        row = r._asdict()
        plan = S.plan_for(s, row)
        if plan is None:
            continue
        i0 = int(np.searchsorted(ot, t.to_datetime64()))
        if i0 >= len(ot):
            break
        rr, _, out, ei = _exit_walk(exit_df, i0, s.side, plan["entry"], plan["sl"], plan["tp"], plan["tp"], hold)
        busy = pd.Timestamp(ot[ei]).tz_localize("UTC") + pd.Timedelta(minutes=bar_minutes)
        risk = abs(plan["entry"] - plan["sl"])
        rows.append({"entry_t": t, "exit_t": busy, "side": s.side, "r": rr - fee_rt * plan["entry"] / risk,
                     "r_gross": rr, "outcome": out, "d_up": bool(row.get("d_up", False)), "risk_pct": plan["risk_pct"]})
    return pd.DataFrame(rows)


def _period(tr: pd.DataFrame, name: str) -> pd.DataFrame:
    a, b = PANEL_PERIODS[name]
    if len(tr) == 0:
        return tr
    return tr[(tr.entry_t >= pd.Timestamp(a, tz="UTC")) & (tr.entry_t < pd.Timestamp(b, tz="UTC"))]


def panel_report() -> None:
    import math as _math
    grids = panel_grids()
    m15 = pd.read_pickle(os.path.join(DATA, "15m.pkl"))
    m5 = pd.read_pickle(os.path.join(DATA, "5m.pkl"))
    rows, curves, stats = [], {}, {}
    bench = {}
    for side in ("LONG", "SHORT"):  # benchmark acak: plan swing standar tiap jam
        s = S.Strategy("bench", "Benchmark acak", side, "swing", lambda g: pd.Series(True, index=g.index), "acak")
        bench[side] = strat_trades(grids["swing"], s, m15, 15, FEE_RT)
    runs = [(s, FEE_RT, "") for s in S.STRATEGIES] + [(s, 0.0004, " (fee maker 0.04%)") for s in S.STRATEGIES if s.grid != "swing"]
    for s, fee, tag in runs:
        exit_df, bm = (m15, 15) if s.grid == "swing" else (m5, 5)
        tr = strat_trades(grids[s.grid], s, exit_df, bm, fee)
        m = {p: metrics(_period(tr, p)) for p in PANEL_PERIODS}
        y = _period(tr, "y1")
        tstat = float(y.r.mean() / (y.r.std() / _math.sqrt(len(y)))) if len(y) > 2 and y.r.std() > 0 else float("nan")
        up = metrics(y[y.d_up]) if len(y) else metrics(y)
        name = f"{s.name}{tag}"
        rows.append({"name": name, "side": s.side, "n": m["y1"]["n"], "wr": m["y1"]["wr"], "avg_r": m["y1"]["avg_r"],
                     "pf": m["y1"]["pf"], "maxdd": m["y1"]["maxdd"], "tot_r": m["y1"]["tot_r"],
                     "is_r": m["is"]["avg_r"], "oos_n": m["oos"]["n"], "oos_r": m["oos"]["avg_r"],
                     "hold_n": m["holdout"]["n"], "hold_r": m["holdout"]["avg_r"], "up_r": up["avg_r"], "t": tstat,
                     "fee_r": float((y.r_gross - y.r).mean()) if len(y) else float("nan")})
        curves[f"{name} {s.side}"] = y
        if not tag:
            alert = bool(m["oos"]["avg_r"] > 0 and m["oos"]["n"] >= 10 and m["y1"]["avg_r"] > 0)
            stats[S.stat_key(s)] = {
                "name": s.name, "side": s.side, "n": m["y1"]["n"], "wr": round(m["y1"]["wr"], 1),
                "avg_r": round(m["y1"]["avg_r"], 3), "pf": round(m["y1"]["pf"], 2) if np.isfinite(m["y1"]["pf"]) else None,
                "oos_n": m["oos"]["n"], "oos_avg_r": round(m["oos"]["avg_r"], 3),
                "holdout_n": m["holdout"]["n"], "holdout_avg_r": round(m["holdout"]["avg_r"], 3),
                "t": round(tstat, 2) if np.isfinite(tstat) else None, "alert": alert,
            }
    for side, tr in bench.items():
        m = {p: metrics(_period(tr, p)) for p in PANEL_PERIODS}
        rows.append({"name": "Benchmark acak", "side": side, "n": m["y1"]["n"], "wr": m["y1"]["wr"], "avg_r": m["y1"]["avg_r"],
                     "pf": m["y1"]["pf"], "maxdd": m["y1"]["maxdd"], "tot_r": m["y1"]["tot_r"], "is_r": m["is"]["avg_r"],
                     "oos_n": m["oos"]["n"], "oos_r": m["oos"]["avg_r"], "hold_n": m["holdout"]["n"],
                     "hold_r": m["holdout"]["avg_r"], "up_r": float("nan"), "t": float("nan"), "fee_r": float("nan")})
        curves[f"Benchmark acak {side}"] = _period(tr, "y1")
    stats["_meta"] = {"generated": pd.Timestamp.now(tz="UTC").isoformat(timespec="minutes"), "fee_rt": FEE_RT,
                      "periods": PANEL_PERIODS, "rule": "alert hanya jika avgR OOS > 0 (n OOS ≥ 10) dan avgR 1 tahun > 0",
                      "plan_swing": "SL 1×ATR H4, TP 2R, maks 48 jam", "plan_scalp": "SL 1×ATR TF, TP 1.5R, maks 4j (15m) / 2j (5m)"}
    with open(STATS_PATH, "w") as fh:
        _json.dump(stats, fh, indent=1, ensure_ascii=False)
    df = pd.DataFrame(rows)
    df.to_pickle(os.path.join(DATA, "panel_results.pkl"))
    with open(os.path.join(DATA, "panel_curves.pkl"), "wb") as fh:
        pickle.dump(curves, fh)
    pd.set_option("display.width", 260)
    for side in ("LONG", "SHORT"):
        print(f"\n=== {side} (urut avgR OOS) ===")
        print(df[df.side == side].sort_values("oos_r", ascending=False).round(2).to_string(index=False))
    print("\nstats →", STATS_PATH)


# ------------------------------------------------------------------ sniper backtest
import market_tools as MT  # noqa: E402


def _sniper_at(t: pd.Timestamp) -> Dict[str, Any]:
    bundle = bundle_at(t)
    a, dfs, px = bundle["analyses"], bundle["dfs"], bundle["price"]
    liq = [MT.liquidity_map(dfs["H1"], "H1", px), MT.liquidity_map(dfs["H4"], "H4", px)]
    return {"t": t, "price": px, "direction": B.htf_direction(a),
            "LONG": MT.sniper_plan(a, dfs, px, "LONG", liq), "SHORT": MT.sniper_plan(a, dfs, px, "SHORT", liq)}


def _resl(p: Optional[Dict[str, Any]], sl_atr_h4: float) -> Optional[Dict[str, Any]]:
    """Variasi SL struktural: buffer k×ATR H4 di balik zona; TP1/TP2 dihitung ulang 2R/3R kalau terlalu dekat."""
    if p is None or sl_atr_h4 <= 0:
        return p
    q = dict(p)
    long = q["side"] == "LONG"
    q["sl"] = q["entry_low"] - sl_atr_h4 * q["atr_h4"] if long else q["entry_high"] + sl_atr_h4 * q["atr_h4"]
    risk = abs(q["entry_mid"] - q["sl"])
    s = 1 if long else -1
    if s * (q["tp1"] - q["entry_mid"]) < 1.5 * risk:
        q["tp1"] = q["entry_mid"] + s * 2 * risk
    q["tp2"] = q["entry_mid"] + s * 3 * risk
    q["risk_pct"] = risk / q["entry_mid"] * 100
    return q


def sniper_backtest(valid_h: int = 12, fee_rt: float = FEE_RT, sl_atr_h4: float = 0.0) -> Dict[str, pd.DataFrame]:
    cache = os.path.join(DATA, "sniper_plans.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            plans = pickle.load(fh)
    else:
        data = _load()
        times = list(pd.date_range("2025-08-01", "2026-11-09", freq="4h", tz="UTC"))
        with Pool(os.cpu_count() or 2, initializer=_init, initargs=(data,)) as pool:
            plans = pool.map(_sniper_at, times, chunksize=16)
        with open(cache, "wb") as fh:
            pickle.dump(plans, fh)
    m15 = pd.read_pickle(os.path.join(DATA, "15m.pkl"))
    ot, hi, lo = m15.open_time.values, m15.high.values, m15.low.values
    out = {}
    for mode in ("with", "against"):
        for side in ("LONG", "SHORT"):
            rows, busy = [], None
            for pl in plans:
                p = _resl(pl[side], sl_atr_h4)
                wt = (side == "LONG" and pl["direction"] == "BULLISH") or (side == "SHORT" and pl["direction"] == "BEARISH")
                if p is None or (mode == "with") != wt or (busy is not None and pl["t"] < busy):
                    continue
                i0 = int(np.searchsorted(ot, pl["t"].to_datetime64()))
                i1 = min(len(ot), i0 + valid_h * 4)
                fill_i = None
                for i in range(i0, i1):
                    if (side == "LONG" and hi[i] >= p["tp1"]) or (side == "SHORT" and lo[i] <= p["tp1"]):
                        break  # lari ke TP1 dulu → batal
                    if (side == "LONG" and lo[i] <= p["entry_mid"]) or (side == "SHORT" and hi[i] >= p["entry_mid"]):
                        fill_i = i
                        break
                if fill_i is None:
                    continue
                r, _, outc, ei = _exit_walk(m15, fill_i, side, p["entry_mid"], p["sl"], p["tp1"], p["tp2"], 192)
                risk = abs(p["entry_mid"] - p["sl"])
                busy = pd.Timestamp(ot[ei]).tz_localize("UTC") + pd.Timedelta(minutes=15)
                rows.append({"entry_t": pd.Timestamp(ot[fill_i]).tz_localize("UTC"), "exit_t": busy, "side": side,
                             "r": r - fee_rt * p["entry_mid"] / risk, "outcome": outc, "risk_pct": p["risk_pct"]})
            out[f"{mode}:{side}"] = pd.DataFrame(rows)
    return out


def sniper_report() -> None:
    # SNIPER_VARIANTS=1 + cache dari plan SL ketat (SNIPER_SL_ATR_H4=0) → tabel variasi SL.
    for k in ((0.0, 0.5, 1.0) if os.getenv("SNIPER_VARIANTS") == "1" else ()):
        for key, tr in sniper_backtest(sl_atr_h4=k).items():
            m = {p: metrics(_period(tr, p)) for p in PANEL_PERIODS}
            print(f"  variasi SL {'0.5×ATR H1' if k == 0 else f'{k:g}×ATR H4':10s} {key:13s} 1y n={m['y1']['n']:3d} "
                  f"WR={m['y1']['wr']:4.0f}% avgR={m['y1']['avg_r']:+.2f} | IS {m['is']['avg_r']:+.2f} OOS {m['oos']['avg_r']:+.2f} holdout {m['holdout']['avg_r']:+.2f}")
    sl_k = 0.0  # plan sudah dibuat dengan SL yang sama seperti bot live (env SNIPER_SL_ATR_H4)
    res = sniper_backtest(sl_atr_h4=sl_k)
    res_maker = sniper_backtest(fee_rt=0.0004, sl_atr_h4=sl_k)
    stats = load_stats_file()
    for k, tr in res.items():
        m = {p: metrics(_period(tr, p)) for p in PANEL_PERIODS}
        mk = metrics(_period(res_maker[k], "y1"))
        print(f"sniper {k:13s} 1y {fmt_m(m['y1'])} | IS {m['is']['avg_r']:+.2f} OOS {m['oos']['avg_r']:+.2f} (n{m['oos']['n']}) "
              f"holdout {m['holdout']['avg_r']:+.2f} (n{m['holdout']['n']}) | maker-fee avgR {mk['avg_r']:+.2f}")
        if k.startswith("with"):
            stats[f"sniper:{k.split(':')[1]}"] = {"name": "Sniper limit (searah tren)", "side": k.split(":")[1],
                                                  "n": m["y1"]["n"], "wr": round(m["y1"]["wr"], 1), "avg_r": round(m["y1"]["avg_r"], 3),
                                                  "oos_n": m["oos"]["n"], "oos_avg_r": round(m["oos"]["avg_r"], 3),
                                                  "holdout_n": m["holdout"]["n"], "holdout_avg_r": round(m["holdout"]["avg_r"], 3),
                                                  "alert": bool(m["oos"]["avg_r"] > 0 and m["oos"]["n"] >= 10 and m["y1"]["avg_r"] > 0)}
    with open(STATS_PATH, "w") as fh:
        _json.dump(stats, fh, indent=1, ensure_ascii=False)
    with open(os.path.join(DATA, "sniper_curves.pkl"), "wb") as fh:
        pickle.dump({k: _period(v, "y1") for k, v in res.items()}, fh)


def load_stats_file() -> Dict[str, Any]:
    try:
        with open(STATS_PATH) as fh:
            return _json.load(fh)
    except (OSError, ValueError):
        return {}


def panel_chart(out: str = "/workspace/artifacts/strategies_compare.png") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(os.path.join(DATA, "panel_curves.pkl"), "rb") as fh:
        curves = pickle.load(fh)
    with open(os.path.join(DATA, "sniper_curves.pkl"), "rb") as fh:
        sn = pickle.load(fh)
    pick = [("EMA Stack 7/25/99 (aturan kamu) SHORT", "-"), ("Momentum SHORT", "-"), ("Trend Following LONG", "-"),
            ("Pullback SHORT", "-"), ("Price Action (wick liquidity) LONG", "-"), ("Scalping Momentum 15m SHORT", ":"),
            ("Benchmark acak SHORT", "--"), ("Benchmark acak LONG", "--")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=110, gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    for name, ls in pick:
        tr = curves.get(name)
        if tr is None or len(tr) == 0:
            continue
        tr = tr.sort_values("exit_t")
        ax.step(tr.exit_t, tr.r.cumsum(), where="post", linestyle=ls, label=f"{name}: {len(tr)} trade, {tr.r.sum():+.1f}R")
    for k, tr in sn.items():
        if k.startswith("with") and len(tr):
            tr = tr.sort_values("exit_t")
            ax.step(tr.exit_t, tr.r.cumsum(), where="post", linestyle="-.", label=f"Sniper searah tren {k[5:]}: {len(tr)} trade, {tr.r.sum():+.1f}R")
    ax.axvline(pd.Timestamp(PANEL_PERIODS["oos"][0], tz="UTC"), color="grey", linestyle="--", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylim(-60, 30)
    ax.set_title("Equity 1 tahun (R setelah fee 0.08%) — scalping 15m dipotong (total di legend)")
    ax.legend(loc="lower left", fontsize=7)
    ax.grid(alpha=0.3)
    df = pd.read_pickle(os.path.join(DATA, "panel_results.pkl"))
    df = df[~df.name.str.contains("maker")].copy()
    df["label"] = df.name.str.replace(r" \(.*\)", "", regex=True).str.replace("Scalping Momentum", "Scalp") + " " + df.side
    df = df.sort_values("oos_r")
    ax2 = axes[1]
    colors = ["#2e7d32" if (o > 0 and y > 0) else "#c62828" for o, y in zip(df.oos_r, df.avg_r)]
    ax2.barh(df.label, df.oos_r.clip(lower=-1), color=colors)
    ax2.axvline(0, color="black", linewidth=0.6)
    ax2.set_title("avgR OOS per strategi (hijau = OOS & 1 tahun > 0)")
    ax2.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    tag = sys.argv[2] if len(sys.argv) > 2 else "current"
    if cmd == "decisions":
        import time
        t0 = time.time()
        d = compute_decisions(tag)
        print("decisions", tag, len(d), f"{time.time() - t0:.0f}s")
    elif cmd == "report":
        report()
    elif cmd == "strategies":
        strategies_report()
    elif cmd == "panel":
        panel_report()
    elif cmd == "sniper":
        sniper_report()
    elif cmd == "panel_chart":
        panel_chart()
