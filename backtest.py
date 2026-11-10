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
from typing import Any, Dict, List, Optional

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
