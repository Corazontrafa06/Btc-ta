"""
Fitur tambahan bot: wick liquidity, saran sniper, EMA harian, berita + kalender, dan render panel strategi.
Fungsi analisis murni (tanpa jaringan) supaya bisa dites dan di-backtest; fetch berita terpisah dan aman gagal.
"""
from __future__ import annotations

import asyncio
import html
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import strategies as S

WIB = timezone(timedelta(hours=7))
STATS_PATH = os.getenv("STRATEGY_STATS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy_stats.json"))


def fp(x: float) -> str:
    return f"{x:,.0f}"


def wib(dt: datetime) -> str:
    return dt.astimezone(WIB).strftime("%d %b %H:%M WIB")


# ------------------------------------------------------------------ swing helper

def _swings(df: pd.DataFrame, order: int) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    hs, ls = [], []
    for i in range(order, len(df) - order):
        if h[i] == h[i - order:i + order + 1].max():
            hs.append((i, float(h[i])))
        if l[i] == l[i - order:i + order + 1].min():
            ls.append((i, float(l[i])))
    return hs, ls


# ------------------------------------------------------------------ wick liquidity

def liquidity_map(df: pd.DataFrame, label: str, price: float, order: int = 3, look_sweep: int = 12,
                  equal_pct: float = 0.001) -> Dict[str, Any]:
    """Pool likuiditas yang BELUM tersentuh (swing high/low) + equal highs/lows + sweep terbaru.
    df berisi candle forming di baris terakhir; analisis hanya pakai candle closed."""
    closed = df.iloc[:-1].reset_index(drop=True) if len(df) > 2 else df.reset_index(drop=True)
    H, L, C, O = (closed[c].to_numpy() for c in ("high", "low", "close", "open"))
    n = len(closed)
    highs, lows = _swings(closed, order)

    def untapped(points, is_high):
        out = []
        for idx, lv in points:
            after = H[idx + 1:] if is_high else L[idx + 1:]
            # toleransi equal_pct: high berikutnya yang hampir sama = equal highs, bukan "tersentuh"
            if len(after) and ((after.max() > lv * (1 + equal_pct)) if is_high else (after.min() < lv * (1 - equal_pct))):
                continue
            out.append((idx, lv))
        return out

    def pools(points, above):
        pts = sorted(untapped(points, above), key=lambda x: x[1])
        groups: List[List[Tuple[int, float]]] = []
        for p in pts:
            if groups and abs(p[1] - groups[-1][-1][1]) / p[1] <= equal_pct:
                groups[-1].append(p)
            else:
                groups.append([p])
        res = []
        for g in groups:
            lv = max(x[1] for x in g) if above else min(x[1] for x in g)
            if (above and lv <= price) or (not above and lv >= price):
                continue
            res.append({"price": lv, "tf": label, "touches": len(g), "equal": len(g) >= 2,
                        "age": n - 1 - max(x[0] for x in g), "dist_pct": (lv - price) / price * 100})
        res.sort(key=lambda r: abs(r["dist_pct"]))
        return res

    sweeps = []
    for i in range(max(order * 2 + 1, n - look_sweep), n):
        rng = max(H[i] - L[i], 1e-9)
        # level yang sudah terkonfirmasi sebelum candle i dan belum tersentuh sampai i-1
        ph = [lv for idx, lv in highs if idx + order < i and (H[idx + 1:i].max() if i - idx > 1 else -np.inf) < lv]
        pl = [lv for idx, lv in lows if idx + order < i and (L[idx + 1:i].min() if i - idx > 1 else np.inf) > lv]
        hit_h = [lv for lv in ph if H[i] > lv and C[i] < lv]
        hit_l = [lv for lv in pl if L[i] < lv and C[i] > lv]
        if hit_h and (H[i] - max(O[i], C[i])) / rng >= 0.4:
            sweeps.append({"side": "SHORT", "kind": "sapu high", "level": max(hit_h), "wick": float(H[i]), "tf": label,
                           "time": closed["open_time"].iloc[i], "bars_ago": n - 1 - i})
        if hit_l and (min(O[i], C[i]) - L[i]) / rng >= 0.4:
            sweeps.append({"side": "LONG", "kind": "sapu low", "level": min(hit_l), "wick": float(L[i]), "tf": label,
                           "time": closed["open_time"].iloc[i], "bars_ago": n - 1 - i})
    return {"above": pools(highs, True), "below": pools(lows, False), "sweeps": sweeps}


def format_liquidity(maps: Sequence[Dict[str, Any]], price: float) -> str:
    above = sorted([p for m in maps for p in m["above"]], key=lambda p: p["price"])[:4]
    below = sorted([p for m in maps for p in m["below"]], key=lambda p: -p["price"])[:4]
    sweeps = sorted([s for m in maps for s in m["sweeps"]], key=lambda s: s["time"], reverse=True)[:4]
    lines = ["💧 <b>WICK LIQUIDITY</b>", f"Harga: <b>${fp(price)}</b>", "",
             "<b>Pool di atas</b> (stop short / buy stop, belum tersentuh):"]
    lines += [f"  • ${fp(p['price'])} ({p['dist_pct']:+.1f}%) · {p['tf']}" + (" · <b>equal highs</b>" if p["equal"] else "")
              for p in above] or ["  • tidak ada dekat harga"]
    lines.append("<b>Pool di bawah</b> (stop long / sell stop, belum tersentuh):")
    lines += [f"  • ${fp(p['price'])} ({p['dist_pct']:+.1f}%) · {p['tf']}" + (" · <b>equal lows</b>" if p["equal"] else "")
              for p in below] or ["  • tidak ada dekat harga"]
    lines.append("")
    lines.append("<b>Sweep terbaru</b> (wick tembus lalu close balik):")
    if sweeps:
        for s in sweeps:
            arrow = "🔴" if s["side"] == "SHORT" else "🟢"
            lines.append(f"  {arrow} {s['tf']} {s['kind']} ${fp(s['level'])} (wick {fp(s['wick'])}) · {wib(s['time'].to_pydatetime())}")
    else:
        lines.append("  • belum ada di candle terakhir")
    lines.append("<i>Pool = kumpulan stop di atas high / bawah low. Harga sering 'menyapu' pool lalu berbalik.</i>")
    return "\n".join(lines)


# ------------------------------------------------------------------ EMA harian

def daily_ema(df_1d: pd.DataFrame, price: float) -> Dict[str, Any]:
    closed = df_1d.iloc[:-1] if len(df_1d) > 2 else df_1d
    c = closed["close"]
    vals = {n: float(c.ewm(span=n, adjust=False).mean().iloc[-1]) for n in (7, 25, 99, 20, 50, 200)}
    e7, e25, e99 = vals[7], vals[25], vals[99]
    stack = "ABOVE" if e7 > e25 and e7 > e99 else ("BELOW" if e7 < e25 and e7 < e99 else "MIXED")
    e20, e50, e200 = vals[20], vals[50], vals[200]
    if e20 > e50 > e200 and price > e20:
        trend = "STRONG BULLISH"
    elif e20 < e50 < e200 and price < e20:
        trend = "STRONG BEARISH"
    elif price > e50 and e20 > e50:
        trend = "BULLISH"
    elif price < e50 and e20 < e50:
        trend = "BEARISH"
    else:
        trend = "NETRAL / TRANSISI"
    above = sorted([(v, n) for n, v in vals.items() if v > price])
    below = sorted([(v, n) for n, v in vals.items() if v < price], reverse=True)
    return {"vals": vals, "stack": stack, "trend": trend, "res": above, "sup": below, "price": price,
            "rsi": float(S._rsi(c).iloc[-1])}


def format_daily(d: Dict[str, Any], stack_stat: Optional[Dict[str, Any]] = None) -> str:
    v, price = d["vals"], d["price"]
    def row(n, color=""):
        pct = (price - v[n]) / v[n] * 100
        return f"{color}EMA{n}: ${fp(v[n])} (harga {abs(pct):.1f}% {'di atas' if pct >= 0 else 'di bawah'})"
    stack_txt = {"ABOVE": "kuning DI ATAS biru & pink", "BELOW": "kuning DI BAWAH biru & pink", "MIXED": "campur"}[d["stack"]]
    lines = ["📅 <b>SARAN EMA HARIAN (Daily)</b>", f"Harga: <b>${fp(price)}</b> · RSI D {d['rsi']:.0f}", "",
             "<b>EMA 7 / 25 / 99</b> (warna Bybit)", row(7, "🟡 "), row(25, "🔵 "), row(99, "🟣 "),
             f"Stack: {stack_txt}", "", "<b>EMA 20 / 50 / 200</b>", row(20), row(50), row(200),
             f"Tren harian: <b>{d['trend']}</b>", ""]
    if d["res"]:
        lines.append("Resistance dinamis: " + " · ".join(f"EMA{n} ${fp(x)}" for x, n in d["res"][:3]))
    if d["sup"]:
        lines.append("Support dinamis: " + " · ".join(f"EMA{n} ${fp(x)}" for x, n in d["sup"][:3]))
    lines.append("")
    t = d["trend"]
    if "BEARISH" in t:
        tip = (f"Bias harian turun. Lebih aman cari SHORT saat harga pantul ke resistance dinamis"
               f"{' (EMA' + str(d['res'][0][1]) + ' $' + fp(d['res'][0][0]) + ')' if d['res'] else ''}; LONG hanya scalp cepat.")
    elif "BULLISH" in t:
        tip = (f"Bias harian naik. Lebih aman cari LONG saat harga turun ke support dinamis"
               f"{' (EMA' + str(d['sup'][0][1]) + ' $' + fp(d['sup'][0][0]) + ')' if d['sup'] else ''}; SHORT hanya scalp cepat.")
    else:
        tip = "Harian belum jelas. Kurangi ukuran posisi, tunggu close Daily di atas/bawah EMA50."
    lines.append(f"💡 {tip}")
    if d["stack"] == "ABOVE":
        lines.append("Aturan kamu (EMA stack) di Daily: kuning di atas → waspada koreksi.")
    if stack_stat:
        lines.append(f"<i>Backtest EMA stack SHORT (H1/H3/H4): WR {stack_stat['wr']:.0f}% · {stack_stat['avg_r']:+.2f}R/trade</i>")
    lines.append("<i>Bukan saran finansial.</i>")
    return "\n".join(lines)


# ------------------------------------------------------------------ sniper

def sniper_plan(analyses: Dict[str, Dict[str, Any]], dfs: Dict[str, pd.DataFrame], price: float, side: str,
                liq: Optional[Sequence[Dict[str, Any]]] = None, max_atr_h4: float = 3.0,
                sl_atr_h4: float = float(os.getenv("SNIPER_SL_ATR_H4", "1.0"))) -> Optional[Dict[str, Any]]:
    """Zona limit sempit di confluence S/R + EMA + likuiditas. SL kecil di balik zona, TP ke level lawan."""
    h1, h4 = analyses.get("H1", {}), analyses.get("H4", {})
    atr1 = float(h1.get("atr") or 0)
    atr4 = float(h4.get("atr") or 0)
    if atr1 <= 0 or atr4 <= 0:
        return None
    long = side == "LONG"
    cands: List[Tuple[float, float, str]] = []
    for tf, w in (("H1", 1.0), ("H4", 2.0), ("Daily", 2.5)):
        a = analyses.get(tf) or {}
        for lv in (a.get("supports") if long else a.get("resistances")) or []:
            cands.append((float(lv), w, f"S/R {tf}" if tf != "Daily" else "S/R D"))
    for tf, w in (("H1", 0.75), ("H4", 1.25), ("Daily", 1.75)):
        a = analyses.get(tf) or {}
        for n in (20, 50, 200):
            lv = a.get(f"ema{n}")
            if lv:
                cands.append((float(lv), w, f"EMA{n} {tf if tf != 'Daily' else 'D'}"))
    for m in liq or []:
        for p in (m["below"] if long else m["above"]):
            cands.append((float(p["price"]), 1.5 + (1.0 if p["equal"] else 0), f"liq {p['tf']}" + (" equal" if p["equal"] else "")))
    lo_b, hi_b = (price - max_atr_h4 * atr4, price - 0.25 * atr1) if long else (price + 0.25 * atr1, price + max_atr_h4 * atr4)
    cands = sorted(c for c in cands if lo_b <= c[0] <= hi_b)
    if not cands:
        return None
    groups: List[List[Tuple[float, float, str]]] = []
    for c in cands:
        if groups and c[0] - groups[-1][-1][0] <= 0.25 * atr1:
            groups[-1].append(c)
        else:
            groups.append([c])
    def score(g):
        return sum(x[1] for x in g) + 0.5 * (len({x[2].split()[0] for x in g}) - 1)
    best = max(groups, key=lambda g: (score(g), -abs(np.mean([x[0] for x in g]) - price)))
    lvls = [x[0] for x in best]
    centre = float(np.average(lvls, weights=[x[1] for x in best]))
    lo, hi = min(lvls), max(lvls)
    if hi - lo > 0.4 * atr1:
        lo, hi = centre - 0.2 * atr1, centre + 0.2 * atr1
    if hi - lo < 0.1 * atr1:
        lo, hi = centre - 0.05 * atr1, centre + 0.05 * atr1
    mid = (lo + hi) / 2
    buf = sl_atr_h4 * atr4 if sl_atr_h4 > 0 else 0.5 * atr1
    sl = lo - buf if long else hi + buf
    risk = abs(mid - sl)
    opp: List[float] = []
    for tf in ("H1", "H4", "Daily"):
        a = analyses.get(tf) or {}
        opp += [float(x) for x in ((a.get("resistances") if long else a.get("supports")) or [])]
    for m in liq or []:
        opp += [float(p["price"]) for p in (m["above"] if long else m["below"])]
    opp = sorted(x for x in opp if (x >= mid + 1.5 * risk if long else x <= mid - 1.5 * risk))
    if not long:
        opp = sorted(opp, reverse=True)
    tp1 = opp[0] if opp else (mid + 2 * risk if long else mid - 2 * risk)
    if abs(tp1 - mid) > 4 * risk:
        tp1 = mid + 2 * risk if long else mid - 2 * risk
    tp2 = mid + 3 * risk if long else mid - 3 * risk
    if (long and tp2 <= tp1) or (not long and tp2 >= tp1):
        tp2 = tp1
    return {"side": side, "entry_low": lo, "entry_high": hi, "entry_mid": mid, "sl": sl, "tp1": tp1, "tp2": tp2,
            "rr1": abs(tp1 - mid) / risk, "rr2": abs(tp2 - mid) / risk, "risk_pct": risk / mid * 100,
            "score": round(score(best), 1), "sources": sorted({x[2] for x in best}),
            "dist_pct": (mid - price) / price * 100, "atr_h1": atr1, "atr_h4": atr4,
            "sl_tight": lo - 0.5 * atr1 if long else hi + 0.5 * atr1}


def format_sniper(plans: Dict[str, Optional[Dict[str, Any]]], price: float, direction: str,
                  stat: Optional[Dict[str, Any]] = None) -> str:
    lines = ["🎯 <b>SARAN SNIPER (limit order)</b>", f"Harga: <b>${fp(price)}</b> · arah HTF: <b>{direction}</b>", ""]
    for side in ("SHORT", "LONG") if direction != "BULLISH" else ("LONG", "SHORT"):
        p = plans.get(side)
        with_trend = (side == "LONG" and direction == "BULLISH") or (side == "SHORT" and direction == "BEARISH")
        tag = "✅ searah tren" if with_trend else ("⚠️ melawan tren, ukuran kecil" if direction != "NEUTRAL" else "↔️ HTF netral")
        icon = "🔴" if side == "SHORT" else "🟢"
        if not p:
            lines += [f"{icon} <b>{side}</b>: tidak ada confluence dalam 3 ATR H4", ""]
            continue
        lines += [
            f"{icon} <b>{side} limit</b> · {tag}",
            f"   Zona : {fp(p['entry_low'])} – {fp(p['entry_high'])} ({p['dist_pct']:+.1f}%)",
            f"   SL   : {fp(p['sl'])} ({p['risk_pct']:.2f}%, 1×ATR H4)  ·  SL ketat: {fp(p['sl_tight'])}",
            f"   TP1  : {fp(p['tp1'])} (1:{p['rr1']:.1f})  ·  TP2: {fp(p['tp2'])} (1:{p['rr2']:.1f})",
            f"   Confluence ({p['score']}): {', '.join(p['sources'][:5])}",
            "",
        ]
    if stat:
        lines.append(f"<i>Backtest sniper searah tren (SL 1×ATR H4): {stat.get('n', 0)} trade, WR {stat.get('wr', 0):.0f}%, "
                     f"{stat.get('avg_r', 0):+.2f}R/trade (OOS {stat.get('oos_avg_r', 0):+.2f}R) ≈ impas, belum ada edge.</i>")
    lines.append("<i>SL ketat (0.5 ATR H1) di backtest sering tersapu: WR 13–18%, -0.5 s/d -0.8R/trade.</i>")
    lines.append("<i>Limit order: batal kalau harga sudah lari ke TP1 sebelum zona terisi. Bukan saran finansial.</i>")
    return "\n".join(lines)


# ------------------------------------------------------------------ berita + kalender

NEWS_FEEDS = [
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Cointelegraph", "https://cointelegraph.com/rss"),
    ("The Block", "https://www.theblock.co/rss.xml"),
    ("Decrypt", "https://decrypt.co/feed"),
]
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_TAGS = {
    "bullish": r"approv|inflow|rate cut|cuts? rates|adopt|record high|all-time high|\bath\b|surge|rall(y|ies)|soar|accumulat|reserve|buyback|bullish",
    "bearish": r"hack|exploit|outflow|\bban\b|lawsuit|sues|crash|plunge|tumbl|liquidat|sell-?off|rate hike|tariff|bankrupt|bearish|slump|drain",
    "risiko": r"\bfomc\b|\bcpi\b|\bfed\b|powell|payroll|\bnfp\b|jobs report|inflation|regulat|\bsec\b|treasury|yield|recession|geopolit|war\b",
}
_BTC_RE = re.compile(r"bitcoin|\bbtc\b|crypto|etf|fed|market|stablecoin|macro|treasury|binance|liquidat", re.I)


def tag_headline(title: str) -> List[str]:
    t = title.lower()
    return [k for k, pat in _TAGS.items() if re.search(pat, t)]


def parse_rss(text: str, source: str) -> List[Dict[str, Any]]:
    items = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return items
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        pub = it.findtext("pubDate") or ""
        try:
            dt = parsedate_to_datetime(pub)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
        if title:
            items.append({"title": html.unescape(title), "link": link, "time": dt, "source": source, "tags": tag_headline(title)})
    return items


def parse_calendar(data: Any, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    now = now or datetime.now(timezone.utc)
    out = []
    if not isinstance(data, list):
        return out
    for ev in data:
        try:
            if ev.get("country") != "USD" or ev.get("impact") != "High":
                continue
            dt = datetime.fromisoformat(ev["date"])
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
        if dt >= now - timedelta(hours=6):
            out.append({"title": ev.get("title", ""), "time": dt, "forecast": ev.get("forecast", ""), "previous": ev.get("previous", "")})
    return sorted(out, key=lambda e: e["time"])


async def fetch_news(session, timeout: float = 12.0) -> Dict[str, Any]:
    import aiohttp
    headers = {"User-Agent": "Mozilla/5.0 (compatible; BTC-TA-Bot/2.0)"}

    async def get(url):
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                if r.status == 200:
                    return await r.text()
        except Exception:
            return None
        return None

    texts = await asyncio.gather(*(get(u) for _, u in NEWS_FEEDS), get(CALENDAR_URL))
    items, ok_src = [], []
    for (name, _), txt in zip(NEWS_FEEDS, texts[:-1]):
        if txt:
            parsed = parse_rss(txt, name)
            if parsed:
                ok_src.append(name)
                items += parsed
    seen, uniq = set(), []
    for it in sorted(items, key=lambda x: x["time"], reverse=True):
        k = re.sub(r"\W+", "", it["title"].lower())[:60]
        if k not in seen:
            seen.add(k)
            uniq.append(it)
    cal = []
    if texts[-1]:
        try:
            cal = parse_calendar(json.loads(texts[-1]))
        except ValueError:
            cal = []
    return {"items": uniq, "sources": ok_src, "calendar": cal, "calendar_ok": texts[-1] is not None,
            "fetched": datetime.now(timezone.utc)}


def format_news(news: Dict[str, Any], max_items: int = 8, hours: int = 24) -> str:
    now = news.get("fetched") or datetime.now(timezone.utc)
    lines = ["📰 <b>BERITA &amp; KALENDER</b>", ""]
    cal = news.get("calendar") or []
    lines.append("<b>📆 Event USD high-impact</b> (ForexFactory)")
    if not news.get("calendar_ok"):
        lines.append("  • kalender gagal diambil")
    elif not cal:
        lines.append("  • tidak ada sisa minggu ini")
    for ev in cal[:6]:
        when = "sudah lewat" if ev["time"] < now else f"{(ev['time'] - now).total_seconds() / 3600:.0f} jam lagi"
        extra = " · ".join(x for x in (f"f {ev['forecast']}" if ev["forecast"] else "", f"p {ev['previous']}" if ev["previous"] else "") if x)
        lines.append(f"  ⚠️ {wib(ev['time'])} — {html.escape(ev['title'])} ({when}){' · ' + extra if extra else ''}")
    lines.append("")
    recent = [it for it in news.get("items", []) if it["time"] >= now - timedelta(hours=hours) and _BTC_RE.search(it["title"])]
    lines.append(f"<b>🗞 Headline {hours} jam terakhir</b> ({', '.join(news.get('sources') or []) or 'semua sumber gagal'})")
    if not recent:
        lines.append("  • tidak ada / sumber gagal")
    icons = {"bullish": "🟢", "bearish": "🔴", "risiko": "⚠️"}
    for it in recent[:max_items]:
        tag = " ".join(icons[t] for t in it["tags"]) or "⚪"
        title = html.escape(it["title"][:140])
        link = html.escape(it["link"], quote=True)
        lines.append(f"{tag} {it['time'].astimezone(WIB).strftime('%H:%M')} · <a href=\"{link}\">{title}</a> <i>({it['source']})</i>")
    counts = {k: sum(1 for it in recent if k in it["tags"]) for k in icons}
    lines += ["", f"Ringkas {len(recent)} headline: 🟢 {counts['bullish']} · 🔴 {counts['bearish']} · ⚠️ {counts['risiko']}",
              "<i>Tag 🟢/🔴/⚠️ = heuristik kata kunci, bukan analisis sentimen. Cek sumbernya.</i>"]
    return "\n".join(lines)


def news_summary_line(news: Optional[Dict[str, Any]]) -> str:
    if not news:
        return "Berita: belum diambil (/berita)"
    now = news.get("fetched") or datetime.now(timezone.utc)
    nxt = [e for e in news.get("calendar") or [] if e["time"] >= now]
    ev = f"event berikut {wib(nxt[0]['time'])} {html.escape(nxt[0]['title'])}" if nxt else "tidak ada event USD high-impact tersisa"
    recent = [it for it in news.get("items", []) if it["time"] >= now - timedelta(hours=24) and _BTC_RE.search(it["title"])]
    b = sum(1 for it in recent if "bullish" in it["tags"])
    r = sum(1 for it in recent if "bearish" in it["tags"])
    return f"Berita 24j: 🟢{b} 🔴{r} · {ev}"


# ------------------------------------------------------------------ panel strategi

def load_stats(path: str = STATS_PATH) -> Dict[str, Any]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def alert_mode(key: str, stats: Dict[str, Any], watch: Sequence[str]) -> str:
    """'alert' (OOS positif), 'pantauan' (watch-only yang diminta), atau 'info'."""
    st = stats.get(key) or {}
    if st.get("alert"):
        return "alert"
    if key in watch:
        return "pantauan"
    return "info"


def _stat_txt(st: Optional[Dict[str, Any]]) -> str:
    if not st or not st.get("n"):
        return "backtest: -"
    return f"WR {st['wr']:.0f}% · {st['avg_r']:+.2f}R · OOS {st['oos_avg_r']:+.2f}R · n{st['n']}"


def format_strategy_panel(status: List[Dict[str, Any]], stats: Dict[str, Any], watch: Sequence[str],
                          families: Optional[Sequence[str]] = None, title: str = "🧩 <b>PANEL STRATEGI</b>") -> str:
    now = datetime.now(timezone.utc)
    by_fam: Dict[str, List[Dict[str, Any]]] = {}
    for st in status:
        by_fam.setdefault(st["strategy"].key, []).append(st)
    lines = [title, "<i>Status di candle closed terakhir. Angka = backtest 1 tahun (fee 0.08%).</i>",
             "🔔 = kirim alert (edge OOS positif) · 👁 = pantauan · ℹ️ = info saja", ""]
    for fam in families or S.FAMILY_ORDER:
        rows = by_fam.get(fam)
        if not rows:
            continue
        lines.append(f"<b>{html.escape(rows[0]['strategy'].name.split(' (')[0])}</b>")
        for st in rows:
            s = st["strategy"]
            key = S.stat_key(s)
            mode = alert_mode(key, stats, watch)
            badge = {"alert": "🔔", "pantauan": "👁", "info": "ℹ️"}[mode]
            icon = "🟢" if s.side == "LONG" else "🔴"
            if not st.get("ok"):
                body = "data kurang"
            elif st["signal"]:
                p = st["plan"]
                body = (f"<b>SINYAL</b> entry {fp(p['entry'])} · SL {fp(p['sl'])} · TP {fp(p['tp'])} (1:{p['rr']:.1f})")
            else:
                ago = ""
                if st.get("last_signal_t") is not None:
                    hrs = (now - st["last_signal_t"].to_pydatetime()).total_seconds() / 3600
                    ago = f" (terakhir {hrs:.0f} jam lalu)" if hrs >= 1 else f" (terakhir {hrs * 60:.0f} menit lalu)"
                body = f"tidak ada sinyal{ago}"
            lines.append(f"  {badge}{icon} {s.side}: {body}")
            lines.append(f"      <i>{_stat_txt(stats.get(key))}</i>")
        lines.append("")
    meta = stats.get("_meta") or {}
    if meta:
        lines.append(f"<i>Plan: {meta.get('plan_swing', '')}; scalp: {meta.get('plan_scalp', '')}. {meta.get('rule', '')}.</i>")
    return "\n".join(lines).rstrip()


def split_message(text: str, limit: int = 4000) -> List[str]:
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for line in text.split("\n"):
        while len(line) > limit:  # baris super panjang
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(line[:limit])
            line = line[limit:]
        if len(cur) + len(line) + 1 > limit:
            parts.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        parts.append(cur)
    return parts
