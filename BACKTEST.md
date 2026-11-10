# Backtest & analisis loss — BTC TA bot

Data: Binance BTCUSDT, 1 Agu 2025 → 9 Nov 2026 (11.161 langkah H1; BTC turun ~115k → ~75k).
In-sample (IS) = Agu 2025 – Apr 2026, out-of-sample (OOS) = Mei – Nov 2026, plus 3 fold walk-forward.

Cara replay (`backtest.py`):
- Tiap close H1 dipanggil `build_decision()` asli dengan candle closed + 1 candle forming sintetis
  (agregasi H1 yang sudah closed). Tanpa lookahead. H3 = resample 1h. Funding historis asli; OI & order book tidak ada historinya.
- State machine seperti bot live: FORMING → TRIGGERED → TP1 → TP2/BE, INVALID/EXPIRED, 1 setup aktif, cooldown 6 jam.
- Exit di candle 15m, SL menang kalau satu candle kena SL dan TP. Fill di harga saat TRIGGERED (bukan tengah zona).
- Hasil realistis: 50% ditutup di TP1, sisa SL ke BE, runner ke TP2. Fee 0.08% round-trip.

```
python fetch_history.py              # ambil data (sekali)
python backtest.py decisions current # replay kode sekarang
python backtest.py report            # tabel + backtest_equity.png
```

## Hasil (R setelah fee)

| Versi | IS n | IS WR | IS avgR | IS PF | IS maxDD | OOS n | OOS WR | OOS avgR | OOS PF | OOS maxDD | avgR per fold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (kode lama) | 47 | 30% | -0.33 | 0.58 | 18.7 | 27 | 22% | -0.49 | 0.44 | 14.3 | -0.35 / -0.46 / -0.35 |
| Improved (kode baru) | 25 | 40% | -0.03 | 0.95 | 7.1 | 18 | 33% | -0.12 | 0.85 | 5.7 | -0.22 / +0.26 / -0.12 |
| EMA stack SHORT sendiri | 41 | 41% | +0.07 | 1.12 | 9.1 | 44 | 43% | +0.19 | 1.30 | 4.7 | +0.41 / -0.20 / +0.18 |
| Benchmark SHORT acak | 419 | 36% | -0.04 | 0.94 | 33.4 | 294 | 36% | -0.05 | 0.93 | 38.2 | +0.00 / -0.05 / -0.07 |

Sebelum fee: baseline -0.22R (IS) / -0.36R (OOS), improved +0.09R / +0.03R. Fee rata-rata 0.12–0.13R per trade karena SL sempit (~0.8%).

**Jujur:** versi improved jauh lebih baik (rugi 28.6R → 2.9R, drawdown OOS 14.3R → 5.7R), tapi setelah fee masih sedikit negatif.
Sampelnya kecil (43 trade). Ini mengurangi kerugian, belum bukti ada edge.

## Penyebab loss terbesar (baseline, 74 trade)

1. Fill terlambat: harga fill rata-rata 0.55R lebih buruk dari tengah zona (median) karena trigger menunggu candle konfirmasi. 73% trade kena SL (median 5 jam).
2. 63% trade yang kena SL sempat untung ≥1R dulu. Tes SL ke BE di +1R: OOS membaik, IS memburuk → tidak dipakai.
3. Kejar RSI ekstrem: LONG saat RSI H4 ≥65 → 0/6 menang (-1.12R rata-rata). SHORT saat RSI H4 ≤35 → 1/8 (-0.87R).
4. Melawan Weekly: LONG saat Weekly bearish → 12 trade, WR 17%, -0.70R. SHORT saat Weekly strong bullish → 6 trade, -0.38R.
5. TP2 tidak realistis: TP2 > 8R → 0/10 tercapai (-1.14R). TP2 3–5R → +0.19R. (Contoh live: TP2 65,618 = ~19R.)
6. Sesi Eropa 08–13 UTC: 20 trade, WR 10%, -0.89R. Volatilitas tinggi (ATR > 1.25× median): 14 trade, WR 14%, -0.76R.
   Keduanya tidak konsisten di OOS → tidak dipakai.
7. LONG lebih buruk dari SHORT (-0.56R vs -0.21R) di periode turun.

Yang dites tapi tidak dipakai (tidak memperbaiki IS dan OOS sekaligus): SL minimal k×ATR, limit order di tengah zona,
SL ke BE di +1R, filter R:R dari harga fill, filter ATR regime, filter sesi. Korelasi ranking IS vs OOS di 128 kombinasi cuma 0.16,
jadi sebagian besar perbedaan adalah noise. Yang dipakai hanya kombinasi terbaik IS yang juga membaik di OOS.

## Perubahan di bot (bisa diatur env)

| Env | Default | Arti |
|---|---|---|
| `RSI_LONG_MAX_H4` | 65 | Tidak LONG kalau RSI H4 ≥ ini |
| `RSI_SHORT_MIN_H4` | 35 | Tidak SHORT kalau RSI H4 ≤ ini |
| `BLOCK_AGAINST_WEEKLY` | 1 | Tidak entry melawan arah Weekly |
| `TP2_MAX_R` | 4 | TP2 maksimal 4R dari tengah zona (0 = tanpa batas) |
| `EMA_STACK_SL_ATR` / `EMA_STACK_TP_R` / `EMA_STACK_VALID_HOURS` | 1.0 / 2.0 / 48 | Rencana yang ditampilkan di alert EMA stack |

## Aturan EMA stack (kuning EMA7 di atas EMA25 & EMA99 di H1+H3+H4 → SHORT)

- Sebagai trade sendiri (SHORT saat baru aktif, SL 1×ATR H4, TP 2R, maks 48 jam): IS +0.07R, OOS +0.19R per trade, 85 trade.
  Benchmark SHORT acak dengan SL/TP sama: -0.04R / -0.05R. Jadi ada keunggulan ~+0.1–0.2R di kedua periode.
- Tapi t-stat ≈ 0.9 → belum signifikan, dan periodenya bear market. TP 1.5R hasilnya ≈ 0.
- Rata-rata harga 48 jam setelah aktif justru naik (+0.37 ATR): banyak koreksi kecil, sesekali lanjut naik kencang. SL wajib.
- Di bot: tetap tampil di /ta, /ema, dan dikirim sekali saat baru aktif dengan rencana SL/TP yang diuji.

---

# Strategy lab 1 tahun (LONG & SHORT)

`python backtest.py strategies`, dengan window 10 Nov 2025 → 9 Nov 2026. IS < 10 Mei 2026 ≤ OOS, 3 fold walk-forward 4 bulanan,
plus holdout tambahan 1 Agu → 9 Nov 2025 yang tidak dipakai memilih apa pun.
Indikator per TF dihitung dari candle closed dan dipetakan ke grid jam memakai waktu close candle, jadi tidak ada lookahead
(flag EMA stack 99.9% sama dengan replay bot). Entry di close H1, SL k×ATR H4, TP n·R, maks 48 jam, satu posisi per strategi.
Exit dicek di 15m, fee 0.08%.

**Konteks:** BTC 104.7k → 74.9k (-29%). Daily bullish (close > EMA50 D) cuma 38% dari jam. Semua strategi LONG dinilai dengan caveat ini.
Kolom "naik" = avgR untuk trade yang masuk saat Daily bullish.

### SHORT (urut avgR 1 tahun)

| Strategi | n 1y | WR | avgR | PF | maxDD | total R | IS avgR | OOS avgR | fold 1/2/3 | holdout Agu–Nov 25 | t |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EMA stack user (kuning di atas → SHORT) SL1 TP3* | 65 | 42% | +0.37 | 1.59 | 12.5 | +24.0 | -0.01 | +0.64 | +1.26/-0.24/+0.63 | -0.11 (n14) | 1.7 |
| **EMA stack user SL1 TP2 (rencana asli)** | 68 | 44% | +0.18 | 1.31 | 13.5 | +12.4 | -0.12 | +0.39 | +0.92/-0.26/+0.33 | -0.04 (n16) | 1.1 |
| Bot sekarang (zona bot, SHORT) | 23 | – | +0.27 | – | – | +6.1 | +0.38 | +0.12 | – | – | – |
| EMA stack "normal" (kuning di bawah → SHORT) | 94 | 39% | +0.03 | 1.06 | 14.1 | +3.1 | -0.08 | +0.15 | – | +0.23 (n23) | 0.2 |
| Sweep reversal tanpa filter | 201 | 36% | -0.03 | 0.95 | 20.5 | -6.5 | -0.05 | -0.01 | – | +0.01 | -0.3 |
| Benchmark acak SHORT | 516 | 36% | -0.04 | 0.93 | 47.8 | -22.5 | -0.05 | -0.04 | – | -0.02 | -0.7 |
| Trend pullback EMA H4+D | 94 | 33% | -0.09 | 0.87 | 21.4 | -8.5 | -0.19 | +0.10 | – | -0.40 | -0.6 |
| RSI pullback / BB mean-rev / sweep+H4 / breakout-retest | 42–89 | 30–42% | -0.15…-0.02 | <1 | – | negatif | – | – | – | – | – |

### LONG (urut avgR 1 tahun)

| Strategi | n 1y | WR | avgR | PF | maxDD | total R | IS avgR | OOS avgR | naik (Daily bull) | holdout Agu–Nov 25 | t |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EMA cross H1 20/50 + filter H4 naik | 25 | 40% | +0.15 | 1.24 | 3.8 | +3.7 | +0.63 (n7) | -0.04 | +0.10 (n21) | -0.51 (n5) | 0.5 |
| EMA stack mirror (kuning di bawah → LONG) SL0.75 TP3* | 82 | 33% | +0.11 | 1.16 | 9.6 | +9.3 | +0.19 | +0.02 | -0.18 | -0.06 (n23) | 0.6 |
| Sweep reversal tanpa filter | 213 | 38% | -0.04 | 0.94 | 18.5 | -8.9 | +0.05 | -0.12 | -0.13 | -0.10 | -0.5 |
| **EMA stack mirror SL1 TP2 (rencana asli)** | 80 | 35% | -0.06 | 0.91 | 9.5 | -4.6 | +0.05 | -0.18 | -0.31 | -0.18 | -0.4 |
| Benchmark acak LONG | 528 | 35% | -0.09 | 0.86 | 54.4 | -49.4 | -0.07 | -0.12 | -0.17 | -0.12 | -1.6 |
| EMA stack "normal" (kuning di atas → LONG) | 74 | 36% | -0.10 | 0.85 | 14.0 | -7.6 | +0.11 | -0.29 | +0.05 | +0.01 | -0.7 |
| RSI pullback in trend | 62 | 35% | -0.17 | 0.76 | 13.6 | -10.2 | +0.12 | -0.33 | -0.15 | -0.25 | -1.0 |
| Bot sekarang (zona bot, LONG) | 10 | – | -0.55 | – | – | -5.5 | -1.14 (n2) | -0.41 | – | – | – |
| Trend pullback EMA, sweep+H4, breakout-retest, BB mean-rev | 31–65 | 24–28% | -0.34…-0.44 OOS | <0.6 | – | negatif | – | – | – | – | – |

\* = variasi SL/TP terbaik dari 9 yang dicoba (dipilih setelah lihat hasil → optimis). Tidak ada strategi dengan t > 2.

### Kesimpulan

1. **EMA stack SHORT versi user (SL 1×ATR H4, TP 2R, 48 jam)** adalah yang terbaik dan paling dekat dengan strategi kita:
   +0.18R/trade setelah fee selama 1 tahun (68 trade, +12.4R). Benchmark SHORT acak = -0.04R. Positif di 2 dari 3 fold.
   Juga positif saat Daily bullish (+0.24R, n=37), jadi bukan cuma ikut bear market. **Tapi** t=1.1, IS (Nov–Mei) negatif,
   dan di holdout Agu–Nov 2025 ≈ 0. Edge mungkin ada tapi kecil dan belum terbukti. TP 3R terlihat lebih bagus, tapi itu hasil pilih-pilih.
2. **Setup SHORT bot (kode sekarang)**: +0.27R/trade, 23 trade, positif di IS dan OOS. Sampel kecil.
3. **LONG:** tidak ada yang robust. Mirror aturan user untuk LONG (SL1 TP2) negatif (-0.06R), dan saat Daily bullish malah lebih buruk (-0.31R).
   Kandidat LONG satu-satunya yang layak dipantau: **EMA20/50 cross H1 searah tren H4**. +0.15R di 1 tahun, positif saat Daily bullish,
   drawdown kecil, tapi cuma 25 trade dan negatif di holdout (5 trade).
4. Bacaan EMA "normal" (kuning di atas = LONG) kalah dari aturan user di SHORT dan tidak menghasilkan apa-apa di LONG.

---

# Panel 8 strategi + sniper (dipakai di /strategi, /scalp, /sniper)

Definisi ada di `strategies.py`. Bot live dan backtest memakai fungsi yang sama, candle closed saja.
Periode: holdout 1 Agu–10 Nov 2025, IS 10 Nov 2025–10 Mei 2026, OOS 10 Mei–9 Nov 2026 (1 tahun = IS+OOS).
Plan swing: SL 1×ATR H4, TP 2R, maks 48 jam. Scalp: SL 1×ATR TF, TP 1.5R, maks 4 jam (15m) / 2 jam (5m), exit dicek di candle 5m.
Fee 0.08% round-trip; scalp juga dihitung dengan fee maker 0.04%.
Perintah: `python backtest.py panel` (tulis `strategy_stats.json`), `python backtest.py sniper`, `python backtest.py panel_chart`.

**Aturan alert (otomatis dari `strategy_stats.json`):** avgR OOS > 0 dengan n OOS ≥ 10 **dan** avgR 1 tahun > 0.
Selain itu info saja, kecuali Trend Following LONG yang diberi label 👁 pantauan (`STRAT_WATCH`).

**LONG** (urut avgR OOS)

| Strategi | n 1y | WR | avgR | PF | maxDD | IS | OOS (n) | holdout (n) | t | fee/trade | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Momentum | 143 | 37% | -0.03 | 0.95 | 16.5 | -0.07 | -0.00 (69) | -0.09 (44) | -0.3 | 0.08R | info |
| Trend Following | 25 | 40% | +0.15 | 1.24 | 3.8 | +0.63 | -0.04 (18) | -0.51 (5) | +0.5 | 0.08R | 👁 pantauan |
| Benchmark acak | 528 | 35% | -0.10 | 0.86 | 54.5 | -0.07 | -0.12 (268) | -0.12 (150) | – | – | – |
| Reversal | 27 | 37% | -0.10 | 0.85 | 5.2 | -0.08 | -0.12 (9) | -0.44 (13) | -0.4 | 0.05R | info |
| Price Action (wick liquidity) | 124 | 40% | -0.03 | 0.96 | 12.4 | +0.09 | -0.15 (61) | -0.05 (34) | -0.2 | 0.07R | info |
| Breakout | 44 | 34% | -0.22 | 0.68 | 16.8 | -0.26 | -0.17 (21) | -0.06 (19) | -1.2 | 0.07R | info |
| EMA Stack 7/25/99 (mirror) | 80 | 35% | -0.06 | 0.91 | 9.5 | +0.05 | -0.18 (38) | -0.18 (23) | -0.4 | 0.07R | info |
| Scalping Momentum 15m (fee maker 0.04%) | 821 | 39% | -0.22 | 0.70 | 180.2 | -0.20 | -0.23 (413) | -0.29 (249) | -5.1 | 0.19R | – |
| Pullback | 60 | 30% | -0.22 | 0.70 | 19.0 | +0.07 | -0.34 (42) | +0.27 (11) | -1.3 | 0.08R | info |
| Range | 65 | 29% | -0.38 | 0.48 | 25.7 | -0.27 | -0.44 (41) | -0.35 (30) | -2.8 | 0.08R | info |
| Scalping Momentum 15m | 821 | 38% | -0.40 | 0.52 | 333.4 | -0.36 | -0.45 (413) | -0.51 (249) | -9.4 | 0.37R | info |
| Scalping Momentum 5m (fee maker 0.04%) | 2180 | 37% | -0.44 | 0.49 | 966.7 | -0.39 | -0.49 (1113) | -0.49 (646) | -16.6 | 0.39R | – |
| Scalping Momentum 5m | 2180 | 34% | -0.83 | 0.26 | 1806.5 | -0.71 | -0.94 (1113) | -0.93 (646) | -28.7 | 0.77R | info |
| Sniper limit searah tren (SL 1×ATR H4) | 37 | 38% | +0.03 | – | – | – | -0.16 (21) | -0.70 (9) | – | – | info |

**SHORT** (urut avgR OOS)

| Strategi | n 1y | WR | avgR | PF | maxDD | IS | OOS (n) | holdout (n) | t | fee/trade | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EMA Stack 7/25/99 (aturan kamu) | 68 | 44% | +0.18 | 1.31 | 13.5 | -0.12 | +0.39 (40) | -0.04 (16) | +1.1 | 0.08R | 🔔 alert |
| Pullback | 94 | 33% | -0.09 | 0.87 | 21.4 | -0.19 | +0.10 (33) | -0.40 (23) | -0.6 | 0.06R | info |
| Reversal | 30 | 33% | -0.10 | 0.85 | 6.1 | -0.23 | +0.10 (12) | +0.29 (8) | -0.4 | 0.07R | info |
| Breakout | 55 | 35% | -0.08 | 0.88 | 12.0 | -0.22 | +0.09 (25) | -0.10 (16) | -0.4 | 0.06R | info |
| Momentum | 155 | 39% | +0.03 | 1.05 | 14.7 | -0.00 | +0.06 (78) | +0.05 (45) | +0.3 | 0.07R | 🔔 alert |
| Trend Following | 44 | 34% | -0.03 | 0.95 | 6.7 | -0.05 | -0.00 (16) | -0.09 (10) | -0.2 | 0.06R | info |
| Price Action (wick liquidity) | 119 | 34% | -0.09 | 0.87 | 18.9 | -0.19 | -0.01 (66) | -0.09 (32) | -0.7 | 0.07R | info |
| Benchmark acak | 515 | 36% | -0.04 | 0.94 | 47.8 | -0.05 | -0.04 (270) | -0.03 (155) | – | – | – |
| Range | 64 | 30% | -0.23 | 0.69 | 16.6 | -0.39 | -0.10 (35) | +0.04 (26) | -1.4 | 0.08R | info |
| Scalping Momentum 15m (fee maker 0.04%) | 887 | 39% | -0.19 | 0.74 | 180.9 | -0.15 | -0.22 (452) | -0.21 (266) | -4.5 | 0.17R | – |
| Scalping Momentum 15m | 887 | 39% | -0.36 | 0.56 | 327.6 | -0.30 | -0.41 (452) | -0.40 (266) | -8.6 | 0.34R | info |
| Scalping Momentum 5m (fee maker 0.04%) | 2387 | 39% | -0.39 | 0.54 | 931.8 | -0.33 | -0.44 (1254) | -0.45 (728) | -14.9 | 0.37R | – |
| Scalping Momentum 5m | 2387 | 36% | -0.76 | 0.31 | 1811.7 | -0.64 | -0.87 (1254) | -0.85 (728) | -26.1 | 0.74R | info |
| Sniper limit searah tren (SL 1×ATR H4) | 62 | 39% | -0.01 | – | – | – | -0.10 (26) | +0.15 (20) | – | – | info |
### Kesimpulan jujur

- **Yang lolos aturan alert cuma 2:** EMA Stack SHORT (aturan kamu, +0.18R, OOS +0.39R) dan Momentum SHORT
  (+0.03R, OOS +0.06R, holdout +0.05R). Momentum SHORT edge-nya sangat tipis (t=0.3), hampir sama dengan benchmark acak (-0.04R).
- **Tidak ada strategi LONG yang lolos.** Trend Following LONG +0.15R di 1 tahun tapi OOS -0.04R dan holdout -0.51R → 👁 pantauan saja.
  Catatan: periode ini bear market (BTC -29%), jadi hasil LONG wajar jelek. Ulangi saat tren berubah.
- **Scalping mati karena fee:** fee makan 0.34–0.37R (15m) dan 0.74–0.77R (5m) per trade. Dengan fee maker 0.04% pun tetap -0.2 s/d -0.4R.
  Sebelum fee 15m hampir impas (≈ -0.03R). Tidak ada alert scalping.
- **Sniper:** SL ketat (0.5 ATR H1) kalah telak (WR 13–18%). SL 1×ATR H4 ≈ impas (SHORT -0.01R, LONG +0.03R),
  OOS negatif di dua arah → info saja. Satu revisi SL dicoba (0.5/1.0 ATR H4); lebih dari itu sudah overfitting.
- **Setup bot:** SHORT tetap alert. LONG diturunkan ke info saja (`BOT_LONG_ALERTS=0`).
- Rule terlihat rapuh: setelah perbaikan kecil di peta likuiditas, OOS sniper SHORT berubah +0.04R → -0.10R.
  Angka di bawah ±0.1R per trade praktis noise.
