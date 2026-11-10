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
