# BTC Technical Analysis Bot

Bot Telegram analisis teknikal profesional khusus **BTCUSDT**.

## Fitur Utama

- Multi-timeframe lengkap: **15m • H1 • H3 • H4 • Daily • Weekly** (H3 = resample 1h, batas UTC 00/03/06/…)
- Deteksi Support & Resistance (swing + clustering)
- Indikator: EMA 20/50/200, RSI, MACD, ATR, Bollinger
- Bias arah per timeframe + overall confluence
- **Dual Entry Idea**:
  - Conservative (tunggu pullback / rejection)
  - Aggressive (ikuti momentum)
- Stop Loss & Take Profit berbasis struktur + ATR
- Format rapi siap baca di Telegram
- **EMA Stack 7/25/99** (aturan pribadi, warna Bybit: kuning EMA7, biru EMA25, pink EMA99):
  kalau kuning di atas biru **dan** pink di H1, H3, dan H4 (candle closed) → sinyal SHORT versi user.
  Tampil di `/ta` dan `/ema`, mematikan aggressive LONG, dan dikirim sekali ke subscriber saat baru aktif
  (cooldown `EMA_STACK_COOLDOWN_HOURS`, default 4 jam). Kebalikan pembacaan tren biasa.
  Periode bisa diganti lewat env `EMA_FAST`, `EMA_MID`, `EMA_SLOW`.

## Perintah

| Command | Fungsi |
|---------|--------|
| `/ta` | Full Multi-TF Analysis + Entry Ideas |
| `/signal` | Sama seperti /ta (quick access) |
| `/h1` | Detail H1 |
| `/h3` | Detail H3 (resample 1h) |
| `/h4` | Detail H4 |
| `/ema` | Status EMA stack H1/H3/H4 |
| `/daily` | Detail Daily |
| `/weekly` | Detail Weekly |
| `/15m` | Detail 15 menit |
| `/sr` | Fokus Support & Resistance |
| `/setup` | Setup yang dipantau |
| `/stats` | Win rate & hasil terakhir |
| `/subscribe` / `/unsubscribe` | Atur alert |
| `/help` | Bantuan |

Tes: `python btc_ta_bot.py --selftest` (pakai DB sementara) · `python btc_ta_bot.py --once`

## Deploy ke Railway

1. Rename file:
   - `btc_ta_requirements.txt` → `requirements.txt`
   - `btc_ta_Procfile` → `Procfile`
2. Upload ke GitHub
3. Deploy di Railway
4. Tambah Variable:
   ```
   BOT_TOKEN = token_dari_BotFather
   ```
5. Deploy → test dengan `/ta`

## Catatan

- Data diambil real-time dari Binance Futures public API (tanpa API key)
- Bukan saran finansial. Selalu pakai risk management sendiri.
- Bisa dikembangkan ke multi-pair nanti jika dibutuhkan.

## Backtest

Lihat `BACKTEST.md` (hasil, penyebab loss, dan alasan tiap filter). Ringkas: `python fetch_history.py`, lalu
`python backtest.py decisions current` dan `python backtest.py report`.
Filter baru bisa diatur lewat env: `RSI_LONG_MAX_H4`, `RSI_SHORT_MIN_H4`, `BLOCK_AGAINST_WEEKLY`, `TP2_MAX_R`.

## Perintah baru

| Command | Fungsi |
|---|---|
| `/harian` | Saran EMA harian: EMA7/25/99 + 20/50/200 Daily, tren, support/resistance dinamis |
| `/sniper` | Zona limit di confluence S/R + EMA + likuiditas, SL 1×ATR H4 (dan SL ketat), TP + R:R |
| `/liquidity` | Pool likuiditas di atas/bawah (termasuk equal highs/lows) dan sweep wick terbaru |
| `/berita` | Headline 24 jam (CoinDesk, Cointelegraph, The Block, Decrypt) + kalender USD high-impact (ForexFactory) |
| `/strategi` | Panel 8 strategi: status live + WR/avgR backtest. 🔔 alert / 👁 pantauan / ℹ️ info |
| `/scalp` | Scalping momentum 15m & 5m (info saja, fee membuatnya negatif) |

`/ta` sekarang punya "Ringkasan cepat" dari semua fitur di atas. Pesan > 4000 karakter dipecah otomatis.
Env: `BOT_LONG_ALERTS` (0), `STRAT_WATCH` (trend:LONG), `STRAT_ALERTS` (1), `STRAT_COOLDOWN_HOURS` (6),
`SNIPER_SL_ATR_H4` (1.0), `SNIPER_ALERT_COOLDOWN_HOURS` (12), `NEWS_CACHE_MINUTES` (10).
Status alert tiap strategi dibaca dari `strategy_stats.json`; jalankan ulang `python backtest.py panel` dan `python backtest.py sniper` untuk memperbaruinya.
