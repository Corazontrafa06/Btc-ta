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
