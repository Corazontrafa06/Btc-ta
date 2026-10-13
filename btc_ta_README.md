# BTC Technical Analysis Bot

Bot Telegram analisis teknikal profesional khusus **BTCUSDT**.

## Fitur Utama

- Multi-timeframe lengkap: **15m • H1 • H4 • Daily • Weekly**
- Deteksi Support & Resistance (swing + clustering)
- Indikator: EMA 20/50/200, RSI, MACD, ATR, Bollinger
- Bias arah per timeframe + overall confluence
- **Dual Entry Idea**:
  - Conservative (tunggu pullback / rejection)
  - Aggressive (ikuti momentum)
- Stop Loss & Take Profit berbasis struktur + ATR
- Format rapi siap baca di Telegram

## Perintah

| Command | Fungsi |
|---------|--------|
| `/ta` | Full Multi-TF Analysis + Entry Ideas |
| `/signal` | Sama seperti /ta (quick access) |
| `/h1` | Detail H1 |
| `/h4` | Detail H4 |
| `/daily` | Detail Daily |
| `/weekly` | Detail Weekly |
| `/15m` | Detail 15 menit |
| `/sr` | Fokus Support & Resistance |
| `/help` | Bantuan |

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
