# Bursa Malaysia Stock Scanner

A real-time technical analysis and buy signal scanner for Bursa Malaysia (KLSE) stocks. Uses RSI, MACD, Bollinger Bands, ATR, and TD Sequential indicators to identify high-probability buy opportunities.

## Features

- **Multi-timeframe RSI Analysis**: RSI1 (7-period), RSI2 (14-period), RSI3 (28-period)
- **Buy Signal Logic**: Identifies bullish trends when RSI3 > RSI1 within 30-55 range
- **Technical Indicators**:
  - MACD & Signal Line
  - Bollinger Bands
  - Average True Range (ATR)
  - Stochastic Oscillator
  - TD Sequential 9 Setup
  - Moving Averages (SMA 20/50/200)
- **Buy Rank Classification**:
  - **Perfect**: All conditions + ≥2 analyst opinions + price ≤5% above 3-day low
  - **Strong**: Buy signal + ≥2 analyst opinions
  - **Watch**: Buy signal only
  - **N/A**: No buy signal
- **Support & Stop-Loss Calculation**: Derived from ATR and 3-day price levels
- **Breakout Detection**: 20-day high breakout with potential return percentage
- **Multiple Output Formats**:
  - Rich console table (styled, readable)
  - Detailed markdown profiles
  - WhatsApp-friendly summary (line-based, copy-paste friendly)
  - TradingView recommendation support when available

## Requirements

- Python 3.8+
- `yfinance` - Market data fetching
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `python-dotenv` - Environment variable management
- `crewai[google-genai]` - LLM agent framework
- `rich` - Console table styling
- `tradingview_ta` - Optional TradingView signal and RSI support

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shannenHere/bursa.git
   cd bursa
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install yfinance pandas numpy python-dotenv crewai[google-genai] rich tradingview_ta
   ```

4. Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

Run the scanner:
```bash
python app.py
```

Or check a specific stock instantly:
```bash
python app.py lookup 0371    # Quick lookup for Camaroe Berhad
python app.py lookup 1651    # Quick lookup for Malaysian Resources Corporation Berhad
python app.py lookup 5555    # Quick lookup for Sunway Healthcare Holdings Berhad
```

## Quick Lookup Output

Quick lookup displays:
- Stock name and Bursa Malaysia code
- Current price
- Trading volume
- RSI levels (7/14/28 periods)
- Buy signal status
- TradingView recommendation (if available)

Example:
```
[Fetching data for 0325.KL...]

Stock Lookup: Northeast Group Berhad (0325)
=====================================
Price Data:
   Current Price: RM 1.0200
   Volume: 1,010,700

Technical Indicators:
   RSI (7/14/28): 42.86 / 56.04 / 58.06
   Buy Signal (RSI3 > RSI1): YES
   RSI Range Check (30-55): FAIL

TradingView Analysis:
   Recommendation: N/A
   RSI: N/A
=====================================
```

## Full Scanner Output

Run the full exchange scanner:
```bash
python app.py
```

The script will:
1. **Always check hardcoded watchlist**: CAMAROE (0371), MRCB (1651), SUNMED (5555) 
2. Scan Bursa Malaysia symbols in liquidity brackets 1000-1150 and 5000-5150
3. Apply technical filters and scoring
4. Attempt TradingView recommendation lookups when available
5. **Display results in priority order**:
   - **Watchlist stocks first** (marked with 📌): 0371, 1651, 5555
   - **Scanner results after** (sorted by score and upside potential)
6. Output:
   - **Rich Console Table**: Styled summary table with 📌 badge on watchlist stocks
   - **Detailed Profiles**: Full technical analysis for each stock (marked [📌 WATCHLIST] or [📊 SCANNER])
   - **WhatsApp Summary**: Line-based format ready for messaging apps

## Output Indicators

In all output formats:
- **📌 Badge** or **[📌 WATCHLIST]**: Stock from the hardcoded watchlist (always checked)
- **[📊 SCANNER]**: Stock discovered through exchange scanning

## Watchlist Configuration

Edit the hardcoded watchlist in `app.py` (around line 134):
```python
watchlist = [1651, 5555, 371] # CAMAROE, MRCB, SUNMED 
```

Replace with your preferred stock codes to always monitor specific stocks.

## Output Formats

### Rich Console Table
Displays in the terminal with colors and borders—ideal for monitoring.

### Detailed Profile
Markdown-formatted profiles with:
- Match category & score
- RSI levels and trend analysis
- MACD & Bollinger Band status
- Analyst recommendations
- Support & stop-loss levels
- Technical verification (volume, MACD, TD count, etc.)

### WhatsApp Summary
Plain-text format (copy-paste friendly):
```
1) 0123 ABC
   Buy: YES | Sell: MYR 0.30 | Target: MYR 0.40 | Support: MYR 0.22 | SL: MYR 0.20
   RSI: 31/45/52 | TV Rec: BUY | TV RSI: 48.5 | Upside: 22.9% | Timeframe: Daily
```

## Buy Signal Logic

A stock generates a buy signal when:
1. **RSI3 > RSI1**: Longer-term RSI stronger than short-term (bullish reversal)
2. **All RSI values 30-55**: Avoids oversold/overbought extremes
3. **Breakout Price**: 20-day high with calculated upside potential
4. **Support Level**: 3-day minimum price
5. **Stop-Loss**: Support - (1.5 × ATR)

### Buy Rank Criteria

- **Perfect**: Buy signal + ≥2 analyst opinions + price within 5% of 3-day low
- **Strong**: Buy signal + ≥2 analyst opinions
- **Watch**: Buy signal only
- **N/A**: No buy signal

## Configuration

Edit the scanner settings in `app.py`:

- `prime_ranges`: Symbol codes to scan (default: 1000-1150, 5000-5150)
- `max_recommendations`: Max stocks to display (default: 5)
- `min_volume`: Minimum trading volume filter (default: 20,000)
- RSI windows: 7, 14, 28 periods (configurable in `calculate_technical_metrics`)

## Example

```
🚀 Gathering data vectors from live market catalog stream...

Rich Console Table:
+-----+--------+------------------+--------+-----+----------+--------+...
| #   | Symbol | Counter Name     | Rank   | Buy | Sell Pr. | Target | ...
+-----+--------+------------------+--------+-----+----------+--------+...
| 1   | 0123   | ABC Corp Bhd     | Strong | YES | MYR 0.30 | 0.40   | ...
+-----+--------+------------------+--------+-----+----------+--------+...

Detailed Profile:
### 1. ABC Corp Bhd (0123) — Match Score: 4/5
* **Match Category:** Near Match
* **Price:** MYR 0.29 (-2.3% vs prior close)
* **Buy Signal (RSI3 > RSI1):** YES
* **Buy Rank:** Strong
* **Recommended Sell Price:** MYR 0.40 (Target Price)
...

WhatsApp Summary:
1) 0123 ABC
   Buy: YES | Sell: MYR 0.30 | Target: MYR 0.40 | Support: MYR 0.22 | SL: MYR 0.20
   RSI: 31/45/52 | Upside: 22.9% | Timeframe: Daily
```
