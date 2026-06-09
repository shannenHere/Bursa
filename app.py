import os
import yfinance as yf
import pandas as pd
import numpy as np
from rich.table import Table
from rich.console import Console
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# ==========================================
# 1. INITIALIZE CONFIGURATION & ENVIRONMENT
# ==========================================
load_dotenv()

if not os.environ.get("GEMINI_API_KEY"):
    raise ValueError("CRITICAL: GEMINI_API_KEY is missing from your .env file!")

gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0.1
)

# ==========================================
# 2. CORE TECHNICAL ANALYSIS ENGINE
# ==========================================
def calculate_technical_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Computes technical indicator arrays (RSI, MACD Crossovers, Bollinger Bands, ATR, and TD Sequential)."""
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # 1. RSI calculations for multiple lookbacks (RSI_1, RSI_2, RSI_3)
    def _rsi(series: pd.Series, window: int) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    # Common choices: short, medium, long windows
    df['RSI_1'] = _rsi(close, 7)
    df['RSI_2'] = _rsi(close, 14)
    df['RSI_3'] = _rsi(close, 28)
    # Backwards-compatible single `RSI` column points to the medium timeframe
    df['RSI'] = df['RSI_2']

    # 2. MACD & Signal Line Generation
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

    # 3. Moving averages and Bollinger Bands
    df['SMA_20'] = close.rolling(window=20).mean()
    df['SMA_50'] = close.rolling(window=50).mean()
    df['SMA_200'] = close.rolling(window=200).mean()
    rolling_std = close.rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (rolling_std * 2)
    df['BB_Lower'] = df['SMA_20'] - (rolling_std * 2)

    # 4. Average True Range (ATR)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # 5. Stochastic oscillator
    low14 = low.rolling(window=14).min()
    high14 = high.rolling(window=14).max()
    df['Stoch_%K'] = np.where(high14 != low14, 100 * ((close - low14) / (high14 - low14)), 50)
    df['Stoch_%D'] = df['Stoch_%K'].rolling(window=3).mean()

    # 6. Tom DeMark (TD) Sequential 9 Setup
    df['TD_Count'] = 0
    count = 0
    for i in range(4, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-4]:
            count += 1
        else:
            count = 0
        df.iloc[i, df.columns.get_loc('TD_Count')] = count

    return df

# ==========================================
# 3. LIVE FULL EXCHANGE COMPREHENSIVE SCANNER
# ==========================================
@tool("Bursa Malaysia Full Exchange Automated Strategy Filter")
def bursa_full_exchange_scanner(target_sector_flag: str) -> str:
    """
    Dynamically maps and sweeps the real-time active catalog of Bursa Malaysia (KLSE).
    Returns perfect matches or fallback near-matches ranked by strategic relevance scoring.
    """
    print("🚀 Gathering data vectors from live market catalog stream...")
    
    # Scanning high liquidity brackets (shorter array to respect free-tier timings)
    prime_ranges = list(range(1000, 1150)) + list(range(5000, 5150))
    perfect_matches = []
    near_matches = []
    max_recommendations = 5
    # diagnostic counters
    scanned_count = 0
    skipped_no_data = 0
    skipped_by_volume = 0
    
    for current_code in prime_ranges:
        symbol = f"{str(current_code).zfill(4)}.KL"
        try:
            scanned_count += 1
            ticker_obj = yf.Ticker(symbol)
            hist = ticker_obj.history(period="3mo")
            
            if hist.empty or len(hist) < 30:
                skipped_no_data += 1
                continue
                
            current_volume = int(hist['Volume'].iloc[-1])
            if current_volume < 20000:  # Relaxed baseline for data availability (debug)
                skipped_by_volume += 1
                continue
                
            df = calculate_technical_metrics(hist)
            current_close = float(df['Close'].iloc[-1])
            yesterday_close = float(df['Close'].iloc[-2])
            # RSI timeframes
            current_rsi1 = float(df['RSI_1'].iloc[-1])
            current_rsi2 = float(df['RSI_2'].iloc[-1])
            current_rsi3 = float(df['RSI_3'].iloc[-1])
            current_macd_hist = float(df['MACD_Histogram'].iloc[-1])
            current_macd_line = float(df['MACD'].iloc[-1])
            current_sma20 = float(df['SMA_20'].iloc[-1])
            current_sma50 = float(df['SMA_50'].iloc[-1])
            current_sma200 = float(df['SMA_200'].iloc[-1])
            bb_upper = float(df['BB_Upper'].iloc[-1])
            bb_lower = float(df['BB_Lower'].iloc[-1])
            current_atr = float(df['ATR'].iloc[-1]) if not np.isnan(df['ATR'].iloc[-1]) else 0.0
            stoch_k = float(df['Stoch_%K'].iloc[-1])
            stoch_d = float(df['Stoch_%D'].iloc[-1])
            td_sequential = int(df['TD_Count'].iloc[-1])
            price_change_pct = ((current_close - yesterday_close) / yesterday_close) * 100
            min_3d_price = float(hist['Close'].iloc[-3:].min())
            price_vs_3d_min_pct = ((current_close - min_3d_price) / min_3d_price) * 100

            pe_ratio = "N/A"
            company_name = f"Bursa Counter {str(current_code).zfill(4)}"
            target_price = None
            analyst_opinion_count = 0
            recommendation_key = None
            recommendation_mean = None
            
            try:
                info = ticker_obj.info
                if info:
                    company_name = info.get('longName', info.get('shortName', company_name))
                    raw_pe = info.get('trailingPE', info.get('forwardPE', None))
                    if raw_pe:
                        pe_ratio = round(raw_pe, 2)
                    target_price = info.get('targetMeanPrice', info.get('targetMedianPrice', None))
                    analyst_opinion_count = int(info.get('numberOfAnalystOpinions', 0) or 0)
                    recommendation_key = info.get('recommendationKey', None)
                    recommendation_mean = info.get('recommendationMean', None)
            except Exception:
                pass
                
            if not target_price:
                target_price = current_close * 1.08

            upside = ((target_price - yesterday_close) / yesterday_close) * 100

            # Buy signal using RSI3 > RSI1 (bullish trend indicator)
            buy_signal = False
            sell_price = None
            sell_price_reason = None
            try:
                if all(pd.notna(v) for v in (current_rsi1, current_rsi2, current_rsi3)):
                    # immediate buy when the longer-term RSI3 is stronger than the short-term RSI1
                    buy_signal = current_rsi3 > current_rsi1
                else:
                    buy_signal = False
            except Exception:
                buy_signal = False
            # ensure compatibility with older checks
            current_rsi = current_rsi2
            
            # --- STRATEGIC RELEVANCE SCORE ---
            match_score = 0
            reasons_failed = []
            insights = []
            
            if current_volume >= 200000:
                match_score += 1
                insights.append("Strong liquidity")
            else:
                reasons_failed.append("Volume < 200k")
                insights.append("Lower liquidity")
                
            # RSI range condition is now stricter: all RSI windows should sit in 30-55 range
            if 30 <= current_rsi1 <= 55 and 30 <= current_rsi2 <= 55 and 30 <= current_rsi3 <= 55:
                match_score += 1
                insights.append("RSI range healthy (30-55)")
            else:
                reasons_failed.append("RSI outside 30-55")
                if current_rsi1 < 30 or current_rsi2 < 30 or current_rsi3 < 30:
                    insights.append("RSI too low")
                elif current_rsi1 > 55 or current_rsi2 > 55 or current_rsi3 > 55:
                    insights.append("RSI too high")
                else:
                    insights.append("RSI not in ideal range")
                
            if current_macd_hist > 0:
                match_score += 1
                insights.append("Bullish MACD momentum")
            else:
                reasons_failed.append("MACD Bearish/Flat")
                insights.append("MACD momentum weak")
                
            if current_close >= current_sma50:
                match_score += 1
                insights.append("Price above SMA50")
            else:
                reasons_failed.append("Below SMA50")
                insights.append("Trend lacking mid-term support")
                
            if current_close >= current_sma200:
                insights.append("Price above SMA200")
            else:
                insights.append("Price below SMA200")

            if upside >= 5.0:
                match_score += 1
                insights.append("Upside target healthy")
            else:
                reasons_failed.append("Upside < 5%")
                insights.append("Limited reward potential")

            if stoch_k > stoch_d and stoch_k < 80:
                insights.append("Stochastic momentum rising")
            else:
                insights.append("Stochastic momentum neutral/bearish")

            trend_bias = "Bullish" if current_close >= current_sma200 and current_close >= current_sma50 else (
                "Neutral/Bullish" if current_close >= current_sma50 else "Neutral/Bearish"
            )
            # Determine recommended sell price when buy signal triggered
            if buy_signal:
                # prefer analyst target price, but if Bollinger upper band is higher, use that
                sell_price = target_price if target_price else current_close * 1.08
                sell_price_reason = 'Target Price'
                try:
                    if bb_upper and bb_upper > sell_price:
                        sell_price = bb_upper
                        sell_price_reason = 'Bollinger Upper'
                except Exception:
                    pass
            else:
                sell_price = None
                sell_price_reason = None

            analyst_recommendation_pass = analyst_opinion_count >= 2
            if buy_signal and analyst_recommendation_pass and 30 <= current_rsi1 <= 55 and 30 <= current_rsi2 <= 55 and 30 <= current_rsi3 <= 55 and price_vs_3d_min_pct <= 5:
                buy_signal_rank = "Perfect"
            elif buy_signal and analyst_recommendation_pass:
                buy_signal_rank = "Strong"
            elif buy_signal:
                buy_signal_rank = "Watch"
            else:
                buy_signal_rank = "N/A"

            if buy_signal:
                insights.append("ALERT: RSI3 > RSI1 — immediate bullish buy signal")

            if price_vs_3d_min_pct <= 2:
                insights.append("Price is within 2% of the 3-day low")
            else:
                insights.append(f"Price is {price_vs_3d_min_pct:.2f}% above the 3-day low")

            # Compute breakout, support and stop-loss levels for technical buy notes
            recent_high_20 = float(hist['Close'].iloc[-20:].max()) if len(hist) >= 20 else current_close
            breakout_price = recent_high_20
            try:
                breakout_potential_pct = round(((target_price - breakout_price) / breakout_price) * 100, 2) if breakout_price and target_price else round(upside, 2)
            except Exception:
                breakout_potential_pct = round(upside, 2)
            support_level = float(min_3d_price)
            stop_loss_level = round(max(0.0, support_level - (current_atr * 1.5)), 4) if current_atr else round(max(0.0, support_level * 0.98), 4)
            timeframe_label = "Daily (3mo)"
            technical_buy_note = f"Breakout @ MYR {breakout_price:.2f} (+{breakout_potential_pct}% potential)"

            stock_payload = {
                "symbol": symbol.replace(".KL", ""),
                "company_name": company_name,
                "price": current_close,
                "last_price": round(current_close, 2),
                "volume": current_volume,
                "price_change_pct": round(price_change_pct, 2),
                "rsi_1": round(current_rsi1, 2),
                "rsi_2": round(current_rsi2, 2),
                "rsi_3": round(current_rsi3, 2),
                "rsi": round(current_rsi2, 2),
                "macd": round(current_macd_line, 3),
                "macd_hist": round(current_macd_hist, 3),
                "sma_20": round(current_sma20, 2),
                "sma_50": round(current_sma50, 2),
                "sma_200": round(current_sma200, 2),
                "bb_upper": round(bb_upper, 2),
                "bb_lower": round(bb_lower, 2),
                "atr": round(current_atr, 4),
                "stoch_k": round(stoch_k, 2),
                "stoch_d": round(stoch_d, 2),
                "trend_bias": trend_bias,
                "pe": pe_ratio,
                "td_count": td_sequential,
                "target_upside_pct": round(upside, 2),
                "target_price": round(target_price, 2) if target_price else None,
                "breakout_price": round(breakout_price, 4),
                "breakout_potential_pct": breakout_potential_pct,
                "support": round(support_level, 4),
                "stop_loss": round(stop_loss_level, 4),
                "timeframe": timeframe_label,
                "technical_buy_note": technical_buy_note,
                "score": match_score,
                "category": "Perfect Setup Match" if match_score == 5 else "Near Match",
                "buy_signal": buy_signal,
                "buy_signal_rank": buy_signal_rank,
                "sell_price": round(sell_price, 2) if sell_price is not None else None,
                "sell_price_reason": sell_price_reason,
                "min_3d_price": round(min_3d_price, 2),
                "price_vs_3d_min_pct": round(price_vs_3d_min_pct, 2),
                "analyst_opinion_count": analyst_opinion_count,
                "recommendation_key": recommendation_key,
                "recommendation_mean": round(recommendation_mean, 2) if isinstance(recommendation_mean, (int, float)) else recommendation_mean,
                "analyst_recommendation_pass": analyst_recommendation_pass,
                "missed": ", ".join(reasons_failed) if reasons_failed else "None",
                "analysis_notes": "; ".join(insights)
            }
            
            if match_score == 5:
                perfect_matches.append(stock_payload)
            elif match_score >= 2:
                near_matches.append(stock_payload)
                
        except Exception:
            continue
            
    # Process output list
    output_pool = perfect_matches + near_matches
    output_pool = sorted(output_pool, key=lambda x: (x['score'], x['target_upside_pct']), reverse=True)[:max_recommendations]
    
    # Formulate output text profile; print a Rich table to console for readability
    clean_report = "\n"
    if output_pool:
        try:
            console = Console()
            table = Table(show_header=True, header_style="bold magenta")
            cols = ["#", "Symbol", "Counter Name", "Rank", "Buy", "Sell Price", "Technical Buy", "Last Price",
                    "Target Price", "Support", "Stop-Loss", "Timeframe", "Current Price", "3d Low", "Δ vs 3d Low", "Analysts", "Rec", "RSI"]
            for c in cols:
                table.add_column(c, overflow="fold")

            for idx, item in enumerate(output_pool, 1):
                recommendation_display = item['recommendation_key'] if item['recommendation_key'] else (
                    str(item['recommendation_mean']) if item['recommendation_mean'] is not None else 'N/A'
                )
                table.add_row(
                    str(idx),
                    item['symbol'],
                    item['company_name'],
                    item['buy_signal_rank'],
                    'YES' if item['buy_signal'] else 'No',
                    f"MYR {item['sell_price']}" if item.get('sell_price') else 'N/A',
                    item.get('technical_buy_note', 'N/A'),
                    f"MYR {item.get('last_price','N/A')}",
                    f"MYR {item.get('target_price','N/A')}" if item.get('target_price') else 'N/A',
                    f"MYR {item.get('support','N/A')}",
                    f"MYR {item.get('stop_loss','N/A')}",
                    item.get('timeframe','N/A'),
                    f"MYR {item['price']:.2f}",
                    f"MYR {item['min_3d_price']:.2f}",
                    f"{item['price_vs_3d_min_pct']}%",
                    str(item['analyst_opinion_count']),
                    recommendation_display,
                    f"{item['rsi_1']}/{item['rsi_2']}/{item['rsi_3']}"
                )

            console.print("\n[bold underline]Buy Signal Summary Table[/]\n")
            console.print(table)
            # Build a WhatsApp-friendly summary that avoids alignment issues
            def build_whatsapp_summary(rows):
                lines = ["```"]
                for item in rows:
                    lines.append(f"{item['index']}) {item['symbol']} {item['company_name']}")
                    lines.append(
                        f"   Buy: {item['buy']} | Sell: {item['sell']} | Target: {item['target']} | Support: {item['support']} | SL: {item['stop_loss']}"
                    )
                    lines.append(
                        f"   RSI: {item['rsi']} | Upside: {item['upside']} | Timeframe: {item['timeframe']}"
                    )
                    lines.append("")
                lines.append("```")
                return "\n".join(lines)

            rows = []
            for idx, item in enumerate(output_pool, 1):
                rows.append({
                    'index': idx,
                    'symbol': item['symbol'],
                    'company_name': item['company_name'],
                    'buy': 'YES' if item['buy_signal'] else 'No',
                    'sell': f"MYR {item['sell_price']}" if item.get('sell_price') else 'N/A',
                    'target': f"MYR {item.get('target_price','N/A')}" if item.get('target_price') else 'N/A',
                    'support': f"MYR {item.get('support','N/A')}" if item.get('support') else 'N/A',
                    'stop_loss': f"MYR {item.get('stop_loss','N/A')}" if item.get('stop_loss') else 'N/A',
                    'rsi': f"{item['rsi_1']}/{item['rsi_2']}/{item['rsi_3']}",
                    'upside': f"{item['target_upside_pct']}%",
                    'timeframe': item.get('timeframe', 'N/A')
                })

            whatsapp_table_text = build_whatsapp_summary(rows)
            print("\nWhatsApp-friendly summary (copy & paste):\n")
            print(whatsapp_table_text)
            clean_report += "\nWhatsApp-friendly summary:\n" + whatsapp_table_text
        except Exception:
            # Fallback to HTML-style string if rich is not available
            clean_report += (
                "### Buy Signal Summary Table\n"
                '<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;width:100%;">\n'
                "<thead><tr>"
                "<th>#</th><th>Symbol</th><th>Counter Name</th><th>Rank</th><th>Buy Signal</th><th>Sell Price</th>"
                "<th>Technical Buy</th><th>Last Price</th><th>Target Price</th><th>Support</th><th>Stop-Loss</th><th>Timeframe</th>"
                "<th>Current Price</th><th>3d Low Price</th><th>Δ vs 3d Low</th><th>Analyst Opinions</th>"
                "<th>Recommendation</th><th>RSI 1/2/3</th>"
                "</tr></thead>\n<tbody>\n"
            )
            for idx, item in enumerate(output_pool, 1):
                recommendation_display = item['recommendation_key'] if item['recommendation_key'] else (
                    str(item['recommendation_mean']) if item['recommendation_mean'] is not None else 'N/A'
                )
                clean_report += (
                    f"<tr><td>{idx}</td><td>{item['symbol']}</td><td>{item['company_name']}</td>"
                    f"<td>{item['buy_signal_rank']}</td><td>{'YES' if item['buy_signal'] else 'No'}</td><td>{('MYR ' + str(item['sell_price'])) if item['sell_price'] else 'N/A'}</td>"
                    f"<td>{item.get('technical_buy_note','N/A')}</td><td>MYR {item.get('last_price','N/A')}</td><td>{('MYR ' + str(item['target_price'])) if item.get('target_price') else 'N/A'}</td>"
                    f"<td>{('MYR ' + str(item['support'])) if item.get('support') else 'N/A'}</td><td>{('MYR ' + str(item['stop_loss'])) if item.get('stop_loss') else 'N/A'}</td><td>{item.get('timeframe','N/A')}</td>"
                    f"<td>MYR {item['price']:.2f}</td><td>MYR {item['min_3d_price']:.2f}</td><td>{item['price_vs_3d_min_pct']}%</td>"
                    f"<td>{item['analyst_opinion_count']}</td><td>{recommendation_display}</td><td>{item['rsi_1']}/{item['rsi_2']}/{item['rsi_3']}</td></tr>\n"
                )
            clean_report += "</tbody></table>\n"
    for idx, item in enumerate(output_pool, 1):
        status = "Perfect Setup Match" if item['score'] == 5 else f"Near Match (Missed: {item['missed']})"
        recommendation_details = item['recommendation_key'] if item['recommendation_key'] else (
            str(item['recommendation_mean']) if item['recommendation_mean'] is not None else 'N/A'
        )
        clean_report += (
            f"### {idx}. {item['company_name']} ({item['symbol']}) — Match Score: {item['score']}/5\n"
            f"* **Match Category:** {item['category']}\n"
            f"* **Price:** MYR {item['price']:.2f} ({item['price_change_pct']}% vs prior close)\n"
            f"* **3-Day Low Comparison:** MYR {item['min_3d_price']:.2f} | +{item['price_vs_3d_min_pct']}%\n"
            f"* **Technical Buy:** {item.get('technical_buy_note', 'N/A')}\n"
            f"* **Last Price:** MYR {item.get('last_price', 'N/A')}\n"
            f"* **Target Price:** {('MYR ' + str(item['target_price'])) if item.get('target_price') else 'N/A'}\n"
            f"* **Support:** {('MYR ' + str(item['support'])) if item.get('support') else 'N/A'}\n"
            f"* **Stop-Loss:** {('MYR ' + str(item['stop_loss'])) if item.get('stop_loss') else 'N/A'}\n"
            f"* **Timeframe:** {item.get('timeframe', 'N/A')}\n"
            f"* **Buy Signal (RSI3 > RSI1):** {'YES' if item.get('buy_signal') else 'No'}\n"
            f"* **Buy Rank:** {item['buy_signal_rank']}\n"
            f"* **Recommended Sell Price:** {('MYR ' + str(item['sell_price']) + ' (' + str(item['sell_price_reason']) + ')') if item.get('sell_price') else 'N/A'}\n"
            f"* **Target Upside:** {item['target_upside_pct']}%\n"
            f"* **RSI (1/2/3):** {item['rsi_1']} | {item['rsi_2']} | {item['rsi_3']}\n"
            f"* **Analyst Coverage:** {item['analyst_opinion_count']} opinions | {recommendation_details}\n"
            f"* **Fundamentals:** P/E Ratio: {item['pe']} | Trend Bias: {item['trend_bias']}\n"
            f"* **Moving Averages:** SMA20: {item['sma_20']} | SMA50: {item['sma_50']} | SMA200: {item['sma_200']}\n"
            f"* **Bollinger Bands:** Lower: {item['bb_lower']} | Upper: {item['bb_upper']}\n"
            f"* **Volatility / Momentum:** ATR: {item['atr']} | Stoch %K: {item['stoch_k']} | %D: {item['stoch_d']}\n"
            f"* **Technical Verification:** Volume: {item['volume']:,} | RSI(med): {item['rsi']} | MACD: {item['macd']} | MACD Hist: {item['macd_hist']} | TD Count: {item['td_count']}\n"
            f"* **Analysis Notes:** {item['analysis_notes']}\n"
            f"* **Relevance Note:** {status}\n"
            f"--- \n\n"
        )
        
    # Append diagnostic summary
    summary = (
        f"\nScan Summary: Scanned {scanned_count} symbols | Perfect matches: {len(perfect_matches)} | "
        f"Near matches: {len(near_matches)} | Skipped (no data): {skipped_no_data} | Skipped (low vol): {skipped_by_volume}\n"
    )

    if not output_pool:
        no_data_msg = (
            f"No data was returned from the Bursa Malaysia Full Exchange Automated Strategy Filter for the target sector flag \"{target_sector_flag}\".\n"
            "Diagnosis: The scanner found few or no tickers passing the data and filter criteria.\n"
        )
        return no_data_msg + summary

    clean_report += summary
    return clean_report

# ==========================================
# 4. CONFIGURE AGENT PERSONA SYSTEMS
# ==========================================
bursa_analyst_agent = Agent(
    role='KLSE Production Strategy Desk Operator',
    goal='Display the ranked stock entries provided by the tool exactly as they are formatted.',
    backstory='You are a prompt, high-speed reporting desk terminal. You output structured profiles directly.',
    tools=[bursa_full_exchange_scanner],
    llm=gemini_llm,
    verbose=True,
    memory=False
)

# ==========================================
# 5. DEFINE STRATEGY EXECUTION FLOW
# ==========================================
quantitative_screening_task = Task(
    description=(
        'Call the bursa_full_exchange_scanner tool with target_sector_flag="ALL" to pull real-time data metrics. '
        'Once the tool prints the layout blocks, replicate that identical Profile Layout structure as your final response.'
    ),
    expected_output='A clean list of stocks formatted in individual profile narrative segments.',
    agent=bursa_analyst_agent
)

trading_desk_crew = Crew(
    agents=[bursa_analyst_agent],
    tasks=[quantitative_screening_task],
    process=Process.sequential
)

if __name__ == "__main__":
    print("🔄 Initializing intelligent scanning system pipeline...")
    try:
        final_output = trading_desk_crew.kickoff(inputs={'target_sector_flag': 'ALL'})
        print("\n=================== FINAL PORTFOLIO LEDGER REPORT ===================\n")
        print(final_output)
    except Exception as e:
        print("\n⚠️ Framework Rate-Limit Hit. Please review the live intercepted stream generated above.")