import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import datetime

# --- MODULES MERGED FOR DEPLOYMENT ---

# 1. DATA LOADER
def get_ticker_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetches historical data for a given ticker.
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', '005930.KS').
        period (str): Data period to download (default '2y' to ensure enough data for indicators).
        interval (str): Data interval (default '1d').
    Returns:
        pd.DataFrame: Dataframe with 'Open', 'High', 'Low', 'Close', 'Volume'.
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return pd.DataFrame()
        # Ensure MultiIndex columns are handled if present (yfinance update quirk)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def get_stock_name(ticker: str) -> str:
    """
    Fetches the full name of the stock.
    For Korean stocks (.KS, .KQ), uses Naver Finance to get the Korean name.
    """
    try:
        # 1. Custom handling for Korean stocks to get Hangul Name
        if ticker.endswith(".KS") or ticker.endswith(".KQ"):
            try:
                code = ticker.split(".")[0]
                url = f"https://finance.naver.com/item/main.naver?code={code}"
                
                import requests
                import re
                
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                r = requests.get(url, headers=headers, timeout=10)
                
                # Naver Finance often sends CP949 even when headers say UTF-8
                try:
                    html_content = r.content.decode('cp949')
                except:
                    html_content = r.text
                
                # Pattern 1: Extract from title tag "삼성전자 : 네이버 페이 증권"
                match = re.search(r'<title>(.*?) : .*?</title>', html_content)
                if match:
                    stock_name = match.group(1).strip()
                    if stock_name and not stock_name.startswith('\ufffd'):
                        return stock_name
                
                # Pattern 2: Global title fallback
                match = re.search(r'<title>(.*?)</title>', html_content)
                if match:
                    full_title = match.group(1)
                    stock_name = full_title.split(':')[0].strip()
                    if stock_name and not stock_name.startswith('\ufffd'):
                        return stock_name
                        
            except Exception as e:
                print(f"Error fetching Korean stock name from Naver: {e}")
                pass # Fallback to yfinance if Naver fails

        t = yf.Ticker(ticker)
        info = t.info
        return info.get('longName') or info.get('shortName') or ticker
    except Exception as e:
        print(f"Error fetching info for {ticker}: {e}")
        return ticker



def get_market_indices() -> dict:
    """
    Fetches key market indices and rates.
    Returns:
        dict: Dictionary containing current price and daily change for each index.
    """
    indices = {
        "S&P 500": "^GSPC",
        "나스닥 (NASDAQ)": "^IXIC",
        "코스피 (KOSPI)": "^KS11",
        "달러/원 (USD/KRW)": "KRW=X",
        "달러 인덱스": "DX-Y.NYB"
    }
    
    data = {}
    tickers = list(indices.values())
    timestamp_str = "N/A"
    
    try:
        df = yf.download(tickers, period="5d", progress=False)
        
        if not df.empty:
            # Get the latest timestamp from the index
            latest_dt = df.index[-1]
            # Convert to string format
            timestamp_str = latest_dt.strftime("%Y-%m-%d %H:%M")
        
        # yfinance returns MultiIndex (Price, Ticker)
        # We need 'Close' for prices.
        closes = df['Close']
        
        for name, ticker in indices.items():
            if ticker in closes.columns:
                series = closes[ticker].dropna()
                if len(series) >= 2:
                    current = series.iloc[-1]
                    prev = series.iloc[-2]
                    change = ((current - prev) / prev) * 100
                    data[name] = {
                        "current": current,
                        "change": change
                    }
                elif len(series) == 1:
                     data[name] = {
                        "current": series.iloc[-1],
                        "change": 0.0
                    }
    except Exception as e:
        print(f"Error fetching market indices: {e}")
        
    return data, timestamp_str

# 2. INDICATORS
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators for the dataframe using pure Pandas.
    Args:
        df: Dataframe with OHLCV data.
    Returns:
        df: Dataframe with added indicator columns.
    """
    if df.empty:
        return df
    
    # 1. RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. MACD (12, 26, 9)
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']

    # 3. Relative Volume
    # Compare current volume to 20-day simple moving average volume
    df['Vol_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Relative_Vol'] = df['Volume'] / df['Vol_SMA_20']

    # 4. ADR (Average Daily Range) - 14 days
    df['Daily_Range'] = df['High'] - df['Low']
    df['ADR'] = df['Daily_Range'].rolling(window=14).mean()
    df['ADR_Percent'] = (df['ADR'] / df['Close']) * 100

    # 5. Supertrend (10, 3)
    # TR = Max(High - Low, abs(High - PrevClose), abs(Low - PrevClose))
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/10, adjust=False).mean() # ATR 10
    
    multiplier = 3
    final_upperband = (high + low) / 2 + (multiplier * atr)
    final_lowerband = (high + low) / 2 - (multiplier * atr)
    
    # Initialize Supertrend columns
    supertrend = pd.Series(0.0, index=df.index)
    supertrend_dir = pd.Series(1, index=df.index) # 1 for up, -1 for down
    
    # Convert to numpy arrays for speed
    close_arr = close.values
    fu_arr = final_upperband.values
    fl_arr = final_lowerband.values
    st_arr = np.zeros(len(df))
    dir_arr = np.zeros(len(df))
    
    # Initial values
    st_arr[0] = fl_arr[0]
    dir_arr[0] = 1
    
    # Variables for recursive state
    prev_fu = fu_arr[0]
    prev_fl = fl_arr[0]
    prev_st = st_arr[0]
    prev_dir = 1
    
    for i in range(1, len(df)):
        curr_close = close_arr[i]
        curr_prev_close = close_arr[i-1]
        
        # Calculate Basic Bands
        curr_fu = fu_arr[i]
        curr_fl = fl_arr[i]
        
        # Calculate Final Bands regarding previous bands
        if (curr_fu < prev_fu) or (curr_prev_close > prev_fu):
            effective_fu = curr_fu
        else:
            effective_fu = prev_fu
            
        if (curr_fl > prev_fl) or (curr_prev_close < prev_fl):
            effective_fl = curr_fl
        else:
            effective_fl = prev_fl
            
        # Determine Direction and Value
        if prev_st == prev_fu: # Previous was downtrend
            if curr_close > effective_fu:
                current_st = effective_fl
                current_dir = 1 # Change to Uptrend
            else:
                current_st = effective_fu
                current_dir = -1 # Stay Downtrend
        else: # Previous was uptrend (prev_st == prev_fl)
            if curr_close < effective_fl:
                current_st = effective_fu
                current_dir = -1 # Change to Downtrend
            else:
                current_st = effective_fl
                current_dir = 1 # Stay Uptrend
                
        st_arr[i] = current_st
        dir_arr[i] = current_dir
        
        # Update previous state
        prev_fu = effective_fu
        prev_fl = effective_fl
        prev_st = current_st
    
    df['Supertrend'] = st_arr
    df['Supertrend_Direction'] = dir_arr

    # 6. Williams %R (14)
    # Formula: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Range: 0 to -100
    highest_high = df['High'].rolling(window=14).max()
    lowest_low = df['Low'].rolling(window=14).min()
    df['Williams_%R'] = ((highest_high - df['Close']) / (highest_high - lowest_low)) * -100

    return df

def find_support_resistance(df: pd.DataFrame, window=20) -> tuple[list, list]:
    """Identifies support and resistance levels using local mins and maxs."""
    supports = []
    resistances = []
    
    recent_data = df.tail(300) 
    if recent_data.empty:
        return [], []

    # Identify local maxima
    local_max = recent_data['High'].rolling(window=window*2+1, center=True).max()
    peaks = recent_data[recent_data['High'] == local_max]['High']
    resistances = peaks.tolist()
    
    # Identify local minima
    local_min = recent_data['Low'].rolling(window=window*2+1, center=True).min()
    valleys = recent_data[recent_data['Low'] == local_min]['Low']
    supports = valleys.tolist()
            
    return consolidate_levels(supports), consolidate_levels(resistances)

def consolidate_levels(levels, threshold=0.02):
    """Merges levels that are within threshold % of each other."""
    if not levels:
        return []
        
    levels.sort()
    merged = []
    current_group = [levels[0]]
    
    for level in levels[1:]:
        if (level - current_group[-1]) / current_group[-1] <= threshold:
            current_group.append(level)
        else:
            merged.append(sum(current_group) / len(current_group))
            current_group = [level]
    merged.append(sum(current_group) / len(current_group))
    
    return merged

# 3. PREDICTOR
def predict_direction(df: pd.DataFrame, supports: list, resistances: list) -> dict:
    """Analyzes the latest data to predict direction."""
    if df.empty:
        return {"signal": "Error", "summary": "No data available", "details": []}

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    score = 0
    details = []
    
    # 1. Supertrend (Trend)
    if 'Supertrend_Direction' in df.columns:
        if latest['Supertrend_Direction'] == 1:
            score += 2
            details.append("📈 **슈퍼트렌드 상승 (Bullish)**: 현재 주가가 상승 추세 위에 있습니다. 전반적인 매수 심리가 살아있는 상태입니다.")
        else:
            score -= 2
            details.append("📉 **슈퍼트렌드 하락 (Bearish)**: 현재 주가가 하락 추세 아래에 있습니다. 매도 압력이 강한 상태이니 주의가 필요합니다.")

    # 2. RSI (Momentum)
    rsi = latest['RSI']
    if rsi < 30:
        score += 2
        details.append(f"🟢 **RSI 과매도 ({rsi:.2f})**: 단기간에 주가가 과도하게 하락했습니다. 기술적 반등(데드캣 바운스)이 나올 가능성이 높습니다.")
    elif rsi > 70:
        score -= 2
        details.append(f"🔴 **RSI 과매수 ({rsi:.2f})**: 단기간에 주가가 과도하게 상승했습니다. 차익 실현 매물로 인한 조정 가능성이 있습니다.")
    elif 50 <= rsi < 70:
        score += 1
        details.append(f"🔼 **RSI 상승세 ({rsi:.2f})**: 매수 세력이 우세하며 추가 상승 여력이 있어 보입니다.")
    else:
        score -= 1
        details.append(f"🔽 **RSI 하락세 ({rsi:.2f})**: 매도 세력이 우세하거나 모멘텀이 약해지고 있습니다.")
        
    # 3. MACD (Momentum/Trend)
    if 'MACD_HIST' in df.columns:
        hist = latest['MACD_HIST']
        prev_hist = prev['MACD_HIST']
        
        if hist > 0:
            score += 1
            if prev_hist < 0:
                score += 2 # Golden Cross signal
                details.append("✨ **MACD 골든크로스**: 단기 이동평균선이 장기를 뚫고 올라갔습니다. 강력한 매수 신호 중 하나입니다.")
            else:
                details.append("👍 **MACD 양수**: 상승 모멘텀이 유지되고 있습니다.")
        else:
            score -= 1
            if prev_hist > 0:
                score -= 2 # Death Cross signal
                details.append("💀 **MACD 데드크로스**: 단기 이동평균선이 장기 아래로 떨어졌습니다. 하락 추세 전환 신호일 수 있습니다.")
            else:
                details.append("👎 **MACD 음수**: 하락 모멘텀이 지속되고 있습니다.")

    # 4. Support/Resistance (Price Action) - High Weight
    close = latest['Close']
    nearest_support = max([s for s in supports if s < close], default=0)
    nearest_resistance = min([r for r in resistances if r > close], default=float('inf'))
    
    # Check proximity (within 1.5% for more sensitivity)
    if nearest_support > 0 and (close - nearest_support) / close < 0.015:
        score += 3 # Increased weight from 2 to 3
        details.append(f"🛡️ **지지선 근접 (약 {nearest_support:,.0f}원/달러)**: 바닥을 다지고 반등할 수 있는 가격대입니다. 매수하기 좋은 위치일 수 있습니다.")
    
    if nearest_resistance != float('inf') and (nearest_resistance - close) / close < 0.015:
        score -= 3 # Increased weight from 2 to 3
        details.append(f"🧱 **저항선 근접 (약 {nearest_resistance:,.0f}원/달러)**: 이 가격대에서 매도 물량이 쏟아져 상승이 막힐 수 있습니다. 돌파 여부를 잘 지켜봐야 합니다.")

    # 5. Williams %R (Momentum)
    # Overbought: > -20, Oversold: < -80
    if 'Williams_%R' in df.columns:
        wr = latest['Williams_%R']
        if wr > -20:
            score -= 2
            details.append(f"🔥 **Williams %R 과매수 ({wr:.2f})**: 매수세가 너무 강해 과열권에 진입했습니다. 조만간 조정이 올 수 있습니다.")
        elif wr < -80:
            score += 2
            details.append(f"💧 **Williams %R 과매도 ({wr:.2f})**: 공포감에 의한 투매가 나왔을 수 있습니다. 저점 매수의 기회일 수 있습니다.")
        else:
            # Neutral zone, mild trend follow?
            pass

    # Interpret Score
    # Range roughly -12 to +12
    # Adjusted thresholds slightly for more signals
    if score >= 6:
        signal = "강력 매수 (Strong Buy)"
    elif score >= 2:
        signal = "매수 (Buy)"
    elif score <= -5:
        signal = "강력 매도 (Strong Sell)"
    elif score <= -2:
        signal = "매도 (Sell)"
    else:
        signal = "중립 (Neutral)"
        
    return {
        "score": score,
        "signal": signal,
        "summary": f"종합 점수: {score}점. 전반적인 기술적 전망은 '{signal}' 입니다.",
        "details": details
    }

import json
import os

# --- PERSISTENCE ---
WATCHLIST_FILE = "watchlist.json"

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading watchlist: {e}")
    # Default initial list
    return [
        {"ticker": "005930.KS", "name": "삼성전자"},
        {"ticker": "AAPL", "name": "Apple Inc."},
        {"ticker": "TSLA", "name": "Tesla, Inc."},
        {"ticker": "NVDA", "name": "NVIDIA Corp."}
    ]

# save_watchlist function removed for local isolation


# --- HELPER UI FUNCTIONS ---
def truncate_text(text, max_len=15):
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text

# --- MAIN APPLICATION ---

# Page configuration
st.set_page_config(layout="wide", page_title="주식 예측 AI", page_icon="📈")

# Custom CSS for styling and responsiveness
st.markdown("""
<style>
    /* Responsive Text Sizes */
    h1 { font-size: clamp(1.5rem, 4vw, 3rem) !important; }
    h2 { font-size: clamp(1.2rem, 3vw, 2.2rem) !important; }
    
    /* Metrics: Default sizes, will be overridden by dynamic style below */
    .metric-card {
        background-color: #1e1e1e;

        padding: 5px; /* Reduced padding for mobile */
        border-radius: 5px;
        border: 1px solid #333;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .buy { background-color: rgba(0, 255, 0, 0.1); border: 2px solid #00ff00; }
    .sell { background-color: rgba(255, 0, 0, 0.1); border: 2px solid #ff0000; }
    .neutral { background-color: rgba(255, 255, 255, 0.1); border: 2px solid #ffffff; }
    
    /* Button Styling in Sidebar */
    .stButton button {
        text-align: left !important;
        width: 100%;
        padding-left: 10px;
    }
    
    /* Custom colored markers for sidebar items if needed, 
       but we will use Emojis for simplicity and robustness */
</style>
""", unsafe_allow_html=True)

# Helper function for loading file
def load_watchlist_from_file():
    uploaded_file = st.session_state.get('uploaded_file_widget')
    if uploaded_file is not None:
        try:
            loaded_data = json.load(uploaded_file)
            if isinstance(loaded_data, list):
                st.session_state.watchlist = loaded_data
                st.toast("✅ 리스트를 성공적으로 불러왔습니다!")
        except Exception as e:
            st.error(f"파일 불러오기 오류: {e}")


# Application Title
st.title("📈 AI 주식 예측기")

# --- Initialize Session State for Watchlist ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_watchlist()
if 'batch_analysis_results' not in st.session_state:
    st.session_state.batch_analysis_results = {}
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = st.session_state.watchlist[0]['ticker'] if st.session_state.watchlist else "005930.KS"

# --- 1. Market Overview Header ---
st.subheader("🌍 시장 현황")
market_indices, market_time = get_market_indices()

if market_indices:
    st.caption(f"🕒 기준 시간: {market_time}")
    cols = st.columns(len(market_indices))
    for i, (name, data) in enumerate(market_indices.items()):
        with cols[i]:
            st.metric(
                label=name,
                value=f"{data['current']:,.2f}",
                delta=f"{data['change']:.2f}%"
            )
else:
    st.info("시장 데이터를 불러오는 중...")

st.markdown("---")

# --- Sidebar Inputs ---
# --- Sidebar Inputs ---
st.sidebar.header("데이터 설정")

# Watchlist Management - Local Import/Export
st.sidebar.subheader("📂 리스트 관리")

# Download (Save to Local)
save_name = st.sidebar.text_input("저장할 파일명", value="stock_watchlist", help=".json 확장자는 자동으로 붙습니다.")
if not save_name.endswith(".json"):
    save_name += ".json"

watchlist_json = json.dumps(st.session_state.watchlist, ensure_ascii=False, indent=2)
st.sidebar.download_button(
    label="💾 리스트 내보내기 (저장)",
    data=watchlist_json,
    file_name=save_name,
    mime="application/json",
    help="현재 리스트를 내 컴퓨터에 JSON 파일로 저장합니다."
)

# Upload (Load from Local)
# Use key and on_change callback to handle loading only when file changes
st.sidebar.file_uploader(
    "📂 리스트 불러오기", 
    type=["json"], 
    help="저장된 리스트 파일을 불러옵니다.",
    key='uploaded_file_widget',
    on_change=load_watchlist_from_file
)


st.sidebar.markdown("---")

st.sidebar.subheader("📋 관심 종목")

# Batch Analysis Button
if st.sidebar.button("🚀 일괄 분석 실행 (Batch Analysis)"):
    progress_bar = st.sidebar.progress(0)
    total = len(st.session_state.watchlist)
    
    results = {}
    for idx, item in enumerate(st.session_state.watchlist):
        ticker = item['ticker']
        # Fetch minimal data for speed (e.g., 6mo or enough for indicators)
        # We need enough for Moving Averages (200 might be safest, so 1y or 2y)
        # Reuse existing function
        _df = get_ticker_data(ticker, period="1y") 
        if not _df.empty:
            _df = calculate_indicators(_df)
            _supports, _resistances = find_support_resistance(_df)
            _pred = predict_direction(_df, _supports, _resistances)
            
            # Parsing signal for color
            sig = _pred['signal']
            if "매수" in sig:
                results[ticker] = "buy"
            elif "매도" in sig:
                results[ticker] = "sell"
            else:
                results[ticker] = "neutral"
        else:
            results[ticker] = "error"
            
        progress_bar.progress((idx + 1) / total)
        
    st.session_state.batch_analysis_results = results
    st.sidebar.success("분석 완료!")

# Add to Watchlist
st.sidebar.markdown("### 종목 추가")

# Market Selection moved here
market_type = st.sidebar.radio(
    "시장 선택",
    ("🇺🇸 미국 (US)", "🇰🇷 한국 (KR)"),
    horizontal=True,
    help="한국 주식은 종목코드(숫자)만 입력하세요."
)

with st.sidebar.form(key="add_stock_form", clear_on_submit=True):
    new_ticker_input = st.text_input("종목 코드/티커", placeholder="예: AAPL 또는 005930")
    submitted = st.form_submit_button("추가")

    if submitted and new_ticker_input:
        final_ticker = new_ticker_input.strip().upper()
        # Variable to store validated name if found during check
        validated_name = None 
        
        # Logic to handle Korean stocks automatically
        if "한국" in market_type:
            # If user entered digits only, we assume it's a code
            if final_ticker.isdigit():
                # Try KOSPI first
                test_ticker = f"{final_ticker}.KS"
                
                with st.spinner("종목 확인 중... (KOSPI/KOSDAQ)"):
                    name_check = get_stock_name(test_ticker)
                    
                    if name_check != test_ticker:
                        final_ticker = test_ticker
                        validated_name = name_check
                    else:
                        # Try KOSDAQ
                        test_ticker_bq = f"{final_ticker}.KQ"
                        name_check_bq = get_stock_name(test_ticker_bq)
                        if name_check_bq != test_ticker_bq:
                            final_ticker = test_ticker_bq
                            validated_name = name_check_bq
                        else:
                            # Both failed, default to KS
                            final_ticker = f"{final_ticker}.KS"
            
        # Check integrity
        exists = any(item['ticker'] == final_ticker for item in st.session_state.watchlist)
        if not exists:
            # Fetch name if we haven't already
            if validated_name is not None:
                fetched_name = validated_name
            else:
                with st.spinner(f"'{final_ticker}' 종목 정보 확인 중..."):
                    fetched_name = get_stock_name(final_ticker)
                
            # If name is same as ticker, it's a fallback
            if fetched_name == final_ticker:
               st.toast(f"⚠️ '{final_ticker}' 이름을 가져오지 못했습니다. 티커로 표시됩니다.")
            
            st.session_state.watchlist.append({"ticker": final_ticker, "name": fetched_name})
            st.success(f"✅ {fetched_name} 추가 완료!")
            st.rerun()

        else:
            st.warning("이미 목록에 있는 종목입니다.")



# Whatchlist Items
st.sidebar.markdown("---")
st.sidebar.caption("종목을 클릭하여 분석하세요:")

for item in st.session_state.watchlist:
    ticker = item['ticker']
    name = item['name']
    
    # Determine Label with Color/Emoji based on Batch Results
    status = st.session_state.batch_analysis_results.get(ticker)
    
    # User requested: Buy=Red, Sell=Blue, Neutral=Yellow
    # We use emojis to simulate this on the button.
    # 🔴: Buy, 🔵: Sell, 🟡: Neutral
    
    prefix = ""
    if status == "buy":
        prefix = "🔴 "
    elif status == "sell":
        prefix = "🔵 "
    elif status == "neutral":
        prefix = "🟡 "
    
    # Truncate long names for display
    display_name = truncate_text(name, 12)
    label = f"{prefix}{display_name}"
    
    col1, col2 = st.sidebar.columns([0.8, 0.2])
    with col1:
        # Show name on button, ticker in tooltip/help if possible, but button text is primary
        if st.button(label, key=f"btn_{ticker}", help=f"{name} ({ticker})", use_container_width=True):
            st.session_state.selected_ticker = ticker
            st.rerun()
    with col2:
        if st.button("❌", key=f"del_{ticker}"):
            st.session_state.watchlist = [i for i in st.session_state.watchlist if i['ticker'] != ticker]
            # save_watchlist removed
            st.rerun()


st.sidebar.markdown("---")
st.sidebar.markdown("---")
timeframe = st.sidebar.selectbox("기간", ["1y", "2y", "5y"], index=0)

st.sidebar.markdown("---")
# Font Size Slider
font_size_scale = st.sidebar.slider("글자 크기 조절 (Font Size)", 0.5, 1.5, 1.0, 0.1)

# Dynamic CSS injection for font size
# Base size: 10pt is approx 13.3px. Assuming 1rem = 16px, 10pt = ~0.8rem
# We apply this to metrics.
st.markdown(f"""
<style>
    /* Force override Key Metrics */
    div[data-testid="stMetricValue"] > div {{
        font-size: {0.8 * font_size_scale}rem !important;
    }}
    div[data-testid="stMetricLabel"] > label {{
        font-size: {0.7 * font_size_scale}rem !important;
    }}
    
    /* Also adjust table text if needed, but primarily metrics */
    
    @media (max-width: 600px) {{
        div[data-testid="stMetricValue"] > div {{
            font-size: {0.7 * font_size_scale}rem !important;
        }}
        div[data-testid="stMetricLabel"] > label {{
            font-size: {0.6 * font_size_scale}rem !important;
        }}
    }}
</style>
""", unsafe_allow_html=True)


# --- Main Analysis Area ---
target_ticker = st.session_state.selected_ticker

if target_ticker:
    # Fetch Name (Double check or use saved)
    stock_name = get_stock_name(target_ticker)
    
    # Placeholders for dynamic header
    header_placeholder = st.empty()
    caption_placeholder = st.empty()
    
    with st.spinner(f"{stock_name} 데이터 분석 중..."):
        # Fetch Data
        df = get_ticker_data(target_ticker, period=timeframe)
        
        if df.empty:
            st.error(f"데이터를 가져올 수 없습니다: {target_ticker}. 종목 코드를 확인해주세요.")
        else:
            # Get latest price and time
            latest_price = df['Close'].iloc[-1]
            latest_time = df.index[-1].strftime("%Y-%m-%d %H:%M")
            
            # Update Header with Price
            header_placeholder.header(f"📊 {stock_name} ({latest_price:,.0f})")
            caption_placeholder.caption(f"Ticker: {target_ticker} | 🕒 기준 시간: {latest_time}")
            # Calculate Indicators
            df = calculate_indicators(df)
            
            # Find Support/Resistance
            supports, resistances = find_support_resistance(df)
            
            # Predict
            prediction = predict_direction(df, supports, resistances)
            
            # --- Display Prediction ---
            signal_color = "neutral"
            if "매수" in prediction['signal']: signal_color = "buy"
            elif "매도" in prediction['signal']: signal_color = "sell"
            
            st.markdown(f"""
            <div class="prediction-card {signal_color}">
                <h2>예측: {prediction['signal']}</h2>
                <p>{prediction['summary']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("예측 상세 사유 보기"):
                for detail in prediction['details']:
                    st.write(f"- {detail}")
            
            # --- Main Chart (Price + Supertrend + S/R) ---
            st.subheader("가격 및 슈퍼트렌드 (Price & Supertrend)")
            
            # Create subplots (Main Price + Volume + RSI + MACD + Williams)
            fig = make_subplots(
                rows=5, cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.05, 
                row_heights=[0.4, 0.1, 0.15, 0.15, 0.15],
                specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
                subplot_titles=(
                    "주가 및 트렌드 (Price & Trend)", 
                    "거래량 (Volume)", 
                    "상대강도지수 (RSI)", 
                    "MACD (추세/모멘텀)", 
                    "윌리엄스 %R (Williams %R)"
                )
            )

            # 1. Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                name='주가 (Price)'
            ), row=1, col=1)
            
            # Supertrend
            if 'Supertrend' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Supertrend'], 
                    mode='lines', name='슈퍼트렌드 (Supertrend)',
                    line=dict(color='orange', width=1)
                ), row=1, col=1)

            # Support/Resistance Lines
            for s in supports:
                fig.add_hline(y=s, line_dash="dot", line_color="green", opacity=0.5, row=1, col=1)
            for r in resistances:
                fig.add_hline(y=r, line_dash="dot", line_color="red", opacity=0.5, row=1, col=1)

            # 2. Volume
            colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in df.iterrows()]
            fig.add_trace(go.Bar(
                x=df.index, y=df['Volume'],
                name='거래량 (Volume)', marker_color=colors, opacity=0.5
            ), row=2, col=1)
            
            # 3. RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
            fig.add_hline(y=70, line_dash='dash', line_color='red', row=3, col=1)
            fig.add_hline(y=30, line_dash='dash', line_color='green', row=3, col=1)

            # 4. MACD
            if 'MACD' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=4, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD_SIGNAL'], name='신호선 (Signal)', line=dict(color='orange')), row=4, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df['MACD_HIST'], name='히스토그램 (Hist)', marker_color='gray'), row=4, col=1)

            # 5. Williams %R
            if 'Williams_%R' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['Williams_%R'], name='Williams %R', line=dict(color='gold')), row=5, col=1)
                fig.add_hline(y=-20, line_dash='dash', line_color='red', row=5, col=1)
                fig.add_hline(y=-80, line_dash='dash', line_color='green', row=5, col=1)

            fig.update_layout(height=1000, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            # --- Additional Data ---
            st.subheader("기술적 지표 상세")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RSI (14일)", f"{df['RSI'].iloc[-1]:.2f}")
            with col2:
                st.metric("ADR 비율", f"{df['ADR_Percent'].iloc[-1]:.2f}%")
            with col3:
                st.metric("상대 거래량", f"{df['Relative_Vol'].iloc[-1]:.2f}x")

else:
    st.info("사이드바에서 종목을 선택하거나 추가해주세요.")
