import streamlit as st
import sys
import os

# Add current directory to path to ensure imports work in deployment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_loader import get_ticker_data, get_market_indices, get_stock_name
from utils.indicators import calculate_indicators, find_support_resistance
from utils.predictor import predict_direction
import datetime

# Page configuration
st.set_page_config(layout="wide", page_title="주식 예측 AI", page_icon="📈")

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 10px;
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
</style>
""", unsafe_allow_html=True)

# Application Title
st.title("📈 AI 주식 예측기")

# --- Initialize Session State for Watchlist ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['005930.KS', 'AAPL', 'TSLA', 'NVDA']
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = '005930.KS'

# --- 1. Market Overview Header ---
st.subheader("🌍 시장 현황")
market_indices = get_market_indices()

if market_indices:
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
st.sidebar.header("데이터 설정")
st.sidebar.markdown("""
**💡 한국 주식 입력 팁:**
- 코스피: 종목코드.KS (예: `005930.KS`)
- 코스닥: 종목코드.KQ (예: `091990.KQ`)
""")

# Watchlist Management
st.sidebar.subheader("📋 관심 종목 (Watchlist)")

# Add to Watchlist
new_ticker = st.sidebar.text_input("종목 추가", placeholder="예: BTC-USD")
if st.sidebar.button("추가"):
    if new_ticker and new_ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_ticker)
        st.rerun()

# Whatchlist Items
st.sidebar.markdown("---")
st.sidebar.caption("종목을 클릭하여 분석하세요:")

for ticker in st.session_state.watchlist:
    col1, col2 = st.sidebar.columns([0.8, 0.2])
    with col1:
        if st.button(ticker, key=f"btn_{ticker}", use_container_width=True):
            st.session_state.selected_ticker = ticker
            st.rerun()
    with col2:
        if st.button("❌", key=f"del_{ticker}"):
            st.session_state.watchlist.remove(ticker)
            st.rerun()

st.sidebar.markdown("---")
timeframe = st.sidebar.selectbox("기간", ["1y", "2y", "5y"], index=1)

# --- Main Analysis Area ---
target_ticker = st.session_state.selected_ticker

if target_ticker:
    # Fetch Name
    stock_name = get_stock_name(target_ticker)
    
    st.header(f"📊 {stock_name} ({target_ticker}) 분석")
    
    with st.spinner(f"{stock_name} 데이터 분석 중..."):
        # Fetch Data
        df = get_ticker_data(target_ticker, period=timeframe)
        
        if df.empty:
            st.error(f"데이터를 가져올 수 없습니다: {target_ticker}. 종목 코드를 확인해주세요.")
        else:
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
