import pandas as pd
import numpy as np

def predict_direction(df: pd.DataFrame, supports: list, resistances: list) -> dict:
    """
    Analyzes the latest data to predict direction.
    Args:
        df: Dataframe with calculated indicators.
        supports: List of support levels.
        resistances: List of resistance levels.
    Returns:
        dict: {
            "score": int,
            "signal": str, # Strong Buy, Buy, Neutral, Sell, Strong Sell
            "summary": str,
            "details": list # list of strings explaining the score
        }
    """
    if df.empty:
        return {"signal": "Error", "summary": "No data available", "details": []}

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    score = 0
    details = []
    
    # 1. Supertrend (Trend)
    # Direction: 1 (Up), -1 (Down)
    if 'Supertrend_Direction' in df.columns:
        if latest['Supertrend_Direction'] == 1:
            score += 2
            details.append("슈퍼트렌드 상승 추세 (Bullish).")
        else:
            score -= 2
            details.append("슈퍼트렌드 하락 추세 (Bearish).")

    # 2. RSI (Momentum)
    rsi = latest['RSI']
    if rsi < 30:
        score += 2
        details.append(f"RSI 과매도 구간 ({rsi:.2f}). 반등 가능성 높음.")
    elif rsi > 70:
        score -= 2
        details.append(f"RSI 과매수 구간 ({rsi:.2f}). 조정 가능성 있음.")
    elif 50 <= rsi < 70:
        score += 1
        details.append(f"RSI 상승세 ({rsi:.2f}).")
    else:
        score -= 1
        details.append(f"RSI 하락세 ({rsi:.2f}).")
        
    # 3. MACD (Momentum/Trend)
    if 'MACD_HIST' in df.columns:
        hist = latest['MACD_HIST']
        prev_hist = prev['MACD_HIST']
        
        if hist > 0:
            score += 1
            if prev_hist < 0:
                score += 2 # Golden Cross signal
                details.append("MACD 골든 크로스 (매수 신호).")
            else:
                details.append("MACD 히스토그램 양수 (상승 모멘텀).")
        else:
            score -= 1
            if prev_hist > 0:
                score -= 2 # Death Cross signal
                details.append("MACD 데드 크로스 (매도 신호).")
            else:
                details.append("MACD 히스토그램 음수 (하락 모멘텀).")

    # 4. Support/Resistance (Price Action) - High Weight
    close = latest['Close']
    nearest_support = max([s for s in supports if s < close], default=0)
    nearest_resistance = min([r for r in resistances if r > close], default=float('inf'))
    
    # Check proximity (within 1.5% for more sensitivity)
    if nearest_support > 0 and (close - nearest_support) / close < 0.015:
        score += 3 # Increased weight from 2 to 3
        details.append(f"🔥 주요 지지선 근접 ({nearest_support:,.0f}). 강력한 반등 자리.")
    
    if nearest_resistance != float('inf') and (nearest_resistance - close) / close < 0.015:
        score -= 3 # Increased weight from 2 to 3
        details.append(f"⚠️ 주요 저항선 근접 ({nearest_resistance:,.0f}). 돌파 실패 시 하락 위험.")

    # 5. Williams %R (Momentum)
    # Overbought: > -20, Oversold: < -80
    if 'Williams_%R' in df.columns:
        wr = latest['Williams_%R']
        if wr > -20:
            score -= 2
            details.append(f"Williams %R 과매수 ({wr:.2f}). 매도 압력 증가.")
        elif wr < -80:
            score += 2
            details.append(f"Williams %R 과매도 ({wr:.2f}). 매수 기회.")
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
