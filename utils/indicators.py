import pandas as pd
import numpy as np

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
    
    # Ensure standard column names
    # yfinance often gives Title Case: Open, High, Low, Close, Volume
    
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
    
    # Iterative calculation needed for Supertrend logic
    # This is slow in python loop but fine for daily data (years of data is small)
    
    # We need to treat them as arrays for speed or just loop
    # Let's loop for correctness as supertrend logic is recursive
    
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
    """
    Identifies support and resistance levels using local mins and maxs.
    Args:
        df: Dataframe with OHLC data.
        window: data window to look for peaks/valleys.
    Returns:
        supports (list), resistances (list) - List of price levels.
    """
    supports = []
    resistances = []
    
    recent_data = df.tail(300) 
    if recent_data.empty:
        return [], []

    # Iterate through data
    # We can use rolling max/min
    
    # Identify local maxima
    local_max = recent_data['High'].rolling(window=window*2+1, center=True).max()
    # If the central value equals the local max, it's a peak
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
