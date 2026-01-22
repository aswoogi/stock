import yfinance as yf
import pandas as pd

def get_ticker_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
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
    """
    try:
        t = yf.Ticker(ticker)
        # Fast access to info is sometimes slow or limited, but .info usually works.
        # Alternatively, use a static mapping or search, but yfinance is best here.
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
    
    # Batch download might be faster, but let's do individual for error isolation first or batch if stable.
    # Batch is better for performance.
    try:
        df = yf.download(tickers, period="5d", progress=False)
        
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
        
    return data
