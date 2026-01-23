import requests
import re

def get_stock_name_fixed(ticker):
    try:
        code = ticker.split(".")[0]
        url = f"https://finance.naver.com/item/main.naver?code={code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        
        # Strategy: Try CP949 first, as Naver Finance frequently uses it
        # even if headers say UTF-8.
        try:
            # Try to decode as CP949
            html_content = r.content.decode('cp949')
        except:
            # Fallback to auto-detected encoding (could be UTF-8)
            html_content = r.text
            
        # Pattern 1: <title> tag
        match = re.search(r'<title>(.*?)</title>', html_content)
        if match:
            full_title = match.group(1)
            stock_name = full_title.split(':')[0].strip()
            if stock_name and not stock_name.startswith(''):
                return stock_name
                
        # Pattern 2: meta og:title
        match = re.search(r'<meta property="og:title" content="([^"]+)"', html_content)
        if match:
            og_title = match.group(1).strip()
            stock_name = og_title.split('-')[0].strip()
            return stock_name
            
        return ticker
    except Exception as e:
        return f"Error: {e}"

# Test with various tickers
tickers = ["005930.KS", "330590.KS", "035720.KS", "AAPL"]
for t in tickers:
    name = get_stock_name_fixed(t)
    print(f"{t}: {name}")
