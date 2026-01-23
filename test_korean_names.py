import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import the function
from app import get_stock_name

# Test Korean stocks
test_tickers = [
    "005930.KS",  # Samsung Electronics
    "035720.KS",  # Kakao
    "000660.KS",  # SK Hynix
    "005380.KS",  # Hyundai Motor
]

print("Testing Korean stock name fetching:")
print("=" * 60)

for ticker in test_tickers:
    name = get_stock_name(ticker)
    print(f"Ticker: {ticker}")
    print(f"Name: {name}")
    print(f"UTF-8 Bytes: {name.encode('utf-8')}")
    print("-" * 60)
