import sys
import os

# Mock streamlit and other things if needed, or just import the function
sys.path.append(os.getcwd())

from app import get_stock_name

test_cases = [
    ("005930.KS", "삼성전자"),
    ("035720.KS", "카카오"),
    ("330590.KS", "롯데리츠"),
    ("000660.KS", "SK하이닉스"),
    ("AAPL", "Apple Inc.")  # Should still work with yfinance fallback
]

print("--- Testing get_stock_name implementation in app.py ---")
for ticker, expected in test_cases:
    name = get_stock_name(ticker)
    print(f"Ticker: {ticker:10} | Expected: {expected:15} | Actual: {name}")

print("\n--- Done ---")
