import requests
import re

# Test different encoding methods
code = "005930"
url = f"https://finance.naver.com/item/main.naver?code={code}"
headers = {'User-Agent': 'Mozilla/5.0'}

r = requests.get(url, headers=headers, timeout=10)

print("=" * 50)
print("Test 1: Auto-detected encoding")
print(f"Encoding: {r.encoding}")
html1 = r.text
match1 = re.search(r'<title>(.*?)</title>', html1)
if match1:
    title1 = match1.group(1).split(':')[0].strip()
    print(f"Title: {title1}")
    print(f"Bytes: {title1.encode('utf-8')}")

print("\n" + "=" * 50)
print("Test 2: Force EUC-KR encoding")
try:
    html2 = r.content.decode('euc-kr', errors='replace')
    match2 = re.search(r'<title>(.*?)</title>', html2)
    if match2:
        title2 = match2.group(1).split(':')[0].strip()
        print(f"Title: {title2}")
        print(f"Bytes: {title2.encode('utf-8')}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 50)
print("Test 3: Force UTF-8 encoding")
try:
    html3 = r.content.decode('utf-8', errors='replace')
    match3 = re.search(r'<title>(.*?)</title>', html3)
    if match3:
        title3 = match3.group(1).split(':')[0].strip()
        print(f"Title: {title3}")
        print(f"Bytes: {title3.encode('utf-8')}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 50)
print("Test 4: Parse company name from div")
try:
    # Try to find company name in the page
    match4 = re.search(r'class="h_company"[^>]*>.*?<strong[^>]*>(.*?)</strong>', html1, re.DOTALL)
    if match4:
        name = match4.group(1).strip()
        print(f"Company name from div: {name}")
    
    # Alternative pattern
    match5 = re.search(r'<meta property="og:title" content="([^"]+)"', html1)
    if match5:
        og_title = match5.group(1).strip()
        print(f"OG Title: {og_title}")
except Exception as e:
    print(f"Error: {e}")
