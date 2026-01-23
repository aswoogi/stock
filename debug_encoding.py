
import requests
import re

def test_encoding(code):
    url = f"https://finance.naver.com/item/main.naver?code={code}"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        
        print(f"--- Testing {code} ---")
        print(f"Content-Type Header: {r.headers.get('Content-Type')}")
        print(f"Apparent Encoding: {r.apparent_encoding}")
        print(f"Requests Default Encoding: {r.encoding}")
        
        encodings = ['euc-kr', 'cp949', 'utf-8']
        
        for enc in encodings:
            try:
                content = r.content.decode(enc)
                match = re.search(r'<title>(.*?)</title>', content)
                if match:
                    print(f"[{enc}]: {match.group(1)}")
                else:
                    print(f"[{enc}]: No title match")
            except Exception as e:
                print(f"[{enc}]: Failed to decode ({e})")
                
    except Exception as e:
        print(f"Error: {e}")

test_encoding("330590") # Lotte REIT
test_encoding("005930") # Samsung Electronics
