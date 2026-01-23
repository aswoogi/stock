
import requests
import re

def get_naver_name(code):
    url = f"https://finance.naver.com/item/main.naver?code={code}"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        print(f"Status Code: {r.status_code}")
        
        # Print the title tag to see what we are getting
        match = re.search(r'<title>(.*?)</title>', r.text)
        if match:
            print(f"Full Title Tag Content: {match.group(1)}")
            
            # Test our specific regex
            title_match = re.search(r'<title>(.*?) : 네이버 페이 증권</title>', r.text)
            if title_match:
                print(f"Matched Portion: {title_match.group(1)}")
                return title_match.group(1)
            else:
                print("Regex did not match ' : 네이버 페이 증권'")
        else:
            print("No title tag found")
            
    except Exception as e:
        print(f"Error: {e}")

print("Testing 330590...")
get_naver_name("330590")
