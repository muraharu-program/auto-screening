import requests
import pandas as pd
from bs4 import BeautifulSoup

def get_nikkei225_tickers():
    url = "https://indexes.nikkei.co.jp/nkave/index/component?idx=nk225"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    table = soup.find("table", class_="cmn-table_style1")
    tickers = []
    if table:
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) >= 2:
                code = cols[1].text.strip()
                if code.isdigit():
                    tickers.append(f"{code}.T")
    return tickers

if __name__ == "__main__":
    try:
        tickers = get_nikkei225_tickers()
        if tickers:
            pd.Series(tickers).to_csv("nikkei225_tickers.csv", index=False, header=False)
            print(f"取得銘柄数: {len(tickers)}")
        else:
            print("銘柄を取得できませんでした。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")