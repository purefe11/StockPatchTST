import time

import requests
from bs4 import BeautifulSoup


# 종목에 대한 테마 조회
def get_stock_theme(stock_code):
    url = f"https://finance.finup.co.kr/stock/{stock_code}"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return f"Failed to retrieve data (status code: {response.status_code})"

    soup = BeautifulSoup(response.text, "html.parser")
    ul = soup.find("ul", {"id": "ulStockRelationTheme"})
    if not ul:
        return []

    themes = []
    for li in ul.find_all("li"):
        data_idx = li.get("data-idx")
        label_tag = li.find("span", class_="label")
        label = label_tag.text if label_tag else "N/A"
        themes.append((data_idx, label))

    time.sleep(0.1)
    return themes