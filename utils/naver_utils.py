import time
from datetime import datetime
from typing import NamedTuple

import dateutil
import requests

from settings import settings


# @dataclass
class StockNewsItem(NamedTuple):
    stock_code: str
    stock_name: str
    sentiment: int
    date: datetime
    title: str
    description: str
    link: str
    original_link: str


# https://developers.naver.com/apps/#/list
# https://developers.naver.com/docs/serviceapi/search/news/news.md#%EB%89%B4%EC%8A%A4
# 네이버 뉴스 조회
def get_news_infos(keyword, target_date, count):
    # 요청 URL 구성
    sort = 'sim'  # sim: 정확도순으로 내림차순 정렬, date: 날짜순으로 내림차순 정렬
    url = f"https://openapi.naver.com/v1/search/news.json?query={keyword}&display={count}&sort={sort}"

    # API 요청 헤더 설정
    headers = {
        "X-Naver-Client-Id": settings.NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": settings.NAVER_CLIENT_SECRET
    }

    # API 요청
    response = requests.get(url, headers=headers)
    news_data = response.json()

    result = []
    for item in news_data['items']:
        pub_date = dateutil.parser.parse(item['pubDate'])
        if pub_date.strftime("%Y%m%d") == target_date:
            result.append((pub_date, item['title'], item['description'], item['link'], item['originallink']))
    time.sleep(0.01)
    return result