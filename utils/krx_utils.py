import numpy as np
import os
import pandas as pd
import requests
import time
import xml.etree.ElementTree as et
import yfinance as yf
from datetime import datetime, timedelta
from pandas import DataFrame
from pykrx.stock import stock_api
from pykrx.website.krx.market.core import (
    전종목시세, 투자자별_거래실적_개별종목_일별추이_일반
)
from pykrx.website.krx.market.ticker import get_stock_ticker_isin
from pykrx.website.naver.core import Sise

from .common_utils import add_days


# n거래일 조회
# 당일 날짜는 오전 9시 이후 가져올 수 있다.
def get_nearest_business_dates(date: str, ndays: int):
    dates = []
    while len(dates) < ndays:
        date = stock_api.get_nearest_business_day_in_a_week(date)
        dates.append(date)
        date = add_days(date, -1)
    return dates


def get_market_cap_by_ticker(date: str, market: str = "ALL", ascending: bool = False) -> DataFrame:
    market2mktid = {
        "ALL": "ALL",
        "KOSPI": "STK",
        "KOSDAQ": "KSQ",
        "KONEX": "KNX"
    }

    df = 전종목시세().fetch(date, market2mktid[market])
    time.sleep(0.2)

    df = df[['ISU_SRT_CD', 'ISU_ABBRV', 'TDD_OPNPRC', 'TDD_LWPRC', 'TDD_HGPRC', 'TDD_CLSPRC', 'FLUC_RT', 'MKTCAP',
             'ACC_TRDVOL', 'ACC_TRDVAL', 'LIST_SHRS', 'MKT_NM']]
    df.columns = ['stock_code', 'stock_name', 'open', 'low', 'high', 'close', 'close_rate', 'market_cap',
                  'trading_volume', 'trading_value', 'shares', 'market_name']

    df = df.replace(r'[^-\w\.]', '', regex=True)
    df = df.replace(r'\-$', '0', regex=True)
    df = df.replace('', '0')
    df = df.astype({
        'open': np.int32,
        'low': np.int32,
        'high': np.int32,
        'close': np.int32,
        'close_rate': np.float32,
        'market_cap': np.int64,
        'trading_volume': np.int64,
        'trading_value': np.int64,
        'shares': np.int64,
    })
    df['date'] = date

    holiday = (df[['close', 'market_cap', 'trading_volume', 'trading_value']] == 0).all(axis=None)
    if holiday:
        return None
    else:
        return df.sort_values('trading_value', ascending=ascending)


def get_kospi_index(start_date: str, end_date: str) -> DataFrame:
    df = stock_api.get_index_ohlcv_by_date(fromdate=start_date, todate=end_date, ticker="1001", freq= 'd', name_display=False)
    time.sleep(0.2)
    df = df.reset_index()
    df = df.rename(columns={
        '날짜': 'date',
        '시가': 'open',
        '고가': 'high',
        '저가': 'low',
        '종가': 'close',
        '거래량': 'trading_volume',
        '거래대금': 'trading_value',
        '상장시가총액': 'market_cap',
    })
    df["date"] = df["date"].dt.strftime("%Y%m%d")
    df['market_name'] = 'KOSPI'
    return df


def get_kosdaq_index(start_date: str, end_date: str) -> DataFrame:
    df = stock_api.get_index_ohlcv_by_date(fromdate=start_date, todate=end_date, ticker="2001", freq= 'd', name_display=False)
    time.sleep(0.2)
    df = df.reset_index()
    df = df.rename(columns={
        '날짜': 'date',
        '시가': 'open',
        '고가': 'high',
        '저가': 'low',
        '종가': 'close',
        '거래량': 'trading_volume',
        '거래대금': 'trading_value',
        '상장시가총액': 'market_cap',
    })
    df["date"] = df["date"].dt.strftime("%Y%m%d")
    df['market_name'] = 'KOSDAQ'
    return df


# https://github.com/sharebook-kr/pykrx
# 종목 날짜별 시가/고가/저가/종가/거래량/등락률
# 당일 데이터는 오후 3시 20분 이후 가져올 수 있다.
def get_ohlcv(ticker, start_date, end_date):
    df = stock_api.get_market_ohlcv_by_date(fromdate=start_date, todate=end_date, ticker=ticker)
    if df.empty:
        return df

    df = df.rename(columns={
        '시가': 'open',
        '고가': 'high',
        '저가': 'low',
        '종가': 'close',
        '거래량': 'trading_volume',
        '등락률': 'change_rate',
    })

    # 값이 None인 경우 제거
    df = df.dropna()

    # 필수 컬럼이 0인 경우 제거
    mandatory_columns = ['open', 'high', 'low', 'close', 'trading_volume']
    df = df[~df[mandatory_columns].eq(0).any(axis=1)]

    # 인덱스 리셋(날짜는 일반 컬럼으로)
    df = df.reset_index()
    df["date"] = df["날짜"].dt.strftime("%Y%m%d")

    # 컬럼 추가
    df.insert(0, 'stock_code', ticker)

    # 타입 변경
    df = df.astype({
        'open': 'int',
        'high': 'int',
        'low': 'int',
        'close': 'int',
        'trading_volume': 'int',
        'change_rate': 'float',
    })

    df = df[['stock_code', 'date', 'open', 'high', 'low', 'close', 'trading_volume', 'change_rate']]
    time.sleep(0.01)
    return df


# 종목 날짜별 투자자 순매수 거래량
# 당일 데이터는 오후 3시 50분 이후 가져올 수 있다.
def get_investor_net_volumes(ticker, start_date, end_date):
    df = stock_api.get_market_trading_volume_by_date(fromdate=start_date, todate=end_date, ticker=ticker, detail=True,
                                                     on="순매수")
    if df.empty:
        return df

    df = df.rename(columns={
        '개인': 'individual',
        '외국인': 'foreign',
    })

    df['institution'] = df.apply(lambda x: x['금융투자'] + x['보험'] + x['투신'] + x['사모'] + x['은행'] + x['기타금융'] + x['연기금'],
                                 axis=1)

    # 인덱스 리셋(날짜는 일반 컬럼으로)
    df = df.reset_index()
    df["date"] = df["날짜"].dt.strftime("%Y%m%d")

    # 컬럼 추가
    df.insert(0, 'stock_code', ticker)

    # 타입 변경
    df = df.astype({
        'individual': 'int',
        'foreign': 'int',
        'institution': 'int',
    })

    df = df[['stock_code', 'date', 'individual', 'foreign', 'institution']]
    time.sleep(0.01)
    return df


def download_market_data(year: int):
    os.makedirs('market', exist_ok=True)
    start_date = datetime(int(year), 1, 1)
    end_date = datetime(int(year), 12, 31)
    current_date = start_date
    df_list = []
    while current_date <= end_date and current_date < datetime.today():
        df = get_market_cap_by_ticker(current_date.strftime("%Y%m%d"))
        if df is not None:
            df_list.append(df)
        current_date += timedelta(days=1)

    file_name = f'market/market_{year}.parquet'
    if os.path.exists(file_name):
        os.remove(file_name)

    df = pd.concat(df_list)
    df.to_parquet(file_name, index=False)
    print(f'{file_name} saved')


def download_kospi_index(year):
    os.makedirs('index', exist_ok=True)
    file_name = f'index/kospi_{year}.parquet'
    if os.path.exists(file_name):
        print(f'{file_name} skipped')
    else:
        df = get_kospi_index(f'{year}0101', f'{year}1231')
        df.to_parquet(file_name, index=False)
        print(f'{file_name} saved')


def download_kosdaq_index(year):
    os.makedirs('index', exist_ok=True)
    file_name = f'index/kosdaq_{year}.parquet'
    if os.path.exists(file_name):
        print(f'{file_name} skipped')
    else:
        df = get_kosdaq_index(f'{year}0101', f'{year}1231')
        df.to_parquet(file_name, index=False)
        print(f'{file_name} saved')


def get_market_trading_value_and_volume_on_ticker_by_date(fromdate: str, todate: str, stock_code: str) -> DataFrame:
    isin = get_stock_ticker_isin(stock_code)

    option_a = "거래량"
    option_b = "순매수"

    option_a = {"거래량": 1, "거래대금": 2}.get(option_a, 1)
    option_b = {"매도": 1, "매수": 2, "순매수": 3}.get(option_b, 3)

    df = 투자자별_거래실적_개별종목_일별추이_일반().fetch(fromdate, todate, isin, option_a, option_b)
    df.columns = ['date', 'institution', '기타법인', 'individual', 'foreign', 'total']

    df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d').dt.strftime("%Y%m%d")
    df = df.replace(r'[^-\w]', '', regex=True)
    df = df.replace('', '0')

    df["stock_code"] = stock_code
    df = df[['stock_code', 'date', 'individual', 'foreign', 'institution']]

    # 타입 변경
    df = df.astype({
        'individual': np.int64,
        'foreign': np.int64,
        'institution': np.int64,
    })

    df = df.sort_values(by='date', ascending=True)
    return df


def download_stock_trading(stock_code: str, start_date: str, end_date: str):
    os.makedirs('stocks/trading', exist_ok=True)
    file_name = f'stocks/trading/stock_trading_{stock_code}.parquet'
    if os.path.exists(file_name):
        df = pd.read_parquet(file_name)
        return df

    df = get_market_trading_value_and_volume_on_ticker_by_date(start_date, end_date, stock_code)
    df.to_parquet(file_name, index=False)
    print(f'{file_name} saved')
    time.sleep(0.05)
    return df


# krx는 가격 보정이 안 되어 있어서 네이버 ohlcv를 사용한다.
def download_stock_ohlcv(stock_code: str, start_date: str, end_date: str):
    os.makedirs('stocks/ohlcv', exist_ok=True)
    start_date = datetime.strptime(start_date, '%Y%m%d')
    days_difference = (datetime.today() - start_date).days
    trading_days = (days_difference * 260) / 365

    file_name = f'stocks/ohlcv/stock_ohlcv_{stock_code}.parquet'
    if os.path.exists(file_name):
        df = pd.read_parquet(file_name)
        return df

    xml = Sise().fetch(stock_code, trading_days)
    result = []
    try:
        for node in et.fromstring(xml).iter(tag='item'):
            row = node.get('data')
            result.append(row.split("|"))

        cols = ['date', 'open', 'high', 'low', 'close', 'trading_volume']
        df = DataFrame(result, columns=cols)
        df = df.astype({
            'open': np.int32,
            'high': np.int32,
            'low': np.int32,
            'close': np.int32,
            'trading_volume': np.int64,
        })

        df["stock_code"] = stock_code
        df = df.iloc[1:].reset_index(drop=True)
        df.to_parquet(file_name, index=False)
        print(f'{file_name} saved')
        time.sleep(0.05)
        return df
    except et.ParseError:
        return None


def get_market_years(file_name_prefix, start_year: int, end_year: int) -> DataFrame:
    df_list = []
    os.makedirs('market', exist_ok=True)
    for year in range(start_year, end_year + 1):
        file_name = f'market/{file_name_prefix}_{year}.parquet'
        if os.path.exists(file_name):
            df = pd.read_parquet(file_name)
            df_list.append(df)
    return pd.concat(df_list)


def get_index_years(file_name_prefix, start_year: int, end_year: int) -> DataFrame:
    df_list = []
    os.makedirs('index', exist_ok=True)
    for year in range(start_year, end_year + 1):
        file_name = f'index/{file_name_prefix}_{year}.parquet'
        if os.path.exists(file_name):
            df = pd.read_parquet(file_name)
            df_list.append(df)
    return pd.concat(df_list)


def index_csv_to_year_parquet(file_name):
    # CSV 파일 읽기
    df = pd.read_csv(f"index/{file_name}.csv")
    df = df.rename(columns={
        '날짜': 'date',
        '종가': 'close',
        '시가': 'open',
        '고가': 'high',
        '저가': 'low',
        '거래량': 'trading_volume',
        '변동 %': 'close_rate',
    })

    df = df.drop(columns=['trading_volume'])

    # 날짜 포맷 정리
    df["date"] = df["date"].str.replace(" ", "")
    df["date"] = df["date"].str.replace("-", "")
    df["datetime"] = pd.to_datetime(df["date"], format="%Y%m%d")

    string_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in string_columns:
        if column in ["close", "open", "high", "low"]:
            df[column] = df[column].str.replace(",", "").astype(float)

    # 변동 % 컬럼 정리 (문자열 % 제거 후 float 변환)
    df["close_rate"] = df["close_rate"].str.replace("%", "").astype(float)

    # 연도별로 데이터 분할하여 parquet 파일로 저장
    for year, year_df in df.groupby(df["datetime"].dt.year):
        year_df = year_df.drop(columns=["datetime"])
        year_df.to_parquet(f"index/{file_name}_{year}.parquet", index=False)
        print(f"index/{file_name}_{year}.parquet 파일 저장 완료")


# 코스피 변동성 지수 조회
def get_kospi_volatility(from_date, to_date):
    # http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201010303
    # https://github.com/FinanceData/FinanceDataReader
    headers = {
        'User-Agent': 'Chrome/135.0.0.0 Safari/537.36', 
        'Referer': 'http://data.krx.co.kr/',
    }
    data = {
        'bld': 'dbms/MDC/STAT/standard/MDCSTAT01201',
        'locale': 'ko_KR',
        'indTpCd': '1',
        'idxIndCd': '300',
        'idxCd': '1',
        'idxCd2': '300',
        'strtDd': from_date,
        'endDd': to_date,
        'csvxls_isNo': 'false',
    }

    url = 'http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd'
    r = requests.post(url, data, headers=headers)
    df = pd.DataFrame(r.json()['output'])

    df = df.rename(columns={
        'TRD_DD': 'date',
        'CLSPRC_IDX': 'close',
        'PRV_DD_CMPR': 'close_rate',
    })
    df['date'] = df['date'].str.replace('/', '', regex=False)
    df = df[['date', 'close', 'close_rate']]
    df = df.sort_values(by=['date'], ascending=[True]).reset_index(drop=True)
    time.sleep(0.05)
    return df


def get_yfinance_ohlcv(ticker: str, ndays: int):
    df = yf.Ticker(ticker).history(period=f"{ndays}d", interval="1d").reset_index()
    df = df.sort_values(by=["Date"], ascending=[True]).reset_index(drop=True)
    df = df.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
    })
    df['date'] = df['date'].dt.tz_convert('Asia/Seoul').dt.strftime('%Y%m%d')

    # FIXME: 전체 코드에 time.sleep 대신 @rate_limit 적용
    time.sleep(0.05)
    return df[['date', 'open', 'high', 'low', 'close', 'volume']]


def get_sp500_futures(ndays: int):
    return get_yfinance_ohlcv("ES=F", ndays)


def get_nasdaq100_futures(ndays: int):
    return get_yfinance_ohlcv("NQ=F", ndays)


def get_sp500_vix(ndays: int):
    return get_yfinance_ohlcv("^VIX", ndays)
