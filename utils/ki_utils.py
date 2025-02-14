import json
import time
from datetime import datetime, timedelta

import requests

from settings import settings


def get_ki_access_token():
    url = "https://openapi.koreainvestment.com:9443/oauth2/tokenP"

    payload = json.dumps({
        "grant_type": "client_credentials",
        "appkey": settings.KI_APP_KEY,
        "appsecret": settings.KI_APP_SECRET
    })
    headers = {
        'content-type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json()['access_token']


if 'KI_ACCESS_TOKEN' not in globals():
    KI_ACCESS_TOKEN = get_ki_access_token()

if 'STOCK_INDUSTRY_CODES' not in globals():
    STOCK_INDUSTRY_CODES = dict()


# 주식 기본 조회 API
def get_stock_info(stock_code: str):
    if stock_code in STOCK_INDUSTRY_CODES:
        return STOCK_INDUSTRY_CODES[stock_code]

    url = f"https://openapi.koreainvestment.com:9443/uapi/domestic-stock/v1/quotations/search-stock-info?PRDT_TYPE_CD=300&PDNO={stock_code}"
    payload = ""
    headers = {
        'content-type': 'application/json',
        'authorization': f'Bearer {KI_ACCESS_TOKEN}',
        'appkey': settings.KI_APP_KEY,
        'appsecret': settings.KI_APP_SECRET,
        'tr_id': 'CTPF1002R'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    output = response.json()['output']
    time.sleep(0.05)
    kospi_listing_date = output['scts_mket_lstg_dt']
    kosdaq_listing_date = output['kosdaq_mket_lstg_dt']
    listing_date = kospi_listing_date if kospi_listing_date > kosdaq_listing_date else kosdaq_listing_date
    output = {
        'industry_code': output['std_idst_clsf_cd'],  # 표준산업분류코드
        'listing_date': listing_date,
    }
    STOCK_INDUSTRY_CODES[stock_code] = output
    return output


# 주식 현재가 시세 API
def get_ki_ohlcv(stock_code: str):
    url = f"https://openapi.koreainvestment.com:9443/uapi/domestic-stock/v1/quotations/inquire-price?fid_cond_mrkt_div_code=J&fid_input_iscd={stock_code}"
    payload = ""
    headers = {
        'content-type': 'application/json',
        'authorization': f'Bearer {KI_ACCESS_TOKEN}',
        'appkey': settings.KI_APP_KEY,
        'appsecret': settings.KI_APP_SECRET,
        'tr_id': 'FHKST01010100'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    output = response.json()['output']
    time.sleep(0.05)
    return {
        'stock_code': stock_code,
        'open': int(output['stck_oprc']),
        'low': int(output['stck_lwpr']),
        'high': int(output['stck_hgpr']),
        'close': int(output['stck_prpr']),
        'trading_volume': int(output['acml_vol']),
        'change_rate': float(output['prdy_ctrt']),
        'change_volume': float(output['prdy_vrss_vol_rate']),
        'per': output['per'],
        'pbr': output['pbr'],
        'eps': output['eps'],
        'bps': output['bps'],
        'high_52w': int(output['w52_hgpr']),
        'high_52w_rate': 100 + float(output['w52_hgpr_vrss_prpr_ctrt']),
        'caution': True if output['invt_caful_yn'] == 'Y' else False,
        'warn': True if int(output['mrkt_warn_cls_code']) else False,
        'overheat': True if output['short_over_yn'] == 'Y' else False,
        'prgm': int(output['pgtr_ntby_qty']),
    }


# 국내주식 종목별 외국인, 기관 추정가집계 API
def get_ki_investor_trend_estimate(stock_code: str):
    url = f"https://openapi.koreainvestment.com:9443/uapi/domestic-stock/v1/quotations/investor-trend-estimate?MKSC_SHRN_ISCD={stock_code}"
    payload = ""
    headers = {
        'content-type': 'application/json',
        'authorization': f'Bearer {KI_ACCESS_TOKEN}',
        'appkey': settings.KI_APP_KEY,
        'appsecret': settings.KI_APP_SECRET,
        'tr_id': 'HHPTJ04160200'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    output = response.json()['output2']
    result = {
        'stock_code': stock_code,
        'individual': 0,
        'foreign': 0,
        'institution': 0,
    }

    # 증권사 직원이 장중에 집계/입력한 자료를 단순 누계한 수치로서
    # 입력시간은 외국인 09:30, 11:20, 13:20, 14:30 / 기관종합 10:00, 11:20, 13:20, 14:30 이며, 사정에 따라 변동될 수 있다.
    bsop_hour_gb = "0"
    # 1: 09시 30분 입력
    # 2: 10시 00분 입력
    # 3: 11시 20분 입력
    # 4: 13시 20분 입력
    # 5: 14시 30분 입력
    for item in output:
        if item['bsop_hour_gb'] > bsop_hour_gb:
            result['foreign'] = int(item['frgn_fake_ntby_qty'])
            result['institution'] = int(item['orgn_fake_ntby_qty'])
            bsop_hour_gb = item['bsop_hour_gb']
    time.sleep(0.05)
    return result


# 국내주식 시간외현재가 API
def get_ki_overtime_price(stock_code: str):
    url = f"https://openapi.koreainvestment.com:9443/uapi/domestic-stock/v1/quotations/inquire-overtime-price?FID_COND_MRKT_DIV_CODE=J&FID_INPUT_ISCD={stock_code}"
    payload = ""
    headers = {
        'content-type': 'application/json',
        'authorization': f'Bearer {KI_ACCESS_TOKEN}',
        'appkey': settings.KI_APP_KEY,
        'appsecret': settings.KI_APP_SECRET,
        'tr_id': 'FHPST02300000'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    output = response.json()['output']
    time.sleep(0.05)
    return {
        'stock_code': stock_code,
        'over_close': int(output['ovtm_untp_prpr']),
        'over_rate': float(output['ovtm_untp_prdy_ctrt']),
    }


# 주식현재가 시간외일자별주가 API
def get_ki_daily_overtime_price(stock_code: str):
    url = f"https://openapi.koreainvestment.com:9443/uapi/domestic-stock/v1/quotations/inquire-daily-overtimeprice?FID_COND_MRKT_DIV_CODE=J&FID_INPUT_ISCD={stock_code}"
    payload = ""
    headers = {
        'content-type': 'application/json',
        'authorization': f'Bearer {KI_ACCESS_TOKEN}',
        'appkey': settings.KI_APP_KEY,
        'appsecret': settings.KI_APP_SECRET,
        'tr_id': 'FHPST02320000'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    output = response.json()['output2']
    time.sleep(0.05)
    return [{
        'date': x['stck_bsop_date'],
        'over_close': int(x['ovtm_untp_prpr']),
        'over_rate': float(x['ovtm_untp_prdy_ctrt']),
    } for x in output]


# 종합 시황/공시 API
def get_ki_stock_notice_count(stock_code: str, ndays: int = 7):
    # 코스피/코스닥 공시 가져오기
    url = f"https://openapi.koreainvestment.com:9443/uapi/domestic-stock/v1/quotations/news-title?FID_NEWS_OFER_ENTP_CODE=FG&FID_COND_MRKT_CLS_CODE=&FID_INPUT_ISCD={stock_code}&FID_TITL_CNTT=&FID_INPUT_DATE_1=&FID_INPUT_HOUR_1=&FID_RANK_SORT_CLS_CODE=&FID_INPUT_SRNO="
    payload = ""
    headers = {
        'content-type': 'application/json',
        'authorization': f'Bearer {KI_ACCESS_TOKEN}',
        'appkey': settings.KI_APP_KEY,
        'appsecret': settings.KI_APP_SECRET,
        'tr_id': 'FHKST01011800'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    output = response.json()['output']
    time.sleep(0.05)

    notice_count = 0
    for x in output:
        # n일 이내 작성된 공시인지 확인
        date = datetime.strptime(x['data_dt'], '%Y%m%d')
        if date >= (datetime.today() - timedelta(days=ndays)):
            notice_count += 1
    time.sleep(0.05)
    return {
        'stock_code': stock_code,
        'notice': notice_count,
    }


# 주식현재가 투자자 API
def get_ki_investor(stock_code: str):
    url = f"https://openapi.koreainvestment.com:9443/uapi/domestic-stock/v1/quotations/inquire-investor?FID_COND_MRKT_DIV_CODE=J&FID_INPUT_ISCD={stock_code}"
    payload = ""
    headers = {
        'content-type': 'application/json',
        'authorization': f'Bearer {KI_ACCESS_TOKEN}',
        'appkey': settings.KI_APP_KEY,
        'appsecret': settings.KI_APP_SECRET,
        'tr_id': 'FHKST01010900 '
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    output = response.json()['output']
    time.sleep(0.05)

    if not output[0]['prsn_ntby_qty']:
        return None

    return {
        'stock_code': stock_code,
        'individual': int(output[0]['prsn_ntby_qty']),
        'foreign': int(output[0]['frgn_ntby_qty']),
        'institution': int(output[0]['orgn_ntby_qty']),
    }


def get_ki_kospi_daily_rate():
    url = f"https://openapi.koreainvestment.com:9443/uapi/domestic-stock/v1/quotations/inquire-index-daily-price?FID_PERIOD_DIV_CODE=D&FID_COND_MRKT_DIV_CODE=U&FID_INPUT_ISCD=0001&FID_INPUT_DATE_1="
    payload = ""
    headers = {
        'content-type': 'application/json',
        'authorization': f'Bearer {KI_ACCESS_TOKEN}',
        'appkey': settings.KI_APP_KEY,
        'appsecret': settings.KI_APP_SECRET,
        'tr_id': 'FHPUP02120000 '
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    output = response.json()['output2']
    time.sleep(0.05)

    return [{
        'date': x['stck_bsop_date'],
        'kospi_rate': x['bstp_nmix_prdy_ctrt'],
    } for x in output]


def get_ki_kosdaq_daily_rate():
    url = f"https://openapi.koreainvestment.com:9443/uapi/domestic-stock/v1/quotations/inquire-index-daily-price?FID_PERIOD_DIV_CODE=D&FID_COND_MRKT_DIV_CODE=U&FID_INPUT_ISCD=1001&FID_INPUT_DATE_1="
    payload = ""
    headers = {
        'content-type': 'application/json',
        'authorization': f'Bearer {KI_ACCESS_TOKEN}',
        'appkey': settings.KI_APP_KEY,
        'appsecret': settings.KI_APP_SECRET,
        'tr_id': 'FHPUP02120000 '
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    output = response.json()['output2']
    time.sleep(0.05)

    return [{
        'date': x['stck_bsop_date'],
        'kosdaq_rate': x['bstp_nmix_prdy_ctrt'],
    } for x in output]
