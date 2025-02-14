import json
from pathlib import Path
import env


class Settings:
    def __init__(self):
        # 한국투자증권 open api
        # https://apiportal.koreainvestment.com/apiservice
        # https://github.com/koreainvestment/open-trading-api/tree/main/postman
        self.KI_APP_KEY = env.KI_APP_KEY
        self.KI_APP_SECRET = env.KI_APP_SECRET

        # 네이버 API 키
        # https://developers.naver.com/products/service-api/search/search.md
        self.NAVER_CLIENT_ID = env.NAVER_CLIENT_ID
        self.NAVER_CLIENT_SECRET = env.NAVER_CLIENT_SECRET

        # OpenAI API 키 설정
        # https://platform.openai.com/settings/organization/usage
        self.OPEN_API_KEY = env.OPEN_API_KEY

        # 구글 서비스 계정 인증 정보
        # https://console.cloud.google.com/marketplace/product/google/drive.googleapis.com
        self.GOOGLE_SERVICE_ACCOUNT_DICT = env.GOOGLE_SERVICE_ACCOUNT_DICT

settings = Settings()
