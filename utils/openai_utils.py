import json
import logging

import openai

from settings import settings


openai.api_key = settings.OPEN_API_KEY

# 로깅 설정 (콘솔 출력 전용)
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # 콘솔 출력 전용
)

# 뉴스 타이틀에 대한 감정(긍정, 중립, 부정) 조회
def analyze_news_sentiment(news_title):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a financial news sentiment classifier."},
            {"role": "user", "content": news_title}
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "analyze_sentiment",
                    "description": "Analyze sentiment of a financial news headline.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sentiment": {
                                "type": "integer",
                                "enum": [-1, 0, 1],
                                "description": "Sentiment classification (-1=negative, 0=neutral, 1=positive)."
                            }
                        },
                        "required": ["sentiment"]
                    }
                }
            }
        ],
        tool_choice={"type": "function", "function": {"name": "analyze_sentiment"}},
    )

    try:
        arguments = response.choices[0].message.tool_calls[0].function.arguments
        result = json.loads(arguments)
        return int(result["sentiment"])
    except (json.JSONDecodeError, KeyError, ValueError, AttributeError):
        # 예외 발생 시 콘솔에 로그 출력
        logging.error(f"Sentiment analysis failed for news title: {news_title}, Error: {str(e)}")
        # 예외 발생 시 기본값 0 반환
        return 0