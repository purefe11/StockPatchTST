# 📈 주요 피처 설명 및 수식 정리

PatchTST 기반 주가 랭킹 모델에 사용된 주요 기술 지표 및 캔들 특성 피처에 대한 정의와 수식

---

## 🕯️ 1. 캔들 차트 관련 피처
<img src=https://github.com/user-attachments/assets/5e6540fc-3ce1-4a93-8b66-e1a2dca4c559 width=600/>

### ✅ `open_to_close`
- 시가 대비 종가의 변화율<br>
![open_to_close](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;\frac{close-open}{open})

### ✅ `high_to_low`
- 고가와 저가의 상대 차이 (변동성 측정)<br>   
![high_to_low](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;\frac{high-low}{low})

### ✅ `candle_upper_tail_ratio`
- 윗꼬리 길이 비율<br>
![upper_tail](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;\frac{high-\max(open,&space;close)}{close})

### ✅ `candle_lower_tail_ratio`
- 아랫꼬리 길이 비율<br>
![lower_tail](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;\frac{\min(open,&space;close)-low}{close})

### ✅ `candle_body_ratio`
- 몸통(body)의 비율<br>
![body](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;\frac{|close-open|}{high-low})

---

## 🔄 2. RSI (Relative Strength Index)

- 최근 N일 동안 상승/하락 강도의 상대 비율<br>
![RSI](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;\text{RSI}=&space;100&space;-&space;\frac{100}{1&space;+&space;\frac{AvgGain}{AvgLoss}})
- 일반적으로 `N = 14` 사용

---

## 📏 3. ATR (Average True Range)

- 일정 기간 동안의 평균 변동 폭<br>
![ATR](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;\text{ATR}_t&space;=&space;\text{EMA}_n(\text{TR}_t))

- TR (True Range)<br>
![TR](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;\text{TR}_t&space;=&space;\max(high_t-low_t,&space;|high_t-close_{t-1}|,&space;|low_t-close_{t-1}|))

---

## ⚡ 4. MACD (Moving Average Convergence Divergence)

- 단기-장기 이동평균 간의 괴리도로 추세 전환을 포착
  - MACD Line<br>
    ![macd](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;\text{MACD}&space;=&space;\text{EMA}_{12}(close)&space;-&space;\text{EMA}_{26}(close))
  - Signal Line<br>  
    ![macd_signal](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;\text{Signal}&space;=&space;\text{EMA}_9(\text{MACD}))
  - Histogram<br>  
    ![macd_hist](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;\text{MACD}_{hist}&space;=&space;\text{MACD}&space;-&space;\text{Signal})

---

## 🟣 5. VWMA (Volume Weighted Moving Average)

- 거래량을 고려한 가중 이동 평균. 대량 거래 시 반영 비중이 커짐<br>
![VWMA](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;\text{VWMA}_t&space;=&space;\frac{\sum_{i=0}^{n-1}&space;price_{t-i}&space;\times&space;volume_{t-i}}{\sum_{i=0}^{n-1}&space;volume_{t-i}})

---

## 📉 6. VWMA 기반 볼린저 밴드

- VWMA를 기준으로 상하단 변동폭을 시각화하여 과매수/과매도 상태 탐지
  - 중심선 (VWMA)<br>
    ![bb_mid](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;\text{BB}_{mid}&space;=&space;\text{VWMA}_n)
  - 상단선<br>
    ![bb_upper](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;\text{BB}_{upper}&space;=&space;\text{VWMA}_n&space;+&space;k&space;\cdot&space;\text{std}(close,\&space;n))
  - 하단선<br>
    ![bb_lower](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;\text{BB}_{lower}&space;=&space;\text{VWMA}_n&space;-&space;k&space;\cdot&space;\text{std}(close,\&space;n))
  - 보통 `k = 2`, `n = 20` 사용

---

## 📌 참고

- EMA: 지수 이동 평균
- std: 표준편차
- close, high, low: 종가, 고가, 저가
- `t`는 시점(index), `n`은 기간(window size)
