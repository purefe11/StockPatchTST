# ⚙️ 손실 함수 LambdaRankLoss 단계별 설명

> 모델이 예측한 점수(`preds`)와 실제 라벨(`targets`)을 기반으로,  
> "쌍(pair) 간 순서가 잘 맞았는가?"를 판단하여  
> **NDCG가 높아지도록 유도하는 손실 함수**

---

### 🔹 1. 입력 정의

- `preds`: 모델이 출력한 예측 점수 (종목 간 우월성, 높을 수록 상위 랭킹 예측)
- `targets`: 실제 수익률 기반 라벨 (5일 후 수익률 n분위수)

```python
preds:  Tensor of shape (N,)
targets: Tensor of shape (N,)
```

---

### 🔹 2. 안전한 relevance 처리

```python
safe_targets = torch.clamp_min(targets, 0.0)
```

- NDCG 계산은 음수 relevance(우월성 점수)를 허용하지 않기 때문에 **음수 값이 들어올 경우 0으로 보정**
- n분위 등급처럼 **0 이상 정수 값만 사용하는 경우 이 단계는 영향 없음**

---

### 🔹 3. 이상적인 DCG (IDCG) 계산

```python
sorted_targets, _ = torch.sort(safe_targets, descending=True)
ideal_dcg = dcg(sorted_targets)
```

- 정답 순서대로 완벽히 정렬했을 때 얻을 수 있는 **최고 점수**
- 이후의 delta-NDCG 계산을 위해 **정규화 기준값**으로 사용

---

### 🔹 4. 모든 쌍에 대해 예측 점수 차이 및 정답 차이 계산

```python
pred_diffs = preds.unsqueeze(1) - preds.unsqueeze(0)
target_diffs = targets.unsqueeze(1) - targets.unsqueeze(0)
S_ij = torch.sign(target_diffs)
```

- `S_ij = +1`: i가 j보다 더 relevance 높음 (우선 순위 높음)
- `S_ij = -1`: j가 i보다 relevance 높음
- `S_ij = 0`: 둘 다 같은 등급 → **loss 계산에서 제외됨**

---

### 🔹 5. 예측된 점수 기준 순위(rank) 계산

```python
rank_positions = compute_ranks(preds) - 1  # 0-based
log_i = log2(pos_i + 2)
log_j = log2(pos_j + 2)
```

- **예측 순위를 기반으로 DCG 계산에서 사용할 log 할인 계수 생성**

---

### 🔹 6. delta NDCG 계산

```python
delta_dcg = |1/log_i - 1/log_j| × |target_diffs|
delta_ndcg = delta_dcg / ideal_dcg
```

- **두 종목의 순서를 바꾸면 얼마나 NDCG가 변화할까?**
- 이 값이 **크면 클수록 중요한 쌍** → 학습에서 더 크게 반영

---

### 🔹 7. 최종 손실 계산

```python
log_loss = F.logsigmoid(S_ij * pred_diffs)
mask = (S_ij != 0).float()
loss_matrix = -log_loss * delta_ndcg * mask
```

- `logsigmoid(S_ij * pred_diff)`:
  - 순서가 맞으면 → 손실 작음
  - 순서가 틀리면 → 손실 큼
- 여기에 **delta-NDCG를 곱해 "얼마나 중요한 실수였는지" 반영**

---

### 🔹 8. 평균 손실 반환

```python
loss_sum / pair_count
```

- 유효한 쌍(`S_ij ≠ 0`)만 평균 내서 손실 계산

---

### ✅ 핵심 요약

| 단계 | 설명 |
|------|------|
| 쌍 생성 | 모든 종목을 비교하여 순서가 맞는지 판단 |
| 중요도 계산 | 순서가 틀렸을 때 NDCG가 얼마나 손해보는지 계산 |
| 학습 유도 | 손해가 큰 쌍은 크게 벌주고, 작은 쌍은 덜 벌줌 |

---

> 🔁 결과적으로, LambdaRankLoss는  
> **"순서를 바꿨을 때 NDCG 점수가 얼마나 좋아지는지를 계산해서,  
> 모델에게 그 방향으로 점수를 조정하라고 학습시키는 방식"**

<br><br>

# 🧠 LambdaRankLoss에서 동일 분위수(공동 등수)가 많은 경우

### 🔢 예시
- 일별 최대 200개 종목  
- 수익률을 5분위로 라벨링 (0, 1, 2, 3, 4)  
- 각 분위에 40개씩 고르게 분포

---

### 📊 쌍(pair)의 개수 계산

- **전체 쌍 수**:

  ![전체쌍수](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;\binom{200}{2}&space;=&space;\frac{200&space;\times&space;199}{2}&space;=&space;19,900)

- **같은 분위 내 쌍 수** (예: 분위 0 내 40개 중 2개 선택):

  ![같은분위수쌍](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;\binom{40}{2}&space;=&space;\frac{40&space;\times&space;39}{2}&space;=&space;780)

- 분위가 5개니까:

  ![전체제외쌍](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;780&space;\times&space;5&space;=&space;3,900)

---

### 🚫 LambdaRankLoss에서 제외되는 쌍

- 같은 분위에 속한 종목들은 **relevance 차이 = 0**

  ![Sij_0](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;S_{ij}&space;=&space;\text{sign}(target_i&space;-&space;target_j)&space;=&space;0)

- → `S_ij = 0`인 쌍은 **loss 계산에서 제외**

---

### ✅ 문제인가?

**여전히 다음과 같은 유효 쌍이 많이 존재하기 때문에 큰 문제는 아니다!**

  ![유효쌍](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;19,900&space;-&space;3,900&space;=&space;16,000)

- 유효한 비교 쌍이 **16,000쌍 이상** 존재
- **학습 신호가 줄어들긴 하지만 충분히 남아 있음**
- 이런 동작은 **pairwise 랭킹 손실 함수의 자연스러운 특성**

---

### 🔁 참고: 개선 아이디어

- **분위 수 늘리기** (예: 20 분위) → 더 많은 유효 쌍 확보

<br><br>

# 📌 Appendix: 손실 함수 비교 정리

**회귀**, **분류**, **랭킹** 모델에서 자주 사용하는 손실 함수들

---

## ✅ 회귀(Regression) 모델용 손실 함수

| 손실 함수 | 설명 | 수식 / 특징 |
|-----------|------|--------------|
| **MSE (Mean Squared Error)** | 평균 제곱 오차 | ![MSE = (1/n) Σ(yᵢ - ŷᵢ)²](https://latex.codecogs.com/svg.image?MSE%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum%28y_i%20-%20%5Chat%7By%7D_i%29%5E2) <br> 이상치에 민감 |
| **MAE (Mean Absolute Error)** | 평균 절대 오차 | ![MAE = (1/n) Σ|yᵢ - ŷᵢ|](https://latex.codecogs.com/svg.image?MAE%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum%20%7C%20y_i%20-%20%5Chat%7By%7D_i%20%7C) <br> 이상치에 덜 민감 |
| **Huber Loss** | MSE + MAE 혼합형 | 작은 오차는 MSE, 큰 오차는 MAE로 처리 <br> 튜닝 가능한 δ 존재 |
| **Log-Cosh Loss** | 부드러운 MAE 형태 | `log(cosh(y - ŷ))` <br> 안정적인 학습 가능 |

---

## ✅ 분류(Classification) 모델용 손실 함수

### 🔸 이진 분류

| 손실 함수 | 설명 | 수식 / 특징 |
|-----------|------|--------------|
| **Binary Cross Entropy (BCE)** | 확률 기반 이진 분류 | ![BCE = - (y log p + (1 - y) log (1 - p))](https://latex.codecogs.com/svg.image?BCE%20%3D%20-%28y%20%5Clog%20p%20&plus;%20%281-y%29%20%5Clog%281-p%29%29) |
| **Focal Loss** | 불균형 이진 분류 대응 | 어려운 샘플에 가중치 부여 <br> `FL = -α(1-p)^γ log(p)` |

### 🔸 다중 클래스 분류

| 손실 함수 | 설명 | 특징 |
|-----------|------|------|
| **Categorical Cross Entropy** | one-hot 라벨용 | softmax + log likelihood |
| **Sparse Categorical Cross Entropy** | 정수 라벨용 | 라벨이 one-hot 아닌 경우 사용 |
| **Label Smoothing Loss** | hard label 완화 | 라벨을 부드럽게 하여 과적합 완화 |
| **Hinge Loss** | SVM 계열 | `max(0, 1 - y·ŷ)` 형태로 마진 기반 |

---

## ✅ 랭킹(Ranking) 모델용 손실 함수

| 손실 함수 | 설명 | 특징 |
|-----------|------|------|
| **Pairwise Ranking Loss** | 두 샘플 간 상대 순위 학습 | `max(0, margin - (sᵢ - sⱼ))` |
| **Triplet Loss** | anchor, positive, negative 관계 학습 | `(‖f(a) - f(p)‖² - ‖f(a) - f(n)‖² + margin)+` |
| **ListNet** | 확률적 랭킹 예측 | softmax 기반 정렬 확률 비교 |
| **ListMLE** | 정렬 정확도 기반 | likelihood 최대화 |
| **LambdaRank / LambdaMART** | NDCG 최적화 기반 | XGBoost / LightGBM 등에서 사용 |
| **RankNet** | 시그모이드 기반 pairwise 비교 | 순위 점수 차이에 sigmoid 적용 |

---

📌 **Tip:**  
- 회귀는 연속값 예측, 분류는 클래스 예측, 랭킹은 상대 순위 예측에 초점  
- 불균형 데이터에는 Focal Loss 권장

