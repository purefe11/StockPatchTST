# Transformer 내부 구조 (쉬운 버전!)
> Transformer는 '여러 친구가 함께 회의하면서, 누가 중요한 말을 했는지를 판단하는 구조'로 이해하면 쉽다.
<img src=https://github.com/user-attachments/assets/72a1f54d-dfdf-49bf-8328-c2fff4ce4043 width=640/>

> 30일 동안의 주식 데이터를 9개 조각(패치)으로 잘랐다고 했을 때
> 각 조각은 6일 동안의 주가 흐름을 담고 있다.
> 이제 Transformer에서 이 9개의 조각에게 무슨 일이 일어나는지 살펴보자.

---

## 🧩 1. Multi-head Self-Attention  
**“각 조각이 다른 조각들을 바라보면서 중요도를 정한다.”**

- 퍼즐 조각들(패치)끼리 서로를 바라본다.  
- "너는 나와 관련 있어!", "넌 덜 중요해" 같은 판단을 여러 방향에서 한다.  
- 여러 개의 시선으로 본다고 해서 **Multi-head (여러 개의 눈)** 이라고 한다.

💡 *비유:*  
> 친구들과 모여서, "우리 중 누가 가장 중요한 이야기를 했는지" 여러 시선에서 생각해보는 것과 비슷하다.

---

## ➕ 2. Add & LayerNorm  
**“자신의 생각과 친구들 의견을 합쳐서 정리한다.”**

- 친구들의 의견(Attention 결과)과 원래 자기 생각을 더한다.  
- 그리고 그 결과가 너무 튀지 않도록 정리한다. (= LayerNorm)

💡 *비유:*  
> 친구 의견 + 내 의견 = 더 똑똑해진 나!

---

## 🔧 3. Feed Forward (선형 → ReLU → 선형)  
**“생각을 더 깊게 정리해서 다시 표현한다.”**

- 두 번의 계산을 통해(숫자 늘렸다가 줄이는 과정) 생각을 정리하고 다듬는다.  
- [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)는 "음수는 싫어!"라고 말하면서 필요한 정보만 남긴다.

💡 *비유:*  
> 한 번 더 생각해서 더 멋진 문장을 만드는 과정과 같다.

---

## ➕ 4. Add & LayerNorm (또 나온다!)  
**“이번에도 내 생각과 새로 정리한 생각을 합쳐서 다시 정리한다.”**

- 정리된 생각 + 원래 생각 → 한 번 더 정리!
- 이런 과정을 겹겹이 쌓으면서 더 똑똑해진다.

💡 *비유:*  
> 내가 한 말 + 새로 정리한 말 = 완성된 멋진 말!

---

## 🎉 Transformer가 뭘 해주는 걸까?

- 9개의 조각을 잘 이해했고(조각들 사이의 관계를 잘 파악하고),
- 어떤 조각이 중요한지도 알아낸다.  
- → 미래를 예측할 준비 완료!

---

## ✨ 한 줄 요약

> **Transformer는 여러 친구(조각)가 서로의 말을 들으며,
> 가장 중요한 정보를 찾아내고 똑똑하게 정리해 예측하는 구조다!**

<br>

# 📘 Transformer 인코더 (심화 버전!)
> ⚠️ 본 섹션에는 아래 출처의 도식 이미지 일부를 인용 또는 재구성하여 사용하였습니다.

- ["Attention is All You Need"](https://arxiv.org/abs/1706.03762)
- [Seq2seq Models With Attention](https://nlpinkorean.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [The Illustrated Transformer by Jay Alammar](https://nlpinkorean.github.io/illustrated-transformer/)

---

### 🧩 1. Query, Key, Value 생성

![QKV](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;Q%20%3D%20XW%5EQ%2C%5Cquad%20K%20%3D%20XW%5EK%2C%5Cquad%20V%20%3D%20XW%5EV)

![transformer_self_attention_vectors](https://github.com/user-attachments/assets/83db602c-417f-4982-a29d-a5a05bf3bf43)
![self-attention-matrix-calculation](https://github.com/user-attachments/assets/45aa8662-704b-426a-8b44-1b89c6a74637)

👉 설명:
- \( X \): 입력 시퀀스 (n개의 벡터, 각 벡터 차원 model_dim)
- 각각에 대해 **Query (질문), Key (정보의 위치), Value (실제 정보)** 를 만들기 위한 선형 변환 
- \( W^Q, W^K, W^V \): Query, Key, Value 생성을 위한 가중치 행렬

📌 핵심 개념:
- 각 입력 벡터를 질문, 정보 색인, 정보 내용의 세 가지로 나눠서 준비함

💡 *비유:*
- 문장: "나는 Transformer를 공부한다"
  - Q(Query): "공부한다"라는 단어에서 다른 단어가 얼마나 중요한지를 판단하기 위한 질문
  - K(Key): "나는", "Transformer", "를", 등 각 단어의 정보 요약 — 나와 관련 있는지를 판단하는 기준
  - V(Value): "나는", "Transformer", "를", 등의 실제 의미 벡터 — 중요한 정보를 가져올 때 사용할 값
- 📌 아래 Attention 과정을 통해 "공부한다"라는 단어는 자신(Q)이 중요하다고 판단한 단어(K)에 해당하는 정보를(V) 가져오게 된다.

---

### 2. Scaled Dot-Product Attention

![Attention](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;\text{Attention}(Q%2C%20K%2C%20V)%20%3D%20\text{softmax}\left(\frac{QK%5ET}{\sqrt{d_k}}\right)V)

![self-attention-output](https://github.com/user-attachments/assets/daf6360a-4ce6-4a0a-a80b-852f26bf0a72)<br>
![self-attention-matrix-calculation-2](https://github.com/user-attachments/assets/128ee31d-a2b3-4ffd-9fa8-c1eea4873345)

👉 설명:
- QK^T: Query와 Key의 내적을 통해 유사도 계산
- 1/sqrt(dk): 내적 값이 너무 커지지 않도록 스케일 조정
- softmax: 0~1 범위의 가중치로 바꿔서 중요도를 확률처럼 표현
- 위 결과(Attention Score)에 𝑉를 가중합(weighted sum)한다.

📌 핵심 개념:
- 입력 시퀀스 간 유사도에 따라 Value를 가중 평균해 주는 연산
- (각 위치의 입력 벡터가, 전체 시퀀스를 스캔하며 어떤 정보를 더 가져올지 결정하는 과정)

---

### 3. Multi-head Attention

![MultiHead](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;\text{MultiHead}(X)%20%3D%20\text{Concat}(\text{head}_1%2C%20\dots%2C%20\text{head}_h)W^O)

![transformer_multi-headed_self-attention-recap](https://github.com/user-attachments/assets/9864cf84-ed63-41f3-acb5-3b711aa94097)

👉 설명:
- 앞에서 계산한 Attention을 여러 개 병렬로 수행
→ 각 head가 서도 다른 관점(예: 단기 흐름, 장기 흐름)에서 Attention을 수행하고
- 마지막에 Concat해서 하나로 이어붙인 다음, W^O로 다시 projection

📌 핵심 개념:
- 여러 개의 시선으로 입력을 바라보고, 이들을 종합해 더 풍부한 정보로 변환

📦 [코드 예시: PyTorch `TransformerEncoderLayer`](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)
- 이 레이어는 내부적으로 multi-head attention과 feedforward block을 모두 포함하고 있음
- 즉, Transformer 인코더 블록 전체 구조를 한 줄로 사용할 수 있음

---

### 4. Add & LayerNorm

![AddNorm1](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;Z_1%20%3D%20\text{LayerNorm}(X%20+%20\text{MultiHead}(X)))

![transformer_resideual_layer_norm_2](https://github.com/user-attachments/assets/8e2a0fb0-ec50-4f81-b565-c72ec3098667)

👉 설명:
- 원래 입력 𝑋에 self-attention 결과를 더함 (Residual Connection)
- 그 다음, 전체 벡터를 정규화 (평균 0, 표준편차 1)
→ 학습 안정성, 그래디언트 흐름 개선

📌 핵심 개념:
- 기존 정보와 새로 계산된 정보를 섞은 뒤, 균형 잡힌 벡터로 다시 정리

---

### 5. Feed Forward Network (FFN)

![FFN](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;\text{FFN}(x)%20%3D%20\text{ReLU}(xW_1%20+%20b_1)W_2%20+%20b_2)

👉 설명:
- 각 위치별로 독립적으로 처리되는 2층 MLP
- \( W_1, W_2 \): 선형 변환 가중치  
- 중간에 [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)를 써서 비선형성 부여

📌 핵심 개념:
- Attention으로 요약된 정보를 한 번 더 생각하고 정제하는 역할

---

### 6. Add & LayerNorm (again)

![AddNorm2](https://latex.codecogs.com/png.image?\fg{gray}\dpi{100}&space;Z_2%20%3D%20\text{LayerNorm}(Z_1%20+%20\text{FFN}(Z_1)))

👉 설명:
- FFN의 출력과 이전 블록의 출력 𝑍1을 다시 더함
- 그 후 정규화 (LayerNorm)로 안정성 유지

📌 핵심 개념:
- 새로 정리된 생각을 이전 생각과 섞어 정돈하는 마무리 단계

---

🧠 전체 흐름 다시 한 번
1. 입력 벡터를 Query/Key/Value로 변환
2. Attention으로 서로를 바라보며 중요도를 학습
3. 여러 head로 다양한 관점을 합침
4. 기존 정보와 섞고 정리
5. FFN으로 추가 처리
6. 다시 섞고 정리

---

### 📚 추가 참고 자료

- [Transformer의 큰 그림 이해: 기술적 복잡함 없이 핵심 아이디어 파악하기](https://medium.com/@hugmanskj/transformer의-큰-그림-이해-기술적-복잡함-없이-핵심-아이디어-파악하기-5e182a40459d)
- [PyTorch Transformer Tutorial (Official)](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [Transformer Playground (Visualization Tool)](https://transformer-playground.tensorflow.org/)
