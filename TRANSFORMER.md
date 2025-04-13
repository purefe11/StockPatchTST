# 🤖 Transformer 내부 구조

> 퍼즐 조각 같은 데이터를 보고 미래를 상상하는 똑똑한 로봇 이야기!

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

- 두 번의 계산을 통해 생각을 다듬는다.  
- [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)는 "음수는 싫어!"라고 말하면서 필요한 정보만 남긴다.

💡 *비유:*  
> 한 번 더 생각해서 더 멋진 문장을 만드는 과정과 같다.

---

## ➕ 4. Add & LayerNorm (또 나온다!)  
**“이번에도 내 생각과 새로 정리한 생각을 합쳐서 다시 정리한다.”**

- 정리된 생각 + 원래 생각 → 다시 정리해서 더 똑똑하게 만든다.

💡 *비유:*  
> 내가 한 말 + 새로 정리한 말 = 완성된 멋진 말!

---

## 🎉 Transformer가 뭘 해주는 걸까?

- 조각들 사이의 관계를 잘 파악하고,  
- 어떤 조각이 중요한지도 알아낸다.  
- → 미래를 예측할 준비 완료!

---

## ✨ 한 줄 요약

> **Transformer는 여러 조각이 서로를 바라보며 중요도를 정하고,  
> 생각을 정리해서 똑똑한 예측을 하는 구조다!**

<br>

## 📚 더 알아보기

- ["Attention is All You Need"](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

  ![transformer_resideual_layer_norm_2](https://github.com/user-attachments/assets/7506baf2-33bb-4d5b-a33f-cd4e2f10bae3)
