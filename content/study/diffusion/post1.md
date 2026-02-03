---
title: '1. Introduction: Deep Generative Modeling(DGM)'
---
## DGM의 목적

실제 데이터의 분포에서 새로운 데이터를 생성하는 것을 목표로 한다. 
크게 두가지 방법으로 나눌 수 있는데

1. explicit modeling : $p_{data}(x)$ 를 직접적으로 모델링 및 추정한 $p_\phi(x)$로부터 새로운 데이터를 Sampling한다.
2. implicit modeling : x의 분포를 직접 추정하지는 않지만, p(z) Sampling -> p(x|z) Sampling과 같은 방식을 통해 새로운 데이터를 Sampling 한다.  


---

### Key Idea

- Data is generated from an unknown distribution $p_{data}(x)$
- We model it using a neural network parameterized distribution $p_\phi(x)$
- Sampling from $p_\phi(x)$ generates new synthetic data

---

### Mathematical Formulation

$p_\phi(x) \approx p_{data}(x)\$

Training objective:

$\max_\phi \; \mathbb{E}_{x \sim p_{data}} [\log p_\phi(x)]$
