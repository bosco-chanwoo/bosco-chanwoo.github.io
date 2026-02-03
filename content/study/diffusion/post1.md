---
title: '1. Introduction: Deep Generative Modeling(DGM)'
---
## DGM의 목적

실제 데이터의 분포에서 새로운 데이터를 생성하는 것을 목표로 한다. 


The ultimate goal of DGM is to learn a probability distribution from which data samples are generated.
True data distribution $p_{data}(x)$ is usually unknown, so we want to approximate it with $p_\phi(x)$.

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
