---
title: 'Introduction: Deep Generative Modeling(DGM)'
---
## Goal of DGM

> **Goal**  
> Learn the underlying data-generating distribution $p_{data}(x)$  
> using a parametric model $p_\phi(x)$.

The ultimate goal of DGM is to learn a probability distribution from which data samples are generated.
True data distribution $p_{data}(x)$ is usually unknown, so we want to approximate it with $p_\phi(x)$.

---

### Key Idea

- Data is generated from an unknown distribution $p_{data}(x)$
- We model it using a neural network parameterized distribution $p_\phi(x)$
- Sampling from $p_\phi(x)$ generates new synthetic data

---

### Mathematical Formulation

$[p_\phi(x) \approx p_{data}(x)\$

Training objective:

\[
\max_\phi \; \mathbb{E}_{x \sim p_{data}} [\log p_\phi(x)]
\]
