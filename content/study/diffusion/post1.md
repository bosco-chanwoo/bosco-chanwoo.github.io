
## 1. Introduction: Deep Generative Modeling(DGM)
> 이 장에서는 Deep Generative Modeling의 목적, 기본 베이스 아이디어에서 시작하여 다양한 방법론에 대해 간단히 소개한다. 또한 특정 방법론들이 Diffusion model과 어떤식으로 연결되는지 개괄하고자 한다. 


### 1.1 DGM의 목적

실제 데이터의 분포에서 새로운 데이터를 생성하는 것을 목표로 한다. 
크게 두가지 방법으로 나눌 수 있는데

1) explicit modeling : $p_{data}(x)$ 를 직접적으로 모델링 및 추정한 $p_\phi(x)$로부터 새로운 데이터를 Sampling한다.
2) implicit modeling : x의 분포를 직접 추정하지는 않지만, p(z) Sampling -> p(x|z) Sampling과 같은 방식을 통해 새로운 데이터를 Sampling 한다.  


---

### 1.2 Mathematical Formulation

$$
p_\phi(x) \approx p_{data}(x)
$$


> 실제 데이터의 분포를 알 수 없기 때문에, 모델 분포가 실제 데이터 분포를 근사하도록 학습한다.

Training objective:

$\max_\phi \; \mathbb{E}_{x \sim p_{data}} [\log p_\phi(x)]$
