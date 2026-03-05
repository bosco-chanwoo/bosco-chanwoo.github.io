+++
title = "Introduction: Deep Generative Modeling"
date = 2026-02-18T21:45:00+09:00
+++



## 1. Introduction: Deep Generative Modeling(DGM)
이 장에서는 Deep Generative Modeling의 목적, 기본 아이디어에서 시작하여 다양한 방법론에 대해 간단히 소개한다. 또한 특정 방법론들이 Diffusion model과 어떤식으로 연결되는지 개괄하고자 한다. 


### 1.1 DGM의 목적

실제 데이터의 분포에서 새로운 데이터를 생성하는 것을 목표로 한다. 
크게 두가지 방법으로 나눌 수 있는데

1) explicit modeling : $p_{data}(x)$ 를 직접적으로 모델링 및 추정한 $p_\phi(x)$로부터 새로운 데이터를 Sampling한다.
2) implicit modeling : x의 분포를 직접 추정하지는 않지만, $p(z)$ Sampling -> $p(x|z)$ Sampling과 같은 방식을 통해 새로운 데이터를 Sampling 한다.  


---

### 1.2 Mathematical Formulation
True Data distbution을 추정하고자 하는데 $\phi$로 모수화한 model을 다음과 같이 표현한다.

$$
p_\phi(x) \approx p_{data}(x)
$$

우리가 추정하고 싶은 pdf $p_{data}(x)$를 그대로 추정하기에는 후보가 광범위하다(Infinity dimesnion) 이러한 이유로 
어떤 모형을 정하고 모수화하여 $\phi \in \Phi$ 의 유한차원 공간에서 추정하고자 한다.

이때, $\hat{\phi}$ 를 찾는과정을 training 이라고 한다.

### 1.3 KL divergence
실제 데이터의 pdf $p_{data}(x)$ 를 모델링한다면, 그 모델은 실제 데이터 분포와 일치하거나 비슷한 성질을 가져야 할 것이다. 이러한 관점에서 두 pdf 사이의 유사도를 측정하려고 KL divergence가 제시되었다.

$$
D_{KL}(p_{data}|q_\phi) = \int p_{data}(x)\log\frac{p_{data}(x)}{q_{\phi}(x)}dx
= \mathbb{E}_{p_{data}}\log\frac{p_{data}(x)}{q_{\phi}(x)}
$$

위 식에서, 데이터가 p_{data}를 따를 때 log likelihood ratio의 기댓값으로 정의되었다.
이때 KL divergence가 최소가 되는 $\phi$ 를 찾는것이 자연스럽다. 

$$
\arg\min_\phi D_{KL}(p_{data}|q_\phi)
= \arg\min_\phi[\mathbb{E}_{p_{data}} \log p_{data}(x) - \mathbb{E}_{p_{data}} \log q_{\phi}(x)]
= \arg\max_\phi\mathbb{E}_{p_{data}}{q_{\phi}(x)}
$$

KL를 최소화하는 $\phi$ 를 찾는것은 우리가 가정한 모델의 log likelihood $\log q_\phi(x)$
를 최대화 하는 것과 같다. 

이것은 통계학에서 많이 사용하는 MLE와 유사한 형태이다.
즉, KL divergence를 최소화하는 파라미터가 MLE $\phi$ 가 된다는 것이다.

실제로 통계학에서 MLE의 정의는 모델의 pdf에 관측 데이터를 넣고 파라미터에 대한 함수로 본
likelihood를 최대화하는 $\hat{\phi}$ 를 찾는 것이지만, 추후에 Information theory가 발전하면서 등장한 KL divergence에서 등장한 MLE 형태에 $P_{data}$를 Empirical distribution
$\hat{P_n}$으로 대체한다면 통계학에서의 MLE와 동일한 형태가 된다. 
(위에서는 ML에서의 관습으로 pdf로 설명했지만 여기서의 P는 measure P , dirac delta measure $\hat{P_n}$을 의미한다 )

### 1.4 Prominent Deep Generative Models
이제 Generative Model에는 어떤 것들이 있는지 살펴보자

**1. Energy Based Model**
$$
p_{\phi}(x) = \frac{1}{c(\phi)} \exp(-E_{\phi}(x))
$$ 

물리법칙으로부터 유도된 Energy function을 이용한 방법이다.
여기서 $c(\phi)$ 를 계산하기 위한 적분이 intractable 한 경우가 많아서 문제가 발생한다.

**2. Autoregressive model**

시계열 데이터를 모델링한 형태이다. 
이전까지의 값들을 given한 상태에서 다음 값이 나올 확률을 모델링한다.
LLM 계열 모델들의 베이스가 되는 transformer 같은 경우에는
prompt, 이전 생성된 데이터 $x_1, x_2, \ldots, x_{t-1}$ 가 $x_t$에 영향을 준다는 것을 가정한 모델이다.
$p({x_1|x_0,prompt})$ 에서 $x_1$을 생성하고 생성한 $x_1$을 given 하여 $p({x_2|x_0,x_1,prompt})$
에서 $x_2$를 생성한다. 이것을 반복하여 최종 $x_t$를 생성하는 방식을 취한다.

**3. Variational AutoEncoder**
기존의 AutoEncoder 모델같은 경우의 목적은 어떤 벡터를 더 낮은 차원의 벡터로 Encoding 하는 것이었다.
AutoEncoder는 확률적인 관점 없이 deterministic fuction을 학습시켰다. 이러한 경우에 임의의 벡터를 Decoder에 입력한다고 해서
Decoded vector들이 $p_{data}$를 따를지 알 수 없고 Sampling 방법으로서 부적절하다.

이러한 문제를 개선하고자 등장한 것이 Variational AutoEncoder(VAE)인데, VAE가 Generative model로서 활용되기 위해서는
두가지가 중요하다. 첫번째로는 Encoder에 태운 encoded vector들의 분포가 우리가 가정한 prior dist과 유사해야 한다는 것이다.
이것이 만족한다면 prior dist에서의 데이터를 sampling하고 그 데이터를 decoder를 태워서 $p_{data}$ 하의 sample을 만들어 낼 수 있을 것이다. 두번째로는 decoder인데 $p_{\phi}(x|z)$를 decoder라고 한다면, 이것은 실제로 데이터를 인코딩했다가 디코딩 했을때, 원래 데이터를 잘 복원해야 할 것이다. 이 두 관점은 추후에 자세히 설명하겠다.

