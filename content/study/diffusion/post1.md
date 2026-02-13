
## 1. Introduction: Deep Generative Modeling(DGM)
이 장에서는 Deep Generative Modeling의 목적, 기본 베이스 아이디어에서 시작하여 다양한 방법론에 대해 간단히 소개한다. 또한 특정 방법론들이 Diffusion model과 어떤식으로 연결되는지 개괄하고자 한다. 


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

위와 같은 model의 모수를 추정할 때에 주로 사용되는 방식은 MLE이다. 다음과 같은 objective function을 최대화하는 $\phi$를 MLE라고 한다.

$$
\mathbb{E}_{x \sim p_{\text{data}}}\big[\log p_\phi(x)\big]
$$

기존의 통계학에서 사용되는 MLE는 데이터 $x_1, \dots, x_n$ 을 관측했을 때, $X1, \dots, Xn$ 에 대한 joint pdf를 parameterize 한 $p_\phi(x1, \dots, xn)$ 에 관측 데이터를 대입하고
$\phi$ 에 대한 함수로 본 likelihood function 을 최대화 하는 $\phi$ 를 찾는 방법이다.
추후에 KL divergence 라고 하는 개념과 함께 위에서 정의된 MLE 가 정의되었다. 




