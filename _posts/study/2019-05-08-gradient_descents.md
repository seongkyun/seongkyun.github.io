---
layout: post
title: Gradient Descent Optimization 알고리즘 정리
category: study
tags: [Gradient Descent]
comments: true
---

# Gradient Descent Optimization 알고리즘 정리
- 참고 글: http://ruder.io/optimizing-gradient-descent/
- 참고 글: http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html

- 딥러닝에선 네트워크 파라미터 $\theta$ 에 대해 실제 값과 예측값의 차이를 정의하는 loss function $J(\theta)$를 최소화하기 위하여 기울기 $\nabla_{\theta} J(\theta)$를 이용하여 loss function을 최소화하도록 네트워크가 학습 됨
- 네트워크의 학습에선 $\theta$에 대해 gradient 반대 방향으로 일정 크기만큼 이동하는것을 반복하며 loss function의 값을 최소화할 수 있게 하는 파라미터 $\theta$를 찾게 됨
- 한 번의 iteration에서 변화 식은 아래와 같음

$$\theta = \theta - \eta \nabla_{\theta} J(\theta)$$

- 여기서 $\eta$는 미리 정해진 step size로 보통 0.01~0.001 정도의 크기를 사용
- Loss function 계산 시 전체 train set을 사용하는것을 Batch Gradient Descent(BGD)라고 함
  - 하지만 이렇게 계산 할 경우 step 한 번을 할 때 전체 데이터에 대한 loss function을 계산해야 하므로 많은 계산량이 필요
  - 이를 방지하기위해 보통은 Stochastic Gradient Descent(SGD)를 사용
- SGD는 loss function을 계산 시 전체 batch에 대한 데이터 대신 일부의 조그마한 데이터 모음(mini-batch)에 대해서만 loss function을 계산하며, 이 방법은 batch gradient descent에 비해 다소 부정확하지만 계산 속도가 훨씬 빠름
  - 훨씬 빠르게 학습 가능
  - 여러 번 반복될 경우 BGD의 결과와 유사하게 수렴함
- SGD를 사용 할 경우 BGD와 다르게 local minima에 빠지지 않고 더 나은 방향으로 수렴 할 가능성도 있음
- 보통 네트워크의 학습시엔 SGD를 이용하나, SGD만을 이용하여 네트워크를 학습시키는 것에는 명확한 한계가 존재함

<center>
<figure>
<img src="/assets/post_img/study/2019-05-08-gradient_descents/fig1.gif" alt="views">
<figcaption>Gradient Descent Optimization Algorithms at Long Valley</figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/post_img/study/2019-05-08-gradient_descents/fig2.gif" alt="views">
<figcaption>Gradient Descent Optimization Algorithms at Beale's Function</figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/post_img/study/2019-05-08-gradient_descents/fig3.gif" alt="views">
<figcaption>Gradient Descent Optimization Algorithms at Saddle Point</figcaption>
</figure>
</center>

- 위 그림들은 각각 SGD 및 SGD 변형 알고리즘들이 minima를 찾는 과정을 시각화 한 것
- 빨간색으로 표현되는 SGD가 우리가 흔히 알고있는 Naive Stochastic Gradient Descent 알고리즘이고, Momentum, NAG, Adagrad, AdaDelta, RMSprop 등은 모두 SGD의 변형된 꼴
- 그림처럼 모든 경우에서 SGD는 다른 알고리즘보다 성능이 월등히 낮음
  - 다른 알고리즘들보다 이동속도가 현저히 낮음
  - 방향을 제대로 잡지 못하고 이상한곳에 수렴하는경우도 관찰 가능
- 즉, 단순하게 SGD를 이용하여 네트워크를 학습시킬 경우 네트워크가 상대적으로 좋은 결과를 얻지 못할 것이라 예측 가능

## Momentum
- Momentum 방식은 gradient descent를 통해 parameter가 update되는 과정에 일종의 관성을 주는 방법임
- 현재 gradient를 통해 이동하는 방향과 별개로 과거에 이동했던 방향등을 기억하면서 그 방향으로 일정 정도를 추가적으로 이동하게 되는 방식
- 수식으로 표현하면 아래와 같으며, $v_t$를 time step t에서 이동 벡터라 할 때 아래와 같은 식으로 이동을 표현 가능함

$$v_t = \gamma v_{t-1} + \eta \nabla_{\theta}J(\theta)$$
$$\theta = \theta - v_t$$

- durltj $\gamma$는 얼마나 momentum을 줄 것인지에 대한 term으로 보통 0.9정도의 값을 사용
- 식을 보면 과거에 얼마나 이동했는지에 대한 이동 항 $v$를 기억하고 새로운 이동항을 구할 때 과거에 이동했던 정도에 관성항만큼 곱한 후 gradient를 이용한 이동 step 항을 더해주게 됨
- 이렇게 할 경우 이동항 $v_t$는 다음과 같은 식으로 정리할 수 있어 gradient들의 지수평균을 이용하여 이동한다고도 해석이 가능해짐

$$v_t = \eta \nabla_{\theta}J(\theta)_t + \gamma \eta \nabla_{\theta}J(\theta)_{t-1} +\gamma^2 \eta \nabla_{\theta}J(\theta)_{t-2} + ....$$

- Momentum 방식은 SGD가 진동하는 현상을 겪을 때 이를 해결하도록 해 줌
- 아래와 같이 SGD가 oscilation을 하고 있는 상황을 살펴보자

<center>
<figure>
<img src="/assets/post_img/study/2019-05-08-gradient_descents/fig4.gif" alt="views">
<figcaption>t</figcaption>
</figure>
</center>

- 현재 SGD는 중앙의 minima로 이동해야 하지만 한번의 step에서 움직일 수 있는 stemp size에 한계가 있어 이러한 oscilation을 겪게 됨

<center>
<figure>
<img src="/assets/post_img/study/2019-05-08-gradient_descents/fig5.gif" alt="views">
<figcaption></figcaption>
</figure>
</center>

- 하지만 momentum 항이 추가될 경우 위와 같이 자주 이동하는 방향에 대해 관성이 생기게 되어 진동을 하더라도 중앙으로 가는 방향에 힘을 얻게됨
- 이로 인해 momentum이 적용된 SGD가 그렇지 않은 optimizer보다 더 빠르게 최적화가 가능하게 됨

<center>
<figure>
<img src="/assets/post_img/study/2019-05-08-gradient_descents/fig6.gif" alt="views">
<figcaption>Avoiding Local Minima. Picture from http://www.yaldex.com</figcaption>
</figure>
</center>

- 또한 momentum방식을 이용하게 되면 위와 같이 local minima를 빠져나올수 있게 하는 효과를 기대 가능함
- 기존의 SGD의 경우 그림의 좌측부분처럼 local minima에 빠지게 되면 gradient가 0이 되어 이동 불가하지만 momentum 방식의 경우 기존에 이동했던 방향에 관성이 있어 local minima를 빠져나와 더 좋은 broad minima에 수렴할 수 있게 될 확률이 큼
- 반면에 momentum 방식을 이용하게 되면 기존의 변수들 $\theta$ 외에도 과거 이동했던 양을 변수별로 저장해야 하므로 변수에 대한 메모리가 기존의 두 배로 필요하게 됨



