---
layout: post
title: Gradient Descent Optimization 알고리즘 정리
category: study
tags: [Gradient Descent]
comments: true
---

# Gradient Descent Optimization 알고리즘 정리
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
