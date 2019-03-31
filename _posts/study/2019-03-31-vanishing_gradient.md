---
layout: post
title: Vanishing gradient problem
category: study
tags: [Vanishing gradient]
comments: true
---

# Vanishing gradient problem
- 참고글: https://www.quora.com/What-is-the-vanishing-gradient-problem
- 갑자기 vanishing gradient에 대해 정리하고싶어져서..

---

- Vanishing gradient problem은 인공신경망을 기울기값(gradient)을 베이스로 weight parameter를 update하며 모델을 학습시키는 방법(back-propagation)에서 발생하는 문제이다.
- 특히 이 문제는 네트워크의 앞 쪽 레이어들의 weight parameter들을 올바르게 학습시키고 tunning하는데에 대해 큰 영향을 끼친다.
  - 즉, gradient가 네트워크의 뒷단에서 계산되어 앞으로 흘러가며 앞 단으로 갈수록 vanishing gradient에 의해 weight parameter가 올바르게 update되기 힘들어진다는 의미
- Vanishing gradient problem은 네트워크가 깊어질수록 그 영향력이 더 커진다.

- 이는 인공신경망(neural network)의 근본적인 문제점이 아니라 특정한 activation function을 이용하여 gradient를 계산하여 weight parameter를 올바른 방향으로 update하는 back-propagation 학습 방법에 대해서 문제가 발생한다.
- 아래에서는 직관적으로 이러한 문제를 이해하고 그것으로 인해 네트워크에 끼쳐지는 영향에 대해 알아본다.

### 문제
- Gradient 기반의 네트워크 학습 방법은 파라미터 값의 작은 변화가 네트워크의 출력에 얼마나 영향을 미칠지를 이해하는 것을 기반으로 파라미터 값을 학습시킨다.
- 만약 파라미터 값의 변화가 네트워크의 출력에 매우 적은 변화를 야기한다면 네트워크는 파라미터를 효과적으로 학습 시킬 수 없게 되는 문제가 발생한다.
  - 즉, gradient는 결국 미분값, 그러니까 변화량을 의미하는데 이 변화량이 매우 작다면 네트워크를 효과적으로(효율적으로!) 학습시키지 못하게 되며, 이는 곧 loss function이 적절히 수렴하지 못하고 높은 error rate 상태에서 수렴하게 되는 문제가 발생하게 된다.

- 이게 바로 vanishing gradient problem이며, 이로인해 초기 레이어들 각각의 파라미터들에 대한 네트워크의 출력의 gradient는 매우 작아지게 된다.
  - 초기 레이어에서 파라미터값에대해 큰 변화가 발생해도 output에 대해 큰 영향을 주지 못한다는 것을 의미한다.

- 그렇다면 언제, 왜 이러한 문제가 발생하나?


<center>
<figure>
<img src="/assets/post_img/study/2019-03-31-vanishing_gradient/fig1.jpeg" alt="views">
<figcaption></figcaption>
</figure>
</center>
