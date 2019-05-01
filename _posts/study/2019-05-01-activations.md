---
layout: post
title: Activation function 종류 및 특징
category: study
tags: [Deep learning, Activation function]
comments: true
---

# Activation function 종류 및 특징

- 참고 글
  - http://nmhkahn.github.io/NN

### Sigmoid

<center>
<figure>
<img src="/assets/post_img/study/2019-05-01-activations/sigmoid.jpg" alt="views">
<figcaption></figcaption>
</figure>
</center>

- Sigmoid 비선형 함수는 아래와 같은 수식을 따름

$$\sigma (x)=\frac{1}{1+e^{-x}}$$

- 함수의 출력값(y)을 [0, 1]로 제한(squash)시키며 입력값(x) 이 매우 크면 1, 작으면 0에 수렴
- 함수의 입력과 출력의 형태가 인간 뇌의 뉴런과 유사한 형태를 보여 많이 사용되었던 actiavtion이지만 아래와 같은 이유로 잘 사용되지 않음
  - __Vanishing gradient:__ Gradient가 backpropagation시 사라지는 현상이 발생함. Sigmoid의 gradient는 $x=0$ 일 때 가장 크며, $x=0$ 에서 멀어질수록 gradient가 0에 수렴하게 됨. 이는 이전의 gradient와 local gradient를 곱해서 error를 전파하는 backpropagation의 특성 상 앞단의 뉴런에 gradient가 적절하게 전달되지 못하게 되는 현상을 초래함.
  - __Non-zero centered:__ 함수값의 중심이 0이 아님. 어떤 뉴런의 입력값(x)이 모두 양수라 가정하면, 편미분의 chain rule에 의해 파라미터 $w$의 gradient는 아래와같이 계산됨
    - $\frac{\partial L}{\partial w}=\frac{\partial L}{\partial a}* \frac{\partial a}{\partial w}$
  - 여기서 $L$은 loss function, $a=w^{T}x+b$를 의미
    - 위 식에 $w$에 대해 편미분하면 $\frac{\partial a}{\partial w}=x$가 성립
  - 결론적으로 아래의 수식이 성립하게 됨
    - $\frac{\partial L}{\partial w}=\frac{\partial L}{\partial a}* x$
  - 파라미터의 gradient는 입력값에 의해 영향을 받으며, 만약 입력값이 모두 양수일 경우 파라미터의 모든 부호는 같게 됨
  - 이럴 경우 gradient descent 시 정확한 방향으로 가지 못하고 지그재그로 발산하는 문제가 발생할 수 있음
  - Sigmoid를 거친 출력값은 다음 레이어의 입력값이 되기에 함수값이 not-centered 특성을 가진 sigmoid는 성능에 좋지 않은 영향을 끼치게 됨
  
