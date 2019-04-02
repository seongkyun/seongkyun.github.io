---
layout: post
title: A Gift from Knowledge Distillation- Fast Optimization, Network Monimization and Tranfer Learning
category: papers
tags: [Deep learning]
comments: true
---

# A Gift from Knowledge Distillation: Fast Optimization, Network Monimization and Tranfer Learning

Original paper: http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf

Authors: Junho Yim1 Donggyu Joo1, Jihoon Bae, Junmo Kim (KAIST)

## Abstract
- 본 논문에서는 knowledge transfer를 하는 새로운 방법을 제안한다. Knowledge transfer는 pre-trained DNN(deep neural network)를 다른 DNN을 이용하여 distillation 되는 것을 의미한다.
  - 미리 학습된 DNN의 정보를 다른 DNN으로 transfer 하는것을 distillation이라 한다.
- 네트워크에 순차적으로 쌓인 layer들로 인해 input space부터 output space까지 DNN map들에 대해, 저자는 layer 사이의 흐름(flow)의 관점에서 distilled knowledge가 전달(transfer)되도록 정의한다. 이는 두 layer간의 feature에 대해 inner product를 계산하여 수행된다.
- 논문에서 동일한 size의 teacher network가 없이 학습된 DNN과 본 논문에서 제안하는 방법을 적용시킨 student DNN을 비교한다. 논문에서 제안하는 방법인 두 layer 간의 flow를 이용하여 distilled knowledge을 student DNN으로 전달 할 때, 제안하는 방법이 적용된 모델은 그렇지 않은 모델에 비해 세 가지 중요한 phenomena에 대한 차이를 보인다.
  - (1): Student DNN에 대해 distilled knowledge가 전달되는 경우가 그렇지 않은 모델보다 훨씬 더 빨리 optimize된다.
  - (2): 논문에서 제안하는 방법이 적용된 student DNN이 일반 DNN보다 성능이 더 우수하다.
  - (3): Student DNN은 다른 task에 대해 training 된 teacher DNN으로부터 distillation된 정보(knowledge)를 학습 할 수 있으며, 이렇게 학습 된 student DNN은 처음부터(from scratch) 학습 된 DNN 모델에 비해 우수한 성능을 보인다.

## Conclusion
- 본 논문에서는 DNN으로부터 distilled knowledge를 생성하는 새로운 접근방식을 제안했다. Distilled knowledge를 논문에서 제안하는 FSP matrix로 계산 된 solving procedure의 흐름(flow)으로 결정함으로써 제안하는 방법의 성능이 여타 SOTA knowledge transfer method의 성능을 능가하였다. 논문에서는 3가지 중요한 측면에서 제안하는 방법의 효율성을 검증하였다. 제안하는 방법은 DNN을 더 빠르게 optimize시키며(빠른 학습), 더 높은 level의 성능을 만들어 내게 한다. 게다가 제안하는 방법은 transfer learning task에도 적용 가능하다.

<center>
<figure>
<img src="/assets/post_img/papers/2019-04-03-a_gift_from_distillation/fig1.jpg" alt="views">
<figcaption></figcaption>
</figure>
</center>
