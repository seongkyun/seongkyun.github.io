---
layout: post
title: Deep Mutual Learning
category: papers
tags: [Deep learning]
comments: true
---

# Deep Mutual Learning

Original paper: https://arxiv.org/pdf/1706.00384.pdf

Authors: Ying Zhang, Tao Xiang, Timothy M. Hospedales, Huchuan Lu

## Abstract
- Model distillation은 teacher의 정보를 student network로 전달하도록 많이 사용되는 효과적인 기술이다. 일반적으로 small network로 성능좋고 큰 네트워크나 앙상블 네트워크를 transfer 하는것은 low-memory나 빠른 동작이 필요할 때 더 필요하다. 본 논문에서는 미리 학습(정의된) teacher와 student 사이에 단 방향으로 transfer 되는 방식이 아닌 student의 앙상블이 협력적으로 training 과정 전반에 걸쳐 서로를 가르치는 deep mutual learning(DML) strategy를 제안한다. 실험에서는 논문에서 제안하는 mutual learning이 다양한 network 구조에 대해 CIFAR-100 recognition과 Market-1501 person re-identification benchmark에서 매우 좋은 결과를 보였다. 논문에선 이전처럼 표현력 좋은 강력한 teacher network가 필요하지 않다는 것을 밝혔다. 단순히 student network들로 이루어진 collection간에 서로 상호 학습을 하도록 하는것이 더 효과적이며, 더욱 강력하면서도 teacher net의 distillation보다 더 좋은 성능을 보여준다.

## 1. Introduction
- Deep neural network는 다양한 분야에서 SOTA 성능을 보였지만 대개는 depth나 width가 넓어 많은 파라미터들을 갖고 있다[6, 25]. 이는 연산량이 많아 속도가 느리거나 메모리를 많이 필요로하므로 제한된 성능을 갖는 환경에서의 적용이 어렵다. 따라서 빠른 모델을 만들어내는 연구가 활발히 진행되었다. 크기는 작지만 정확한 모델을 얻는방법에 대해 간단한(frugal) 모델 설계[8], model compression[2], pruning[13], binarisation[18], model distillation[7]등의 연구가 진행되었다.
- Distillation based 모델 압축방식은 작은 네트워크가 대때로 큰 네트워크만큼의 표현량(representation capacity)을 갖는 경우가 많다는 obwervation과 관계되어 있다[3, 2]. 하지만 큰 네트워크와 비교했을 때 desired function을 실현시키는 올바를 파라미터를 갖도록 모델을 학습시키고 찾는것은 더 어려워지게 된다. 즉 limitation은 네트워크의 크기보다 적절한 optimization을 하는것이 더 어려운 문제다[2]. 작은 네트워크를 잘 학습시키기 위한 distillation 방법에서는 deep하고 wide하거나 앙상블로 이루어진 teacher net이 필수요소이며 작은 student network는 이러한 teacher net을 흉내내도록 학습되어진다 [7, 2, 16, 3]. Teacher net의 class probabilities[7]나 feature representation[2, 19]을 흉내내거나 하는것은 기존의 supervised learning target의 목표를 넘어 추가적인 정보를 이용 할 수 있게 되는것이다. Teacher를 흉내내도록 학습시키기 위한 optimization 문제는 target function을 다이렉트로 학습하는것보다 더 쉬운 것으로 밝혀졌으며, 이로인해 훨씬 작은 student가 larger teacher의 성능 또는 그 성능을 능가할 수 있게 된다[19].
- 논문에선 model distillation과 관련되었지만 다른 방법을 제안하며, 이를 mutual learning으로 정의한다. 

## Conclusion
- 논문에선 DNN을 집단(cohort)으로 만들어 peer와 mutual distillation 을 통해 DNN의 성능을 향상시키는 간단하지만 general하게 적용 가능한 방법을 제안하였다. 이 방법을 이용해 static(단독학습, pre-trained) teacher로부터 distilled된 네트워크보다 성능이 더 좋은 compact network를 얻을 수 있었다. Deep mutual learning(DML)을 활용하는 한가지 예로 compact하고 빠른 효율적인 네트워크를 얻을 수 있다. 또한 논문에선 이 방식을 이용해 크고 powerful한 네트워크의 성능도 향상시킬 수 있었으며, 논문에서 제안하는 방식을 따라 학습된 network cohort(네트워크 그룹)은 더 성능 향상을 위한 앙상블 모델로 사용될 수 있다.

<center>
<figure>
<img src="/assets/post_img/papers/2019-04-04-deep_mutual_learning/fig1.jpg" alt="views">
<figcaption></figcaption>
</figure>
</center>
