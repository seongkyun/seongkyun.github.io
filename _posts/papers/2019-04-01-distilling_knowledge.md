---
layout: post
title: Distilling the knowledge in a Neural Network
category: papers
tags: [Deep learning]
comments: true
---

# Distilling the knowledge in a Neural Network

Original paper: https://arxiv.org/pdf/1503.02531.pdf

Authors: Geoffrey Hinton, Oriol Vinyals, Jeff Dean (Google Inc)

- 참고글: https://www.youtube.com/watch?v=tOItokBZSfU
- 참고글: https://jamiekang.github.io/2017/05/21/distilling-the-knowledge-in-a-neural-network/

- 논문의 내용 들어가기에 앞서, 아래와 같이 간단한 개념을 이해하고 시작하는게 도움 됨.
  - Google과 같이 큰 회사에서 ML로 어떤 서비스를 개발 할 때, 개발 단계에 따라 training에 사용되는 모델과 실제 서비스로 deploy 되는 모델에는 차이 존재
  - 즉, training에 사용되는 모델은 대규모 데이터를 갖고 batch 처리를 하며 리소스를 비교적 자유롭게 사용하고, 최적화를 위한 비슷한 여러 변종이 존재 할 것임
  - 반면에 실제 deployment 단계의 모델은 데이터의 실시간 처리가 필요하며 리소스의 제약을 받는 상태에서의 빠른 처리가 중요함
  - 이 두가지 단계의 모델을 구분하는 것이 이 논문에서 중요한데, 그때 그때 다른 이름으로 부르므로 
    - 1번 모델을 "cumbersome model"로 하며 training stage 에서 large ensemble 구조를 갖고 동작하며, 느리고 복잡하나 정확히 동작함.(teacher)
    - 2번 모델을 "small model"로 하며, deployment stage에서 single small 구조를 갖고 동작하며, 빠르고 간편하나 비교적 정확도가 떨어짐.(student)
  - 머신 러닝 알고리즘의 성능을 올리는 아주 쉬운 방법은 많은 모델을 만들어 같은 데이터셋으로 training 한 다음, 그 prediction 결과 값을 average 하는 것임.
  - 이러한 다중 네트워크를 포함하는 모델의 집합을 ensemble(앙상블) 모델이라 하며, 위에서 1번의 "cumbersome model"에 해당함
  - 하지만 실제 서비스에서 사용할 모델은 2번 모델인 "small model"이므로, 어떻게 1번의 training 결과를 2번에게 잘 가르치느냐 하는 문제가 발생함
- 이 논문의 내용을 한 문장으로 말하면, 1번 모델(cumbersome model)이 축적한 지식(dark knowledge)을 2번 모델에 효율적으로 전달하는(기존 연구보다 더 general 한) 방법에 대한 설명임.

- 이 논문은 아래와 같이 크게 두 부분으로 나누어짐
  - Model compression: 1번 ensemble 모델의 지식을 2번 모델로 전달하는 방법
  - Specialist networks: 작은 문제에 특화된 모델들을 training 시켜 ensemble training 시간을 단축하는 방법
- 본 글에서는 distillation을 이용한 model compression에 대해 설명

## Model compression

<center>
<figure>
<img src="/assets/post_img/papers/2019-04-01-distilling_knowledge/fig1.jpg" alt="views">
<figcaption></figcaption>
</figure>
</center>
