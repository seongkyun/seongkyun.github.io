---
layout: post
title: Inception v4
category: papers
tags: [Deep learning]
comments: true
---

# Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning

Original paper: https://arxiv.org/pdf/1602.07261.pdf

Authors: Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi

- 참고 글
  - https://norman3.github.io/papers/docs/google_inception.html

- 2015년 ResNet을 Inception에 붙여보려는 시도를 보인 논문
  - 하지만 해당 모델은 Inception v4가 아니라 Inception-resnet 이라는 별도의 모델로 생성시킴
  - Inception v4는 기존의 v3모델에 몇 가지 기능을 추가시켜 업그레이드한 모델
  - 따라서 이 논문은 Inception v4와 Inception-resnet 둘 다 다루고 있음
    - 특히 resnet을 도입한 모델을 Inception-resnet이라 명명
    - 마찬가지로 Inception-resnet v1, Inception-resnet v2와 같이 별도의 버전들이 존재함
  - 실제로는 ad-hoc한 모델로 이 모델의 한계점을 시사
- Residual connections
  - 깊은 망의 학습시에는(classification) residual connection이 꼭 필요한것인지 구글 내에선 논의중이라고 함
  - 하지만 residual connection이 존재하는 구조의 경우 확실히 학습속도가 빨라지게됨(장점)

<center>
<figure>
<img src="/assets/post_img/papers/2019-07-12-inception_v4/fig1.png" alt="views">
<figcaption></figcaption>
</figure>
</center>

- 위 그림은 residual connection의 예시를 보여줌
  - 첫 번째 그림은 가장 간단한 형태의 residual connection 구조
  - 두 번째는 1x1 conv를 추가하여 연산량을 줄인 residual connection 구조
  - 즉, residual의 개념은 이전 몇 단계 전 레이어의 결과를 현재 레이어의 결과와 합쳐서 내보내는것을 의미
  
## Inception v4, Inception-resnet v2

