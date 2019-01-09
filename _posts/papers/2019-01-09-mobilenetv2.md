---
layout: post
title: MobileNetV2: Inverted Residuals and Linear Bottlenecks
category: papers
tags: [Deep learning, mobilenetv2, linear bottleneck]
comments: true
---

# MobileNetV2: Inverted Residuals and Linear Bottlenecks

Original paper: https://arxiv.org/abs/1801.04381

Authors: Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen (Google Inc.)

## Introduction
- Neural Network(NN)의 정확도를 향상시키는 것은 컴퓨터의 연산량에 직결적으로 연결된 문제(성능이 좋을수록 더 많은 연산량 요구)
- 대부분의 NN은 제한된 성능에서 FLOPs와 같은 indirect metric computation complexity를 고려
  - Direct metric computation complexity: Memory access cost, platform 특성에 의존하는 속도
- 본 논문에서는, FLOPs를 떠나 direct metric을 평가해 효율적인 네트워크 구축을 위한 구조를 제안(ShuffleNet V2)
  - 메모리 연산에 효율적인 Inverted Residual Block을 제안
- Inverted Residual Block
  - Input으로 low-dimensional compressed representation을 받아 high-dimension으로 확장시킨 후, depthwise convolution을 수행
  - 이렇게 나온 feature를 다시 linear-convolution을 통해 low-dimensional 정보로 만듦.
- 이러한 Module(Inverted Residual Block) 덕분에 Memory 사용량이 줄어, mobile design에 최적화 됨

## Depthwise Separable Convolution
- 일반적인 convolution 연산의 경우, 3개의 채널에 대해 3x3의 공간방향의 convolution을 수행 시 1개의 채널이 생성 됨.
<center>
<figure>
<img src="/assets/post_img/papers/2019-01-09-mobilenetv2/fig1.png" alt="views">
<figcaption>Normal convolution</figcaption>
</figure>
</center>

- Depthwise separable convolution의 경우, 일반적인 convolution 연산과 달리 공간/채널(깊이)을 따로 연산
  - 채널별로 convolution 연산 후, 이를 1x1 convolution 연산으로 채널 간(깊이 방향) 연산을 수행
<center>
<figure>
<img src="/assets/post_img/papers/2019-01-09-mobilenetv2/fig2.png" alt="views">
<figcaption>Depthwise separable convolution / 공간 방향의 convolution</figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/post_img/papers/2019-01-09-mobilenetv2/fig3.png" alt="views">
<figcaption>Depthwise separable convolution / 채널(깊이) 방향의 convolution</figcaption>
</figure>
</center>


---

- [참고 글]

http://eremo2002.tistory.com/48

http://hugrypiggykim.com/2018/11/17/mobilenet-v2-inverted-residuals-and-linear-bottlenecks/
