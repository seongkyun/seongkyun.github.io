---
layout: post
title: MobileNetV2- Inverted Residuals and Linear Bottlenecks
category: papers
tags: [Deep learning, Mobilenetv2, Linear bottleneck]
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
<figcaption>Standard convolution</figcaption>
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

- 두 연산방법간의 연산량 차이는 아래와 같음. 
  - Input: $h_i*w_i*d_i$ 크기의 Tensor
  - Output: $h_i*w_i*d_j$ 크기의 Feature map
  - Kernel size: $k*k$
  - Sandard convolution의 연산량: $h_i*w_i*d_i*d_j*k*k$
  - Depthwise separable convolution의 연산량: $h_i*w_i*d_i*(k^2+d_j)$
  - 즉, Standard convolution에 비해 $\frac{1}{d_j}+\frac{1}{k^2}$배 만큼의 연산량 감소 효과가 있음.
  - $3*3$컨벌루션 연산 시 약 8~9배 가량의 연산량 감소 효과

## Linear Bottleneck
- 각 layer에서 focusing하는 feature들은 조금씩 다름.
- Conv 연산으로 feature map에 담긴 정보를 전체 채널을 따져 고려하면, 결국 중요한 정보는 몇 몇 manifold에 존재.
  - 이것을 다시 low-dimensional sub-space로 만들 수 있음.

- 즉, feature map에 존재하는 모든 값들이 의미있는 정보를 나타내지 않고, 의미있는 정보를 나타내는 값은 특정 부분에 몰려있거나 전체에 여러 영역에 걸쳐 나타날 수 있다는 것을 의미.

- MobileNet V1에선 이런 manifold를 low-dimensional sup-space로 만듦
  - MobileNet V1의 Width-multiplier를 이용해 채널을 조정하는 것
- 이러한 방법으로 어떤 manifold of interest 부분이 entire space가 될 때까지 차원을 축소 시킬 수 있음
  - 의미있는 정보와 의미 없는 정보가 담긴 정보량들에 대해, 의미없는 정보를 버리고 의미있는 정보만이 남도록 정보가 존재하는 차원을 줄이고, 이로 인해 메모리의 효율성이 좋아지는것을 의미



---

- [참고 글]

http://eremo2002.tistory.com/48

http://hugrypiggykim.com/2018/11/17/mobilenet-v2-inverted-residuals-and-linear-bottlenecks/
