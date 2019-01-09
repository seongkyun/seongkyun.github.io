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
  - Input: $h_{i}\times w_{i}\times d_{i}$ 크기의 Tensor
  - Output: $h_{i}\times w_{i}\times d_{j}$ 크기의 Feature map
  - Kernel size: $k\times k$
  - Sandard convolution의 연산량: $h_{i}\times w_{i}\times d_{i}\times d_{j}\times k\times k$
  - Depthwise separable convolution의 연산량: $h_{i}\times w_{i}\times d_{i}\times (k^{2}+d_{j})$
  - 즉, Standard convolution에 비해 $\frac{1}{d_j}+\frac{1}{k^2}$배 만큼의 연산량 감소 효과가 있음.
  - $3\times 3$컨벌루션 연산 시 약 8~9배 가량의 연산량 감소 효과

## Linear Bottleneck
- 각 layer에서 focusing하는 feature들은 조금씩 다름.
- Conv 연산으로 feature map에 담긴 정보를 전체 채널을 따져 고려하면, 결국 중요한 정보는 몇 몇 manifold에 존재.
  - 이것을 다시 low-dimensional sub-space로 만들 수 있음.

- 즉, feature map에 존재하는 모든 값들이 의미있는 정보를 나타내지 않고, 의미있는 정보를 나타내는 값은 특정 부분에 몰려있거나 전체에 여러 영역에 걸쳐 나타날 수 있다는 것을 의미.

- MobileNet V1에선 이런 manifold를 low-dimensional sup-space로 만듦
  - MobileNet V1의 Width-multiplier를 이용해 채널을 조정하는 것
- 이러한 방법으로 어떤 manifold of interest 부분이 entire space가 될 때까지 차원을 축소 시킬 수 있음
  - 의미있는 정보와 의미 없는 정보가 담긴 정보량들에 대해, 의미없는 정보를 버리고 의미있는 정보만이 남도록 정보가 존재하는 차원을 줄이고, 이로 인해 메모리의 효율성이 좋아지는것을 의미
- 하지만, 차원을 줄이는 과정(Sub-space로 엠베딩 시키는 과정)에서 쓰이는 ReLU와 같은 비선형 함수에 의해 문제가 발생
  - if) ReLU를 지난 출력이 S인 경우, S가 non-zero라면, S에 맵핑 된 input의 각 값은 선형변환 된 것.
  - 하지만, 0 이하의 값은 버려져 정보의 손실이 발생하게 됨.(쓸모있는 정보가 손실 됨)
  - 이를 방지하고자 channel reduction이 아닌 channel expansion을 이용
  - 채널을 확장시키는 경우, 채널의 깊이가 깊어져 low-dimensional sub-space로 embed해도 비선형 변환(ReLU)에 의한 정보의 손실이 적어짐.

- 정리하자면,
1. Manifold of interest 부분이 ReLU(비선형 변환)를 거친 후에도 Non-zero volume를 갖는다면, linear transformation에 해당
2. Input manifold가 input의 low-dimensional sub-space에 놓여있다면, ReLU가 input manifold에 대해 완전한 정보의 보호가 가능해짐

- 만약 Manifold of interest가 low-dimension이라 가정하면, convolution block에 linear bottleneck layer를 삽입해 이를 capture 할 수 있게 됨.
  - ReLU(비선형 변환)의 정보의 손실을 방지할 수 있음

## Inverted Residuals
<center>
<figure>
<img src="/assets/post_img/papers/2019-01-09-mobilenetv2/fig5.png" alt="views">
<figcaption>(a) Residual block  (b) Inverted residual block</figcaption>
</figure>
</center>

- Resudual block과 비슷하나, 반대의 구조를 띔
  - Resnet의 Residual block은 bottleneck 이후 채널을 축소시키지만, 본 논문에서 제시하는 Inverted Residual Block은 bottleneck 이후 채널을 확장시킴
- Bottleneck이 필요한 정보를 모두 갖고있고, expansion layer는 tensor의 비선형 변형과 같은 역할만 하기에 bottleneck 사이를 연결하는 구조로 사용됨.
- Residual block은 채널이 큰 특징맵끼리 연결되어 있으나, Inverted Residual Block은 bottleneck 끼리 연결
- Shortcut 연결은 Resnet처럼 gradient 전파를 효율적으로 하기 위함임.
- Inverted 구조는 메모리 사용측면에서 효율적

## Inverted Residual Block Structure

<center>
<figure>
<img src="/assets/post_img/papers/2019-01-09-mobilenetv2/fig6.png" alt="views">
<figcaption>Inverted Residual Block basic structure</figcaption>
</figure>
</center>

- 기본적인 구조는 위와 같음.
  - t: Channel expansion parameter
  - ReLU6 사용. Mobile 환경에서는 ReLU6가 더 연산/메모리측면에서 효율적이라고 사용한다던데...
- 연산량의 차이
  - Input size: $hwd'$, kernel size: $k$, output channel: $d''$
  - Bottleneck convolution
    - 수식 수식 





---

- [참고 글]

http://eremo2002.tistory.com/48

http://hugrypiggykim.com/2018/11/17/mobilenet-v2-inverted-residuals-and-linear-bottlenecks/
