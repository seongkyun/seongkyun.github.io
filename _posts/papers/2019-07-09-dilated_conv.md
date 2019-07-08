---
layout: post
title: Multi-Scale Context Aggregation by Dilated Convolutions
category: papers
tags: [Deep learning]
comments: true
---

# Multi-Scale Context Aggregation by Dilated Convolutions

Original paper: http://vladlen.info/papers/dilated-convolutions.pdf

Authors: Fisher Yu, Vladlen Koltun

- 참고 글
  - https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=220991967450&proxyReferer=https%3A%2F%2Fwww.google.com%2F

- FCN(Fully Convolutional Network)에 대한 분석과 약간의 구조 변경을 통해 FCN의 성능을 좀 더 끌어올리게 할 수 있는 방법을 제시
  - Dilated convolution
- 본 논문에서는 dilated convolution이라는 단어를 Fisher Yu의 segmentation 방법을 지칭하는것과 실제 dilation 개념이 적용된 convolution을 섞어서 표현함

## Dilated convolution이란?
- Dilated convolution 자체는 Fisher Yu가 처음 언급한것이 아니라, FCN을 발표한 Jonathan Long의 논문에서 잠깐 언급이 있었고, FCN 개발자들은 dilated convolution 대신 skip layer와 upsampling 개념을 사용함
- 그 후 DeepLab의 논문 "Semantic image segmentation with deep convolutioonal nets and fully connected CRFs"에서 dilated convolution이 나오지만, Fisher Yu의 방법과 조금 다른 방법으로 사용
- Dilated convolution의 개념은 wavelet decomposition 알고리즘에서 "Atrous algorithm" 이라는 이름으로 사용되었으며, DeepLab 팀은 구별하여 부르기 위해 atrous convolution이라고 불렀는데, Fisher Yu는 이를 dilated convolution이라고 불렀으며 주로 이렇게 표현됨
  - 참고로 atrous는 프랑스어로 a trous고 trous는 hole(구멍)의 의미를 갖음.
- Dilated convolution이란 아래 그림처럼 기본적인 conv와 유사하지만 빨간색 점의 위치에 있는 픽셀들만 이용하여 conv 연산을 수행함
- __이렇게 하는 이유는 해상도의 손실 없이 receptive field의 크기를 확장할 수 있기 때문임__
- Atrous convolution이라고 불리는 이유는 전체 receptive field에서 빨간색 점의 위치만 계수가 존재하고, 나머지는 모두 0으로 채워지기 때문임
- 아래 그림에서 (a)는 1-dilated convolution이며 이는 흔히 알고있는 convolution과 동일함.
- (b)는 2-dilated convolution이며, 빨간 점들의 값만 conv 연산에서 참조되어 사용되고 나머지는 0으로 채워짐.
  - 이렇게 되면 receptive field의 크기는 7x7 영역으로 커지게 됨(연산량의 증가 없이 RF가 커짐)
- (c)는 4-dilated convolution이며, receptive field의 크기는 15x15로 커지게 됨

<center>
<figure>
<img src="/assets/post_img/papers/2019-07-09-dilated_conv/fig1.PNG" alt="views">
<figcaption></figcaption>
</figure>
</center>

- Dilated conv를 사용하여 얻을 수 있는 이점이 큰 receptive field(RF)를 취하려면 일반적으로 파라미터의 개수가 많아야 하지만(large kernel size conv를 사용 후 pooling해야함) dilated convolution을 사용하면 receptive field는 커지지만 파라미터의 개수는 늘어나지 않기에 연산량 관점에서 탁월한 효과를 얻을 수 있음
- 위 그림의 (b)에서 RF는 7x7이기 때문에 normal filter로 구현 시 필터의 파라미터 개수는 49개가 필요하며, conv 연산이 CNN에서 가장 많은 연산량을 차지한다는점을 고려한다면 이는 부담이 상당한 연산이 됨
- 하지만 dilated conv를 사용하면 49개의 파라미터중 빨간점에 해당하는 9개의 파라미터만 사용하고 나머지 값 40개는 모두 0으로 처리되어 연산량 부담이 3x3 filter 처리량과 같아지게 됨

## Dilated convolution을 하면 좋은점
- 우선 RF의 크기가 커지게 된다는 점이며, dilation 계수 조절 시 다양한 scale에 대한 대응이 가능해짐.
- 다양한 scale에서의 정보를 끄집어내려면 넓은 receptive field를 볼 수 있어야 하는데, dilated conv를 사용하면 별 어려움이 없이 이것이 가능해짐
- __기존의 일반적인 CNN에서는 RF의 확장을 위해 pooling layer를 통해 feature map의 크기를 줄인 후 convolution 연산을 수행하는 방식으로 계산__
  - 기본적으로 pooling을 통해 크기가 줄었기에 동일한 크기의 filter를 사용하더라도 CNN망의 뒷단으로 갈수록 넓은 RF를 커버할 수 있게 됨
- Fisher Yu의 논문에서는 자세하게 설명하지 않았기에 아래의 DeepLab 논문 그림을 참조하여 이해하면 이해가 훨씬 쉬워짐

<center>
<figure>
<img src="/assets/post_img/papers/2019-07-09-dilated_conv/fig2.PNG" alt="views">
<figcaption></figcaption>
</figure>
</center>

- 그림에서 위쪽은 앞서 설명한 것처럼 down-sampling(pooling) 후 conv를 통해 large RF를 갖는 feature map을 얻고, 이를 이용해 픽셀 단위 예측을 하기위해 다시 up-sampling을 통해 영상의 크기를 키운 결과임
- 아래는 dilated convolution(atrous conv)을 통해 얻은 결과
- Fisher Yu는 context module이라는것을 개발하여 segmentation의 성능을 끌어 올렸으며, 여기에 dilated convoluton을 적용함




