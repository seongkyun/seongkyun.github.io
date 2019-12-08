---
layout: post
title: Semantic Segmentation
category: study
tags: [CNN, Deep learning]
comments: true
---

# Semantic Segmentation
- 참고
  - https://medium.com/hyunjulie/1%ED%8E%B8-semantic-segmentation-%EC%B2%AB%EA%B1%B8%EC%9D%8C-4180367ec9cb
  - https://medium.com/nanonets/how-to-do-image-segmentation-using-deep-learning-c673cc5862ef
  - https://www.youtube.com/watch?v=nDPWywWRIRo&feature=youtu.be

## Introduction

<center>
<figure>
<img src="/assets/post_img/study/2019-12-08-segmentation/fig1.png" alt="views">
<img src="/assets/post_img/study/2019-12-08-segmentation/fig2.png" alt="views">
<figcaption>from DeepLab V3+</figcaption>
</figure>
</center>

- 사진을 보고 분류하는 것이 아니라, 장면을 완벽히 이해해야 하는 높은 수준의 문제임
  - 자율주행 등 적용 가능 분야가 매우 큼
- 실제 구글 Pixel 2와 Pixel 2X 스마트폰에선 구글의 DeepLab V3+ 논문의 방법을 이용해 Portrait Mode를 구현함
  - 아이폰의 Portrait Mode와 유사하지만, 2개의 스테레오 카메라를 이용하는 아이폰 X의 방법과는 다르게 단안 카메라의 영상을 이용해 딥러닝을 적용시켜 동일한 결과를 얻어냄

<center>
<figure>
<img src="/assets/post_img/study/2019-12-08-segmentation/fig3.png" alt="views" height="300">
<figcaption>Google Pixel 2의 Portrait Mode</figcaption>
</figure>
</center>

- 즉, Semantic Segmentation은 영상 내 모든 픽셀의 레이블을 예측하는 task를 의미함
  - FCN, SegNet, DeepLab 등
- 이미지에 있는 모든 픽셀에 대한 예측이므로 dense prediction이라고도 불림

- Semantic Segmentation은 같은 class의 instance를 구별하지 않음
  - 즉, 아래의 짱구 사진처럼 같은 class에 속하는 사람 object 4개를 따로 구분하지 않음

<center>
<figure>
<img src="/assets/post_img/study/2019-12-08-segmentation/fig4.png" alt="views" height="200">
<figcaption></figcaption>
</figure>
</center>

- Semantic segmentation에선 해당 픽셀 자체가 어떤 class에 속하는지에만 관심이 있음

<center>
<figure>
<img src="/assets/post_img/study/2019-12-08-segmentation/fig5.png" alt="views" height="200">
<figcaption></figcaption>
</figure>
</center>
  
- 위의 좌측 사진처럼 각각 객체의 종류와 어떤 객체인지에 대한 분류가 아니라
  - 이처럼 instance를 구별하는 task는 Instance Segmentation이라고 불림
- 위의 우측 사진처럼 각각 객체의 종류의 분류만을 목적으로 함

- MS COCO detection challenge는 80개의 class, PASCAL VOC challenge는 21개의 class를 갖고 있음

<center>
<figure>
<img src="/assets/post_img/study/2019-12-08-segmentation/fig6.png" alt="views">
<figcaption></figcaption>
</figure>
</center>

## Semantic Segmentation 이해하기
- Input: RGB color 이미지 ($height\times width\times 3$) 또는 흑백 이미지
- Output: 각 픽셀별 어느 class에 속하는지를 나타내는 레이블을 나타낸 Segmentation Map
  - 아래에선 단순화된 segmentation map 예시를 확인 가능하며, 실제론 입력과 동일한 resolution의 map을 가짐

<center>
<figure>
<img src="/assets/post_img/study/2019-12-08-segmentation/fig7.png" alt="views" height="200">
<figcaption></figcaption>
</figure>
</center>

- One-Hot encoding으로 각 class에 대해 출력 채널을 만들어서 segmentation map을 생성함
- Class의 개수 만큼 만들어진 채널을 `argmax`를 통해 위에 있는 이미지처럼 하나의 출력물을 내놓음
  - `argmax`? channel 방향으로 가장 큰 값의 index를 출력
    - ex. x = [0, 1, 4, 3] 일 때, argmax(x)의 결과는 2
  - 직관적 이해는 아래의 사진 참고

<center>
<figure>
<img src="/assets/post_img/study/2019-12-08-segmentation/fig8.png" alt="views" height="200">
<figcaption>각 Class별로 출력 채널을 만든 후 argmax 적용</figcaption>
</figure>
</center>

