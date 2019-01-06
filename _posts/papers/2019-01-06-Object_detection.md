---
layout: post
title: R-CNN/Fast R-CNN/Faster R-CNN/SSD 가볍게 알아보기
category: papers
tags: [Deep learning, Object detection]
comments: true
---

# [Object detector] R-CNN/Fast R-CNN/Faster R-CNN/SSD 가볍게 알아보기

객체 탐지(Object detection)에 대해 공부하면서 정리해놓았던 내용들을 업로드 해 보았다.

## Introduction
<center>
<figure>
<img src="/assets/post_img/papers/2019-01-06-Object_detection/fig1.PNG" alt="views">
<figcaption>Object detection</figcaption>
</figure>
</center>

- 객체 탐지(Object detection)은 사진처럼 영상 속의 어떤 객체(Label)가 어디에(x, y) 어느 크기로(w, h) 존재하는지를 찾는 Task를 말한다.
- 수 많은 객체 탐지 딥러닝 논문들이 나왔지만, 그 중 Base가 될 법한 기본적인 모델들인 R-CNN, Fast R-CNN, Faster R-CNN, 그리고 SSD에 대해 알아본다.

## R-CNN
<center>
<figure>
<img src="/assets/post_img/papers/2019-01-06-Object_detection/fig2.PNG" alt="views">
<figcaption>R-CNN의 구조</figcaption>
</figure>
</center>

1. Hypothesize Bounding Boxes (Proposals)
  - Image로부터 Object가 존재할 적절한 위치에 Bounding Box Proposal (Selective Search)
  - 2000개의 Proposal이 생성됨.
2. Resampling pixels / features for each boxes
  - 모든 Proposal을 Crop 후 동일한 크기로 만듦 (224*224*3)
3. Classifier / Bounding Box Regressor
  - 위의 영상을 Classifier와 Bounding Box Regressor로 처리
  
- 하지만 모든 Proposal에 대해 CNN을 거쳐야 하므로 연산량이 매우 많은 단점이 존재

## Fast R-CNN
<center>
<figure>
<img src="/assets/post_img/papers/2019-01-06-Object_detection/fig3.PNG" alt="views">
<figcaption>상: R-CNN의 구조, 하: Fast R-CNN 구조</figcaption>
</figure>
</center>

- Fast R-CNN은 모든 Proposal이 네트워크를 거쳐야 하는 R-CNN의 병목(bottleneck)구조의 단점을 개선하고자 제안 된 방식
- 가장 큰 차이점은, 각 Proposal들이 CNN을 거치는것이 아니라 전체 이미지에 대해 CNN을 한번 거친 후 출력 된 특징 맵(Feature map)단에서 객체 탐지를 수행

<center>
<figure>
<img src="/assets/post_img/papers/2019-01-06-Object_detection/fig4.PNG" alt="views">
<figcaption>Fast R-CNN 구조</figcaption>
</figure>
</center>

- R-CNN
  - Extract image regions
  - 1 CNN per region(2000 CNNs)
  - Classify region-based features
  - Complexity: ~224 x 224 x 2000
- Fast R-CNN
  - 1 CNN on the entire image
  - Extract features from feature map regions
  - Classify region-based features
  - Complexity: ~600 x 1000 x 1
  - ~160x faster than R-CNN





---
- [참고 글]

[1] Ross Girshick, “Rich feature hierarchies for accurate object detection and semantic segmentation”, 2013

[2] Ross Girshick, “Fast R-CNN”, 2015

[3] Shaoqing Ren, “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks”, 2015

[4] Wei Liu, “SSD: Single Shot MultiBox Detector”, 2015

[5] Karen Simonyan, “Very Deep Convolutional Networks for Large-Scale Image Recognition”, 2014

