---
layout: post
title: CenterNet (Objects as Points)
category: papers
tags: [Deep learning]
comments: true
---

# CenterNet (Objects as Points)

Original paper: https://arxiv.org/pdf/1904.07850v2.pdf

Authors: Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi

- 참고 글
  - https://nuggy875.tistory.com/34
- 코드
  - https://github.com/xingyizhou/CenterNet

- 기존의 real-time object detection 모델들의 성능을 가볍게 제치는 single shot 방식의 detector

<center>
<figure>
<img src="/assets/post_img/papers/2019-10-28-centernet/fig1.png" alt="views" height="300">
<figcaption>YOLO v3의 성능을 뛰어넘는다.</figcaption>
</figure>
</center>

- CenterNet은 기존의 anchor box를 사용하는 single-stage detector인 SSD, YOLO와 비슷하지만 크게 다름
  1. CenterNet은 box overlap이 아닌, 오직 위치만 갖고 anchor를 할당
  2. CenterNet은 오직 한 크기의 anchor만을 사용
  3. CenterNet은 더 큰 output resolution을 갖음
- DSSD의 경우엔 40k가 넘는 anchor box를, RetinaNet에선 100k개가 넘는 anchor box들을 사용했음
  - 이는 할당된 anchor box가 실제값(Grount-Truth box)과 충분히 겹처지도록 하기 위해서였음
- 이렇게 많은 anchor box를 사용하면 정확해지긴 하지만 positive anchor box와 negative anchor box 사이의 불균형이 발생
  - 이로 인해 모델의 학습 속도가 느려짐
- 또한, anchor box의 aspect ratio를 결정해야 하는 등, 말 그대로 "비 논리적인" 결정과 hyperparameter를 설정해야 함
  - Anchor box의 aspect ratio는 어떻게 보면 비 논리적인, 비 합리적이라고 생각 할 수 있음
    - 단순히 해당 비율로 존재하는 객체만 찾도록 학습되어지므로 다양한 비율을 갖는 객체들을 찾는데에는 문제가 있음
  
<center>
<figure>
<img src="/assets/post_img/papers/2019-10-28-centernet/fig4.png" alt="views" height="300">
<figcaption>기존의 방식(a)과 CenterNet(b)의 차이</figcaption>
</figure>
</center>

- CornetNet은 anchor box를 사용하는 것에 대한 단점을 위와 같이 열거했으며, Key point estimation을 사용해 고정적이지 않은 단 하나의 anchor를 사용하는 방법을 소개했음
- CornetNet은 Key point estimation을 사용해 detection을 수행

### Keypoint Estimation for Object Detection
- CenterNet은 단 하나의 anchor를 keypoint estimation을 통해 얻어내고, 이러한 개념은 CornerNet에서 처음 소개됨
