---
layout: post
title: CNN의 parameter 개수와 tensor 사이즈 계산하기
category: study
tags: [CNN parameter, tensor size]
comments: true
---

# CNN의 parameter 개수와 tensor 사이즈 계산하기
- 이번 글에서는 네트워크의 텐서 사이즈와 파라미터의 갯수를 계산하는 공식에 대해 다루려 한다.
- 아래의 AlexNet을 이용하여 예시를 든다.

<center>
<figure>
<img src="/assets/post_img/study/2019-01-25-num_of_parameters/fig1.png" alt="views">
<figcaption>Alexnet의 구조</figcaption>
</figure>
</center>

- AlexNet의 구조
  - Input: 227*227*3 크기의 컬러 이미지. 논문의 224*224 사이즈는 오타임
  - Conv-1: 11*11 크기의 커널 96개, stride=4, padding=0
  - MaxPool-1: stride 2, 3*3 max pooling layer
  - Conv-2: 5*5 크기의 커널 256개, stride=1, padding=2
  - MaxPool-2: stride 2, 3*3 max pooling layer
  - Conv-3: 3*3 크기의 커널 384개, stride=1, padding=1
  - Conv-4: 3*3 크기의 커널 384개, stride=1, padding=1
  - Conv-5: 3*3 크기의 커널 256개, stride=1, padding=1
  - Maxpool-3: stride 2, 3*3 max pooling layer
  - FC-1: 4096개의 fully connected layer
  - FC-2: 4096개의 fully connected layer
  - FC-3: 1000개의 fully connected layer

- 본 글에서는 AlexNet을 이용하여 아래의 내용을 알아본다.
  - 각 단계에서 텐서의 크기를 계산하는 방법
  - 네트워크에서 총 파라미터 개수를 계산하는 방법
  

---
- [참고 글]

https://www.learnopencv.com/number-of-parameters-and-tensor-sizes-in-convolutional-neural-network/
