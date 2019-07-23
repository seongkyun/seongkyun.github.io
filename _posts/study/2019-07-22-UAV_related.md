---
layout: post
title: Vehicle detection in the UAV imagery related papers
category: study
tags: [Convolutional Neural Network, Training]
comments: true
---

# Vehicle detection in the UAV imagery related papers

## Hand crafted methods
### Car Detection in Low Resolution Aerial Image (2001)
- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=937593&tag=1

- Top-view 방식의 저해상도 사진에서 vehicle을 detection하는 방법
- 차량의 외곽과 앞유리 등의 edge 정보와 그림자 정보를 이용하여 차량을 detection하는 traditional 한 방법
  - 차량의 전체적인 shape에 대한 특징을 학습하도록 함
  - 이러한 feature들의 학습에는 bayesian network를 이용
- 일반화성능이 매우 좋지 못하므로 제한된 상황에서만 적용 가능하며, 오탐률이 큼

### Autonomous Real-time Vehicle Detection from a Medium-Level UAV (2009)
- https://pdfs.semanticscholar.org/277c/adfadc4550fc781be7df8cb4ec89e54b793e.pdf?_ga=2.254030475.2079888018.1563806469-1988452867.1561287261

- Bird-view 방식의 UAV imagery에서 vehicle detection하는 방법
- Cascaded Haar classifier를 학습시켜 vehicle detection을 수행
  - 일반화성능이 좋지 못하므로 정확도가 떨어짐
- 일반화성능이 매우 좋지 못하며 제한된 상황에서만 적용 가능하고 오탐률이 큼

### Real-time people and vehicle detection from UAV imagery (2011)
- https://www.spiedigitallibrary.org/conference-proceedings-of-spie/7878/78780B/Real-time-people-and-vehicle-detection-from-UAV-imagery/10.1117/12.876663.short?SSO=1

- Vehicle detection을 위해 cascaded Haar classifier와 thermal image를 사용
  - 일반화 성능이 떨어지는 cascaded Haar classifier, multivariate Gaussian shape matching을 사용
  - Thermal image를 얻기 위한 별도의 고가의 장비가 필요

### Car Detection from High-Resolution Aerial Imagery Using Multiple Features (2012)
- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6350403

- Top-view 방식의 고해상도 사진에서 vehicle을 detection하는 방법
- Histogram of Oriented Gradients (HOG), Local Binary Pattern (LBP), Opponent Histogram, Intersection kernel support vector machine (IKSVM) 등을 이용하여 학습한 후, exhaustive search를 이용하여 vehicle detection을 수행 후, non-maximum suppression(NMS)를 이용하여 겹치는 결과들을 제거
- 일반화 성능이 떨어지므로 사용에 제한이 있으며 정확도가 떨어짐

### Vehicle detection methods from an unmanned aerial vehicle platform (2012)
- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6294294

- Top-view 방식의 UAV imagery에서 vehicle detection하는 방법
- 움직이는 객체는 scale invariant feature transform (SIFT)와 Kanada-Lucas-Tomasi (KLT) matching algorithm을 조합하여 feature point tracking을 이용하여 찾으며, 움직이지 않는 객체는 도로를 찾은 뒤 blob information을 이용하여 탐지함
- 일반화성능이 떨어지므로 사용에 제한이 있고 정확도가 떨어짐

### Detecting Cars in UAV Images With a Catalog-Based Approach (2014)
- https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6719494

- Top-view UAV imagery에서 vehicle detection 수행
- 아스팔트 위의 차량만 찾도록 고안됨
- 아스팔트를 일단 찾은 후 해당 영역에서 수직/수평 방향의 필터링 연산을 수행하여 생성된 HOG feature를 이용하여 catalog에 등록된 reference car feature와의 유사도를 계산하여 vehicle detection을 수행
- 일반화성능이 떨어지고 사용에 제한이 있으며 정확도가 낮음

### An Enhanced Viola–Jones Vehicle Detection Method From Unmanned Aerial Vehicles Imagery (2017)
- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7726065

- 원래의 Viola-Jones obejct detection method를 개선시켜 저 고도에서의 UAV imagery vehicle detection이 잘되도록 함
  - 다양한 방향을 갖는 영상에 대비되도록 도로를 수평방향으로 정렬시키는 과정이 필요함
  - Detection 후 tracker를 이용하여 tracking
- 일반화 성능이 떨어지는 hand crafted method

## Deep learning methods
### Fast Vehicle Detection in UAV Images (2017)
- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7958795

- 기존의 YOLO V2 구조를 그대로 이용하여 UAV imagery에서의 vehicle detection을 수행
  - YOLO V2의 구조적 단점을 거론해야 함 (작은 객체를 찾도록 최적화되어있지 않음)
- Image annotation과 data augmentation을 적용하였으며, 영상에서 vehicle을 annotate하기위해 CSK tracking을 적용함. 
- 입력으로 416x416 크기 영상을 사용하며, GeForce GTX 1060에서 48ms의 속도로 동작함
  - 제안하는 방법은 33ms의 속도로 동작

### Object Recognition in Aerial Images Using Convolutional Neural Networks (2017)
- https://webcache.googleusercontent.com/search?q=cache:Fic0MXgNDy0J:https://www.mdpi.com/2313-433X/3/2/21/pdf+&cd=4&hl=en&ct=clnk&gl=kr

- YOLO를 이용하여 UAV imagery dataset을 제안하고 모델을 학습시킴.
- 다만, YOLO의 hyper parameter들을 이용하여 최적화함
  - YOLO가 갖는 작은 객체를 잘 찾지 못하는 단점등에 대해 논해야 함
  - 또한 YOLO를 이용할 경우 저 고도에서 큰 객체밖에 찾지 못함
- Top-view 방식이며, 150FPS로 동작하지만 그 한계가 명확함

### Vehicle Detection Under UAV Based on Optimal Dense YOLO Method (2018)
- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8599403

- YOLO v2의 구조를 활용하여 작은 객체를 잘 찾도록 최적화 된 네트워크를 제안하고, 이를 DOLO라고 명명
- Top-view 방식의 vehicle detection에서 정확도가 비교적 뛰어나지만, 단순히 YOLO의 네트워크를 깊게 하여 표현력을 증대시키고 Dense YOLO라고 하며 DOLO로 명명
- 또한 네트워크 전체의 깊이가 깊어짐에 따라 연산량의 증가폭이 큼
  - 실시간 동작이 가능하지만, 정확도가 좋은 모델은 연산량이 매우 큼
  
### Real-time Vehicle Detection from UAV Imagery (2018)
- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8499466&tag=1

- 네트워크 구조가 복잡하고 느리지만 성능이 좋은 RefineDet을 최적화하여 UAV Imagery object detection을 수행
  - 구조를 그대로 사용한것이 아니라 default box의 크기를 작게 해 작은 객체 탐지에 최적화시킴

### A Closer Look at Faster R-CNN for Vehicle Detection (2016)
- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7535375

- Faster RCNN을 사용하여 vehicle detection을 수행.(UAV 아님)
  - Faster RCNN 기반의 새로운 detector를 제안함
