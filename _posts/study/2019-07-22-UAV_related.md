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

### An Enhanced Viola–Jones Vehicle Detection Method From Unmanned Aerial Vehicles Imagery
- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7726065

- 원래의 Viola-Jones obejct detection method를 개선시켜 저 고도에서의 UAV imagery vehicle detection이 잘되도록 함
  - 다양한 방향을 갖는 영상에 대비되도록 도로를 수평방향으로 정렬시키는 과정이 필요함\
  - Detection 후 tracker를 이용하여 tracking
- 일반화 성능이 떨어지는 hand crafted method

## Deep learning methods








