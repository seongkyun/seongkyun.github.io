---
layout: post
title: Image Classification 연구동향
category: study
tags: [Deep learning]
comments: true
---

# Image Classification 연구동향
- 참고 글
  - [이호성님 HOYA012 블로그](https://hoya012.github.io/blog/Self-training-with-Noisy-Student-improves-ImageNet-classification-Review/?fbclid=IwAR2Z3v3aBDS1Zc-UEG2YCdmrdlqJG3qn4_qubVoLYvJPjXNYZKsLklXTA1s)

## Image Classification 연구동향
- 2012년 AlexNet을 기점으로 많은 종류의 classification network들이 제안됨
- 시간이 흐를수록 연구방향이 바뀌고 있기에 그 흐름을 간단히 정리하면 아래와 같음
  - 2012년 - 2016년: AlexNet, VGG, GoogLeNet, ResNet, DenseNet, SENet 등 사람이 이런저런 시도를 하며 그럴싸한 네트워크를 디자인
  - 2016년 말 - 2018년: AutoML을 이용한 Neural Architecture Search(NAS)를 이용해 최적의 구조를 찾고, 탐색에 필요한 시간을 획기적으로 줄이고, 줄인 만큼 큰 구조를 만들어내는데 집중
    - Neural Architecture Search with Reinforcement Learning (2016.11)
    - NASNet, ENAS, PNASNet, DARTS, AmoebaNet 등 많은 연구 수행
  - 2018년 - 2019년 초중반: AutoML에서 찾은 구조를 기반으로 사람이 튜닝을 하며 성능을 향상시킴
    - GPipe, EfficientNet 등 많은 연구 진행
  - 2019년 초중반: 수십억장의 web-scale extra labeled images등 무수히 많은 데이터를 잘 활용하여 ResNext로도 SOTA 달성
    - Billion-scale semi-supervised learning for image classification (2019.05)
    - Fixing the train-test resolution discrepancy (2019.06)
  - 2019년 말(현재): Labeled web-scale extra images대신 web-scale extra unlabeled images를 써서 self-training을 활용해 SOTA 달성

- 2016년 NAS 연구가 처음 공개된 이후 많은 논문들이 쏟아져나옴
- NAS 연구 초반엔 비현실적인 GPU cost를 요구하는 네트워크 구조들이였기에 꿈만 같은 연구로 여겨졌음
- 하지만 불과 1년만에 하나의 GPU로 하루만에 학습을 시킬 수 있는 방법들이 제안되며 연구가 굉장히 활발하게 수행됨

- AutoML로 찾은 모델을 사람이 튜닝하여 성능을 대폭 개선시키는 연구들도 활발히 수행됨
- 2019년엔 Web-Scale의 수십억 장의 데이터를 활용해 모델의 임계 성능을 끌어올리기도 함

- Image Classification 분야는 다른 분야에 비해 굉장히 많은 연구가 굉장히 빠르게 진행되는 중
- 현재 ImageNet Top-1 Accuracy가 가장 높은 결과가 87.4% 수준으로, 머지않아 90%를 넘기는 논문이 나올 것으로 예상됨
