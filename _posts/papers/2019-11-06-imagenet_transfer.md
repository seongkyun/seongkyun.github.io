---
layout: post
title: Do Better ImageNet Models Transfer Better?
category: papers
tags: [Deep learning]
comments: true
---

# Do Better ImageNet Models Transfer Better?

Original paper: https://arxiv.org/pdf/1805.08974.pdf

Authors: Simon Kornblith, Jonathon Shlens, and Quoc V. Le (Google Brain)

- 참고 글
  - https://norman3.github.io/papers/docs/do_better_imagenet_models_transfer_better.html

## Abstract
- Pretrained 된 모델을 이용해 다른 task에서 transfer learning을 하는것은 computer vision 분야에서 매우 효과가 좋음
- 하지만 여기엔 슬픈 전설이 있음
  - ImageNet에서의 성능이 좋은 모델일수록 해당 모델을 backbone으로 사용해서 transfer learning을 하면 성능이 더 좋음
- 본 논문에서는 12개의 데이터 셋, 16개의 classification 모델을 비교해서 위의 가설이 사실인지 검증함
  - 실제 backbone과 tranfer task와의 성능에 대한 상관 관계가 매우 높은것을 확인함
 
## Introduction
- 지난 십여년간 computer vision 학계에서는 모델간 성능 비교를 위한 벤치마크 측정 수단을 만들이는데 공을 들임
- 그 중 가장 성공한 프로젝트는 ImageNet
- ImageNet으로 학습된 모델을 이용해 transfer learning, object detection, image segmentation 등의 다양한 task에 대해 성능평가를 수행함
- 여기서 암묵적인 가정은
  1. ImageNet에서 좋은 성능을 보이는 모델은 다른 image task에서도 좋은 성능을 낸다는 것
  2. 더 좋은, 성능이 좋은 모델을 사용할수록 transfer learning에서 더 좋은 성능을 얻을 수 있음
- 이전의 다양한 연구들을 토대로 위 가정들은 어느정도 맞는듯 함

- 본 논문에서는 실험 기준을 세우기 위해 ImageNet feature와 classification model 모두를 살펴봄
  - 16개의 최신 CNN 모델들과 12개의 유명한 classification dataset을 사용해 검증

- 논문에서는 총 3가지 실험을 수행
  1. Pretrained ImageNet에서 고정된 feature 값을 추출한 뒤, 이 결과로 새로운 task를 학습
    - Transfer learning as a fixed feature extractor
      - Feature extractor는 그대로, 뒤 쪽은 학습
  2. Pretrained ImageNet을 다시 fine-tuning 하여 학습
    - Transfer learning
      - 일반적인 전이학습으로, pretrained 모델로 weight parameter 초기화 후 해당값을 시작점으로 하여 재학습
  3. 그냥 각 모델들을 개별 task에서 from scratch로 학습
    - 처음부터 모델을 학습시키는 방법

- Main contributions
  - 더 나은 성능의 imageNet pretrained model을 사용하는것이 linear classification의 transfer learning에서 더 나은 feature extractor의 feature map을 만들어내며(r=0.99), 전체 네트워크가 fine-tuning 되었을 때 더 나은 성능을 보임(r=0.96)
  - ImageNet task에서 모델의 성능을 향상시키는 regularizer들은 feature extractor의 출력 feature map의 관점에서 transfer learning에 오히려 방해가 됨
    - 즉, transfer learning에서는 regularizer들을 사용하지 않는것이 성능이 더 좋았음
  - ImaegNet에서 성능이 좋은 모델일수록 다른 task에서도 비슷하게 성능이 더 좋았음
      
  

<center>
<figure>
<img src="/assets/post_img/papers/2019-11-06-imagenet_transfer/fig1.png" alt="views" height="300">
<figcaption></figcaption>
</figure>
</center>
