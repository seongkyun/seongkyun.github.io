---
layout: post
title: Unsupervised Visual Representation Learning Overview (Self-Supervision)
category: study
tags: [CNN, Deep learning, Unsupervised learning]
comments: true
---

# Unsupervised Visual Representation Learning Overview (Self-Supervision)
- 참고
  - https://www.youtube.com/watch?v=eDDHsbMgOJQ
  - https://www.slideshare.net/HoseongLee6/unsupervised-visual-representation-learning-overview-toward-selfsupervision-194443768

## What is "Self-Supervision"?
- 지도학습(Supervised learning)은 강력하지만 많은 량의 labeled data가 필요함
  - 주로 다양한 문제들을 해결함에 있어서 labeled data를 사용하는 지도학습 모델을 많이 사용함
    - 많은량의 양질의 데이터만 있다면 좋은 성능을 낼 수 있음
  - 하지만 이는 labeled data가 없으면 사용 불가능하다는 문제가 있음
    - 이러한 labeled data를 만드는데도 매우 큰 human factor가 필요함
  - 이처럼 labeled data 문제를 해결하고자 다양한 방법의 un/self supervised learning 방법들이 연구됨
- Self-supervised visual representation learning
  - 본 글에선 비지도 학습(Unsupervised learning)의 한 분야인 self-supervised learning을 이미지 인식분야에 적용한 연구들에 대해 다룸
  - Unlabeled data를 이용해서 모델을 학습시키며, 모델이 이미지의 higher level의 semantic한 정보들을 이해 할 수 있도록 학습됨
    - 이를 위해 필요한게 __pretext task__
  - Pretext task란, 딥러닝 네트워크가 어떤 문제를 해결하는 과정에서 영상 내의 semantic한 정보들을 이해 할 수 있도록 배울 수 있도록 학습되게 하는 임의의 task
    - Pretext task는 딥러닝 네트워크가 semantic 정보를 학습하도록 사용자가 임의대로 정의하게 됨
  - 이렇게 pretext task를 학습하며 얻어진 feature extractor들을 다른 task로 transfer시켜 주로 사용함
- Self-supervised visual representation learning에서 사용하는 pretext task들
  - Exempler, 2014 NIPS
    - Exampler가 처음에 나오고, 이와 관련된 연구들이 계속해서 진행됨
  - Relative Patch Location, 2015 ICCV
  - Jigsaw Puzzles, 2016 ECCV
  - Autoencoder Base Approaches - Denoising Autoencoder(2008), Context Autoencoder(2016), Colorization(2016), Split-brain Autoencoders(2017)
  - Count, 2017 ICCV
  - Multi-task, 2017 ICCV
  - Rotation, 2018 ICLR

## Exemplar
- Discriminative unsupervised feature learning with exemplar convolutional neural networks, 2014 NIPS
- STL-10 데이터셋을 사용해 실험
- STL-10의 96x96크기의 영상 내에서 considerable한 gradient가 있는(객체가 존재할 만한 영역) 부분을 32x32 크기로 crop
- 이렇게 crop된 32x32 크기의 패치를 seed patch로 하고, 이 seed patch가 하나의 class를 의미하도록 함
- Seed patch를 data augmentation에 사용하는 transformation들을 적용시켜 추가영상을 만들어냄

<center>
<figure>
<img src="/assets/post_img/study/2019-11-29-unsupervised/fig1.PNG" alt="views">
<img src="/assets/post_img/study/2019-11-29-unsupervised/fig2.PNG" alt="views">
<figcaption></figcaption>
</figure>
</center>

- 위처럼, seed image를 data augmentation을 적용시켜 여러개로 부풀리고, 부풀려진 data를 seed image로 판독하도록 네트워크를 학습시킴
- 예를들어
  - Deer seed 1번 영상으로 총 23장의 영상을 생성한 후, 해당 seed를 1번 클래스로 지정
  - Deer seed 2번으로(1번과 다름) 총 23장의 영상을 생성한 후, 해당 seed를 2번 클래스로 지정
  - 이런식으로 총 N개의 영상에 대해 N개의 클래스가 존재하게 됨
- Classifier는 하나의 seed image랑 augmented 영상들을 같은 class로 분류하도록 학습됨
- 하지만 이런 경우, ImageNet과 같이 백만장의 영상으로 구성된 데이터셋은 총 백만 개의 class가 존재하게 되는 꼴
  - 학습 난이도가 올라가며 파라미터수도 매우 많아지게 됨
- 따라서 이런 Exemplar 방식은 큰 데이터셋에는 적합하지 않음
