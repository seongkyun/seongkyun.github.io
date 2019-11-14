---
layout: post
title: Self-training with Noisy Student improves ImageNet classification
category: papers
tags: [Deep learning]
comments: true
---

# Self-training with Noisy Student improves ImageNet classification

Original paper: https://arxiv.org/pdf/1911.04252.pdf

Authors: Qizhe Xie, Eduard Hovy, Minh-Thang Luong, Quoc V. Le

- 참고 글
  - [이호성님 HOYA012 블로그](https://hoya012.github.io/blog/Self-training-with-Noisy-Student-improves-ImageNet-classification-Review/?fbclid=IwAR2Z3v3aBDS1Zc-UEG2YCdmrdlqJG3qn4_qubVoLYvJPjXNYZKsLklXTA1s)

## Introduction
- 앞에서 다뤘던 EfficientNet 논문을 기반으로 ImageNet 데이터셋에 대해 SOTA 성능을 갱신
- 실험 결과를 먼저 보면 아래와 같음

<center>
<figure>
<img src="/assets/post_img/papers/2019-11-14-self_training/fig1.png" alt="views">
<figcaption></figcaption>
</figure>
</center>

- EfficientNet의 실험 결과 그래프와 굉장히 유사한 모양의 그래프
- EfficientNet의 성능이 기존 방법들에 비해 월등히 높아 성능 향상이 적은것 같지만, 그래도 SOTA 성능
- 또한 SOTA 성능 달성을 위해 사용한 방법이 굉장히 간단해서 의미있는 결과임
- 본 논문은 NAS처럼 AutoML로 만들은 네트워크를 사람이 튜닝해 성능을 개선시키고, Web-scale의 수십억 장의 데이터를 활용해 self-training 하도록 하는 방식으로 최고성능을 달성

## Self-training with Noisy Student
- 본 논문의 핵심 아이디어는 아래 사진으로 간단하게 설명 가능

### Self-training

<center>
<figure>
<img src="/assets/post_img/papers/2019-11-14-self_training/fig2.png" alt="views">
<figcaption></figcaption>
</figure>
</center>

- Labeled 데이터셋인 ImageNet을 이용해 teacher model을 학습시킴
- 그 뒤, Unlabeled dataset인 JFT-300M을 teacher model에 흘려보내 prediction값을 구한 되, 이를 pseudo label로 사용함
- 이렇게 얻은 JFT-300M 데이터셋의 pseudo label 과 기존에 사용하던 ImageNet의 label을 이용해 student model을 학습시킴
- 여기서 student model의 학습에 noise성분을 추가해서 학습시킴
- 이 과정을 반복하며 iterative하게 학습시키면 알고리즘이 끝남

- Teacher-student 구조를 보면 Knowledge Distillation을 떠올릴 수 있음
- 하지만 다양한 Knowledge Distillation 방법들과 달리 teacher model로 추론된 pseudo label을 추가적으로 student network가 학습하게 된다는 차이점이 존재

### Noise 기법
- Self-training 기법 외에도 Noisy Student Model이 이 논문의 또다른 핵심 아이디어
- Student Model을 학습시킬 때 아래와 같은 Random한 학습 기법들을 사용함
  - [Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf)
  - [Dropout](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)
  - [RandAugment](https://arxiv.org/pdf/1909.13719.pdf)

### Fix train-test resolution discrepancy
- 최근 좋은 성능을 보인 [Fixing the train-test resolution discrepancy](https://arxiv.org/pdf/1906.06423.pdf) 또한 적용시킴
  - 첫 350epoch동안 작은 resolution으로 학습
  - 이 후 1.5epoch동안 unaugmented labeled images에 대해 큰 resolution으로 fine-tuning 시킴
- 위 방법을 제안한 논문과 유사하게 fine-tuning동안 shallow layer를 freeze시켜서 실험함

### Iterative Training
- 반복적으로 새로운 pseudo label을 만들고, 이를 이용해 student model을 학습시키는 방법
  - 이 과정에서 약간의 트릭이 들어감

- 트릭에 사용된 3가지 EfficientNet의 모델은 각각 B7, L0, L1, L2이며 뒤로 갈수록 모델의 size가 커지는것을 의미함
- 각 모델에 대한 세부 구조는 아래에서 확인 가능

<center>
<figure>
<img src="/assets/post_img/papers/2019-11-14-self_training/fig3.png" alt="views">
<figcaption></figcaption>
</figure>
</center>

- 처음엔 teacher와 student 모두 EfficientNet-B7로 학습시킴
- 그 뒤, teacher는 EfficientNet-B7, student는 EfficientNet-L0로 학습
- 다음에 teacher는 EfficientNet-L0, student는 EfficientNet-L1로 학습
- 다음에 teacher는 EfficientNet-L1, student는 EfficientNet-L2로 학습
- 마지막으로 teacher는 EfficientNet-L2, student는 EfficientNet-L2로 학습

## 실험결과
- Appendix의 추가 결과 외에 본문에 있는 결과 위주로 설명됨

### ImageNet result

<center>
<figure>
<img src="/assets/post_img/papers/2019-11-14-self_training/fig4.png" alt="views">
<figcaption></figcaption>
</figure>
</center>

- ImageNet 데이터셋에 대해 다른 선행 연구들을 모두 제치고 가장 높은 Top-1, Top-5 Accuracy를 달성
- 가장 좋은 성능을 보였던 Noisy Student(L2)는 기존 SOTA 성능을 달성했던 모델들보다 더 적은 파라미터수를 가지며, 학습에 사용된 Extra Data의 크기도 더 적고, Label도 사용하지 않고 달성한 결과라 더 유의미함

<center>
<figure>
<img src="/assets/post_img/papers/2019-11-14-self_training/fig1.png" alt="views">
<figcaption></figcaption>
</figure>
</center>

- 위 그림은 Iterative Training을 적용시키지 않은 결과
- EfficientNet-B0부터 EfficientNet-B7까지 Noisy Student 알고리즘으로 학습 시켰을 때의 결과를 보여줌
- 제안하는 알고리즘들이 모든 경우에서 효과적임을 보여줌

### Robustness 실험결과
- 모델의 신빙성, robustness측정을 위한 test set인 ImageNet-C, ImageNet-P, ImageNet-A를 이용한 실험결과

<center>
<figure>
<img src="/assets/post_img/papers/2019-11-14-self_training/fig5.png" alt="views">
<figcaption></figcaption>
</figure>
</center>

- ImageNet-C, ImageNet-P 데이터셋은 (Benchmarking Neural Network Robustness to Common Corruptions and Perturbations)[https://arxiv.org/pdf/1903.12261.pdf] 에서 제안되었으며, 영상에 blurring, fogging, rotation, scaling등 흔히 발생 가능한 왜곡등의 변화요소를 반영시켜 만든 데이터셋
- ImageNet-A 데이터셋은 (Natural Adversarial Examples)[https://arxiv.org/pdf/1907.07174.pdf]에서 제안됐으며 기존 classification network들이 공통으로 분류를 어려워하는 실제 natural image들을 모아 만든 데이터셋

<center>
<figure>
<img src="/assets/post_img/papers/2019-11-14-self_training/fig6.png" alt="views">
<figcaption></figcaption>
</figure>
</center>

- 각 데이터셋에 대한 자세한 실험결과는 위 사진에서 확인 가능함
- ImageNet-C의 평가에 사용된 mCE 지표와 ImageNet-P의 평가에 사용된 mFR 지표는 낮을수록 좋은 결과를 의미함
  - 본 논문에서 제안하고 있는 방식이 기존 모델들 대비 좋은 성능을 보여주고 있음(mCE와 mFR 모두 가장 낮음)
- ImageNet-A에 대해선 가장 높은 정확도를 보여줌
  - Noisy Student 방식처럼 외부의 데이터셋을 사용하는 ResNeXt-101 WSL모델은 ImageNet-A의 Top-1 accuracy가 매우 낮음(Top-1 acc 16.6%)
  - 논문에서 제안하는 Noisy Student(L2)의 경우 굉장히 높은 정확도(Top-1 acc 74.2%)를 보임
  - EfficientNet-L2 역시 괜찮은 정확도(Top-1 acc 49.6%)를 보임
  - 이는 EfficientNet 자체가 natural adversarial example에 꽤 견고한 모델임을 보여주며, 견고한 baseline architecture에 Noisy Student를 적용시킨다면 결과가 훨씬 더 좋아질 수 있음을 의미함
  
  
