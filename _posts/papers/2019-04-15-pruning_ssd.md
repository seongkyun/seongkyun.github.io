---
layout: post
title: Multi-layer Pruning Framework for Compressing Single Shot MultiBox Detector
category: papers
tags: [Deep learning]
comments: true
---

# Multi-layer Pruning Framework for Compressing Single Shot MultiBox Detector

Original paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8658776&tag=1

Authors: Pravendra Singh, Manikandan R, Neeraj Matiyali, Vinay P. Namboodiri (IIT)

## Abstract
- 본 논문에선 Single Shot MultiBox Detector를 compressing하는 framework를 제안한다. 제안하는 SSD compressing framework는 Sparsity Induction, Filter Selection, Filter Pruning의 stage를 거쳐서 모델을 압축시킨다. Sparsity Induction stage에선 object detector model이 improved global threshold에 따라 sparsified(희박해진다는 의미)되어지며, Filter selection과 Pruning stage에선 두 연이은 convolutional layer의 filter weight에서 sparsity statistics에 근거해서 제거할 filter를 선택하고 제거하게 된다. 이로인해 기존의 구조보다 모델이 더 작아지게 된다. 제안하는 framework를 적용한 모델을 이용하여 다양한 데이터셋을 이용하여 검증하였으며, 그 결과를 다른 compressing 방법과 비교했다. 실험결과 PASCAL VOC에 대해 제안하는 방법을 적용했을 때 각각 SSD300과 SSD512에 대해서 6.7배, 4.9배의 모델 압축을 이뤄냈다. 게다가 SSD512모델에 대한 GTSDB 데이터셋 실험에선 26배의 모델 압축을 이뤄낼 수 있었다. 또한 일반적인 classification task에 대해 VGG16 모델을 이용하여 CIFAR, GTSRB 데이터셋을 학습시킨 모델에 적용시킨 결과 각각 125배, 200배의 모델압축과 90.5%, 96.6%의 연산량(flops)감소를 이루어냄과 동시에 정확도는 하락되지 않았다. 그리고 제안하는 framework를 모델에 적용시키는데엔 다른 추가적인 하드웨어나 라이브러리가 필요없다.

### Conclusion
- 본 논문에선 Multi-layer pruning 방법을 제안했다. 주요 아이디어는 training loss function을 수정해서 첫 번째 procedure로 sparsity(희소성)를 유도하는것이다. 두 번째로, sparsity inducing loss function이 0에 수렴하도록 학습되어진 모델의 filter를 pruning한다. 이로인해 classifier와 detector가 구조적으로 크기가 줄어들면서 정확도가 약간 감소하지만 retraining을 통해 그 정확도를 recover할 수 있게된다. 이로인해 SSD 객채탐지모델을 이용하여 SOTA의 압축률을 달성했다. 더 나아가 visiual recognition task와 다른 관련된 loss를 이용하여 filter를 pruning하는것에 대해 연구할 예정이다.

<center>
<figure>
<img src="/assets/post_img/papers/2019-04-15-pruning_ssd/fig1.jpg" alt="views">
<figcaption></figcaption>
</figure>
</center>
