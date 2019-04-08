---
layout: post
title: Learning Efficient Object Detection Models with Knowledge Distillation
category: papers
tags: [Deep learning]
comments: true
---

# Learning Efficient Object Detection Models with Knowledge Distillation

Original paper: https://papers.nips.cc/paper/6676-learning-efficient-object-detection-models-with-knowledge-distillation.pdf

Authors: Guobin Chen, Wongun Choi, Xiang Yu, Tony Han, Manmohan Chandraker

## Abstract
- CNN을 이용하여 object detector의 성능이 크게 향상되었지만, 연산시간또한 크게 증가하게 되는 단점이 존재했다. SOTA 모델들은 매우 deep하고 많은 파라미터들을 갖는 구조를 사용한다. 이러한 모델들의 파라미터 수를 줄이는 연구들이 많이 수행되었지만, 성능또한 같이 줄어드는 단점이 존재했다. 본 논문에서는 knowledge distillation[20]과 hint learning[34]을 이용하여 정확도가 향상된 compact하고 fast한 object detection network의 학습을 위한 framework를 제안한다. 앞선 연구인 knowledge distillation에서는 간단한 classification task에 대한 성능 향상만을 보였다. 하지만 object detection의 경우 regresson, region proposal과 less boluminous label등의 challenging한 문제들이 존재한다. 논문에선 이러한 문제들에 대한 개선을 위해 class imbalance 해결을 위한 weighted cross-entropy loss, regression component를 해결하기 위한 teacher bounded loss, 그리고 intermediate teacher distributions을 더 잘 학습하기 위한 adaptation layer등을 제안한다. 실험에선 PASCAL, KITTI, ILSVRC, MS-COCO 데이터셋을 이용하여 다른 distillation configuration들에 대한 실험을 진행했으며, 일반적인 mAP의 측정으로 평가되었다.  실험 결과는 최근의 multi-class detection models에 대한 accuracy-speed trade-off의 향상을 보여준다. 

## Conclusion
- 논문에선 knowledge distillation을 이용한 compact하고 빠른 CNN based object detector의 학습 framework를 제안했다. 매우 복잡한 teacher detector를 guide로 하여 효율적인 student model을 학습시켰다. Knowledge distillation, hint framework와 제안하는 loss function을 이용하였을 때 다양한 실험 setup에 대하여 모두 성능이 향상되었다. 특히 제안하는 framework로 학습된 compact model은 PASCAL VOC 데이터셋에 대한 teacher model의 정확도와 매우 비슷한 수준으로 훨씬 빠르게 동작하였다. 논문의 실험을 통해 object detector의 학습에 under-fitting 문제가 있음을 확인했으며, 이는 해당 연구분야에서 더 연구 가능한 insight를 준다.

<center>
<figure>
<img src="/assets/post_img/papers/2019-04-08-efficient_detection/fig1.jpg" alt="views">
<figcaption>Figure</figcaption>
</figure>
</center>
