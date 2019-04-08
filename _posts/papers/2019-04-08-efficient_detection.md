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

## 1. Introduction
- CNN등의 발전으로 인해 object detection의 성능은 매우 많이 발전했지만, 실제로 이를 적용하기에는 속도가 너무 느린 단점이 존재했다. 많은 연구들이 정확한 객체검출기 모델의 개발을 위해 진행되었지만 모두 deep한 구조를 사용함으로 인해 runtime의 computational expense가 증가했다. 하지만 이는 deep neural network들이 일반화를 위한 over-parameterized 문제로 알려져있다. 따라서 속도 향상을 위해 몇몇의 연구에선 새로운 구조의 fully convolutional network를 만들어 이를 해결하고자 하거나 작은 filter와 적은 channel을 이용해 파라미터 수를 줄여보고자 했다[22, 25]. 이로인해 객체검출기의 속도가 매우 빨라졌지만 아직도 real-time으로 사용하기엔 무리가 있고 성능향상을 위한 tunning이나 까다로운 redesign등이 필요하다.
- 표현력이 충분한 deep 네트워크의 경우 학습이 잘 되면 더 나은 성능을 보여준다. 하지만 적은 class를 위한 객체검출 모델들은 이런 표현력이 큰 모델이 필요없다. 이로인해 [9, 26, 41, 42]논문에선 모델 압축을 이용해 layer-wise reconstrunction을 따라 각 레이어의 weight를 분해하거나 fine-tunning을 이용해 정확도를 조금 올리게 된다. 이러한 방법들로 인해 속도는 많이 향상되지만 여전히 original model과 압축된 모델간의 성능차가 존재하며, object detection에 이를 적용할 경우 압축모델과 원래모델의 성능차이는 더 벌어지게 된다. 반면 knowledge distillation이 적용된 연구들의 경우 깊고 복잡한 모델의 behavior를 흉내내도록 shallow or compressed 모델이 학습되어지며 knowledge distillation을 통해 대개의 모델들이 정확도 하락을 복구하게 된다[3, 20, 34]. 하지만 이러한 결과들은 dropout과 같은 strong regularization도 적용하지 않고 간단한 네트워크를 이용해 간단한 classification문제에 대한 결과만들 보여준다.
- Distillation을 복잡한 multi-class object detection에 적용하기에는 아래의 몇몇 문제가 따른다. 우선, 객체검출 모델의 성능은 모델 압축 시 classification 모델에 비해 성능이 더 떨어지게 된다. 이는 detection label이 더 expensive 하고 양이 많지 않기 때문이다(expensive and less boluminous). 두번째로, 모든 class가 동등하게 중요하다고 가정되는 classification을 위해 distillation이 제안되지만, object detection과 같이 background class가 훨씬 더 많이 사용되는 경우에는 knowledge distillation이 별 소용이 없게된다. 세번째로, detection은 각 classification과 bounding box regression을 한데로 묶는 복잡한 task이다. 마지막으로, 추가 challenge로써 다른 도메인의 데이터(high-quality and low-quality image domains, or image and depth domains)에 의존하는 다른 task와는 다르게 추가 데이터나 label이 없이 동일한 도메인(images of the same dataset) 내에서 정보(knowledge)를 이전하는데에 집중하는것이다. 
- 위의 challenges에 대해 논문에선 knowledge distillation을 이용한 fast object detection 모델의 학습을 위한 방법을 제안한다. 논문의 main contributions는 아래에 있다.
  - 






## Conclusion
- 논문에선 knowledge distillation을 이용한 compact하고 빠른 CNN based object detector의 학습 framework를 제안했다. 매우 복잡한 teacher detector를 guide로 하여 효율적인 student model을 학습시켰다. Knowledge distillation, hint framework와 제안하는 loss function을 이용하였을 때 다양한 실험 setup에 대하여 모두 성능이 향상되었다. 특히 제안하는 framework로 학습된 compact model은 PASCAL VOC 데이터셋에 대한 teacher model의 정확도와 매우 비슷한 수준으로 훨씬 빠르게 동작하였다. 논문의 실험을 통해 object detector의 학습에 under-fitting 문제가 있음을 확인했으며, 이는 해당 연구분야에서 더 연구 가능한 insight를 준다.

<center>
<figure>
<img src="/assets/post_img/papers/2019-04-08-efficient_detection/fig1.jpg" alt="views">
<figcaption>Figure</figcaption>
</figure>
</center>
