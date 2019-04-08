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
- CNN을 이용하여 object detector의 성능이 크게 향상되었지만, 연산시간또한 크게 증가하게 되는 단점이 존재했다. SOTA 모델들은 매우 deep하고 많은 파라미터들을 갖는 구조를 사용한다. 이러한 모델들의 파라미터 수를 줄이는 연구들이 많이 수행되었지만, 성능또한 같이 줄어드는 단점이 존재했다. 본 논문에서는 knowledge distillation[20]과 hint learning[34]을 이용하여 정확도가 향상된 compact하고 fast한 object detection network의 학습을 위한 framework를 제안한다. 앞선 연구인 knowledge distillation에서는 간단한 classification task에 대한 성능 향상만을 보였다. 하지만 object detection의 경우 regresson, region proposal과 less boluminous label등의 challenging한  문제들이 존재한다.

## Conclusion
- contents

<center>
<figure>
<img src="/assets/post_img/papers/2019-04-08-efficient_detection/fig1.jpg" alt="views">
<figcaption>Figure</figcaption>
</figure>
</center>
