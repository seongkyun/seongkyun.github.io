---
layout: post
title: A Fast and Accurate System for Face Detection,Identification, and Verification
category: papers
tags: [Deep learning, Face detection]
comments: true
---

# A Fast and Accurate System for Face Detection,Identification, and Verification

Original paper: https://arxiv.org/pdf/1809.07586.pdf

Authors: Rajeev Ranjan, Ankan Bansal, Jingxiao Zheng, Hongyu Xu, Joshua Gleason, Boyu Lu, Anirudh Nanduri, Jun-Cheng Chen, Carlos D. Castillo, Rama Chellappa

## Abstract
- 많은 데이터셋과 컴퓨터의 연산능력 증가로 인해 CNN을 이용한 object detection, recognition benchmark의 성능이 향상됨. 이로 인해 딥러닝을 이용한 face detection의 성능이 많이 향상 됨. CNN을 이용해 얼굴을 찾을 수 있으며 landmark의 위치를 정의하고 pose나 얼굴 인식을 할 수 있다.
- 본 논문에서는 몇몇의 benchmark 데이터셋을 이용하여 SOTA face detection의 성능을 검증하고 몇몇 face identification에 대한 detail에 대해 다룸.
- 또한 새로운 face detector인 Deep Pyramid Single Shot Face Detector(DPSSD)를 제안한다. 이는 빠르고 face detection시 다양한 scale 변환에 유용하다. 논문에서 automatic face recognition에 관련된 다양한 모듈의 design detail을 설명하며, 이는 face detection, landmark localization and alignment, 그리고 face identification/verification을 포함한다.
- 논문에서는 제안하는 face detector를 이용하여 각종 데이터셋들에 대한 evaluation 결과를 제공하며, IARPA Janus Benchmarks A, B, C(IJB-A, B, C)와 Janus Challenge Set 5(CS5)에 대한 실험 결과를 제공한다.

## 5. Conclusion
- 논문에서는 현재의 CNN을 이용하는 face recognition system을 이용한 모델들의 overview를 제공했다. 논문에서는 face recognition pipeline에 대해 논했으며 SOTA 기술들이었다. 또한 논문에서는 feature representation을 위한 두 네트워크의 앙상블(ensemble)을 사용하는 face recognition system을 제안하고 이에 대한 detail들에 대해 설명했다. 논문에서 제안하는 모델에서 pipeline에서의 Face detection과 keypoint localization은 모두 CNN을 이용하여 한번에 이루어졌다. 논문에서는 시스템을 위한 training과 dataset에 대한 detail에 대해 논했으며 이게 어떻게 face recognition과 연관되어 잇는지 논했다. 논문에서 IJB-A, B, C와 CS5의 4개 challenging dataset에 대한 제안하는 시스템의 실험결과를 제시했다. 그리고 앙상블 based 시스템이 SOTA 결과에 근접하게 나왔다.
- 하지만 풀어야 할 몇몇 issue들이 존재한다. DCNN 기반의 face recognition system에 대한 이론적인 이해를 위한 연구가 필요하다. 주어지는 다양한 loss function들은 이러한 network의 학습을 위해 사용되어지며, 모든 loss function을 같은 맥락으로 통합하는 framework를 개발해야 한다. Domain adaptation과 dataset bias또한 현재의 face recognition system의 issue다. 이러한 시스템은 보통 dataset을 이용하여 학습되어지며 유사한 test set에 대해 잘 동작한다. 하지만 하나의 도메인에 대해서 학습 되어진 네트워크는 다른 도메인에서 잘 동작하지 않는다. 논문에서는 다른 서로 다른 dataset들을 조합하여 학습시켰다. 이로인해 학습되어진 모델이 더 강건(robust)해 졌다. CNN의 학습에는 현재 몇시간에서 몇일이 걸린다. 따라서 더 빠른 학습을 위한 효율적인 구조(architecture)나 CNN의 implementation이 필요하다.


<center>
<figure>
<img src="/assets/post_img/papers/2019-03-21-Fast_accurate_faced_detection/fig1.PNG" alt="views">
<figcaption></figcaption>
</figure>
</center>
