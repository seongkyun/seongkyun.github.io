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

## 1. Introduction
-  

## 2. A brief survey of existing literature
- 본 챕터에서는 간단하게 현존하는 다른 방법을 사용하는 face identification/verification pipeline 모듈들에 대한 overview를 제시한다. 우선 최근의 face detection  method에 대한 것을 다룬다. 다음으로 두 번째 모듈인 facial keypoint detection에 대해 고려한다. 마지막으로 feature learning에 관한 최근의 여러 연구에 대해 논하고 face verification과 identification에 대한 SOTA 연구들을 요약한다.

### 2.1 Face Detection
- Face detection은 face recognition/verfication pipeline의 첫 번째 step이다. Face detection 알고리즘은 주어진 입력 영상에 대해 모든 얼굴들의 위치를 출력하며, 보통 bouning box 형태로 출력한다. Face detector는 pose, illumination, view-point, expression, scale, skin-color, some occlusions, diguises, make-up 등에 대한 변화요인으로부터 강건해야한다. 대부분의 최근 DCNN-based face detector들은 일반적인 object detection의 접근 방법으로부터 영감 받아졌다(inspired by). CNN detector 들은 두 개의 sub-category로 나뉘어지는데, 하나는 Region-base와 다른 하나는 Sliding window-based 방식이다.
- __Region-based__ 접근방식은 우선 object-proposal들을 생성하며 CNN classifier를 이용해 각각의 proposal을 분류하여 그것이 얼굴인지 아닌지를 판별한다. 첫 번째 step은 보통 off-the-shelf(기성품인) proposal generator인 Selective search[26] 을 사용한다. 최근의 다른 HyperFace[10] detector나 All-in-One Face[18]같은 경우 이러한 방법을 사용한다. Generic method를 이용하여 object proposal을 생성하는 방법 말고, Faster R-CNN[5]는 Region Proposal Network(RPN)을 사용한다. Jiang and Learned-Miller는 Faster-RCNN 네트워크를 face detect에 사용하였다[27]. 유사하게 [28]에서는 Faster-RCNN framework를 사용하여 mulit-task face detector를 제안하였다. Chen et al.[29]에서는 multi-task RPN을 face detection과 facial keypoint localization을 위해 학습 시켰다. 이로인해 낭비되어지는(너무 많이 제안되는) face proposal을 줄일 수 있었으며 그 결과 face proposal의 quality가 좋아졌다. Single Stage Headless face detector[7] 또한 RPN 기반이다.
- __Sliding window-based__ 접근방식은 주어진 scale에서의 feature map에서의 모든 location에서 얼굴을 탐지하여 출력한다. 이러한 detection 방식은 face detection score와 bounding box의 형태로 구성되어 있다. 이러한 접근방식은 proposal generation step으로 분리되어 있지 않으므로(한번에 detection) region-based 방식(2-step)에 비해 더 빠르게 동작한다(one-step).  [9]나 [30]같은 경우 multiple scale에서 image pytamid를 생성하여 multi-scale detection을 수행한다. 유사하게 [31]에선 multiple resolution을 위한 cascade architecture을 사용한다. Single Shot Detector(SSD)[6] 또한 multi-scale sliding-window 기반의 object detector다. 하지만 multi-scale procession을 위한 object pyramid를 사용하는 대신에, SSD는 deep CNN들의 계층적 특성을 이용한다(utilizes the hierarchal nature of deep CNNs). ScaleFace[32]나 S3FD[33]과 같은 방법 또한 face detection을 위한 유사한 방법을 사용한다.
- 더해서 detection 알고리즘을 개선하기 위해, large annotated datset을 사용 가능하게 됨으로써 face detection의 성능이 급진적으로 좋아지고 있다. FDDB[34]는 2,845개의 이미지를 갖고 있으며 전체 5,171개의 얼굴들을 포함한다. 유사한 scale인 MALF[35] 데이터셋은 5,250개의 이미지로 구성되었으며 11,931개의 얼굴을 갖고 있다. 더 큰 데이터셋은 WIDER Face[22]다. WIDER Face는 32,000개가 넘는 이미지를 갖고 있으며 expression, scale, pose, illuminsation 등에 대한 다양한 변화를 갖는 이미지를 갖고있다. 대부분의 SOTA face detector들은 WIDER Face 데이터셋을 이용하여 학습되어졌다. 이 데이터셋은 작은(tiny) 크기의 얼굴들을 많이 갖고있다. 위에서 논해진 몇몇의 face detector들은 아직도 이미지에서 작은 얼굴들을 잘 찾기 위해 노력하고 있다.(작은 얼굴들에 대한 탐지 결과가 좋지 못함). [36]은 이러한 작은 얼굴들의 탐지가 왜 중요한지에 대하여 보여준다.
- [37]에서는 2014년 이전에 개발된 수많은 face detection 방법들에 대한 survey의 연장선을 보여준다(extensive survey). [12]에서는 video에서 face recognition을 위한 face associtaion의 중요성을 논한다. Association은 다른 비디오 프레임에서의 다른 얼굴들에대한 관련성을 찾는 과정이다.

### 2.2 Facial Keypoints Detection and Head Orientation
- Facial keypoints는 corners of eyes나 nose tip, ear lobes, mouth corner 등을 포함한다. 

### 2.3 Face Identification and verification

### 2.4 Multi-task Learning for Facial Analysis

## 3. A state-of-the-art face verification and recognition pipeline
- 

## 4. Experimental results
- 

## 5. Conclusion
- 논문에서는 현재의 CNN을 이용하는 face recognition system을 이용한 모델들의 overview를 제공했다. 논문에서는 face recognition pipeline에 대해 논했으며 SOTA 기술들이었다. 또한 논문에서는 feature representation을 위한 두 네트워크의 앙상블(ensemble)을 사용하는 face recognition system을 제안하고 이에 대한 detail들에 대해 설명했다. 논문에서 제안하는 모델에서 pipeline에서의 Face detection과 keypoint localization은 모두 CNN을 이용하여 한번에 이루어졌다. 논문에서는 시스템을 위한 training과 dataset에 대한 detail에 대해 논했으며 이게 어떻게 face recognition과 연관되어 잇는지 논했다. 논문에서 IJB-A, B, C와 CS5의 4개 challenging dataset에 대한 제안하는 시스템의 실험결과를 제시했다. 그리고 앙상블 based 시스템이 SOTA 결과에 근접하게 나왔다.
- 하지만 풀어야 할 몇몇 issue들이 존재한다. DCNN 기반의 face recognition system에 대한 이론적인 이해를 위한 연구가 필요하다. 주어지는 다양한 loss function들은 이러한 network의 학습을 위해 사용되어지며, 모든 loss function을 같은 맥락으로 통합하는 framework를 개발해야 한다. Domain adaptation과 dataset bias또한 현재의 face recognition system의 issue다. 이러한 시스템은 보통 dataset을 이용하여 학습되어지며 유사한 test set에 대해 잘 동작한다. 하지만 하나의 도메인에 대해서 학습 되어진 네트워크는 다른 도메인에서 잘 동작하지 않는다. 논문에서는 다른 서로 다른 dataset들을 조합하여 학습시켰다. 이로인해 학습되어진 모델이 더 강건(robust)해 졌다. CNN의 학습에는 현재 몇시간에서 몇일이 걸린다. 따라서 더 빠른 학습을 위한 효율적인 구조(architecture)나 CNN의 implementation이 필요하다.


<center>
<figure>
<img src="/assets/post_img/papers/2019-03-21-Fast_accurate_faced_detection/fig1.PNG" alt="views">
<figcaption></figcaption>
</figure>
</center>
