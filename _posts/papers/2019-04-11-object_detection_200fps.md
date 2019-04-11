---
layout: post
title: Object detection at 200 Frames Per Second
category: papers
tags: [Deep learning]
comments: true
---

# Object detection at 200 Frames Per Second

Original paper: https://arxiv.org/pdf/1805.06361.pdf

Authors: Rakesh Mehta, Cemalettin Ozturk

## Abstract
- 논문에선 수백FPS로 동작하는 효율적이고 빠른 object detector를 제안한다. 이를 위해 network architecture, loss function, traning data(labeled and unlabeled)의 세 가지 관점에서 연구했다. 작은 network architecture를 얻기 위해 몇몇의 연산량이 적은 light-weight 모델이면서 성능은 합리적인 연구들에 근거한 몇몇의 제안점들을 설명한다. 연산복잡도는 유지하면서 성능의 향상을 위해 distillation loss를 활용한다. Distillation loss를 사용함으로써 더 정확한 teacher network의 정보(knowledge)를 제안하는 light-weight student network에 전달한다. 논문에선 제안하는 one stage detector pipeline의 distillation이 효율적으로 동작하게 하기위해 objectness scaled distillation loss, feature map non-maximal suppression, detection을 위한 single unified distillation loss function을 제안한다. 마지막으로 distillation loss이 unlabeled data를 활용하여 얼마나  모델의 성능을 끌어올릴 수 있는지에 대해 탐구한다. 제안하는 모델은 teacher network의 soft label을 사용하는 unlabeled data도 이용하여 학습되어진다. 제안하는 네트워크는 VGG based object detector보다 10배 적은 파라미터를 갖고, 속도는 200FPS를 넘어서며 PASCAL dataset에 대해 제안하는 방법을 적용하여 14mAP의 정확도를 달성하는것이다.

### 1. Introduction
- [25, 27, 22, 26, 7, 10, 21]의 연구들에선 deep convolutional network를 이용하여 정확도가 좋은 모델들이 제안되었다. [22(SSD), 26(YOLO9000)]등의 최신 연구에선 일반적인 객체를 정확하고 합리적인 속도로 검출했다. 이러한 연구들로 인해 감시, 자동주행, 로보틱스 등 다양한 산업분야에서 객채탐지가 사용가능해졌다. 해당 분야의 주요 연구들은 pubil benchmarks [21, 10]에 대해 SOTA의 성능을 내도록 집중되었다. 이러한 연구들로 인해 네트워크는 점점 더 깊어져만 갔고(Inception[33], VGG[32], Resnet[11]) 그결과 연산비용이 비싸지며 메모리를 많이 필요로하게 되었다. 이러한 연구들로 연산량증가로 인해 산업에 full-scale로 적용하는것은 불가능하다는 문제가 존재했다. 예를들어 30FPS로 동작하는 50개의 카메라들로 구성된 시스템에 SSD512를 적용하기 위해선 60개의 GPU가 필요하다[22]. 이는 처리해야 할 데이터량이 많아질경우 연산능력이 critical하며 필요한 GPU의 수는 더 많아진다. 하지만 몇 연구들에선 이러한 빠르고 low memory만 요구하며 효율적인 object detector의 중요성을 간과하였다[17]. 본 논문에선 단일GPU로 low memory만 사용하는 효율적이고 빠른 연산시간을 갖는 object detector에 대해 연구한다.
- 빠르고 효율적인 object detector를 디자인하기 위해 딥러닝 object detector에 필수 요소, 해당 필수요소를 만족하는 객채탐지기의 개발가능여부에 대해 중점적으로 탐구했다. [25, 27, 22, 26, 7]에 근거하여 딥러닝 기반의 객채탐지기 famework의 key component를 다음과 같이 정의했다. (1) Network architecture, (2) Loss function, (3) Traning data. 논문에선 각 component들에 대해 개별적으로 연구하고 본 논문에서 제안하거나 related work에 근거한 broad set of customizations을 설명하며, 그것들 중 뭐가 속도와 정확도의 trade-off에 있어서 가장 중요한지에 대해 연구한다.
- 네트워크 architecture는 객체탐지기의 속도와 정확도를 결정짓는 key factor다. 최근의 탐지기들[15]은 VGG나 Resnet등의 깊은 구조를 기반으로 하며 이는 정확도를 좋게하나 연산량이 많아지는 단점이 존재한다. Network comparison과 puning[9, 5, 13]은 detection architecture가 더 간단하도록 해주는 방법을 제안햇다[36, 31, 38]. 하지만 이러한 방법들은 아직도 속도의 향상이 필요한데, 예를들어 [31]의 작은 네트워크는 17FPS로밖에 동작하지 않으면서 17M의 파라미터를 갖는다. 본 연구에선 compact archicture 뿐만아니라 고속으로 동작하는 object detector의 design principle을 개발한다. Densenet[14], Yolo-v2[26], SSD[22]의 deep하고 narrow한 네트워크 구조에서 영감을 받았다. 깊은 구조를 사용하면 정확도는 향상되며 narrow한 layer로 인해 네트워크의 복잡도를 조절할 수 있게 된다. Architecture의 변경을 통해 정확도가 5mAP정도 향상 가능하다는것을 확인했다. 이러한 연구들을 기반으로 architecture design의 main contribution은 딥러닝 기반의 객체검출기를 간단하지만 효율적인 네트워크 아키텍쳐를 개발하면서 200 FPS 속도로 동작하게 하도록 한다. 게다가 제안하는 모델은 15M의 파라미터만을 사용하여 VGG16기반의 모델[3]이 138M개의 파라미터를 사용하는것에 비해 매우 작으므로 가장 작은 모델이 된다. 제안하는 모델의 속도는 Figure 1에서 비교 가능하다.





## Conclusion
- 논문에선 효율적이고 빠른 object detector를 제안했다. 객체검출모델의 speed performance의 trade-off를 조절하기위해 네트워크의 구조, loss function, training data의 역할에 대해 연구했다. 네트워크의 설계에는 이전에 수행되었던 연구들을 이용하여 계산복잡도를 적게 유지하기 위해 몇 가지의 간단한 idea들을 확인하고, 이 아이디어들의 방법을 활용하여 light-weight network를 개발했다. 네트워크 학습 과정에서 FM-NMS와 objectness scaled loss와 같이 carefully하게 설계된 components와 더불어 disitillation이 powerful한 idea임을 보였고, 이를 통해 light-weight single stage object detector의 성능이 향상되었다. 마지막으로 distillation loss를 기반으로 unlabeled data의 traning에 대한 연구를 수행했다. 논문의 실험에선 제안하는 design principle이 적용된 모델이 SOTA object detector들보다 훨씬 빠르게 동작하며 동시에 resonable한 성능을 얻을 수 있다는것을 보였다. 
