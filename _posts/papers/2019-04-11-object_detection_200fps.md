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

## Conclusion
- 논문에선 효율적이고 빠른 object detector를 제안했다. 객체검출모델의 speed performance의 trade-off를 조절하기위해 네트워크의 구조, loss function, training data의 역할에 대해 연구했다. 네트워크의 설계에는 이전에 수행되었던 연구들을 이용하여 계산복잡도를 적게 유지하기 위해 몇 가지의 간단한 idea들을 확인하고, 이 아이디어들의 방법을 활용하여 light-weight network를 개발했다. 네트워크 학습 과정에서 FM-NMS와 objectness scaled loss와 같이 carefully하게 설계된 components와 더불어 disitillation이 powerful한 idea임을 보였고, 이를 통해 light-weight single stage object detector의 성능이 향상되었다. 마지막으로 distillation loss를 기반으로 unlabeled data의 traning에 대한 연구를 수행했다. 논문의 실험에선 제안하는 design principle이 적용된 모델이 SOTA object detector들보다 훨씬 빠르게 동작하며 동시에 resonable한 성능을 얻을 수 있다는것을 보였다. 
