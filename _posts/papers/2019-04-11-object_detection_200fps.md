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

<center>
<figure>
<img src="/assets/post_img/papers/2019-04-11-object_detection_200fps/fig1.jpg" alt="views">
<figcaption>Figure 1. 제안하는 detector와 다른 방식들에 대한 속도와 성능 비교. SSD와 Yolo-v2보다 정확하다.(SSD보다??)</figcaption>
</figure>
</center>

- 간단하고 빠른 architecture만을 사용해야 하기에 성능 향상을 위한 효율적인 학습 방법에 대해 조사했다. 어느정도 정확한 light-weight detector에 대해 더 정확한 모델을 이용하여 training 방법을 더 잘 하게 만든다. 이러한 목적으로 [12, 2, 1]의 network distillation 방법을 고려하였으며, 이는 큰 네트워크의 정보(knowledge)가 효율적으로 더 작은 네트워크의 표현력이 좋아지도록 사용되는 것이다. 비록 이러한 아이디어가 object detection에 최근 사용되었지만[3, 20], 본 논문에선 주요 contribution을 distillation의 적용의 측면에서 중요하게 적용되었다. (1) 본 논문의 방법은 첫 번째 single pass detector(Yolo)에 대한 적용이며, 이전의 연구들이 RCNN계열을 사용한것과 차이점을 갖는다. (2) 본 논문의 접근방식에서 가장 중요한 것은 object detection이 end-to-end 학습과정 외부의 non-maximal suppression(NMS)를 포함한다는 observation(관찰)에 기반한다는 점이다. NMS step 이전에 detection network의 마지막 layer는 detection된 region의 dense activation으로 구성되어지며 만약 student network에 직접 teacher의 이러한 정보(dense activation)가 전달되게 된다면 overfitting으로 이어져 성능이 떨어지게 될 것이다. (3) Teacher detection에서 더 질좋은 탐지를 강조하여 problem을 objectness scaled distillation loss로 공식화하였다. 논문의 실험결과들은 distillation이 연산복잡도는 낮게하면서 성능은 향상시키는 효율적인 접근방법임을 보여준다.
- 마지막으로 object detection의 관점에서 "the effectiveness of data"[8]에 대해 연구했다. 레이블링 된 데이터들은 그 양이 제한적이나, 매우 정확한 object detector와 엄청나게 많은 unlabeled data를 적용하게 될 경우 제안하는 light-weight detector의 성능이 얼마나 향상될지에 대해 연구했다. 제안하는 아이디어는 object detector 분야에 대해서만 연구되지 않은 semi-supervised learning 방법들[29, 35, 4]를 따른다. 최근에 object detector 앙상블을 사용하여 annotation들을 생성되도록 하는 [23]의 방식은 본 연구와 매우 관련성이 높다. 제안하는 방법은 [23]과 다음의 차이점들을 보인다. (1) Network distillation에서 더 효율적이게 teacher network에서의 convolutional feature map에서 나온 soft label을 전달한다[28]. (2) Objectness scaling과 distillation weight을 이용하는 본 논문의 loss공식은 teacher label의 weight given을 조절 할 수 있게 한다. 이러한 공식을 통해 ground truth의 학습에서의 중요성을 더 높게하면서도 상대적으로 정확도가 낮게 예측한 teacher의 결과를 flexible하게 이용 할 수 있게 된다. 게다가 제안하는 traning loss formulation은 detection loss와 distillation loss를 균일하게 이어주며, 이로인해 network는 labeled data와 unlabeled data의 정보를 모두 학습 할 수 있게 된다. 논문에선 이 논문이 labeled/unlabeled data를 jointly하게 이용하여 네트워크를 학습시키는 첫 번째 deep learning object detector 모델이라고 한다.

## 2. Architecture customizations
- 대부분의 성능좋은 객체탐지모델은 좋은성능을 위해 깊은 모델을 쓴다. 하지만 가장 빠른모델이라해도 속도가 20~60FPS로 제한된다[22, 26]. 더 빠른 객체탐지모델을 개발하기 위해 Tiny-Yolo[26]과 같은 성능은 중간정도지만 제일 빠른 모델을 baseline으로 선택했다. 이는 더 적은 convolutional layer를 가지는 Yolo-v2 모델이며, Yolo-v2와 동일한 loss function, optimization strategies(batch normalization[16]), dimenstion cluster, anchor box등을 갖는다. 이 모델을 기반으로 모델이 더 정확하고 빨라지도록 적용된 몇 가지 구조적 커스터마이징에 대해 설명한다.

### Dense feature map with stacking
- [(Densenet)14, (SSD)22]와 같은 최근의 연구에서 영감받아 이전 레이어의 feature map을 합침으로써 성능 향상이 되는 것을 관찰했다. 논문에선 주요 convolution layer의 마지막부분에서 몇몇의 이전 layer에서의 feature map을 합쳤다. 이전 레이어에서 온 feature map의 차원이 뒤쪽것과 다르다. [22]와 같은 논문에선 max pooling을 통해 resize 후 concat 했다. 하지만 max pooling은 곧 정보의 손실이므로 논문에선 더 큰 feature map을 크기를 조절해서 activation이 다른 feature map에 분산되도록 feature map을 stacking했다[26].
- 게다가 feature를 merging하면서 넓은 범위에서 bottleneck layer를 적용하였다. Bottleneck layer를 사용하는 이유는 정보를 더 적은 layer에 넣을 수 있기 때문이다[14, 34]. Bottleneck의 1x1 convolution layer는 depth를 추가시키는 동시에 compression ratio를 유연하게 표현하도록 해준다. Advanced layer(뒤쪽 레이어)들의 feature map들을 합치게 되면 성능이 향상되므로 initial layer들에 대해선 높은 compression ratio를, advanced layer들에 대해선 낮은 ratio를 사용했다.

#### Deep but narrow
- Baseline인 Tiny-Yolo의 구조는 2014라는 큰 숫자의 feature channel을 마지막 몇 convolution layer에서 사용한다. 이전 layer에서의 feature map concat을 사용할 때는 이렇게 많은 feature map이 필요 없다. 따라서 filter의 숫자를 줄였으며, 이로인해 속도가 향상되었다.
- SOTA 객체탐지기와 비교할 때 제안하는 구조는 depth가 좀 모자르다. 몇 conv layer를 쌓으면서 depth를 늘릴 순 있지만 이는 곧 연산량의 증가로 이어지므로 본 논문에선 연산량 제한을 위해 1x1 conv layer를 사용하였다. 마지막 major conv layer 이후 1x1 conv layer를 추가하여 network의 depth는 늘리면서도 연산복잡도는 증가시키지 않을 수 있었다.

### Overall architecture

<center>
<figure>
<img src="/assets/post_img/papers/2019-04-11-object_detection_200fps/fig2.jpg" alt="views">
<figcaption>Figure 2. 제안하는 detector의 기본 architecture. 구조를 간단하게 하기 위해 네트워크의 depth는 제한하면서도 feature map의 수는 낮게 유지하고 작은 kernel size를 사용했다.(1x1이나 3x3)</figcaption>
</figure>
</center>

- 앞의 간단한 concept들을 묶어서 light-weight 객체탐지기를 제안한다. 이러한 modification들로 인해 baseline인 Tiny-Yolo에 비해 성능이 5 mAP정도 향상되었다. 게다가 속도도 20%가 더 빨라졌다. 이는 더 적은 convolutional filter 갯수덕분이라고 판단된다. Baseline인 Tiny-Yolo의 PASCAL dataset 성능은 54.2 mAP이지만, 제안하는 구조는 59.4 mAP를 달성했다. 전체적인 구조는 Figure 2에서 보여준다. 그리고 이 구조를 F-Yolo라고 명명한다.

## 3. Distillation loss for Training
- 논문에선 고속 동작을 위해 네트워크의 구조를 간단하게 했다. 그리고 network distillation[28, 12]방법의 적용을 통해 연산량은 늘리지 않으면서도 정확도의 향상을 꾀했다. Network distillation의 적용은 light-weight network를 student로 하여 large accurate network(teacher)의 정보를 이용하여 student 모델을 학습시키는것이다. 정보의 전달은 teacher network의 추론결과를 soft label 식으로 넘겨주게 된다.
- Distillation 접근법을 묘사하기 전에 Yolo loss function이 어떻게 구성되었고 last convolution layer가 어떤지 간단하게 짚고 넘어간다. Yolo is based on one stage architecture, therefore, unlike RCNN family of detectors, both the bounding box coordinates and the classification probabilities are predicted simultaneously as the output of the last layer. Each cell location in the last layer feature map predicts N bounding boxes, where N is the number of anchor boxes. Therefore, the number of feature maps in the last layers are set to N × (K + 5), where K is the number of classes for prediction of class probabilities and 5 corresponds to bounding box coordinates and objectness values (4+1). Thus, for each anchor box and in each cell, network learns to predict: class probabilities, objectness values and bounding box coordinates. 전체적인 objective는 다음의 3개 파트로 구분되며, 각각 regression loss, objectness loss, 그리고 classification loss이다. 이를 수식적으로 표현하면 아래와 같다.

$$L_{Yolo}=f_{obj}(o_{i}^{gt},\hat{o_{i}})+f_{cl}(p_{i}^{gt},\hat{p_{i}})+f_{bb}(b_{i}^{gt},\hat{b_{i}})\qquad\mbox{(1)}$$

- 각각 $\hat{o_{i}}, \hat{p_{i}}, \hat{b_{i}}$는 student의 objectnessm class probability, boundinb box coordinates이며 $o_{i}^{gt}, p_{i}^{gt}, b_{i}^{gt}$는 ground truth value들이다. The objectness is defined as IOU between prediction and ground truth, class probabilities are the conditional probability of a class given there is an object, the box coordinates are predicted relative to the image size and loss functions are simple $L_{1}$ or $L_{2}$ functions see [25, 26] for details.
- Distillation을 적용하기 위해 teacher network의 마지막 레이어의 출력을 가져와서 ground truth $o_{i}^{gt}, p_{i}^{gt}, b_{i}^{gt}$값을 대체하면 된다. Loss는 teacher network의 activation을 student network로 전파하게 된다. 하지만 single stage detector의 dense sampling으로 인해 이러한 간단한 방식의 application은 distillation을 효율적이지 못하게 만든다. 이에 대한 해결 방안은 아래에서 논해질것이며 이를통해 간단히 single stage detector에 distillation을 적용할 수 있게된다.

### 3.1 

## Conclusion
- 논문에선 효율적이고 빠른 object detector를 제안했다. 객체검출모델의 speed performance의 trade-off를 조절하기위해 네트워크의 구조, loss function, training data의 역할에 대해 연구했다. 네트워크의 설계에는 이전에 수행되었던 연구들을 이용하여 계산복잡도를 적게 유지하기 위해 몇 가지의 간단한 idea들을 확인하고, 이 아이디어들의 방법을 활용하여 light-weight network를 개발했다. 네트워크 학습 과정에서 FM-NMS와 objectness scaled loss와 같이 carefully하게 설계된 components와 더불어 disitillation이 powerful한 idea임을 보였고, 이를 통해 light-weight single stage object detector의 성능이 향상되었다. 마지막으로 distillation loss를 기반으로 unlabeled data의 traning에 대한 연구를 수행했다. 논문의 실험에선 제안하는 design principle이 적용된 모델이 SOTA object detector들보다 훨씬 빠르게 동작하며 동시에 resonable한 성능을 얻을 수 있다는것을 보였다. 
