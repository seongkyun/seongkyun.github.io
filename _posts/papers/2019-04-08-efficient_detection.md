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
  - Knowledge distillation을 이용한 compact multi-class object detection 모델의 end-to-end 학습 framework를 제안한다(section 3.1). 아는바로는 이게 첫 번째 성공적인 multi-class object detection의 knowledge distillation의 적용이다.
  - 앞에서 말한 challenges를 효과적으로 다루는 새로운 loss를 제안한다. 세세적으로 object class의 반대인 background class를 위한 misclassification의 impact의 imbalance를 계산하는 classification을 위한 _weighted cross entropy loss_ 를 제안한다(section 3.2). 다음으로 knowledge distillation을 위한 _teacher bounded regression loss_ 를 제안하며(section 3.3) 마지막으로 teacher의 intermediate layer의 neuron에서의 distribution을 student가 학습하도록 해주는 _adaptation layers for hint learning_ 을 제안한다(section 3.4).
  - Public benchmark를 이용하여 제안하는 모델을 평가한다. 모든 benchmark에 대해 성능이 모두 향상되었다. 각 제안하는 방법에 대해 모두 성능이 향상되었다(section 4.1 - 4.3).
  - 논문에서 제안하는 framework에 대해 generalization과 under-fitting problem에 관련된 insight를 제안한다.

## 2. Related Works
- __CNNs for Detection.__ Deformable Part Model (DPM) [14] was the dominant detection framework before the widespread use of Convolutional Neural Networks (CNNs). Following the success of CNNs in image classification [27], Girshick et al. proposed RCNN [24] that uses CNN features to replace handcrafted ones. Subsequently, many CNN based object detection methods have been proposed, such as Spatial Pyramid Pooling (SPP) [19], Fast R-CNN [13], Faster-RCNN [32] and R-FCN [29], that unify various steps in object detection into an end-to-end multi-category framework.
- __Model Compression.__ CNNs are expensive in terms of computation and memory. Very deep networks with many convolutional layers are preferred for accuracy, while shallower networks are also widely used where efficiency is important. Model compression in deep networks is a viable approach to speed up runtime while preserving accuracy. Denil et al. [9]은 인공신경망이 때때로 과하게 parameterized 되므로 중복되는 parameter들에 대해 제거가 가능한 것을 증명한다. Subsequently, various methods [5, 7, 10, 15, 17, 30] have been proposed to accelerate the fully connected layer. Several methods based on low-rank decomposition of the convolutional kernel tensor [10, 23, 28] are also proposed to speed up convolutional layers. To compress the whole network, Zhang et al. [41, 42] present an algorithm using asymmetric decomposition and additional fine-tuning. 유사하게 Kim et al.[26]은 one-shot whole network compression을 제안하였으며, 정확도의 큰 하락 없이 1.8배의 속도가 향상되었다. 실험에서는 [26]에서 제안한 방법을 사용했다(_Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications_). Besides, a pruning based approach has been proposed [18] but it is challenging to achieve runtime speed-up with a conventional GPU implementation. Additionally, both weights and input activations can be the quantized( [18]) and binarized ( [21, 31]) to lower the computationally expensive.
- __Knowledge Distillation.__ Knowledge distillation is another approach to retain accuracy with model compression. Bucila et al. [3] propose an algorithm to train a single neural network by mimicking the output of an ensemble of models. Ba and Caruana [2] adopt the idea of [3] to compress deep networks into shallower but wider ones, where the compressed model mimics the ‘logits’. Hinton et al. [20] propose knowledge distillation as a more general case of [3], which applies the prediction of the teacher model as a ‘soft label’, further proposing temperature cross entropy loss instead of L2 loss. Romero et al. [34] introduce a two-stage strategy to train deep networks. In their method, the teacher’s middle layer provides ‘hint’ to guide the training of the student model.
- Other researchers [16, 38] explore distillation for transferring knowledge between different domains, such as high-quality and low-quality images, or RGB and depth images. 본 논문의 연구방향과 비슷하게 Shen et al.[36]에선 distillation과 hint framework에 대한 compact object detection model의 학습의 영향에 대하여 연구를 했다. 하지만 앞의 방법은 pedestrian(보행자) object detection task에 대하여만 진행된 연구이므로, 이는 multi-category object detection에는 general하게 적용하지 못한다. [36]과는 다르게 제안하는 방법은 multi-category object detection에 적용 가능하다. 게다가 [36]은 외부 region proposal 방법을 사용하지만 논문의 방법은 distillation과 hint learning이 최근의 end-to-end object detection framework의 region proposal과 classification 모두에 적용 가능함을 보인다.

## 3. Method
- 본 논문에선 Faster-RCNN[32]를 meth-architecture로 사용했다. Faster-RCNN is composed of three modules: 1) A shared feature extraction through convolutional layers, 2) a region proposal network (RPN) that generates object proposals, and 3) a classification and regression
network (RCN) that returns the detection score as well as a spatial adjustment vector for each object proposal. 각 RCN과 RPN 모두 1)의 출력을 사용하며, RCN은 RPN의 출력을 입력으로 사용한다. 정확한 객체검출을 위해 앞의 세 요소에 대해 표현력 strong한 모델을 학습하는것이 중요하다.

### 3.1 Overall Structure
### 3.2 Knowledge Distillation for Classification with Imbalanced Classes
### 3.3 Knowledge Distillation for Regression with Teacher Bounds
### 3.4 Hint Learning with Feature Adaptation

## 4. Experiment
- In this section, we first introduce teacher and student CNN models and datasets that are used in the experiments. 다양한 데이터셋에 대한 전체 결과는 section 4.1에서 보여진다. Section 4.2에선 제안하는 방법을 더 작은 네트워크와 lower quality input을 이용한 실험결과를 보여준다. Section 4.3에서는 classification/regression, distillation 및 hint learning의 세 가지 component에 대한 albation study를 설명한다. Distillation, hint learning에 대해서 얻어진 insight는 section 4.4에서 다뤄진다. Details에 대해선 supplementary materials에서 설명된다.

### Datasets
- We evaluate our method on several commonly used public detection datasets, namely, KITTI [12], PASCAL VOC 2007 [11], MS COCO [6] and ImageNet DET benchmark (ILSVRC 2014) [35]. Among them, KITTI and PASCAL are relatively small datasets that contain less object categories and labeled images, whereas MS COCO and ILSVRC 2014 are large scale datasets. KITTI와 ILSVRC 2014 데이터셋은 test set의 ground-truth annotation을 제공하지 않으므로 [39]와 [24]의 tranining/validation을 사용했다. For all the datasets, we follow the PASCAL VOC convention to evaluate various models by reporting mean average precision (mAP) at IoU = 0.5 . For MS COCO dataset, besides the PASCAL VOC metric, we also report its own metric, which evaluates mAP averaged for IoU 2 [0.5 : 0.05 : 0.95] (denoted as mAP[.5, .95]).

### Models
- Models The teacher and student models defined in our experiments are standard CNN architectures, which consist of regular convolutional layers, fully connected layers, ReLU, dropout layers and softmax layers. We choose several popular CNN architectures as our teacher/student models, namely, AlexNet [27], AlexNet with Tucker Decomposition [26], VGG16 [37] and VGGM [4]. We use two different settings for the student and teacher pairs. In the first set of experiments, we use a smaller network (that is, less parameters) as the student and use a larger one for the teacher (for example, AlexNet as student and VGG16 as teacher). In the second set of experiments, we use smaller input image size for the student model and larger input image size for the teacher, while keeping the network architecture the same.


## Conclusion
- 논문에선 knowledge distillation을 이용한 compact하고 빠른 CNN based object detector의 학습 framework를 제안했다. 매우 복잡한 teacher detector를 guide로 하여 효율적인 student model을 학습시켰다. Knowledge distillation, hint framework와 제안하는 loss function을 이용하였을 때 다양한 실험 setup에 대하여 모두 성능이 향상되었다. 특히 제안하는 framework로 학습된 compact model은 PASCAL VOC 데이터셋에 대한 teacher model의 정확도와 매우 비슷한 수준으로 훨씬 빠르게 동작하였다. 논문의 실험을 통해 object detector의 학습에 under-fitting 문제가 있음을 확인했으며, 이는 해당 연구분야에서 더 연구 가능한 insight를 준다.

<center>
<figure>
<img src="/assets/post_img/papers/2019-04-08-efficient_detection/fig1.jpg" alt="views">
<figcaption>Figure</figcaption>
</figure>
</center>
