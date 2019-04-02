---
layout: post
title: A Gift from Knowledge Distillation- Fast Optimization, Network Monimization and Tranfer Learning
category: papers
tags: [Deep learning]
comments: true
---

# A Gift from Knowledge Distillation: Fast Optimization, Network Monimization and Tranfer Learning

Original paper: http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf

Authors: Junho Yim1 Donggyu Joo1, Jihoon Bae, Junmo Kim (KAIST)

## Abstract
- 본 논문에서는 knowledge transfer를 하는 새로운 방법을 제안한다. 
  - Knowledge transfer는 pre-trained DNN(deep neural network)의 정보(knowledge)를 distillation 하여 다른 DNN에 전달하는것을 의미한다.
- 네트워크에 순차적으로 쌓인 layer들의 input space부터 output space까지 DNN map들에 대해, 저자는 layer 사이의 흐름(flow)의 관점에서 distilled knowledge가 전달(transfer)되도록 정의한다. 이는 두 layer간의 feature에 대해 inner product를 계산하여 수행된다.
- 논문에서 동일한 size의 teacher network가 없이 학습된 DNN과 본 논문에서 제안하는 방법을 적용시킨 student DNN을 비교한다. 논문에서 제안하는 방법인 두 layer 간의 flow를 이용하여 distilled knowledge을 student DNN으로 전달 할 때, 제안하는 방법이 적용된 모델은 그렇지 않은 모델에 비해 세 가지 중요한 phenomena에 대한 차이를 보인다.
  - (1): Student DNN에 대해 distilled knowledge가 전달되는 경우가 그렇지 않은 모델보다 훨씬 더 빨리 optimize된다.
  - (2): 논문에서 제안하는 방법이 적용된 student DNN이 일반 DNN보다 성능이 더 우수하다.
  - (3): Student DNN은 다른 task에 대해 training 된 teacher DNN으로부터 distillation된 정보(knowledge)를 학습 할 수 있으며, 이렇게 학습 된 student DNN은 처음부터(from scratch) 학습 된 DNN 모델에 비해 우수한 성능을 보인다.

## 1. Introduction
- 근 몇년동안 DNN이 제안되었으며, computer vision[8, 23]이나 NLP[1, 19]등의 다양한 task에 대해 SOTA 성능을 보인다.최근 knowledge transfer 기술에 대한 몇몇 연구들[11, 20]이 수행되었다. Hinton의 방법[11]은 처음으로 knowledge distillation(KD)에 대한 개념을 제안하였으며, 논문에서는 teacher-student framework에서 soften된 teacher output을 이용하는 방법을 제안했다. 비록 KD training이 몇몇 dataset에 대해서 정확도의 향상 달성했지만, very deep network의 optimizing이 어렵다는 문제점이 존재했다. 이러한 deep network에 대한 KD training의 optimizing 문제를 해결하기 위해 Romero의 방법[20]은 pretrained teacher의 hint layer와 student의 guided layer를 이용하는 hint-based training 접근방법을 제안했다. Hint-based training 방법 덕분에 학습된 deep student network는 original wide teacher network에 비해 더 적은 parameter 개수로 기존(Romero 방법이 적용되지 않은 모델)에 비해 더 나은 정확도를 보였다. 
- Knowledge transfer의 성능은 어떻게 distilled knowledge가 정의되느냐에 따라 매우 민감하게 바뀐다. Distilled knowledge는 pretrained DNN의 다양한 feature들에 의해 추출되어진다. 실제 teacher가 student를 어떻게 문제를 해결하는지를 가르친다는 점을 고려할 때, 논문에서는 high-level distilled knowledge를 문제 해결의 흐름(flow)으로써 정의한다. DNN은 구조상 input space부터 output space까지 많은 layer를 sequential하게 사용하므로, 문제 해결의 흐름(the flow of solving problem)은 곧 두 layer의 feature간의 관계(relationship)로써 정의되어질 수 있다.
- Gatys의 방법[6]은 Gramian matrix를 input image의 texture information을 표현하기 위해 사용하였다. Gramian matrix는 feature vector들 간의 inner product를 계산하여 생성되므로 texture information로 생각 할 수 있는 feature간의 방향성(directionality)을 포함 할 수 있다. Gatys의 방법과 유사하게, 저자들은 두 개의 layer의 feature 사이의 inner product로 구성된 Gramian matrix를 사용하여 flow of solving problem을 나타냈다. [6]에서 사용한 Gramian matrix와 논문의 방법 사이의 주요 차이점으로는, 논문에서 제안하는 방법은 Gramian matrix를 layer들을 가로질러 계산하며, 이는 [6]의 Gramian matrix가 한 layer 안의 feature들 사이에서만 inner product를 계산하는것과는 대조적이다. Figure 1에서는 논문에서 제안하는 distilled knowledge를 전달하는 방법의 concept diagram을 보여준다. 두 layer들 사이에서 추출된 feature map들은 flow of solution procedure(FSP) matrix를 생성하기 위해 사용되어진다. 학습 과정에서 student DNN는 student DNN에서 계산되어지는 FSP matrix가 teacher DNN의 FSP matrix와 유사하도록 학습되어지게된다.
  - 즉 기존의 방법([6])은 하나의 layer의 feature들에 대해서만 Gramian matrix를 계산하였다면, 본 논문에서 제안하는 방법은 전체 layer들 중 입력단과 출력단의 두 layer에서 각 feature들에 대한 Gramian matrix를 계산하고 이를 FSP matrix라고 정의하고, student DNN이 teacher DNN의 FSP matrix를 닮도록 네트워크가 학습되어진다. Figure 1을 보면 좀 더 직관적인 이해가 쉽다.

<center>
<figure>
<img src="/assets/post_img/papers/2019-04-03-a_gift_from_distillation/fig1.jpg" alt="views">
<figcaption>Figure 1. 제안하는 transfer learning의 concept diagram. Teacher DNN의 distilled knowledge를 표현하는 FSP matrix는 두 layer들의 feature들을 사용하여 생성된다. FSP matrix를 만들기 위해 방향성을 나타내는 inner product를 계산함에 있어서 두 layer 사이의 flow는 FSP matrix로 표현되어질 수 있다.</figcaption>
</figure>
</center>

- 이 논문에선 논문에서 제안하는 distilled knowledge의 유용성을 3가지 task를 이용하여 검증하였다.
- 첫 번째로, 빠른 optimization이다. Flow of solving problem을 이해하는 DNN은 기본 main task를 해결하는데에 좋은 initial weight가 될 수 있으며, 이로인해 일반적인 DNN모델보다 빠른 속도로 학습이 가능해진다. 빠른 optimization은 매우 유용한 기술이다. 다양한 논문에서 advanced learning rate scheduling 기법을 사용하는 것 뿐만 아니라 fast optimizing을 적용에 대해 연구하였고[13, 27, 4], 뿐만 아니라 좋은 initial weight를 찾는것에 대한 다양한 연구도 진행되었다[5, 9, 18, 20]. 논문의 방식은 initial weight method를 기반으로 하므로 다른 initial weight method와의 비교를 수행했다. 저자들은 training iteration의 횟수와 논문의 scheme에 대한 성능을 다양한 다른 기술들과 비교하였다.
- 두 번째 task는 적은 parameter 수를 갖는 shallow network(small network)의 성능을 향상시키는 것이다. Small network가 teacher network에서 온 distilled knowledge를 학습하므로 student network(small network)가 단독으로 학습하는 것 보다 teacher network에서 나온 distilled knowledge를 이용하여 학습하는것이 성능향상에 더 도움이 된다. 저자들은 original network와 다양한 knowledge transfer 기술이 적용된 network들에 대한 성능을 비교했다.
- 세 번째 task는 transfer learning이다. 비록 어떠한 new task가 small dataset만 사용 가능할지라도 deep하고 heavy하며 huge dataset에 대해 pretrained된 DNN을 이용하여 transfer learning을 적용한다면 small dataset만으로도 좋은 성능을 이루어 낼 수 있을 것이다. 제안하는 방법은 distilled knowledge를 작은 DNN으로 transfer할 수 있다는 이점이 있으므로 작은 네트워크는 일반적인 transfer learning에서 사용되는 large DNN과 유사한 정확도로 동작 할 수 있게 된다.
- 본 논문에서는 아래의 contribution을 만든다.
  1. Knowledge distillation을 하는 새로운 기술을 제안
  2. 이 방법은 fast optimization에 유용함
  3. 제안하는 distilled knowledge를 initial weight를 찾기 위해 사용한다면 small network의 성능을 향상 시킬 수 있음
  4. 만약 student DNN이 teacher DNN과 다른 task에 대해 학습되었더라도 제안하는 distilled knowledge는 student DNN의 성능을 향상 시킬 수 있음

## 2. Related Work
### Knowledge Transfer
- 보통 computer vision task에서는 많은 파라미터를 갖는 deep network의 성능이 좋다. 대부분 architecture의 깊이는 성능 향상을 위해 깊어진다. 딥러닝이 시초인 AlexNet[16]은 5개의 conv 레이어뿐이었지만, 근래의 GoogleNet[23]같은 경우 22개의 conv 레이어나 ResNet[8]은 152개의 conv 레이어를 갖는다.
- 많은 파라미터를 갖는 deep network는 training이나 testing에서 무거운 연산량을 필요로 한다. 이러한 이유로 deep network들은 모바일과 같은 일반적인 computing platform에 적용이 불가능하다. 그러므로 많은 연구들이 network의 성능은 유지하면서 크기는 작게 만들려고 시도되었다. 이러한 것을 가능하게 하는 일반적인 방법은 학습된 deep network의 정보를 small network로 distilled 정보를 transfer 하는 것이다. 최근에 Hinton의 방법[11]은 dark knowledge에 기반한 model compression 방법을 설명했다. 이 방법은 small student network를 학습시키기 위해 teacher network의 soften된 최종 output 정보를 사용한다. 이러한 teaching 과정에서 small network는 어떻게 large network가 주어진 task에 대해 잘 학습했는지 압축된 형식으로 정보를 전달받게 된다. Romero 방식[20]은 final output도 사용하면서 동시에 teacher network의 중간 hidden layer의 값을 student network의 학습에 사용하였으며, 동시에 이러한 중간 layer 정보를 사용하는것이 deep하고 thin한 student network의 성능을 향상 시킬 수 있었다. Net2Net[3]도 teacher network의 parameter에 따라 student network의 parameter를 초기화(initialize)하기 위해 function-preserving transform을 적용시킨 teacher-student network system을 사용하였다.
### Fast Optimization
- Deep CNN은 좋은 local optima나 global optimum을 찾기 위해 비교적 시간이 많이 소요된다. 보통은 MNIST[17]나 CIFAR10[15]와 같이 작은 데이터셋은 학습시키기 쉽다. 하지만 ILSVRC[21] 데이터셋과 같은 거대한 데이터셋의 경우 big network의 경우 학습에 몇 주가 소요되기도 한다. 따라서 fast optimization은 최근 연구에서의 중요한 분야중 하나가 되었다. 주로 good initial weight를 찾거나 SGD 방법 외에 다른 기술을 사용하여 optimal point에 도달하는 몇몇 접근방식이 존재한다.
- 초기엔 unit variance와 zero mean을 갖는 Gaussian noise initialization이 매우 많이 사용되었다. 또는 Zavier initialization[7] 등도 광범위하게 사용되었다. 하지만 이러한 간단한 initialization 방법들은 deep network를 학습시킬 때 poor한 성능을 보인다. 이로인해 [18, 22, 14]와 같은 몇몇 새로운 방법들이 수학적인 접근방식으로 이를 해결코자 했다. 좋은 initialization으로 인해 training이 적절한 starting point에서 시작 될 때 parameter들은 빠르게 global optimum에 수렴 할 수 있게 된다.
- Optimization 알고리즘들 또한 딥러닝의 발전과 함께 많이 진화되어왔다. 관습적으로 SGD 알고리즘이 기본적으로 많이 사용되어왔다. 하지만 SGD는 다양한 saddle point에서 탈출하기 힘들다는 단점이 존재한다. 이러한 문제로 인해 [13, 27, 4]와 같은 몇몇 알고리즘들이 제안되었다. 이러한 알고리즘들은 saddle point에서 벗허나도록 도와주며 global optimum에 빠르게 도착하도록 도와준다.
### Transfer Learning
- Transfer learning은 어떠한 task에 대해 미리 학습된 network의 파라미터를 이용하여 새로운 task에 적용가능하도록 해 주는 간단한 기술이다. 전형적으로 feature extraction을 하는 입력단의 레이어들은 pre-trained network로부터 파라미터 변경이 되지 않는 frozen형태나 fine-tuned 형태로 복사되어지며, 반면에 상단의 classifier들(fc layer들)은 새로운 task를 위해 random하게 initialize되어 slow learning rate로 학습되어진다. Fine-tuning은 때때로 처음부터 학습시키는 경우보다 성능이 좋을 수 있으며, 이는  이미 pretrained model이 정보들을 다루는데에 대한 능력이 좋기 때문이다. 예를 들어 [19, 28, 1, 2]과 같은 논문들에서는 ILSVRC데이터셋으로 pretrained된 model을 이용해 VQA[1]나 CUV200[25]와 같은 task에 대해 fine-tuninning을 적용하여 성능을 향상 시켰다. Detection이나 segmentation과 같은 많은 다른 task들에 대해서도 이러한 ImageNet pre-trained model을 initial value로 하여 사용되어지며, 이는 ILSVRC 데이터셋이 generalization에 대해 매우 도움이 되기 때문이다. 논문에서 제안하는 approach 또한 이러한 fine-tunning 기술을 제안하는 good initialization method에 적용하였다.

## 3. Method
### 3.1. Proposed Distilled Knowledge
### 3.2. Mathematical Expression of the Distilled Knowledge
### 3.3. Loss for the FSP Matrix
### 3.4. Learning Procedure

## 4. Experiment
### 4.1. Fast optimization
#### 4.1.1 CIFAR-10
#### 4.1.2 CIFAR-100
### 4.2. Performance improvement for the small DNN
#### 4.2.1 CIFAR-10
#### 4.2.2 CIFAR-100
### Transfer Learning

## Conclusion
- 본 논문에서는 DNN으로부터 distilled knowledge를 생성하는 새로운 접근방식을 제안했다. Distilled knowledge를 논문에서 제안하는 FSP matrix로 계산 된 solving procedure의 흐름(flow)으로 결정함으로써 제안하는 방법의 성능이 여타 SOTA knowledge transfer method의 성능을 능가하였다. 논문에서는 3가지 중요한 측면에서 제안하는 방법의 효율성을 검증하였다. 제안하는 방법은 DNN을 더 빠르게 optimize시키며(빠른 학습), 더 높은 level의 성능을 만들어 내게 한다. 게다가 제안하는 방법은 transfer learning task에도 적용 가능하다.

<center>
<figure>
<img src="/assets/post_img/papers/2019-04-03-a_gift_from_distillation/fig1.jpg" alt="views">
<figcaption></figcaption>
</figure>
</center>
