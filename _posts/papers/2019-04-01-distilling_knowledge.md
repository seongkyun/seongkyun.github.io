---
layout: post
title: Distilling the knowledge in a Neural Network
category: papers
tags: [Deep learning]
comments: true
---

# Distilling the knowledge in a Neural Network

Original paper: https://arxiv.org/pdf/1503.02531.pdf

Authors: Geoffrey Hinton, Oriol Vinyals, Jeff Dean (Google Inc)

- 참고글: https://www.youtube.com/watch?v=tOItokBZSfU
- 참고글: https://jamiekang.github.io/2017/05/21/distilling-the-knowledge-in-a-neural-network/

- 논문의 내용 들어가기에 앞서, 아래와 같이 간단한 개념을 이해하고 시작하는게 도움 됨.
  - Google과 같이 큰 회사에서 ML로 어떤 서비스를 개발 할 때, 개발 단계에 따라 training에 사용되는 모델과 실제 서비스로 deploy 되는 모델에는 차이 존재
  - 즉, training에 사용되는 모델은 대규모 데이터를 갖고 batch 처리를 하며 리소스를 비교적 자유롭게 사용하고, 최적화를 위한 비슷한 여러 변종이 존재 할 것임
  - 반면에 실제 deployment 단계의 모델은 데이터의 실시간 처리가 필요하며 리소스의 제약을 받는 상태에서의 빠른 처리가 중요함
  - 이 두가지 단계의 모델을 구분하는 것이 이 논문에서 중요한데, 그때 그때 다른 이름으로 부르므로 
    - 1번 모델을 "cumbersome model"로 하며 training stage 에서 large ensemble 구조를 갖고 동작하며, 느리고 복잡하나 정확히 동작함.(teacher)
    - 2번 모델을 "small model"로 하며, deployment stage에서 single small 구조를 갖고 동작하며, 빠르고 간편하나 비교적 정확도가 떨어짐.(student)
  - 머신 러닝 알고리즘의 성능을 올리는 아주 쉬운 방법은 많은 모델을 만들어 같은 데이터셋으로 training 한 다음, 그 prediction 결과 값을 average 하는 것임.
  - 이러한 다중 네트워크를 포함하는 모델의 집합을 ensemble(앙상블) 모델이라 하며, 위에서 1번의 "cumbersome model"에 해당함
  - 하지만 실제 서비스에서 사용할 모델은 2번 모델인 "small model"이므로, 어떻게 1번의 training 결과를 2번에게 잘 가르치느냐 하는 문제가 발생함
- 이 논문의 내용을 한 문장으로 말하면, 1번 모델(cumbersome model)이 축적한 지식(dark knowledge)을 2번 모델에 효율적으로 전달하는(기존 연구보다 더 general 한) 방법에 대한 설명임.

- 이 논문은 아래와 같이 크게 두 부분으로 나누어짐
  - Model compression: 1번 ensemble 모델의 지식을 2번 모델로 전달하는 방법
  - Specialist networks: 작은 문제에 특화된 모델들을 training 시켜 ensemble training 시간을 단축하는 방법
- 본 글에서는 distillation을 이용한 model compression에 대해 설명

## Model compression
- Distilling ensemble model to single model을 이용
  - 계산시간이 오래걸리는 앙상블 모델의 정보를 single model로 이전
  - 앙상블 모델의 정보를 single model로 이전하는 방법을 distilling이라고 함
- 일반적인 인공신경망의 경우 파라미터 수가 많아 training data에 대해 overfitting이 쉬움.
  - 이를 해결하기 위해 앙상블 모델을 사용하게 됨

### Expensive ensemble

<center>
<figure>
<img src="/assets/post_img/papers/2019-04-01-distilling_knowledge/fig1.jpg" alt="views">
<figcaption></figcaption>
</figure>
</center>

- 앙상블 모델이란, 위 사진처럼 Input에 대해 여러 모델들이 각각 계산을 한 후, 그 계산을 합쳐서 결과물을 내놓는 방법
  - 쉽게 말해서 데이터셋을 여러개의 training set으로 나누어서 동시에 학습을 진행한 후, 모든 training set에 대한 학습이 끝난 후 그 결과를 통합하는 방식
- Hinton(저자)은 인공신경망 앙상블 모델의 학습과정에서 다른 앙상블처럼 hidden layer의 갯수를 각 구조가 서로 다르게 하기 보다는 단순이 initial parameter를 다르게 설정하는게 효과가 더 좋다고 설명함
- 이러한 앙상블 모델을 이용 할 경우 실험 결과는 최소 2에서 4~5%까지의 정확도 향상을 기대 할 수 있지만, weight parameter의 개수가 매우 많아져 저장공간이 매우 많이 필요하고 계산시간이 매우 오래걸린다는 단점이 존재함.
- GoogleNet의 경우 test set의 inference time조차도 크며, 결국 GPU를 이용한 병렬처리를 한다 해도 앙상블 안의 모델 개수가 core의 개수보다 많은 경우 한 모델에 대한 계산보다 시간이 훨씬 오래 걸리므로 연산에 부담이 됨
- Mobile device 등에서도 저장공간등의 이유로 mobile device에 앙상블 모델을 적용하는것은 불가능함
- 일반적인 딥러닝 모델도 저장공간이 많이 차지해서 mobile device에 적합하지 않음
  - MobileNet 과 같은 특화된 구조 말고, VGG와 같은 모델은 모델의 크기가 500MB가 넘어가게 됨

### Distilling ensemble: Single model

<center>
<figure>
<img src="/assets/post_img/papers/2019-04-01-distilling_knowledge/fig2.jpg" alt="views">
<figcaption></figcaption>
</figure>
</center>

- 위에서 발생하는 문제들을 해결하기 위해 여러 앙상블 모델들의 정보를 전달받은 single model을 만들어야 함
- 이 모델은
  - 앙상블 만큼의 좋은 성능을 보여야 하며
  - 계산시간 및 저장공간을 조금 차지해 computation cost가 적은
- 위의 두 가지 조건을 만족시켜야 하며, 이러한 single shallow model을 얻는 것이 최종 목표임
- 이러한 여러 모델을 포함하는 앙상블 모델을 하나의 single model로 정보들을 이전하는 방식을 "distillation" 이라고 지칭

#### Distillation method 1
- Method from: Buciluǎ, C., Caruana, R., & Niculescu-Mizil, A. (2006, August). Model compression. In Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 535-541). ACM.
- 관측치가 많다면, 구지 앙상블 모델을 쓰지 않아도 일반화 성능이 좋아진다는 점에서 착안

<center>
<figure>
<img src="/assets/post_img/papers/2019-04-01-distilling_knowledge/fig3.jpg" alt="views">
<figcaption></figcaption>
</figure>
</center>

- 앙상블 모델의 경우 학습 데이터셋이 많지 않더라도 일반화 성능이 좋아 실제 test단에서 성능이 좋음
  - Generalization이 잘 된 모델
- 일반적인 single model의 경우, training set에만 적합하도록 overfitting되는 경향이 있음

<center>
<figure>
<img src="/assets/post_img/papers/2019-04-01-distilling_knowledge/fig4.jpg" alt="views">
<figcaption></figcaption>
</figure>
</center>

- 하지만 training data가 많아진다면 overfitting이 일부 발생하더라도 전반적으로 모델의 generalization 성능이 좋아지게 됨.
- 이러한 원리로 접근한 것이 첫 번째 방법.

<center>
<figure>
<img src="/assets/post_img/papers/2019-04-01-distilling_knowledge/fig5.jpg" alt="views">
<figcaption></figcaption>
</figure>
</center>

- 적은 량의 training dataset에 대해 oversampling을 통해 새로운 데이터셋을 생성(사진상의 물음표 처진 training data가 oversampling된 데이터)
  - 이렇게 oversampling 된 데이터는 현재 label이 정의되어있지 않은 상태임
  - 또한 기존의 data에 약간의 noise를 추가해 기존 data와 유사하지만 완전히 같지 않지만 유사한 새로운 unlabeled data를 생성
- 생성된 oversampling data는 label 정보가 없으므로, 이 생성된 data들에 대해 label을 붙이는 과정이 필요함
  - __이 과정에서 기존의 training data로 학습된 앙상블 모델을 사용하여 label 정보를 생성시킴__

<center>
<figure>
<img src="/assets/post_img/papers/2019-04-01-distilling_knowledge/fig6.jpg" alt="views">
<figcaption></figcaption>
</figure>
</center>

- 앙상블 모델을 이용하여 새로 생성된 oversampling data에 레이블 정보를 모두 생성함
- 위 과정을 통해 label정보를 모두 갖고있는 많은 data가 포함된 새로운 training dataset이 정의됨
- 이렇게 생성된 training dataset에는 앙상블 모델의 정보가 담겨있음.
- 새로운 training dataset을 이용하여 다시 single shallow model을 학습시킴
- 이로 인해 앙상블 모델의 정보가 전해진 single shallow network가 만들어짐
  - 좀 더 generalization이 잘 되고, 좀 더 overfitting에 강인한 single shallow model이 학습됨

<center>
<figure>
<img src="/assets/post_img/papers/2019-04-01-distilling_knowledge/fig7.jpg" alt="views">
<figcaption></figcaption>
</figure>
</center>

- 평가지표로 RMSE를 사용
- Training dataset size가 커질수록(x축에서 우측으로 갈수록) single shallow model의 성능이 best ensemble 모델과 비슷해지는것을 확인 가능
  - 빨간 선이 best ensemble model, 파란 선이 single shallow model
  
#### Distillation method 2
- Method from: Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.
- 두 번째 방법으로, training dataset의 양을 늘리지 않고 class의 확률분포 정보를 이용하여 학습을 더 잘 시키는 방법
