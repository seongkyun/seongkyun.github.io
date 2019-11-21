---
layout: post
title: CNN이 동작하지 않는 이유
category: study
tags: [CNN, Convolution]
comments: true
---

# CNN이 동작하지 않는 이유
- 참고
  - https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607
  - http://samediff.kr/wiki/index.php/Neural_net%EC%9D%B4_working%ED%95%98%EC%A7%80_%EC%95%8A%EB%8A%94_37%EA%B0%80%EC%A7%80_%EC%9D%B4%EC%9C%A0

## 먼저 시도 해 볼 것들
1. 해당 task에서 잘 동작한다고 알려진 모델들을 가져다가 시도
  - Multi-box loss, CE loss 등 잘 동작한다고 알려진 평이한 loss를 이용해 첫 학습을 시도하는편이 좋음
2. Regularization(Generalization), Normalization 등의 일반화/정규화 방법들을 모두 off
3. Model을 fine-tuning하는 중이라면 preprocessing을 다시한번 확인
  - 원래 모델에 주어진 preprocessing들을 그대로 적용시켜야 함
4. Input data가 제대로 들어가는지를 확인
5. 샘플링 된 작은 크기의 데이터셋으로 우선 학습을 시작해보기
  - 해당 샘플 데이터셋에 모델을 overfitting 시킨다음 조금씩 데이터셋 크기를 키워가며 학습
6. 마지막으로 위에서 제거한 사항들을 하나씩 더해가며 학습
  - Regularization, Normalization, Custom loss, Complex model 등등
7. 그래도 제대로 학습되지 않는다면 아래의 사항들을 확인

## 1. Dataset Issues
- 어이없는 실수들을 확인해보기
  - Dimension을 뒤바뀌어 있거나
  - 모두 0으로만 만들어진 vector가 네트워크로 들어가고 있거나
  - 같은 data batch만 반복해서 네트워크로 들어가고 있는 등
  - 하나하나 값들을 출력해보며 확인하기
- 랜덤 데이터를 넣어보고 loss의 변화를 살펴보기
  - 만약 계속해서 비슷하다면, network의 중간 어디에선가 값들이 쓰레기 값들로 변하고 있다는 것을 의미함
    - Vector의 값이 모두  0이 된다던지 등등
- 전체 데이터 중 몇 개만 네트워크에 입력시켰을 때, 네트워크가 추론하는 label과 동일 데이터를 shuffle하고 입력시켰을 때 네트워크가 추론하는 label과 동일한지를 점검해보기
- 네트워크가 올바른 문제를 풀고 있는지를 점검해보기
  - 주식 데이터같은것은 원래 랜덤 데이터로 패턴이 존재할 수 없음
  - 즉, 주어진 task, 문제 자체가 올바른 것인지를 확인해야 함
    - Classifier에게 시계열 데이터를 학습시켜 미래 값을 추론하게 시키는 등..
- 데이터 자체가 더러운(noisy) 데이터인지를 확인하기
  - Noise가 너무 많다거나 mis-label이 너무 많다거나 하는 문제들
  - 이러한 문제들은 데이터셋을 일일히 분석해 확인하는 방법밖엔 없음
- 데이터 셋에 shuffle을 꼭 적용하기
  - Ordered data가 들어가게 될 경우 학습이 잘 안됨!
- Class imbalance 확인하기
  - https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
- 트레이닝 셋은 충분한지
  - Fine-tuning 말고 네트워크를 from scratch로 학습시키기 위해선 많은 양의 데이터가 필요함
  - Classification 기준 class당 최소 1000개의 영상은 있어야 한다고 함
- Batch안에 최대한 많은(다양한) label이 들어가도록 하기
  - 이는 shuffle 시켜서 해결 가능
- Batch size 줄이기
  - Batch size가 너무 크면 generalization 능력을 떨어트리게 됨
  - https://seongkyun.github.io/papers/2019/01/04/Dont_decrease_the_lr/
  - https://arxiv.org/abs/1609.04836

## 2. Data Regularization/Normalization
- 데이터 셋 정규화 시킬것(Normalization)
  - 데이터 셋을 training, validation, test set으로 나누고
  - 나뉘어진 training set에 대해 평균, 분산을 구해 data normalization을 수행
- 일반화 방법인 data augmentation을 너무 많이 시키게 되면 under-fitting이 발생하게 됨
- Pre-trained model을 사용할 때는 항상 입력되는 값을 신경써야 함
  - 학습된 네트워크에 입력되는 데이터가 어느 값 분포를 갖도록 정규화 되었는지 신경써야 함
    - 0~1 사이의 분포인지, -1~1 사이의 분포인지, 0~255 사이의 분포인지..

## 3. Implementation Issues
- 좀 더 간단한 문제를 풀어보기
  - Object detection을 하고 싶다면, 일단 객체의 위치만 찾거나 classification만 하도록 학습시켜보기
- 우연히 찍어서 정답이 맞을 확률을 확인해보기
  - 예를 들어 10개의 클래스를 맞추는 문제에서 우연히 답을 맞추는 negative log loss는 -ln(0.1)=2.302가 됨
- Loss function을 custom하게 만들어 적용한다면, 해당 loss가 잘 동작하는지 일일이 확인할 필요가 있음
  - 라이브러리가 제공하는 loss를 사용한다면 해당 loss function이 어떤 형식의 input을 받는지를 명확히 해야 함
    - Pytorch의 경우, NLL Loss와 CrossEntropy Loss는 다른 입력을 받음
      - CE loss의 경우 one-hot 입력을 사용함
  - Total loss가 여러 작은 loss function term들의 합이라면, 각 term의 scale factor를 조절해야 할 수 있음
  - Loss 외에도 accuracy를 사용해야 할 수 있음
    - Metric을 loss로 잡는것이 정확한지, accuracy로 잡는것이 정확할지를 판단해야 함
- Network architecture를 직접 만들었다면, 하나하나 제대로 동작하는지 확실히해야 함
  - 학습중 weight parameter update가 수행되지 않는 frozen layer가 있는지 확인하기
  - 네트워크의 표현력(expressiveness, expressive power)이 부족 할 수 있음
    - 이는 네트워크의 효율화나 크기를 키워 극복 가능
  - Input이 (k, H, W)=(64, 64, 64)로 모두 같은 dimension 크기를 갖는다면 학습이 잘 되고 있는지 중간에서 확인하기 어려움
    - Prime number로 vector(feature map)를 생성하게 조절해서 잘 동작하는지 확인해보기
- Optimizer를 직접 만들었다면(gradient descent algorithm), gradient가 떨어지도록 잘 동작하는지를 확인해보기
  - http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/
  - http://cs231n.github.io/neural-networks-3/#gradcheck
  - https://www.coursera.org/lecture/machine-learning/gradient-checking-Y3s6r
  
## 4. Training Issues
- 네트워크에 샘플링된 매우 작은 크기의 데이터를 입력시켜 학습시킨다음 잘 동작하는지를 확인하기
- Network weight parameter의 초기화가 중요함
  - 초기화에 따라 네트워크가 잘 학습되지 않을 수 있음
  - (Xavier)[http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf]나 (He)[https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf]의 방법을 사용
  - 하지만 어떤 구조라도 최고의 parameter initialization 방법은 pre-trained model의 weight 값으로 초기화 시킨 후 fine-tuning 하는 방법
    - https://seongkyun.github.io/papers/2019/11/06/imagenet_transfer/
- Hyperparameter를 조절해보기
  - 일반적으로 최적의 학습용 하이퍼파라미터를 찾기위해 하나하나 직접 시도해보긴 어렵지만, 대부분의 논문에서 실험적으로 사용한 세팅을 그대로 가져다 사용하는것을 추천
    - Learning rate, LR decaying 등등
  - 하이퍼파라미터를 잘못 설정해도 네트워크가 학습되지 않고 gradient exploding이 되는경우가 허다함!
- Regularization(data augmentation, dropout, weight decay 등)과 Normalization(Batch normalization 등) 방법들을 줄여보기
  - (여기)[https://course.fast.ai/] 에서는 overfitting 문제 해결 전 underfitting시키는 요소들을 제거시켜 우선적으로 overfitting 상태로 네트워크를 만들어보라고 함
- Loss가 전반적으로 우하향 하고 있다면(줄고 있다면), 일단 좀 더 기다려보기
- 각 딥러닝 framework들은 phase에 따라(training/test) batch normalization, dropout등이 다르게 동작하는것을 인지하기
- __학습 과정 시각화하기(중요)__
  - 각 layer의 activations, weights, updates를 모니터링 할 것
  - 변화량이 적어야(약 0.001정도) 학습이 다 된것임
    - https://cs231n.github.io/neural-networks-3/#summary
  - Tensorboard나 Crayon을 사용하면 됨
- Activation 값들의 평균이 0을 상회하는지 확인하기
  - Batch norm이나 ELU activation function 사용하기
- Weight, biases의 분포는 gaussian 분포를 따르는것이 좋음
  - 단, LSTM은 예외
  - 해당 값들이 inf로 발산하는지를 주시해야 함
- Optimizer를 잘 쓰면 빠르게/효율적으로 학습 가능함
  - 매우 정리 잘 되어있는 좋은 글: https://ruder.io/optimizing-gradient-descent/
  - Momentum SGD + step lr decaying을 사용하거나, Adam을 사용하는것이 일반적
- Exploding/Vanishing gradients
  - Gradient clipping을 처리해야 할 수도 있음
  - Activation의 표준편차는 주로 0.5에서 2 사이이며, 이를 벗어나면 vanishing/exploding activation을 의심해봐야 함
- Increase/Decrease Learning Rate
  - 현재 lr에서 0.1이나 10을 곱하면서 학습 진행시키기
- RNN의 학습 시의 NaN은 큰 문제
  - 처음 100 iteration안에 NaN이 출력된다면 lr을 줄여봐야 함
  - 0으로 나눌 때 뿐만 아니라 log에 0이나 음수가 들어가도 나올 수 있으니 알고있어야 함
  - Layer by layer로 조사해보면서 NaN을 찾아야 할 수도 있음
