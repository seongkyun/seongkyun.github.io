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
