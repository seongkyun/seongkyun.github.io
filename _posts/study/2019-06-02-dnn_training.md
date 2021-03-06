---
layout: post
title: 인공신경망 학습 레시피
category: study
tags: [Convolutional Neural Network, Training]
comments: true
---

# 인공신경망 학습 레시피
- 참고 글
  - https://karpathy.github.io/2019/04/25/recipe/
  - https://medium.com/@bntejn/%EC%9D%B8%EA%B3%B5%EC%8B%A0%EA%B2%BD%EB%A7%9D-%ED%95%99%EC%8A%B5-%EB%A0%88%EC%8B%9C%ED%94%BC-%EB%B2%88%EC%97%AD-70c5e58341ec
    - 원글의 번역본을 참고

- 네트워크의 학습시 시 발생할 수 있는 오류들을 줄이기 위한 프로세스를 정리
- 본문 시작전에 다음의 두 가지 중요 관찰사항에 대해 논함

## 1. 인공신경망의 학습과정은 딥러닝 라이브러리를 이용하여 완벽히 추상화하는것이 불가능함
- 보통의 딥러닝 라이브러리를 이용한 네트워크의 학습은 아래와 같은 흐름을 따름

```
{%raw%}
dataset = my_dataset
model = my_model(my_transform, dataset, ResNet50, SGDOptimizer)
{%endraw%}
```

- 복잡한 실제 연산의 과정을 딥러닝 라이브러리는 위의 코드처럼 쉽게 표현되어있음
  - 하지만 실제로는 그렇지 않음
- 일반적인 딥러닝 모델의 학습에 역전파(back-propagation)와 SGD만 적용한다고 해서 인공신경망이 자동으로 동작하지 않음
  - Batch-norm을 적용한다고 해서 optimization이 마술처럼 이루어지는것도 아니고
  - RNN을 도입한다 해서 텍스트를 마술처럼 자동으로 이해하게 되는것도 아니며
  - 강화학습으로 문제를 정의할 수 있다 하여 꼭 그렇게 풀어야 하는것도 아님

## 2. 학습의 실패는 예고없이 등장함
- 코딩할 때 코드를 잘못 짜거나 설정을 잘못한다면 종종 예외처리문(error)을 만나게 됨
  - 문자열 자리에 정수를 넣거나, 매개변수 갯수가 다르다거나, import가 실패하거나, 혹은 입출력의 모양이 다르거나..
- 인공신경망 학습에 있어서 위와같은 쉬운 오류 외에 어디서 잘못되었는지도 모르는 오류가 발생할 확률이 매우 큼
  - 오류 자체가 구문적이며 단위적으로 테스트하며 디버깅하기가 매우 까다로움
  - 예를들어, 
    - Data augmentation시 이미지의 좌우를 뒤집으면서 lable도 뒤집는것을 깜빡해서 인공신경망이 이러한 잘못된 정보까지 학습하는 바람에 오히려 모델의 정확도가 향상 될 가능성도 존재하게 됨.
    - 자기회기모델(auto-regressive model)을 학습시키며 off-by-one bug(배열이나 루프에서 바운더리 컨디션 체크와 관련한 논리적 에러)로 인해 예측하려는 값을 input으로 취하는 실수
    - Gradient를 잘라낸다는게 loss를 잘라내서 학습과정에서 outlier data들이 무시되게 될 수도 있음
    - 미리 학습해둔 checkpoint로부터 weight를 초기화했는데 해당 평균값을 쓰지 않는 실수
    - 정규화 세팅, 학습률, 학습률 감소율, 모델 크기 등에 대한 설정 잘못
  - 위의 경우 외에도 어떠한 잘못된 설정으로 네트워크를 학습시켰을 때 정확도가 좋아질 수 있고, 이는 순전히 운이 좋은경우에 해당
  - 대부분의 경우 학습은 문제없다는듯이 잘 진행되며 성능만 살짝 덜어지게 됨

- 이로인해 __신경망을 학습하기 위한 방법으로 빠르고 강력한 방법은 전혀 효과적이지 않고__ 오히려 학습에 어려움을 줄 수 있음
  - 네트워크를 빠르게 학습시키는 여러 방법들이 실제로는 효용성이 떨어진다는 의미로 판단됨.
- 따라서 네트워크의 학습에는 조심스럽고 방어적이고 시각화를 중요하게 여기는 방향으로 접근해야 학습에 좀 더 효과적임
  - 글쓴이의 경험에 따르면.. 참을성 있게 디테일에 집착하는 태도가 딥러닝 모델의 학습을 성공시키는데 가장 중요한 요소라고 함.

## 레시피
- 위에서 언급한 두 사실에 근거하여 저자가 제안하는 늘 참고하는 구체적인 학습의 프로세스를 소개함
  - 각 단계를 거치며 모델은 단순하게 시작해서 점차 복잡해지며, 매 단계마다 어떤 response가 발생할지에 대한 구체적 가설을 세우고, 각 단계가 적용된 후 혹시 발생했을지 모를 문제를 찾아내기 위해 실험과 검증을 반복
- 이렇게 보수적(조심스럽게)으로 접근하는 이유는 검증되지 않은 복잡성(새로운 것을 적용하는것)의 증가를 최대한 방지하고, 발견이 힘들거나 발견자체가 불가능할 수 있는 버그나 오류를 최대한 예방하기 위함임
- 네트워크의 학습시키는 코드를 짜는것을 인공지능망 학습과정에 비유하면 작은 학습률로 학습을 시작한 뒤 매 iteration마다 테스트 데이터셋 전체를 평가하듯이 각 단계를 진행해야 함.
  - 하나의 방법을 적용하고 effectiveness validation 후 다음 방법을 적용해야 함을 의미

### 1. 데이터와 하나가 되기
- __인공신경망 학습의 첫 단계는 코드는 건드리지 않고 학습용 데이터셋을 철저하게 살피는것!(매우 중요)__
  - 수천개의 데이터를 훑어보며 분포를 이해하고 패턴을 찾는 과정
  - 중복된 데이터, 손상된 이미지, 손상된 레이블 등을 발견하고 제거
  - 데이터의 불균형, 편향을 발견하고 조절
- 데이터의 분석을 통해 궁극적으로 사용하게 될 아키텍쳐에 대한 힌트를 얻을 수 있음
  - 예를 들어 아주 지역적인 특성들 만으로 충분한지, 혹은 전역적인 맥락이 필요한지, 얼마나 많은 변화가 있고 어떤 형태들을 갖는지, 어떠한 비정상적인 변화가 감지되고, 전처리를 통해 제거가 가능한지, 공간적 위치가 중요한지, 어떤 pooling 방식이 좋을지, 세부사항이 얼마나 중요하고 얼마나 많은 이미지들을 샘플링을 통해 줄일 수 있을지, 레이블에 얼마나 노이즈가 많은지 등을 살펴볼 수 있음

- 또한 인공신경망이란 결국 데이터를 압축하고 일반화시켜주는 도구이기에 네트워크의 에러를 보고 어디가 잘못된 것인지도 알 수 있음
  - 만약 정확도가 원래의 것보다 떨어진다면 무언가 잘못되었다는걸 직관적으로 알 수 있게 됨

- 데이터의 품질에 대한 대략적인 감을 잡았다면 학습시킬 데이터를 찾고, 걸러내고, 정렬하기 위한 간단한 코드를 작성
  - 그 기준은 레이블의 타입, annotation,의 크기와 숫자 등 고려 가능한 어떠한 것이 될 수 있음
  - 각 기준에 따른 데이터의 분포를 시각화해보고 각 기준에 따라 시각화 했을 때 분포를 벗어나는 튀는 outlier들을 찾아봄
  - Outlier는 대부분 데이터 품질이나 전처리 과정의 오류로 인해 발생했을 가능성이 큼

### 2. 학습에서 평가까지 전 단계를 아우르는 골격을 먼저 짜고 기준 성능을 측정
- 이번 단계에선 확실하게 성능이 검증되는 작은 크기의 신경망이나 선형 분류기를 선택해서 테스트하는게 좋음
  - 위의 모델을 학습시키고 loss를 시각화하며 metric 측정 후 모델이 추론한 결과를 이용
- __Random seed 이용.__ 랜덤한 seed를 이용하여 초기값을 설정하면 코드를 돌릴때마다 항상 동일한 결과가 나오게 할 수 있음.
  - 가변 변수를 하나 줄임으로써 안정성 확보
- __단순화.__ 학습에 필수적이지 않은 data augmentation등을 적용하지 않도록 해야 함.
  - 일반화 성능을 향상시켜주는 data augmentation은 debugging을 더 어렵게 만듦
- __Validation에서의 loss 확인.__ Test loss를 plot할 땐 전체 테스트셋에 대한 결과를 plot해야 함. 단순히 배치에 따른 test loss를 시각화하면서 loss을 매끄럽게 하는것보다 전체 testset에 대한 loss를 확인하는것이 중요. 또한 validation loss가 낮아야 학습이 잘 된 것으로 validation loss가 낮도록 모델을 학습
- __초기 손실값 확인.__ 초기 loss가 올바르게 잘 감소하고 있는지 확인. 예를들어 마지막 레이어를 잘 초기화했다면 softmax 결과에서 -log(1/clas갯수)가 측정되어야 함. L2 regression등에도 이와같이 기본값을 유도 가능
- __초기화 잘 하기.__ 마지막 레이어의 weight를 잘 초기화. 예를들어 평균이 50인 값에 근사시킨다면 최종 bias를 50으로 설정. 이러한값들을 잘 설정해서 학습 초기에 하키스틱 모양의 발산하는 loss 그래프를 피할 수 있음
- __Human baseline.__ Loss 외에도 정확도같이 사람이 해석 가능한 metric을 모니터링. 또는 매 테스트마다 주석을 달아서 하나는 예측값, 하나는 GT로 간주해 사용
- __Input-independent baseline.__ 입력과 독립적인 기준값을 학습. 예를 들어 모든 입력을 0으로 하게되면 real data가 들어가면 성능이 떨어지게 됨. 아무 정보가 없는 데이터로부터 모델이 아무런 정보도 얻지 못하는 것.
- __Overfit one batch.__ 아주 작은 예제에 한해서 하나의 배치에만 과적합 시키기. 이를 위해 layer나 필터를 더해서 모델의 용량을 키우고 0에 근접한 loss를 얻을 수 있는지 확인해야 함. 레이블과 예측값을 하나의 그림에 시각화시켜 손실값이 최소일 때에도 둘이 잘 일치하는지를 확인. 그렇지 않다면 어디엔가 문제가 있는 상황
- __학습 loss가 감소하는지 확인.__ 조금 네트워크 용량을 키운 모델에 대해 학습 loss가 잘 감소하는지 확인
- __모델에 feed되기 전의 데이터를 시각화.__ 네트워크에 학습 데이터가 들어가기 바로 직전 단계에서 tensor에 저장되어있는 테이터와 레이블을 해석.
- __학습 중 예측값의 변화를 시각화.__ 미리 정해둔 임의의 test batch에 대해 예측값이 어떻게 변하는지를 시각화. 이를 통해 직관적 이해가 가능. 데이터가 특정한 방향이 아닌 오락가락하게 움직이면 모델이 instable함을 의미하게 됨. Learning rate가 너무 낮거나 높은 경우에도 이를통해 알아차릴 수 있음
- __Dependency를 알기 위해 역전파 이용.__ 보통 대부분의 실수는 배치사이 차원간에 정보를 섞어버리는 실수. 이렇게되도 학습은 보통 계속됨.(다른 case로부터 섞인 필요없는 정보를 무시하도록 네트워크가 학습되므로) 특정 예제 i에 대한 loss를 1.0으로 설정하고 입력단까지 back propagation시켜서 해당 i번째 입력에 대해서만 0이 아닌 gradient가 계산되는지를 출력해보면 됨. 즉, 경사도를 이용해 신경망이 어떤 정보에 의존적인가에 대한 정보를 얻을 수 있음.
- __Generalize a special case.__ 보통 코딩시 처음에는 매우 구체적인 함수부터 시작해서 잘 동작하는지 확인한다음 일반적으로 작동하는 함수를 다시 짜서 올바른 결과가 나오는지 확인하는 방법을 사용. 벡터 연산 코드에도 종종 쓰임.

### 3. Overfitting 시키기
- 이 단계에 다다르면 데이터셋에 대한 이해도도 충분하며 학습과 평가를 해주는 파이프라인도 잘 동작한다는 의미임.
- 좋은 모델을 찾기위해 택하는 방법은 두 단계로 나뉘어짐.
  - 첫째, 과적합에 용이한 큰 모델을 학습시키면서 학습 손실값을 최소화하는데만 집중
  - 둘째, 정규화를 적당히해서 validation에 대한 손실값을 줄임. 여기서 loss의 손실이 약간 발생함.
- 위의 일련의 과정을 통해 어떤 모델을 사용했건 오류값 자체가 줄어들지 않는다면 이를통해 버그나 잘못된 설정 등의 이슈를 잡아 낼 수 있기 때문
- __모델 선정.__ 학습에 적당한 모델을 선정하며, 다른 사람들이 일반적으로 사용했을 때 결과가 좋은 구조를 그대로 사용하는것을 추천함.
- __Adam 사용 추천.__ 학습 초기단계에서 Adam을 learning rate 3e-4 정도로 하여 사용하는것을 추천함. 이유는 Adam이 lr을 포함한 다양한 하이퍼파라미터 설정에 영향을 가장 적게 받기 때문. CNN에선 튜닝 잘 된 SGD가 거의 모든 경우에 adam보다 더 나은 성능을 보여주나 최적의 learning rate 구간은 매우 좁고 task에 따라 가변적임. 즉, SGD가 적절히 동작하는 lr 구간을 찾기가 힘듦.
- __복잡도는 한번에 하나씩 더하기.__ 분류기 모델의 성능 향상을 위해 적용될 각종 알고리즘이 여러개일 경우 한번에 하나씩만 적용하고, 늘려갈 때마다 실제 기대한것처럼 정확도가 향상되는지를 확인해봐야함.
- __Learning rate decay를 너무 믿진 말기.__ 다른 도메인에서 쓰이던 코드를 가져와서 재사용하는 경우 learning rate decay를 매우 조심해서 사용해야 함. 서로 다른 task에 대해 다른 learning rate decay function을 쓰는건 당연한 일. 게다가 lr decaying function은 보통 현재 epoch 숫자에 맞춰 계산되도록 구현되어 있는데, 적정 epoch은 dataset의 종류에 따라 크게 달라짐. 예를들어 ImageNet일때 30 epoch에서 1/10 decaying이지만, 다른 데이터셋은 다른 epoch에 적용되어야 함. 이로인해 lr이 너무 빨리 작아져서 적절하게 모델이 학습되지 않을 수 있음. 따라서 데이터셋에 따라 decaying을 일단 적용하지 말고, 마지막에 튜닝하는식으로 적용시켜야 함.

### 4. 일반화(Regularize)
- 지금까지 문제가 없었다면 최소한 학습용 데이터셋에는 확실하게 맞춰진 큰 모델을 갖고있게됨을 의미.
- 이제 regularization을 적용해서 학습정확도는 좀 잃더라도 테스트 정확도를 올릴 차례.
- __더 많은 학습용 데이터셋.__ 어떠한 경우에도 모델의 일반화에 최선의 방법은 더 많은 실제 데이터를 모으는것임. 더 많은 데이터를 모을 수 있는 상황에서 엔지니어링에 많은 노력을 소모하는것은 매우 흔한 실수. 더 많은 학습용 데이터를 통해 네트워크의 성능을 지속적으로 향상 시킬 수 있음. 차선책으로는 앙상블모델이 있지만 앙상블도 5개정도 모델 이후로는 성능이 증가하지 않음
- __Data augmentation.__ 실제 데이터셋을 더 확보하는 방법 다음으로 좋은 방법은 가짜 데이터셋을 만들어 내는 것임. Resize, crop, flipping 등의 distortion을 이용하여 원래 학습 데이터셋보다 더 많은 학습용 데이터셋을 확보 가능
- __Creative augmentation.__ Data augmentation 외에도 도메인 랜덤화, 시뮬레이션, 데이터와 배경을 합성시키는 하이브리드 기법, GAN등을 이용한 데이터의 증대 가능
  - http://vladlen.info/publications/playing-data-ground-truth-computer-games/?fbclid=IwAR3Bbb2-JTT46xou6-ueUpAXdqM3bL2i9UivKKRn05oRWCNnp7QRupTBZWA
  - https://arxiv.org/abs/1708.01642?fbclid=IwAR2pQV7k2oj8H6j2ndxTNCx5IC46VZMGgg2uFElT0CuzOAufePq-ea6JdU8
- __Pre-training.__ 미리 학습시킨 네트워크를 사용하는 경우 웬만해선 성능이 더 좋음. 이미 충분한 데이터가 존재하더라도 pre-trained model을 불러와서 재학습시킬 때의 정확도가 훨신 높음.
- __지도학습 고수하기.__ 비지도학습에 현혹되지 말 것. 최신 결과물 중 쓸만한 방법이 없음. 반면 자연어처리 분야는 BERT 등의 비지도학습 기법들의 성능이 좋은 편.
- __입력 차원은 낮게.__ 이상한 정보를 포함하는 feature를 제거시켜야 함. 이상한 입력이 추가될수록 학습용 데이터셋이 충분하지 않은 경우 overfitting이 발생하게 됨. 이미지 내의 정보중 세세한 디테일한 내용이 중요하지 않다면 작은 이미지를 이용해 학습시키는것도 좋은 방법이 될 수 있음.
- __모델 크기는 작게.__ 대부분의 경우 도메인에 대한 knowledge를 이용해 신경망의 크기를 줄일 수 있음. 예를 들어 ImageNet의 backbone 가장 마지막에 fc layer를 사용하곤 했지만 훗날 average pooling으로 대체되며 파라미터 수를 많이 줄일 수 있게되었음.
- __Batch size는 작게.__ Batch normalization 안의 정규화 때문에 작은 크기의 batch가 더 일반화 성능을 더 좋게 함. 이는 batch의 평균/분산 값들이 전체 데이터셋의 평균/분산 값의 추정치이기 때문인데, 작은 크기의 batch를 사용하면 scale과 offset 값이 batch에 따라 더 많이 wiggle되기 때문임.
  - 즉, 작은 크기의 batch를 사용함에 따라 학습과정에서의 불확실성(noise)이 증가하게 되어 일반화 성능이 향상될 수밖에 없음. 작은 크기의 batch는 그만큼 많은 parameter update가 수행되므로 noise성분을 더 많이 갖게 됨.
- __Drop.__ 드롭아웃을 추가. ConvNet의 경우 Dropout2d (spatial dropout)을 적용. 단, Dropout은 batch norm과 잘 어울리지 못하므로 주의해서 적당히 써야 함.
  - https://arxiv.org/abs/1801.05134?fbclid=IwAR01IhZoe7yftt9oml_-DHSWqvHXqwjOzAXGtus1ZTCYvoEff8IuUQoNBi8
- __Weight decaying.__ Weight decaying penalty를 증가시킴.
- __Early stopping.__ 지금까지 측정된 validation loss를 이용하여 overfitting이 시작되려는 시점에 학습을 종료
- __더 큰 모델을 시도.__ 맨 마지막으로 큰 모델을 사용할 경우 overfitting될 수 있지만 그 전에 학습을 미리 멈추게 될 경우 작은 모델보다 훨씬 나은 성능을 보여 줄 수 있음

- 마지막으로, 학습시킨 분류기가 잘 동작한다는 추가적 확신을 얻기 위해 네트워크의 첫 번째 레이어의 weight값을 시각화해서 깔끔한 모서리가 나오는지를 확인. 만약 filter(weight 값)가 noise처럼 보인다면 제대로 학습된 것이 아니라고 의심 해 볼 수 있음. 비슷하게 네트워크의 중간 weight값을 시각화해서 이로인해 뭐가 잘못되었는지를 파악 할 수 있음.

### 5. Tunning
- __그리드 탐색보단 무작위 탐색.__ 여러 개의 하이퍼파라미터를 그리드 탐색방법으로 동시에 튜닝하게 될 경우 경우의 수가 너무 많음. 무작위 탐색방법을 적용하는게 가장 효율적임. 이유는 인공신경망의 성능이 특정한 소수의 파라미터에 훨씬 민감하게 반응하기 때문임. 예를 들어 파라미터 a를 변경했을 때 loss가 달라졌지만 b는 변경해도 아무런 영향이 없다면 a를 더 철저히 sampling해보는게 더 좋은 방법.
  - http://jmlr.csail.mit.edu/papers/volume13/bergstra12a/bergstra12a.pdf?fbclid=IwAR3DcdRfOMcrro_RyetpXE-CARWpn9fpYvLLIZjP2qXnyEGBDDYnvkTlNUk
- __하이퍼파라미터 최적화.__ 정말 다양하고 많은 베이지안 하이퍼파라미터 최적화 도구들이 존재. 최고의 방법은 노가다.

### 6. Squeeze out the juice
- 최적의 아키텍처와 하이퍼파라미터를 찾아낸 후에도 마지막 한방울의 성능까지 짜낼 수 있는 몇 가지 트릭이 존재.
- __앙상블 모델.__ 앙상블 모델은 어떠한 경우에서라도 2%정도의 정확도를 올려주는 확실한 방법임. 계산량 부담이 불가능한 경우 dark knowledge distillation을 통해 앙상블 모델의 정보를 작은 단일 모델로 증류(distillation)하는 기법을 시도.
  - https://arxiv.org/abs/1503.02531?fbclid=IwAR2HSli0-ilYp5SVP6avCmIyYV95KpSAm-nrJZ7w5wDn-MnDl6nRnHb9Edw
- __계속 학습 시키기.__ Validation loss가 줄어들지 않으면 대부분 학습을 중단시킴. 하지만 경험상 아무리 오랜시간 학습시켜도 학습은 계속되게 되어있음. 저자의 일례로 휴가기간 내내 실수로 돌려둔 학습모델의 성능이 엄청 좋아진 경험이 있다고 함.

## 결론
- 위의 일련의 과정을 통해 성공적인 학습을 위한 모든 요소들을 갖추게 됨. 기술, 데이터셋, 해결하고자 하는 문제에 대한 깊은 이해, 학습과 평가를 위한 총체적인 인프라를 갖추었으며 더 복잡해지는 모델들도 탐색하고, 각 단계에서 예측가능한 만큼의 성능 향상도 이루었을 것임. Good luck!

