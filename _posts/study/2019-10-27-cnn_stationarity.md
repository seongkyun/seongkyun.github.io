---
layout: post
title: CNN의 stationarity와 locality
category: study
tags: [CNN, stationarity, locality]
comments: true
---

# CNN의 stationarity와 locality
  - 출처: https://medium.com/@seoilgun/cnn%EC%9D%98-stationarity%EC%99%80-locality-610166700979

- 딥러닝 모델들은 각각 어떤 가정하에 만들어짐
  - 어떠한 특징을 갖는 데이터만 다루겠다는 가정이 들어감
- 이로 인해 각 모델의 가정을 잘 아는것과 데이터 특성에 맞는 모델을 잘 선택하는것이 중요함

## 영상정보가 가진 특성에 대한 CNN의 가정
- AlexNet 논문에서는 영상정보가 갖는 특성에 대한 CNN의 가정이 별다른 설명 없이 한 줄로 언급된다.
  - Stationarity of statistics과 locality of pixel dependencies

### Stationarity of statistics
- CNN이 stationarity 가정하에 convolution 연산을 사용하는 이유를 이해하기 위해서 필요한 개념들을 설명하면 아래와 같다.
  - Stationarity, Parameter sharing

### Stationarity
- Stationarity는 주로 통계의 시계열 분석에서 사용되는 용어
  - 데이터들이 시간에 관계 없이 데이터의 확률 분포는 일정하다는 가정
  - 즉, 시간이 이동되어도 동일한 패턴이 반복됨을 의미함
- 더 정확하게는 시간이 지나도 데이터 분포의 statistics들(평균, 분산 등)이 변하지 않는다는 가정을 베이스로 예측한다.

<center>
<figure>
<img src="/assets/post_img/study/2019-10-27-cnn_stationarity/fig1.jpg" alt="views">
<figcaption>시간이 지나도 분산, 평균값은 동일하게 유지됨</figcaption>
</figure>
</center>

- 영상정보에서 stationarity of statistics의 의미는 아래와 같다.
  - 이미지의 한 부분에 대한 통계가 어떤 다른 부분들과 동일하다는 가정
  - 즉, 이미지에서 한 특징이 위치에 상관없이 다수 존재할 수 있으며, 결국 어떤 위치에서 학습한 특징 파라미터를 이용해 다른 위치에서도 동일한 특징을 추출할 수 있음을 의미
- 아래 사진을 보면 입이라는 동일한 특징이 서로 다른 위치에 동시에 두 개 존재한다.
  - 물론 모양은 조금 다르지만, 이것이 영상정보 내에서의 stationarity 특성을 의미한다.

<center>
<figure>
<img src="/assets/post_img/study/2019-10-27-cnn_stationarity/fig2.jpg" alt="views">
<figcaption>Stationarity of statistics를 표현한 사진으로, 한 이미지 안에서 위치에 상관없이 입이라는 동일한 특징이 2개 존재함</figcaption>
</figure>
</center>
  
### Parameter sharing
- 위 사진에서 입이라는 특징을 뽑아낼 때 두 가지 방법이 있음
  1. 2개의 입을 하나의 특징으로 취급해 위치와 상관없이 입 2개를 추출하는것과
  2. 위치를 고려해 서로 다른 특징 2개를 추출하는 것
    - 특징 1: 왼쪽에 위치한 입
    - 특징 2: 오른족에 위치한 입

- 위 두 가지 방법 중 1번 방법이 훨씬 효율적인 방법!
  - 파라미터를 공유하는 이유가 바로 위치에 상관없이 특징들을 추출하기 위함임
  - 2번은 동일한 특징이라도 위치가 다르다면 다른 특징으로 인식해야 해서 위치마다 모두 다른 필터를 사용해야 하기에 매우 비효율적이게 됨
- 따라서 파라미터 공유는 feature map 하나를 출력하기 위해 필터를 단 한장만 유지하기 때문에 FC layer보다 훨씬 적은 파라미터 수를 사용하여 메모리를 아끼고 연산량도 적어지고 statistical efficiency 또한 향상됨
  - Statistical efficiency란 예측 모델의 품질 측정 방법으로, 더욱 효율적인 모델은 상대적으로 데이터 수를 적게 학습시켜도 더 좋은 성능을 내야한다는 의미
- 파라미터 공유 덕분에 한 장의 feature map을 만드는데 동일한 특징을 여러 곳에서 볼 수 있기 때문에 fc 레이어에 비해 동일한 학습 데이터셋 상황에서도 더 많은 데이터를 학습한 효과를 갖게 되고, 결국 statistical efficiency가 향상됨

<center>
<figure>
<img src="/assets/post_img/study/2019-10-27-cnn_stationarity/fig4.png" alt="views">
<figcaption>파라미터 공유</figcaption>
</figure>
</center>

- 위 그림의 상단이 파라미터를 공유하는 네트워크로 검은 화살표는 3칸짜리 필터에서 가운데 파라미터가 연결된 입력과 출력을 보여준다.
- 작은 필터 하나가 인풋의 여러 곳을 보기에 마치 더 많은 데이터를 보게 되는 느낌으로 통계적인 효율성이 향상된다.
- 위 그림의 하단은 파라미터를 공유하지 않기에 x3과 s3 하나만 연결되어 있는 상태로, 비교적 데이터를 덜 보게되어 통계적 효율이 떨어진다.

<center>
<figure>
<img src="/assets/post_img/study/2019-10-27-cnn_stationarity/fig5.png" alt="views">
<figcaption>파라미터 공유를 활용한 엣지 디텍션의 효율성</figcaption>
</figure>
</center>

- 위 그림의 우측 사진은 원본 이미지의 각 픽셀에서 왼쪽 값을 빼서 만들어진 이미지다.
- 두이미지는 모두 높이는 280픽셀이며, 왼쪽의 입력 이미지는 가로가 320, 오른쪽의 출력 이미지는 319다.
- 이 영상은 두 칸짜리 커널을 이용해 convolution하여 얻어질 수 있다(왼쪽 -1, 오른쪽 1)
- Convolution 연산 시 319x280x3=267,960번의 연산이 필요하지만, 행렬곱으로 동일한 연산을 하려면 320x280x319x280번의 연산이 필요하다.
- 결론적으로 convolution은 전체 인풋에 걸쳐 작은 local 영역 선형변환을 하는 훨씬 효율적인 방법이다.
  
  
