---
layout: post
title: Inception v4
category: papers
tags: [Deep learning]
comments: true
---

# Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning

Original paper: https://arxiv.org/pdf/1602.07261.pdf

Authors: Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi

- 참고 글
  - https://norman3.github.io/papers/docs/google_inception.html

- 2015년 ResNet을 Inception에 붙여보려는 시도를 보인 논문
  - 하지만 해당 모델은 Inception v4가 아니라 Inception-resnet 이라는 별도의 모델로 생성시킴
  - Inception v4는 기존의 v3모델에 몇 가지 기능을 추가시켜 업그레이드한 모델
  - 따라서 이 논문은 Inception v4와 Inception-resnet 둘 다 다루고 있음
    - 특히 resnet을 도입한 모델을 Inception-resnet이라 명명
    - 마찬가지로 Inception-resnet v1, Inception-resnet v2와 같이 별도의 버전들이 존재함
  - 실제로는 ad-hoc한 모델로 이 모델의 한계점을 시사
- Residual connections
  - 깊은 망의 학습시에는(classification) residual connection이 꼭 필요한것인지 구글 내에선 논의중이라고 함
  - 하지만 residual connection이 존재하는 구조의 경우 확실히 학습속도가 빨라지게됨(장점)

<center>
<figure>
<img src="/assets/post_img/papers/2019-07-12-inception_v4/fig1.png" alt="views">
<figcaption></figcaption>
</figure>
</center>

- 위 그림은 residual connection의 예시를 보여줌
  - 첫 번째 그림은 가장 간단한 형태의 residual connection 구조
  - 두 번째는 1x1 conv를 추가하여 연산량을 줄인 residual connection 구조
  - 즉, residual의 개념은 이전 몇 단계 전 레이어의 결과를 현재 레이어의 결과와 합쳐서 내보내는것을 의미
  
## Inception v4, Inception-resnet v2
- Inception v4의 전체 망 구조는 아래와같음

<center>
<figure>
<img src="/assets/post_img/papers/2019-07-12-inception_v4/fig2.png" alt="views">
<figcaption></figcaption>
</figure>
</center>

- Inception v3와 마찬가지로 거의 유사한 형태의 네트워크 구성을 갖지만 세부적인 inception 모듈의 구성이 다름

## Versioning
- Inception에 resnet이 추가되면서 버저닝이 생김
- Inception v3를 확장한게 Inception v4
- 여기에 Inception v3와 v4에 각각 residual connection을 적용한 버전이 Inception-resnet v1, Inception-resnet v2다.
  - Inception v3를 적용한 resnet은 Inception-resnet v1
  - Inception v4를 적용한 resnet은 Inception-resnet v2

## Stem Layer
- Inception v3에서 앞단의 conv 레이어를 stem영역이라고 부름
- Inception v4에서는 이 부분을 약간 변경함
  - Inception-resnet v2 (Inception v4)에서도 stem영역은 동일하게 아래의 구조를 사용
- Stem 영역의 구조는 아래와 같음

<center>
<figure>
<img src="/assets/post_img/papers/2019-07-12-inception_v4/fig3.png" alt="views">
<figcaption></figcaption>
</figure>
</center>

- 이런 구조가 나오게 된 배경지식은 Inception v3에서 다루었고, Inception v4에서는 앞단의 영역에도 이런 모델이 추가로 적용되어있음
  - 아마도 이것저것 테스트해보다가 결과가 더 좋게 나오기에 이를 채용한듯 하다..

### 4 x Inception-A

<center>
<figure>
<img src="/assets/post_img/papers/2019-07-12-inception_v4/fig4.png" alt="views">
<figcaption></figcaption>
</figure>
</center>

### 7 x Inception-B

<center>
<figure>
<img src="/assets/post_img/papers/2019-07-12-inception_v4/fig5.png" alt="views">
<figcaption></figcaption>
</figure>
</center>

### 3 x Inception-C

<center>
<figure>
<img src="/assets/post_img/papers/2019-07-12-inception_v4/fig6.png" alt="views">
<figcaption></figcaption>
</figure>
</center>

- 위에서 설명된 Inception 모듈은 모두 입출력 크기의 변화가 없음
  - Inception 모듈의 input, output 사이즈를 의미
- 실제 크기 변화가 발생되는 부분은 아래처럼 reduction이라는 이름을 사용함

### Reduction-A

<center>
<figure>
<img src="/assets/post_img/papers/2019-07-12-inception_v4/fig7.png" alt="views">
<figcaption></figcaption>
</figure>
</center>

### Reduction-B

<center>
<figure>
<img src="/assets/post_img/papers/2019-07-12-inception_v4/fig8.png" alt="views">
<figcaption></figcaption>
</figure>
</center>

- Inception v4 모듈엔 새로운 컨셉이 등장하지 않고 기존의 Inception v3 모델을 이것저것 실험해보며 좋은 성능을 보인 case들을 조합해놓은것임

## Resnet
- 논문에선 버전을 Inception-resnet v1과 v2로 구분하여 그림으로 설명함

<center>
<figure>
<img src="/assets/post_img/papers/2019-07-12-inception_v4/fig9.png" alt="views">
<figcaption></figcaption>
</figure>
</center>

- 특별한 내용 없이 residual connection만 추가된 구조

## 결과
- Resnet 도입으로 학습 속도가 빨라짐

<center>
<figure>
<img src="/assets/post_img/papers/2019-07-12-inception_v4/fig10.png" alt="views">
<figcaption></figcaption>
</figure>
</center>

- 위 그림은 Inception v3와 Inception-resnet v1의 error rate 수렴 속도를 나타냄
  - Resnet이 적용된 모델의 수렴속도가 훨씬 빠른것을 볼 수 있음
- 성능지표는 아래와같음

<center>
<figure>
<img src="/assets/post_img/papers/2019-07-12-inception_v4/fig11.png" alt="views">
<figcaption></figcaption>
</figure>
</center>
