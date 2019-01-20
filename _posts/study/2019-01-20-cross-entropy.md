---
layout: post
title: Cross entropy loss in machine learning
category: study
tags: [cross entropy loss, deep learning, machine learning]
comments: true
---

# Cross entropy loss in machine learning

맨날 그냥 사용하기만 하는 cross entropy loss에 대해 공부해보려 한다.
- 손실함수는 신경망 학습의 목적
- Cross-entropy는 squared loss와 더불어 양대 손실함수
- Cross-entropy는 신경망 출력을 확률로 간주할 수 있는 경우에 사용되는 매우 중요한 손실함수

## Cost/Loss function
### Squared loss
- $$E(w)=\frac{1}{2}\sum_{d\in{D}}(y_{d}-\hat{y}_{d})^{2} \quad where \abs{D}=학습집합의\; 크기$$
### Mean squared loss
- $$E(w)=\frac{1}{2}\frac{1}{\left\vert D \right\vert}\sum_{d\in{D}}(y_{d}-\hat{y}_{d})^{2} \quad where \abs{D}=학습집합의\; 크기$$

## Information
- 어떤 사건을 수치로 나타낸 것
- 확률을 이용한다
- $x$: 동전의 앞면이 나올 때(event)
- $X$: 확률변수, random process
- $$P(X=x)=P_{X}(x)=p(x)$$ (모두 같은 표현)
### 사건을 수치화 하기
- 사건이 드물게 발생할수록 정보가 커야 함.
- $$\frac{1}{p(x)}\times \frac{1}{p(y)}$$의 확률을 수치화 해 보면?
  - log는 곱셈을 덧셈 연산으로 바꿔줌
- $\frac{1}{p(x)}\times \frac{1}{p(y)}$의 양 변에 log를 취한게 최종 정보량으로 정의 됨
- $$information=log(\frac{1}{p(x)}\times \frac{1}{p(y)})=log\frac{1}{p(x)}+log\frac{1}{p(y)}$$
- $$information\equiv log(p(x))^{-1}=-log(p(x))$$

## Entropy






---
- [참고글]

https://www.slideshare.net/jaepilko10/ss-91071277
