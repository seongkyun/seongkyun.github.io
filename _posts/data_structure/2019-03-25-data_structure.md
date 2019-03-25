---
layout: post
title: CH6. 스택 (Stack) 1
category: data_structure
tags: [data structure, 자료구조]
comments: true
---

# CH6. 스택 (Stack) 1
- 자료구조중에서 제일 쉬운 구조!
  - 앞에서 공부했던 단방향 연결리스트로도 쉽게 구현 가능함
- 스택(Stack)은 흔히 사용되는 자료구조형은 아님
  - 계산기의 구현(예시), 운영체제 등등.. 에서 사용됨

## 6-1 스택의 이해와 ADT 정의
### 스택의 이해

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig3.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 먼저 들어간것이 나중에 나오는 자료구조(Last-in First-out, LIFO)
- 이를 만족하는 모든 자료구조는 stack이라고 함
- 스택의 기본구조
  - push: 스택에 값을 넣기
  - pop: 스택에서 값을 얻기 + 삭제
    - 값을 반환 받은 후 반환 받은 값을 제거
  - peek: 스택에서 값을 얻기
    - 값을 반환만 받을뿐 삭제하지 않음
- 스택의 구현 자체는 어렵지 않지만 활용의 측면에서 고려해야할것들이 많음

### 스택의 ADT 정의

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig4.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- ADT를 대상으로 배열 기반/연결 리스트 기반의 스택을 모두 구현할 수 있음
- `SPop` 함수와 `SPeek`의 경우, `SIsEmpty` 함수로 스택이 비어있지 않은 경우에만 동작하게끔 작동
  - 데이터가 저장되어 있는 경우에만 pop과 peek 연산이 가능함

## 6-2 스택의 배열 기반 구현

### 구현의 논리

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig6.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 인덱스가 0인 위치를 스택의 바닥으로 정의해야 배열 길이에 상관없이 인덱스값이 동일해짐
  - 0번째 인덱스에 첫 번째 데이터가 저장됨
- Top의 위치를 변수 `topIndex`로 관리하고, -1로 초기화해야 데이터가 입력되었을 때 index 변수로 데이터 입/출력 관리가 용이함

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig7.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig8.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig9.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig10.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig11.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig12.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig13.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig14.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig15.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>
