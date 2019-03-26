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
  - 다른 형식으로 초기화(-1이 아니고) 되어도 상관 없지만, 별도의 top의 위치를 정의하는 변수가 필요하는 등 코드가 지저분해지고 별도 변수의 관리가 필요
  - Stack에서는 top의 위치가 제일 중요함
    - Top의 위치를 기준으로 데이터의 추가/참조가 이루어짐
- Index 값이 큰값에서 작은값으로 감소하며 추가되는것이 틀린것은 아니지만, 길이에 대한 제한이 생기므로 좋지 못한 방법
- Top의 위치는 항상 맨 위가 되어야 함
- `push`: Top을 위로 한 칸 올리고, Top이 가리키는 위치에 새 데이터를 저장
- `pop`: Top이 가리키는 데이터를 나환하고, Top을 한 칸 아래로 내림

### 스택의 헤더파일

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig7.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 기본적으로 위의 구조를 따름
  - 스택의 구조체는 배열 기반의 스택이므로 이를 고려하여 정의
- `StackInit` 함수에서는 topIndex를 -1로만 초기화 해주면 됨
- `SIsEmpty` 함수에서는 스택이 비었으면 True, 스택에 값이 존재하면 False를 반환
- `SPush`, `SPop`, `SPeek`은 각각 값 입력, 값 참조 후 삭제, 값 참조 를 수행

### 배열 기반 스택의 구현: 초기화 및 기타 함수

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig8.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 스택의 topIndex 멤버의 값이 -1일 경우 스택이 비었다고 판단함
- `SIsEmpty` 함수에서는 입력된 스택의 topIndex 멤버의 값을 기준으로 스택이 빈 경우를 판단

### 배열 기반 스택의 구현: PUSH, POP, PEEK

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig9.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- `SPush` 함수는 top인덱스 값을 증가시킨 후 배열의 해당 번째 인덱스(topIndex)에 값을 저아
- `SPop` 함수는 스택 맨 위의 값을 참조 후 해당 값을 삭제
  - 함수는 내부에서 입력된 스택이 비었는지를 판단 한 후 연산이 실행되어야 함
    - `SIsEmpty` 함수를 이용
  - 함수에서 삭제의 의미는 값을 0같은 값으로 초기화 하는 것이 아니라 topIndex의 값을 1을 내려주면 됨
    - 구지 값을 초기화하지 않아도 스택에 저장된 값의 참조나 새 값의 저장은 topIndex 값을 기준으로 수행되기 때문에 topIndex 값만 조절(1 내리기)하면 됨
- `SPeek` 함수는 `SPop`과 동일하지만 삭제 기능이 빠져있음

### 배열 기반 스택의 활용: main 함수

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig10.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

## 스택의 연결 리스트 기반 구현

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig11.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 단순 연결 리스트를 이용하여 손쉽게 구현 가능
  - 스택에서 top이 하는 역할을 단순 연결 리스트에서 head가 하도록 하면 스택을 쉽게 구현 가능
  
### 연결 리스트 기반 스택의 논리와 헤더파일 정의

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig12.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 그림만 놓고 보면(메모리 구조) 스택인지 연결 리스트인지 구별 불가
  - ADT를 어떻게 정의하고 결정하느냐에 따라 자료구조의 형태가 달라짐
- 단순 연결 리스트의 구조체를 그대로 가져다 놓고 Stack으로 응용시킬 수 있음
  - 구조체 멤버에는 head 포인터 변수만 필요하며
    - 실제 스택의 모든 참조 및 쓰기 연산은 head 위치인 top에서만 이루어지므로

- __문제 6-1 꼭 해보기!_
  - 기본적으로 배열 및 연결 리스트 자료구조는 다른 자료구조들의 기본형으로 도구로써 많이 활용됨
    - 배열 및 연결 리스트를 이용하여 다른 자료구조를 구현
  - 문제에서 `CLinkedList.h`와 `CLinkedList.c`를 변형하지 않고 그대로 이용하여 스택을 구현시킬 수 있음!
    - 각각 list.h와 list.c가 연결되어있고, stack.h와 stack.c가 연결되어 있는 상태에서 stack.c가 list.c를 참조하여 stack.c에서 스택을 구현

### 연결 리스트 기반 스택의 구현

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig13.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- `StackInit`에서는 head의 위치만 NULL로 초기화
- `SPop`에서는 head가 다음 노드를 가리키고, head가 가리키던 노드만 삭제하면 됨

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig14.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 연결 리스트 기반의 스택을 직접 구현하는것보다 문제 6-1 내용처럼 구현하는것이 더 유의미함

### 연결 기반 스택의 활용: main 함수

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig15.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- __문제 6-1 구현해보기__
