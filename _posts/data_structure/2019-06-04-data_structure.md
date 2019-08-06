---
layout: post
title: CH7. 덱 (Deque) 3
category: data_structure
tags: [data structure, 자료구조]
comments: true
---

# CH7. 덱 (Deque) 3

## 7-5 덱(Deque)의 이해와 구현
- '디큐'가 아닌 '덱'으로 읽는 이유? -> 발음상 '디큐' 가 맞지만, dequeue와의 구별을 위해 보통 '덱'으로 읽음
- Deque는 Double-end queue의 줄임말
- Stack과 queue의 자료구조적 특성을 모두 가짐
  - 활용에 따라 stack 또는 queue처럼 사용 가능하므로, 기능상 특성만을 생각했을 때 두 자료구조적 특성을 갖는다고 표현 가능

- 덱을 큐(queue) 단원에서 설명하는 이유?
  - 덱에 대해서 다룰게 많지 않기 때문임

- 덱(deque)이란?
  - Stack은 선입 후출, queue는 선입 선출적 특성을 갖는다면 deque은 queue 구조에서 양쪽으로 삽입 및 출력이 모두 가능한 구조임
    - 양쪽 아무데로나 data를 넣고 꺼낼 수 있음
  - 방향의 상관 없이, data를 어디로 넣었건 꺼내는 방향은 상관 없음
    - `F`로 넣고 `F`로 꺼내건, `F`로 넣고 `R`로 꺼내건 상관 없음
  - 단, 넣은 순서나 위치(`F`또는 `R`)에 따라 꺼낼때의 data는 달라짐

### 덱의 이해
- 덱은 앞으로, 뒤로도 넣을 수 있고, 앞으로, 뒤로도 뺄 수 있는 자료구조
  - 앞으로 넣기, 앞에서 빼기, 뒤로 넣기, 뒤에서 빼기 4가지 연산 가능
  - 모든 연산은 짝을 이뤄 수행되지 않으며 개별적으로 연산 가능
 
### 덱의 ADT
- 각각 연산의 종류에 따라 앞/뒤 위치의 입력 및 출력에 대한 4개의 삽입/참조 연산 함수와 위치에 따른 peek 연산의 정의 필요

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-06-04-data_structure/fig1.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

### 덱의 구현: 헤더파일 정의
- 덱의 구현에 가장 적절한 자료구조는 __양방향 연결 리스트__
  - 단순 연결리스트로 구현 할 경우?
    - 새로운 데이터의 삽입에는 `F`나 `R`의 위치에 상관없이 가능
    - 기존 데이터의 삭제 시, `R` 부분의 데이터를 삭제 할 때 문제 발생!
- 단순 연결리스트로 구현된 덱의 `R`부분에서 데이터를 삭제한다면?
  - 삭제 하는 해당 노드 이전 노드의 주소로 `R`의 값을 초기화해줘야 하지만 이전 노드의 주소를 알 수 없음
    - 실제로 알 수 없는건 아니지만 알기에 과정이 복잡함 (`F` 위치부터 다시 주소를 찾아 나가야 하므로..)
  - 따라서 양쪽 노드(해당 노드의 앞, 뒤)의 주소를 모두 아는 __양방향 연결리스트__ 가 덱을 구현하기에 가장 적절한 자료구조형

- 덱 구조체에는 앞/뒤 모두 입/출력이 이루어지므로 `head` 뿐만 아니라 `tail` 도 정의되어야 함

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-06-04-data_structure/fig2.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

### 덱의 구현: 함수의 정의
- 양방향 연결리스트에서 `tail` 포인터 변수만 추가된 꼴!

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-06-04-data_structure/fig3.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>
