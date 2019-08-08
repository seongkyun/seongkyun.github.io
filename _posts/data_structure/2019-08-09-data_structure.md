---
layout: post
title: CH9. 우선순위 큐 (Priority Queue) 2-2 (힙의 구현과 우선순위 큐의 완성)
category: data_structure
tags: [data structure, 자료구조]
comments: true
---

# CH9-2. 힙의 구현과 우선순위 큐의 완성
## 원리 이해 중심의 힙 구현: HDelete

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-09-data_structure/fig1.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 편의를 위해 약간의 연산과정 변경을 거쳐 코드로 구현
  - 논리상 삭제 시 맨 마지막 노드의 값이 1번 자리로 이동해야 하지만, 이렇게 될 경우 소모적인 swap 연산이 계속 반복되어야 함
  - 따라서 구지 옮기지 않고, 가야할 자리의 인덱스(1)와 마지막 노드의 값을 임시로 저장해 이를 이용한 비교연산 수행
- `while` 문에서 우선순위 높은 자식의 인덱스 값을 `childIdx`로 받아
  - 마지막 노드의 우선순위와 비고하여 마지막 노드의 우선순위가 높으면 while문 탈출
    - `parentIdx`엔 마지막 노드가 저장될 인덱스(위치)가 저장됨
  - 아니면 레벨을 올리기 위한 대입연산 수행
- 최종적으로 삭제된 노드의 값을 반환

## 원리 이해 중심의 힙 구현: HInsert

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-09-data_structure/fig2.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 마지막 노드 자리에 새 노드를 넣으면 delete와 마찬가지로 반복/소모적인 swap 연산을 계속 해야함
- 따라서 delete와 동일한 방법으로 insert 수행

- 새 노드가 저장될 인덱스 값을 임시로 받고, 새 노드의 생성 및 초기화(임시)
- `idx`가 1이면 root 자리이므로, 1이 아닐 동안 while문을 돌음
  - if문으로 새 노드와 부모노드의 우선순위를 비교해 새 노드의 우선순위가 높다면
    - 부모 노드를 한 레벨 내림(대입연산으로 실제로 내림)
    - 새 노드를 한 레벨 올림(인덱스 값만 갱신하여 실제로 올리지 않음)
  - 아닐 경우 break
- 마지막으로 새 노드를 만들어진 자리(`idx`)에 집어넣고 전체 데이터 갯수 `numOfData` 1개 증가

## 힙의 확인을 위한 main 함수와 단점

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-09-data_structure/fig3.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 위와 같은 구성은 사용자가 직접 데이터의 우선순위를 결정해야 하는 방식
- 일반적인 경우, 일관성을 위해 입력된 data를 기반으로 자동으로 프로그램이 알아서 우선순위를 결정하는것이 합리적인 구성

## 쓸만한 수준의 힙 구현: 구조체 변경

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-09-data_structure/fig4.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 기존에는 pr과 data를 따로 `HeapElem` 에서 받아 `Heap` 구조체가 완성됨
- 이를 data와 data에 근거해 자동으로 우선순위를 결정하는 함수를 멤버로 선언한 구조체를 생성하고, 이를 `Heap`으로 정의
  - `* comp` 멤버에 우선순위 결정하는 함수의 주솟값 저장
  - 우선순위 결정 함수에 의해 자동으로 data에 따라 우선순위가 반환됨
- `HeapInit` 함수에서도 노드의 갯수 초기화 외에 우선순위 결정하는 함수의 등록도 수행되어야 함

- 함수 포인터?
  - 일반적인 경우 `typedef int *FunctionName(int A, int B)` 또는 `typedef *int FunctionName(int A, int B)`로 구성
  - 이럴 경우, 구조체 멤버 변수 선언 시 `*`가 포함되면 안됨
  - 반대로 위와 같이 함수가 `typedef int FunctionName(int A, int B)`로 선언 된 경우, 구조체 멤버 변수 선언시 멤버 변수에 `*`가 포함되어야 함
  - __이는 일종의 약속, 단 매개변수 선언은 예외가 됨(위 약속을 따르지 않음)__

## 쓸만한 수준의 힙 구현: PriorityComp

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-09-data_structure/fig5.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 함수에 인자가 두개 전달될 경우 경우에 따라 반환되는 값들이 달라짐
- 또한 Insert 함수에서도 `pr`을 별도 인자로 받지 않고 구조체 내의 `comp`함수와 `data`로 자동으로 계산되어 우선순위 설정되도록 함
- 이는 임의로 설정된 조건으로, 사용자의 편의에 따라 아무렇게나 바뀌어도 됨

## 쓸만한 수준의 힙 구현: Helper 함수의 변경

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-09-data_structure/fig6.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- if문의 조건에서 대소가 바뀌는것은 앞의 `PriorityComp`함수의 조건에 따라 바뀌는 것임
  - 따져보면 동일한 조건

## 쓸만한 수준의 힙 구현: HInsert의 변경

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-09-data_structure/fig7.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 위와 마찬가지로 비교함수 조건에 따른 if문의 대소만 변경

## 쓸만한 수준의 힙 구현: HDelete의 변경

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-09-data_structure/fig8.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 위와 동일함

## 쓸만한 수준의 힙 구현: main 함수

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-09-data_structure/fig9.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 구조체의 `data`를 이용하여 자동으로 우선순위를 결정하는 함수를 정의
  - 아스키 코드 값이 작은 문자의 우선순위가 더 높도록 설정

## 쓸만한 수준의 힙을 이용한 우선순위 큐의 구현

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-09-data_structure/fig10.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 앞의 힙은 우선순위 큐에만 최적화하여 구현됨
- 다라서 껍데기만 우선순위 큐로 씌워주면 그대로 우선순위 큐가 됨










