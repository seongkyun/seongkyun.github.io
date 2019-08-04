---
layout: post
title: CH8-4. 수식 트리 (Expression Tree) 의 구현
category: data_structure
tags: [data structure, 자료구조]
comments: true
---

# CH8-4. 수식 트리 (Expression Tree) 의 구현
- 지난 시간까지 만들어놓은 이진 트리를 구성하는 도구를 이용하여 수식트리를 구현
- 수식트리는 이진트리의 일종으로, 수식을 트리 형식으로 구현하는것을 의미
- 전, 중, 후위 연산자처럼 도 하나의 수식 표기 방법
  - 다만, 메모리 관점에서 내용이 설명됨(루트 노드의 주소를 저장)

## 수식 트리의 이해

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-04-data_structure/fig1.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 중위 표기법의 수식을 수식 트리로 변환하는 프로그램을 작성하는것이 목적
- 중위 표기법 수식은 사람이 이해하기 쉽지만 기계가 인식하기에는 어려움
- 따라서 중위 표기법 수식을 수식 트리로 재구성
- 수식 트리는 해석이 쉬움
  - 연산 과정에서 연산자의 우선순위 고려가 필요 없음
- 가운데 루트노드 자리가 연산자, 아래의 하위 노드가 피연산자 역할을 하게 됨

## 수식 트리의 계산과정

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-04-data_structure/fig2.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 자식 노드가 피연산자, 루트 노드가 연산자 역할을 함

## 수식 트리를 만드는 절차

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-04-data_structure/fig3.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 중위 표기법의 수식을 후위 표기법의 수식으로 변환 후, 이를 기반으로 수식 트리를 만듦
  - 후위 표기법 수식으로 수식트리를 만드는게 가장 쉽기 때문에 두 단계로 나눔
- 앞 스택 단원에서 사용한 코드를 이용

## 수식 트리의 구현과 관련된 헤더파일

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-04-data_structure/fig4.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 앞에서 정의된 트리를 만드는 도구를 기반으로 함수를 정의함
  - 본 단원에선 수식 트리를 만드는데 필요한 함수들을 정의
- 전위, 중위, 후위 순회는 앞에서 다룬대로 참조의 순서만 바꾸면 됨

## 수식 트리의 구성 방법: 그림 이해

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-04-data_structure/fig5.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 후위 표기법의 수식이 들어오면
- 앞에서부터 차례대로 참조하여 숫자가 등장하면 stack에 저장
- 저장하다가 연산자가 나오면 연산자를 root 노드로 하여 트리를 구성
- 구성된 트리를 다시 숫자가 저장되는  stack에 통째로 저장(피연산자처럼!)
  - 저장에는 root 노드의 주소만 저장하면 됨
- 만들어진 하위 트리를 피연산자로 하여 앞의 연산을 반복하고, 상위 트리를 구성
- 전체 완료된 경우 이를 또 stack에 넣음
  - root 주소만 저장

- 후위 표기법 수식에서 먼저 등장하는 피연산자와 연산자를 이용해 트리의 하단부터 구성해가고, 이어서 윗부분을 구성하면 됨

## 수식 트리의 구성 방법: 코드로 옮기기

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-04-data_structure/fig6.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- Rule 1. 피연산자는 무조건 스택으로
- Rule 2. 연산자 만나면 스택에서 피연산자 두개 꺼내서 트리 구성

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-04-data_structure/fig7.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- Rule 3. 형성된 트리는 다시 스택으로 (루트 주소 저장)
- Rule 4. 최종 결과도 스택으로

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-04-data_structure/fig8.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 함수에선 피연산자인지, 아닌지에 따라 if문으로 동작을 구분
  - 피연산자(숫자)가 등장한 경우 문자를 정수로 바꿔서 stack에 저장할 준비를 하고
  - 연산자라면, pop으로 stack에서 두 피연산자를 꺼낸 후 해당 연산자로 tree를 구성 해 stack에 저장할 준비를 하고
- stack에 저장할 준비가 된 값을 stack에 저장 (`Spush(&stack, phone);`)
- 최종적으로 수식 트리의 루트 노드의 주솟값을 반환
  - stack에는 무조건 root 노드의 주솟값이 저장되어있을것이므로 `Spop` 함수를 이용

## 수식 트리의 순회 결과
- 전위 순회하여 데이터를 출력하면 -> 전위 표기법 수식
  - 중위, 후위도 순회 방법에 따라 각각 중위 표기법, 후위 표기법의 수식이 출력됨
- 수식을 수식 트리로 구성하면 전위, 중위, 후위 표기법으로의 수식 표현이 쉬움
- 전위, 중위, 후위 순회하며 출력되는 결과물을 통해 `MakeExpTree` 함수를 검증 할 수 있음

## 수식 트리의 순회 방법

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-04-data_structure/fig9.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 수식 출력 함수는 연산자/피연산자에 따라 구분해 출력

## 수식 트리 관련 예제 실행

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-04-data_structure/fig10.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

## 수식 트리의 계산: 기본 구성

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-04-data_structure/fig11.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 서브 트리가 고려되지 않은 구성

## 수식 트리의 계산: 재귀적 구성

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-04-data_structure/fig12.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 재귀적 구성을 통해 하위 트리를 구성했지만, 최 하단 단말 노드에 대한 고려가 되어있지 않음
  - 즉, 재귀 함수에 탈출 조건이 존재하지 않음

## 수식 트리의 계산

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-04-data_structure/fig13.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 좌, 우 서브트리가 각각 NULL 포인터 주소를 가질 경우, 해당 노드의 데이터를 반환하도록 구성
