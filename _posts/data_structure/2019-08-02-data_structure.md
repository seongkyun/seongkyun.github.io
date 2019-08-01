---
layout: post
title: CH8-3. 이진 트리의 순회 (Traversal)
category: data_structure
tags: [data structure, 자료구조]
comments: true
---

# CH8-3. 이진 트리의 순회 (Traversal)
- 재귀를 사용하면 어렵지 않게 구현 가능

## 순회의 세 가지 방법

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-02-data_structure/fig1.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 순회 기준은 루트 노드를 언제 방문하느냐에 있음
  - 루트 노드 방문하는 시점에 따라 중위, 후위, 전위 순회로 나뉨
- 각 하위 노드는 노드가 아니라 하나의 또다른 이진트리로 보면 됨
  - 각 이진 트리 접근 시 최 말단 이진트리의 접근이 끝나야 상위 트리의 접근이 가능함

## 순회의 재귀적 표현

- 중위 순회
  - 왼쪽 서브 트리의 순회 -> 루트 노드 방문 -> 오른쪽 서브 트리의 순회 순으로 참조

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-02-data_structure/fig2.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 위 코드는 탈출 조건이 명시되지 않은 불완전한 코드

## 순회의 재귀적 표현의 완성

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-02-data_structure/fig3.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 탈출 조건
  - 왼쪽/오른쪽에 하위 이진트리가 아닌 노드가 있는 경우 탈출

## 결과물

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-02-data_structure/fig4.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 하위 트리 모두를 참조한 후 상위 트리가 순차적으로 참조됨

## 전위 순회와 후위 순회

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-02-data_structure/fig5.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 동일한 코드에서 순서만 바꾸면 됨

## 노드의 참조를 자유롭게 구성하기

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-02-data_structure/fig6.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 여태까지 코드는 노드의 참조를 print로만 한정시켜 동작
- 실제로는 포인터함수를 이용하여 다양한 연산이 가능함
  - 즉, 트리의 사용자가 노드 방문 목적을 직접 정의 할 수 있음
- 위 예제에선 void 형 함수를 사용했지만, int형 등 다양하게 구성하여 사용 



