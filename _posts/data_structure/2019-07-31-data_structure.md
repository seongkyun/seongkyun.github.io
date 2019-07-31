---
layout: post
title: CH8-1. 트리(Tree)의 개요 2
category: data_structure
tags: [data structure, 자료구조]
comments: true
---

# CH8-1. 트리(Tree)의 개요 2
## 서브 트리의 이해
- 서브 트리는 서브트리로 구성된 재귀적인 구조를 가짐
  - 서브트리는 자신을 부모노드로 하는 또다른 자식노드들을 가짐
- 기본적으로 트리는 자식노드의 갯수제한이 없음

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-07-31-data_structure/fig1.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

## 이진 트리의 이해
- 이진트리는 자식 노드의 갯수가 2개로 제한됨
- 조건
  - 루트 노드를 중심으로 두 개의 서브 트리로 나뉜다
  - 나뉜 두 서브 트리 모두 이진 트리여야 한다.
- __즉, 이진트리는 서브트리를 어떻게 자르던 무조건 이진트리 구조를 갖는다.__
  - 재귀적인 특징을 갖고 있음
- 마지막 단의 노드 하나도 모두 이진트리로 볼 수 있음

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-07-31-data_structure/fig2.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

## 이진 트리와 공집합 노드
- 이진트리? -> 하나의 노드에 최대 두 개의 노드만 추가 가능
  - 규칙을 벗어나지 않기 위해 공집합 노드 개념이 추가됨
- 공집합 노드는 공간은 존재하지만 유효한 데이터가 들어있지 않은 꼴
- 공집합 노드 또한 하나의 노드로 보면, 아래 사진의 모든 구조는 이진트리가 됨

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-07-31-data_structure/fig3.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

## 레벨과 높이, 그리고 포화, 완전 이진 트리
- 트리의 레벨은 최 상단 부모 노드 인덱스를 0으로 시작
- 트리의 높이는 하단으로 뻗어가는, 자식노드로 이어진 길의 길이를 생각하면 됨
  - 트리의 톺이와 레벨의 최대 값은 항상 같음
- 포화 이진 트리?
  - 모든 레벨에 노드가 꽉 차서 더 이상 레벨이 유지된 상태에서 노드 추가가 불가능한 형태
- 완전 이진 트리?
  - 위에서 아래로, 좌에서 우로 채워진 이진 트리를 의미
  - 완전 이진트리는 새로운 노드(공노드에 유효한 데이터)가 추가됨에 따라 포화 이진 트리가 될 수 있음
- 특, 포화 이진트리는 완전 이진 트리지만, 완전 이진 트리는 포화 이진 트리가 될 수 없음.

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-07-31-data_structure/fig4.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>
