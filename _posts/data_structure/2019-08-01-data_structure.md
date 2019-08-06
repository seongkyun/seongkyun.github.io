---
layout: post
title: CH8. 트리(Tree) 2 (이진 트리의 구현)
category: data_structure
tags: [data structure, 자료구조]
comments: true
---

# CH8. 트리(Tree) 2 (이진 트리의 구현)
- 이진 트리를 구현할 수 있는 '도구'를 만드는 작업

## 이진 트리 구현의 두 가지 방법

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-01-data_structure/fig1.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 배열 기반의 방법
  - 노드에 번호를 부여하고, 해당하는 값을 배열의 인덱스로 사용
    - 완전 이진트리 순서와 동일하게 인덱싱
    - 접근이 용이함
  - 편의상 배열의 첫 번째 요소는 사용하지 않음
  - 배열을 이용했을 때 쉬운 적용상황이 있으므로 쓸 데가 있음
  - 배열 기반의 이진 트리의 구조와 실제 리스트 구현에는 구조가 일지하지 않음
- 연결 리스트를 이용한 방법
  - 트리의 구조와 실제 구현된 구조가 일치
    - 양방향 연결리스트로 구현
  - 직관적인 이해가 좋은 편

## 헤더파일에 정의된 구조체의 이해
- 양방향 연결 리스트의 구현에는 두 구조체가 정의되어야 함
  - 구조체 1. 노드 구조체
  - 구조체 2. 양방향 연결리스트 구조체
- 이진트리는 노드 자체가 이진트리이므로 하나의 구조체로 정의됨
- 핵심

```
{%raw%}
typedef struct _bTreeNode
{
  BTData data;
  struct _bTreeNode *left;
  struct _bTreeNode *right;
} BTreeNode;
{%endraw%}
```

- 위의 구조체가 노드 자체이자 이진 트리이며, 모든 노드는 직/간접적으로 연결되어 있음
  - 즉, 루트 노드의 주소값만 이용하여 트리 내 모든 노드에 접근 가능함

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-01-data_structure/fig2.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

## 헤더파일에 선언된 함수들

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-01-data_structure/fig3.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-01-data_structure/fig4.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 함수 이름에 `SubTree`가 들어가는 이유?
  - 노드 자체도 하나의 하위 이진트리로 보기때문에 더 올바른 표현!

## 정의된 함수들의 이해를 돕는 main 함수

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-01-data_structure/fig5.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 앞에서 정의된 함수들을 이용해 이진트리를 만드는 도구(노드/이진트리) 정의하고, main 함수에서 노드를 만들고 노드간에 연결을 하여 이진 트리를 완성

## 이진 트리의 구현

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-01-data_structure/fig6.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 구현 자체에는 큰 어려움이 없으나, 기존에 연결된 노드를 삭제하는 과정에 대한 학습이 필요
  - 말단 노드의 경우 코드처럼 `if`문에서 `free`를 이용하여 간단하게 구현 가능
  - 하지만, 하위 노드가 존재하는 경우 해당 노드만 삭제하면 memory leakage가 발생!
    - __이진 트리의 순회로 해결__
    - __나중에 memory leakage 없게 free 될 수 있도록 구현해보기__

## 이진 트리 관련 main 함수

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-08-01-data_structure/fig7.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 트리를 완전히 소멸시키는 방법으로는 이진 트리의 순회를 사용하며, 다음 세션에서 공부하고 적용해보기.
