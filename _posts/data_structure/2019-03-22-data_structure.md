---
layout: post
title: CH5. 연결 리스트 (Linked list) 3-2
category: data_structure
tags: [data structure, 자료구조]
comments: true
---

# CH5. 연결 리스트 (Linked list) 3-2

## 5-2 양방향 연결 리스트
- 단일 연결리스트 -> 단일 연결리스트 + Dummy node -> 원형 연결리스트 -> 양방향 연결리스트
  - 양방향 연결리스트가 앞의 모든 내용을 총괄하는 내용!

### 양방향 연결 리스트의 이해

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-22-data_structure/fig1.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 양방향 연결 리스트는 노드를 양쪽 참조 가능해야 하기에 자료형의 구조가 다름
  - 각각 오른쪽과 왼쪽 노드의 참조가 가능하게끔 두 개의 포인터 변수가 존재

```c
typedef struct _node
{
  Data data;
  struct _node *next;
  struct _node *prev;
} Node;
```

- 기본적인 양방향 연결 리스트
  - 양방향의 연결 구조를 갖고 있음
  - Head 말고도 필요에 따라 tail도 추가 가능
- 더미 노드 양방향 연결 리스트
  - Head 또는 tail 포인터가 dummy node를 가리키도록 설정 가능
    - 각각 dummy node가 존재할경우 dummy node를 가리키는 head/tail 포인터가 필요
- 원형 연결 기반의 양방향 연결 리스트
  - 원형 연결 리스트를 기반으로 한 양방향 연결리스트
  - 보편적인 라이브러리에서 많이 사용되는 타입

- 노드가 양방향으로 연결된다는 점만 지켜진다면 다양한 형태로 양방향 연결리스트를 구성/정의 가능

### 양방향으로 노드를 연결하는 이유

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-22-data_structure/fig2.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 기존에는 포인터에 대한 연산이 1문장으로 끝
- 양방향은 2배 더 소요
- 하지만 `before` 포인터 변수가 필요없어짐
  - 굉장한 이점!
- next 멤버를 통해 오른쪽으로 이동, before 없이 previous를 이용해 왼쪽으로 이동 및 삭제 가능
  - 삭제 시 필요한 before를 prev 포인터가 대체 가능하므로 before의 필요성이 없어짐
- 실제로 양방향 연결 리스트 구현이 그렇게 어렵지 않음!

### 구현할 양방향 연결 리스트 모델

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-22-data_structure/fig3.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 구현의 복잡도때문에 LRemove 함수를 ADT에서 제외시킴!
  - 문제 5-2 풀이를 통해 직접 구현해보기
- 대신 왼쪽 노드의 데이터를 참조하는 LPrevious 함수를 ADT에 추가
- 새 노드는 머리쪽에 추가시키는 방식을 따름

- 문제 5-2 풀어보기!

### 양방향 연결 리스트의 헤더파일

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-22-data_structure/fig4.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- DBLinkedList를 정의하는 구조체를 보면 head와 cur만 존재하며 before가 사라진 것을 확인 가능
  - Node를 정의하는 구조체에 prev 포인터 멤버변수가 before 역할 대체 가능함
- LPrevious 함수는 LNext 함수와 반대로 왼쪽으로 이동(이전 값 참조)

### 양방향 연결 리스트 활용의 예

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-22-data_structure/fig5.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 항상 코드를 볼땐 ADT를 보고 main함수를 먼저 본 다음 세부 함수 보기!

- Head 추가 방식을 따르므로 1~8 입력시 8~1 순서대로 출력됨
- 각 LNext와 LPrevious 함수는 주어진 리스트에 참조할 값이 없으면 False를 반환

### 연결 리스트의 구현: 리스트의 초기화

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-22-data_structure/fig6.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- DBLinkedList의 멤버중 반드시 초기화가 필요한 객체에 대한 초기화를 진행
- cur은 조회 과정에서 초기화되므로 구지 초기화 필요없음

### 연결 리스트의 구현: 노드 삽입의 구분

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-22-data_structure/fig7.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- SOC을 따라 큰 문제를 잘게 쪼개서 해석하는것이 프로그래밍에서는 중요
  - 새 노드 삽입의 문제를 첫 번째 노드의 추가와 두 번째 이후 노드의 추가로 나누어서 고려
- 두 번째 이후의 노드 추가에서부턴 양쪽 연결을 형성시켜주는 정의가 반드시 필요
- 더미 노드를 기반으로 하지 않기 때문에 노드의 삽입 방법은 두가지로 구분됨
  - 더미 노드가 있으면 구지 구분 필요 없음

### 연결 리스트의 구현: 첫 번째 노드의 삽입

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-22-data_structure/fig8.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- `newNode->next = plist->head;`에서 새 노드의 next가 NULL로 초기화 되지 않는 이유?
  - 두 번째 이후 노드부턴 당연히 plist->head로 초기화 됨
    - 새 노드와 기존 노드의 연결 역할
  - 또한 첫 번째 노드를 추가 시 head 포인터가 NULL로 이미 초기화 되어있으므로 결국 plist->head는 NULL임

### 연결 리스트의 구현: 두 번째 이후 노드의 삽입

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-22-data_structure/fig9.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 코드에서 1의 과정을 통해 새 노드의 next가 head가 가리키던 노드를 가리키도록 설정
- 만약 head 포인터가 NULL일 경우 그대로 끝(첫 번째 노드의 추가 과정이므로)
  - 만약 head가 NULL이 아닐경우(두 번째 이후 노드의 추가 과정)
    - 코드에서 2의 과정을 통해 prev가 새 노드를 가리키도록 설정
- 새 노드는 head쪽에 추가되었으므로 newNode->prev=NULL이 됨
- 새 노드가 새로운 head이므로 plist->head = newNode로 설정

### 연결 리스트의 구현: 데이터 조회

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-22-data_structure/fig10.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- LFirst, LNext는 단방향 연결리스트와 차이가 없고 LPrevious는 LNext와 이동 방향만 다름

### 문제 5-2
- 조건
  - Head, tail 포인터 변수 사용
  - 더미 노드를 사용
  - 새 노드는 꼬리에 추가
- 위 조건을 만족하는 LRemove 함수를 포함하는 양방향 연결 리스트를 구현

- __풀이__
- 모든 상태는 그림을 그려가며 예시를 들어 풀면 쉽게 풀린다.
- 초기상태, 노드의 추가, 노드의 삭제에 대해 dummy node가 존재하므로 이를 기반으로 일반적인 사례를 예로 들어 포인터변수를 초기화하면 된다.

- 초기상태
  - 리스트가 생성되고 초기화가 완료된 상태
  
(head) -> [dummy] <-> [dummy] <- (tail)

- 노드의 추가
 - 1단계: 새 노드를 생성하고 데이터를 저장
 - 2단계: 새 노드와 새 노드의 왼쪽에 위치할 노드가 서로 가리키도록 함
 - 3단계: 새 노드와 새 노드의 오른쪽에 위치할 노드가 서로를 가리키도록 함
 
(head) -> [dummy] <-> [2, old] <-> [4, new] <-> [dummy] <- (tail)

- 노드의 삭제
  - 위 노드관계에서 [2, old] 노드를 삭제하는 과정
  - 1단계: head가 가리키는 dummy node의 next가 [4, new]를 가리키도록 함
  - 2단계: [4, new]의 prev가 head가 가리키는 dummy node를 가리키도록 함
  - 3단계: [2, old] 노드를 삭제하고 

```c
#include <stdio.h>
#include <stdlib.h>
#include "DBLinkedList.h"

void ListInit(List * plist)
{
  plist->head = (Node *)malloc(sizeof(Node));
  plist->tail = (Node *)malloc(sizeof(Node));

  plist->head->prev = NULL;
  plist->head->next = plist->tail;

  plist->tail->prev = plist->head;
  plist->tail->next = NULL;

  plist->numOfData = 0;
}

void LInsert(List * plist, Data data)
{
  Node * newNode = (Node*)malloc(sizeof(Node));
  newNode->data = data;

  newNode->prev = plist->tail->prev;
  plist->tail->prev->next = newNode;

  newNode->next = plist->tail;
  plist->tail->prev = newNode;

  (plist->numOfData)++;
}

int LFirst(List * plist, Data * pdata)
{
  if (plist->head->next == plist->tail)
    return FALSE;

  plist->cur = plist->head->next;
  *pdata = plist->cur->data;

  return TRUE;
  }

int LNext(List * plist, Data * pdata)
{
  if(plist->cur->next == plist->tail)
    return FALSE;

  plist->cur = plist->cur->next;
  *pdata = plist->cur->data;

  return TRUE;
}

int LPrevious(List * plist, Data * pdata)
{
  if(plist->cur->prev == plist->head)
    return FALSE;

  plist->cur = plist->cur->prev;
  *pdata = plist->cur->data;

  return TRUE;
}

int LCount(List * plist)
{
  return plist->numOfData;
}

Data LRemove(List *plist)
{
  Node *rpos = plist->cur;
  Data rm = rpos->data;

  plist->cur->prev->next = rpos->next;
  plist->cur->next->prev = rpos->prev;

  plist->cur = plist->cur->prev;

  free(rpos);
  (plist->numOfData)--;
  return rm;
}
```
