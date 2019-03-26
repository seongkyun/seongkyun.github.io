---
layout: post
title: CH6. 스택 (Stack) 2
category: data_structure
tags: [data structure, 자료구조]
comments: true
---

# CH6. 스택 (Stack) 2

## 6-2 스택의 연결 리스트 기반 구현

### 문제 6-1
- 총 5개의 파일로 구성
  - `CLinkedList.h`, `CLinkedList.c`: 원형 연결 리스트 구현파일
    - tail 멤버를 갖고 있으며, 꼬리쪽(tail)에 새 노드가 연결되는 변형된 원형 연결 리스트
  - `CLLBaseStack.h`, `CLLBaseStack.c`: 스택 구현결과
  - `CLLBaseStackMain.c`: 메인파일
- 구현되어있는 원형 연결 리스트 파일을 가져다가 이용하여 스택을 구현

- `CLLBaseStack.h` 파일
  - 원형 연결 리스트를 이용한 스택 구조체 정의 및 ADT 정의

```c
#ifndef __CLL_STACK_H__
#define __CLL_STACK_H__

#include "CLinkedList.h"

#define TRUE	1
#define FALSE	0

typedef int Data;

typedef struct _listStack
{
  List * plist;
} ListStack;


typedef ListStack Stack;

void StackInit(Stack * pstack);
int SIsEmpty(Stack * pstack);

void SPush(Stack * pstack, Data data);
Data SPop(Stack * pstack);
Data SPeek(Stack * pstack);

#endif
```

- `CLLBaseStack.c` 파일
  - 원형 연결 리스트를 이용한 스택의 구현

```c
#include <stdio.h>
#include <stdlib.h>
#include "CLinkedList.h"
#include "CLLBaseStack.h"

// 스택의 초기화
void StackInit(Stack * pstack)
{
  pstack->plist = (List*)malloc(sizeof(List)); // 입력된 스택을 구현할 새 원형 연결 리스트를 할당
  ListInit(pstack->plist); // 할당된 새 원형 리스트를 초기화(CLinkedList 구현)
}

// 스택이 비어있는지 판단
int SIsEmpty(Stack * pstack)
{
  if(LCount(pstack->plist)==0) // 스택이 비어있어서 입력된 스택을 구현하는 원형 연결 리스트의 멤버 갯수가 0개면 True 반환
    return TRUE;
  else
    return FALSE;
}

// 스택에 값을 추가
void SPush(Stack * pstack, Data data)
{
  LInsertFront(pstack->plist, data); // LInsertFront 함수를 이용하여 원형 연결 리스트의 앞에 새 값을 추가
}

// 스택의 값을 참조한 후 삭제
Data SPop(Stack * pstack)
{
  Data data; // pop할 값
  LFirst(pstack->plist, &data); // 스택의 첫 번째 값만 참조하면 되므로 LFirst만 필요, 참조를 통한 pstack->plist->cur의 값 참조 위치로 초기화
  LRemove(pstack->plist); // 참조된 값 삭제, LRemove함수는 현재 위치(cur)의 값만 삭제함
  return data; // 참조된 값 반환
}

// 스택의 값을 참조
Data SPeek(Stack * pstack)
{
  Data data;
  LFirst(pstack->plist, &data); // SPop와 동일하지만 삭제가 되지 않음
  return data;
}
```

## 6-4 계산기 프로그램 구현
- 앞에 구현된 Stack의 내용을 갖고 계산기를 구현

### 구현할 계산기 프로그램의 성격

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig_17.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 사칙연산 우선순위, 괄호연산을 포함하여 계산의 우선순위를 따져서 계산이 가능해야 함
  - 소괄호를 파악하여 그 부분을 먼저 연산
  - 연산자의 우선순위(+,- 연산보다 \*, / 연산 먼저)를 근거로 연산의 순위를 결정
- 계산기 구현 자체는 Stack의 알고리즘과 별개지만, 알고리즘의 구현에 Stack이 매우 중요하게 쓰임
  - 계산기 연산의 구현만큼 Stack의 활용능력을 잘 보여주는 예시는 없음!
  
### 세 가지 수식의 표기법: 전위, 중위, 

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig_18.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig_19.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig_20.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig_21.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig_22.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig_23.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig_24.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig_25.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig_26.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-25-data_structure/fig_27.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>
