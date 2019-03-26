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
  - `CLLBaseStack.h`, `CLLBaseStack.c`: 스택 구현결과
  - `CLLBaseStackMain.c`: 메인파일
- 구현되어있는 원형 연결 리스트 파일을 가져다가 이용하여 스택을 구현

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

```c
#include <stdio.h>
#include <stdlib.h>
#include "CLinkedList.h"
#include "CLLBaseStack.h"

void StackInit(Stack * pstack)
{
  pstack->plist = (List*)malloc(sizeof(List));
  ListInit(pstack->plist);
}

int SIsEmpty(Stack * pstack)
{
  if(LCount(pstack->plist)==0)
    return TRUE;
  else
    return FALSE;
}

void SPush(Stack * pstack, Data data)
{
  LInsertFront(pstack->plist, data);  
}

Data SPop(Stack * pstack)
{
  Data data;
  LFirst(pstack->plist, &data);
  LRemove(pstack->plist);
  return data;
}

Data SPeek(Stack * pstack)
{
  Data data;
  LFirst(pstack->plist, &data);
  return data;
}
```

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-26-data_structure/fig1.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>
