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

