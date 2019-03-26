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
<img src="/assets/post_img/data_structure/2019-03-26-data_structure/fig_17.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 사칙연산 우선순위, 괄호연산을 포함하여 계산의 우선순위를 따져서 계산이 가능해야 함
  - 소괄호를 파악하여 그 부분을 먼저 연산
  - 연산자의 우선순위(+,- 연산보다 \*, / 연산 먼저)를 근거로 연산의 순위를 결정
- 계산기 구현 자체는 Stack의 알고리즘과 별개지만, 알고리즘의 구현에 Stack이 매우 중요하게 쓰임
  - 계산기 연산의 구현만큼 Stack의 활용능력을 잘 보여주는 예시는 없음!
  
### 세 가지 수식의 표기법: 전위, 중위, 후위

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-26-data_structure/fig_18.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 중위 표기법: 연산자를 중간에 표기
  - 수식 내에 연산의 순서에 대한 정보가 담겨있지 않음
  - 상대적으로 구현하기 어려움
- 전위 표기법: 연산자를 앞에 표기
  - 수식 내에 연산의 순서가 반영되어 있음
- 후위 표기법: 연산자를 뒤에 표기
  - 수식 내에 연산의 순서가 반영되어 있음
- 전위/후위 표기법이 상대적으로 구현 난이도가 낮음
  - 전위/후위 표기법은 연산자의 우선순위 뿐만 아니라 소괄호의 우선순위까지 반영되어있음
  - 즉, 나열되어있는 순서대로 연산하면 됨
- 후위 연산자의 연산방법
  - ex. 중위 표기법 5 + 2 / 7 -> 후위 표기법 5 2 7 / +
    - 2, 7을 /로 연산 후 그 결과와 5를 + 연산
- 중위 연산자 형태로 받은 입력을 변형하여 후위 표기법으로 만들고, 만들어진 후위 표기법 식을 계산하여 계산기를 완성

### 중위 -> 후외: 소괄호 고려하지 않는 경우

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-26-data_structure/fig_19.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 입력으로 저장된 중위 표기법의 수식을 이용하여 후위 표기법으로 변환
- 입력으로 저장된 수식을 왼쪽 문자부터 시작해서 하나씩 처리해나감
- 피 연산자를 만나면 무조건 변환된 수식이 위치할 자리로 이동시킴
- 연산자는 무조건 가운데 쟁반(Stack)으로 우선 이동

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-26-data_structure/fig_20.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 가운데 쟁반(Stack)으로 보내지는 연산자는 쟁반으로 갈지 아니면 쟁반의 연산자가 변환된 수식이 있는곳으로 옮겨진 다음 새 연산자가 들어갈지 결정해야 함
- 숫자는 무조건 변환된 수식이 위치할 자리로 이동

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-26-data_structure/fig_21.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 쟁반(Stack)에는 연산자가 올라가는 경우가 있고 그렇지 않은 경우가 존재
- 연산자 기준
  - 쟁반(Stack)에 위치한 연산자의 우선순위가 높다면
    - 쟁반에 위치한 연산자를 꺼내서 변환된 수식이 위치할 자리로 옮김
    - 새 연산자는 쟁반으로 옮김
  - 쟁반(Stack)에 위치한 연산자의 우선순위가 낮다면
    - 쟁반에 위치한 연산자의 위에 새 연산자를 쌓는다
  - 즉, 연산자의 우선순위가 낮은게 높은것 위에 있을 수 없음
    - 무조건 우선순위가 낮은것 위에 높은것이 쌓이게 되어야 함

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-26-data_structure/fig_22.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 본래 수식의 숫자가 다 들어갔으면 마지막으로 쟁반(Stack)에서 차례대로 연산자를 꺼내 붙임

### 중위 -> 후위: 정리

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-26-data_structure/fig_23.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 피 연산자는 그냥 옮긴다
- 연산자는 우선 쟁반(Stack)으로 옮긴다
- 연산자가 쟁반(Stack)에 있다면, 우선순위를 비교하여 처리방법을 결정한다
  - 쟁반에 들어가려는 연산자가 쟁반에 들어있는 연산자보다 우선순위가 높다면 쟁반에 쌓는다
  - 쟁반에 들어가려는 연산자가 쟁반에 들어있는 연산자보다 우선순위가 낮다면 우선순위가 높은 연산자를 꺼내 변환된 수식의 뒤에 붙인 후 쟁반에 들어가려는 연산자를 넣는다
- 피연산자를 다 옮긴 후 쟁반에 남아있는 연산자들을 하나씩 꺼내서 모두 옮긴다

### 중위 -> 후위: 고민 될 수 있는 상황

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-26-data_structure/fig_24.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 상황 1: 만약 쟁반(Stack)에 동등한 우선순위를 갖는 연산자가 존재할 경우?
  - 기존에 쟁반(Stack)에 들어있던 연산자를 꺼내 변환된 수식에 붙이고, 들어가려는 새 연산자를 넣는다.
- 상황 2: 만약 쟁반(Stack)에 연산 우선순위대로 쌓여있는 상태에서 새 연산자가 중요도가 stack의 top보다 떨어지면?
  - 기존에 쟁반(Stack)에 들어있던 연산자를 모두 꺼내 변환된 수식에 붙이고, 들어가려는 새 연산자를 넣는다.

### 중위 -> 후위: 소괄호 고려

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-26-data_structure/fig_25.jpg" alt="views">
<img src="/assets/post_img/data_structure/2019-03-26-data_structure/fig_26.jpg" alt="views">
<img src="/assets/post_img/data_structure/2019-03-26-data_structure/fig_27.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 소괄호를 처리하며, 아래 예제는 이해를 위해 소괄호가 맨 앞에 나온 상황에 대한 예시이지만 실제로는 앞에 다른 연산이 있어도 정상 작동함
- 소괄호 안에 있는 연산자들이 후위 표기법의 수식에서 앞부분에 위치해야 함
- 이를 위해 소괄호의 시작을 알리는 부분을 `newfloor`로 하여 쟁반(Stack)에 쌓고, 그 위로 소괄호 안의 연산에 대한 동일 작업 수행
  - `newfloor`의 역할은 수식에서 "("가 하게 되며, ")"가 등장하면 소괄호 수식에 대한 연산이 끝난것을 의미
- 위 그림의 기본논리 숫자 순서대로 계산이 진행됨
  - 식: (1 + 2 \* 3) / 4
    - Stack: empty
    - 변환된 수식: empty
  - 1: "("가 등장(괄호가 시작)하므로 쟁반에 newfloor 생성
    - Stack: (_newfloor <- top
    - 변환된 수식: 
  - 2: 1이 변환된 수식으로 이동
    - Stack: (_newfloor <- top
    - 변환된 수식: 1
  - 3: +가 newfloor 위에 쌓임 (newfloor 위는 비어있는 상태이므로 그냥 들어감)
    - Stack: (_newfloor, + <- top
    - 변환된 수식: 1
  - 4: 2가 변환된 수식으로 이동
    - Stack: (_newfloor, + <- top
    - 변환된 수식: 1, 2
  - 5: \*가 newfloor 위에 쌓임 (바로 아래의 +보다 우선순위가 높은 연산이므로)
    - Stack: (_newfloor, +, \* <- top
    - 변환된 수식: 1, 2
  - 6: 3이 변환된 수식으로 이동
    - Stack: (_newfloor, +, \* <- top
    - 변환된 수식: 1, 2, 3
  - 7: ")"가 등장(괄호의 끝)하므로 쟁반에 newfloor 윗부분의 연산에 대한 변환된 수식으로의 이동
    - Stack: empty
    - 변환된 수식: 1, 2, 3, \*, +
  - 8: /가 쟁반으로 이동
    - Stack: / <- top
    - 변환된 수식: 1, 2, 3, \*, +
  - 9: 4가 변환된 수식으로 이동
    - Stack: / <- top
    - 변환된 수식: 1, 2, 3, \*, +, 4
  - 10: 원래 식이 모두 끝났으므로 쟁반(Stack)을 비움
    - Stack: empty
    - 변환된 수식: 1, 2, 3, \*, +, 4, /
- 식에서 "("은 또 다른 바닥, ")"은 변환되어야 하는 수식의 끝을 의미한다고 생각하면 됨
  - ")" 연산자를 만나면 "("를 만날 때 까지 연산자를 이동시키고, 만나면 newfloor를 만들고 새로 그 위에 쌓으면 됨
  
- __해당 챕터 연습문제 풀어보기!__
  - 후위 표기법의 수식으로 바꾸는 
