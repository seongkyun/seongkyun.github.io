---
layout: post
title: CH3. 연결 리스트 (Linked list) 1-2
category: data_structure
tags: [data structure, 자료구조]
comments: true
---

# CH3. 연결 리스트 (Linked list) 1-2

## 3-2 배열을 이용한 리스트의 구현

### 배열 기반 리스트의 헤더파일 정의
- ""

```c
#pragma warning(disable:4996)
#ifndef __ARRAY_LIST_H__ // 헤더 파일 중복선언 방지
#define __ARRAY_LIST_H__

#define TRUE	1
#define FALSE	0

/*** ArrayList의 정의 ****/
#define LIST_LEN	100
typedef int LData; // 실제로 저장할 데이터의 자료형을 결정하기 위한 typedef 선언

typedef struct __ArrayList // 배열기반 리스트를 정의한 구조체
{
	LData arr[LIST_LEN]; // 리스트의 저장소인 배열
	int numOfData;       // 저장된 데이터의 수
	int curPosition;     // 데이터를 참조하는 위치를 기록
} ArrayList; // 배열 기반 리스트를 의미하는 구조체


/*** ArrayList와 관련된 연산들 ****/
typedef ArrayList List; // 리스트의 변경을 용이하게 하기 위한 typedef 선언

void ListInit(List * plist);
void LInsert(List * plist, LData data);

int LFirst(List * plist, LData * pdata);
int LNext(List * plist, LData * pdata);

LData LRemove(List * plist);
int LCount(List * plist);

#endif
```

### 배열 기반 리스트의 초기화

```c
void ListInit(List * plist) // 구조체 변수가 선언되고 그 주솟값이 인자로 전달
{
	(plist->numOfData) = 0; // 해당 주소가 가리키는 구조체의 numOfData 멤버를 0으로 초기화
	(plist->curPosition) = -1; // LF/LN 함수에서 사용하며, -1은 아무런 위치도 참조하지 않았음을 의미
}

```

- `(plist->curPosition) = -1;`에서 -1은 아무런 위치도 참조하지 않았음을 의미
- `(plist->curPosition)`가 0일 때 첫 번째 데이터의 참조가 진행되었음을 의미
  - LFirst 함수를 참조하고 나면 0이 됨(첫 번째 인덱스 참조했다는 의미)

### 배열 기반 리스트의 삽입

```c
typedef struct __ArrayList
{
	LData arr[LIST_LEN]; // 실제로 데이터가 저장되는 위치
	int numOfData; // 0으로 초기화된 상태
	int curPosition; // -1로 초기화된 상태
} ArrayList;
```

- `int numOfData`는 저장된 데이터가 없으므로 0
  - 실제 다음 데이터가 0번째 인덱스에 저장 되어야 한다는 것을 의미
  - 이를 이용해 실제 다음 데이터가 저장될 인덱스를 이 값을 이용해 얻어 올 수 있음

```c
void LInsert(List * plist, LData data)
{
	if(plist->numOfData > LIST_LEN) // 더 이상 저장할 공간이 없는 경우
	{
		puts("저장이 불가능합니다.");
		return;
	}

	plist->arr[plist->numOfData] = data;  // 데이터 저장
	(plist->numOfData)++; // 저장된 데이터의 수 증가(다음 데이터가 저장될 인덱스 값을 의미함)
}
```

### 배열 기반 리스트의 조회
- LFirst 함수와 LNext 함수의 가장 큰 차이점은 `(plist->curPosition)`의 정의 및 사용
  - LFirst는 `(plist->curPosition)`를 0으로 초기화하고 0번째 인덱스의 값을 참조
  - LNext는 `(plist->curPosition)` 값에 해당하는 인덱스의 값을 참조하고 `(plist->curPosition)`의 크기를 1 증가
  

```c
int LFirst(List * plist, LData * pdata) // 초기화 및 첫 번째 데이터 참조
{
	if(plist->numOfData == 0) // 저장된 데이터가 하나도 없다면 False 반환
		return FALSE;

	(plist->curPosition) = 0; // 참조 위치 초기화(첫 번째 데이터 참조)
	*pdata = plist->arr[0]; // pdata가 가리키는 공간에 데이터 저장
	return TRUE;
}

int LNext(List * plist, LData * pdata)  // 그 다음 데이터 참조
{
	if(plist->curPosition >= (plist->numOfData)-1)  // 더 이상 참조할 데이터가 없는경우
		return FALSE;

	(plist->curPosition)++; // 현재 위치 1 증가
	*pdata = plist->arr[plist->curPosition];  // 값의 반환은 매개변수로, 함수의 반환은 성공여부 반환
	return TRUE;
}
```

### 배열 기반 리스트의 삭제
- 배열의 삭제 원칙
  - {A, B, C, D, E, F, G, ...} 에서 C를 삭제
    - C를 제거하고 뒤의 데이터를 하나씩 앞칸으로 이동시킴
    - {A, B, , D, E, F, G, ...} -> {A, B, D, E, F, G, H, ...}
  - curPosition의 값 또한 한칸 앞으로 당겨줘야 함
  - 삭제가 되는 데이터의 값은 반환되어야 함 (원칙)

```c
LData LRemove(List * plist)
{
	int rpos = plist->curPosition;  // 삭제할 데이터 인덱스 값 참조
	int num = plist->numOfData;
	int i;
	LData rdata = plist->arr[rpos]; // 삭제할 데이터 임시 저장

	for(i=rpos; i<num-1; i++) // 삭제를 위한 데이터의 이동을 진행
		plist->arr[i] = plist->arr[i+1];

	(plist->numOfData)--; //데이터의 수 감소
	(plist->curPosition)--; // 참조위치를 하나 앞으로
	return rdata; // 삭제 데이터 반환
}
```

### 리스트에 구조체 변수 저장하기
- 자료형의 저장 대상을 바꿔보는게 핵심 내용
- Point 라는 내용의 구조체를 정의 및 ADT를 "Point.h"에 정의

```c
typedef struct_point
{
  int xpos;
  int ypos;
} Point;

void SetPointPos(Point *ppos, int xpos, int ypos); // Point 변수의 xpos, ypos 설정
{
  ppos->xpos = xpos;
  ppos->ypos = ypos;
}
void ShowPointPos(Point *ppos); // Point 변수의 xpos, ypos 정보 출력
{
  printf("[%d, %d]\n", ppos->xpos, ppos->ypos);
}
int PointComp(Point *pos1, Point *pos2);  // 두 Point 변수 비교
{
  if(pos1->xpos == pos2->xpos && pos1->ypos == pos2->ypos)
    return 0;
  else if(pos1->xpos == pos2->xpos)
    return 1;
  else if(pos1->ypos == pos2->ypos)
    return 2;
  else
    return -1;
}
```

- 위처럼 정의된 Point 구조체의 주솟값을 저장 해 보기
  - "ArrayList.h" 에서 변경사항 두가지
  - `typedef int LData;` 를 `typedef Point *LData;`로
  - `#include "Point.h"`추가
  
- 리스트 구조체 선언 및 초기화, 저장
  - 동적할당 이용
  - ppos 변수에 malloc으로 할당된 주소값을 임시 저장 후 list에 저장(LInsert)
  
```c
List list;
Point compPos;
Point * ppos;

ListInit(&list);

/*** 4개의 데이터 저장 ***/
ppos = (Point*)malloc(sizeof(Point)); //Point 구조체 변수를 동적할당
SetPointPos(ppos, 2, 1);
LInsert(&list, ppos);  // 리스트에 주소값 저장
...
```

- 저장된 데이터 참조 및 조회

```c
printf("현재 데이터의 수: %d \n", LCount(&list));

if(LFirst(&list, &ppos))
{
  ShowPointPos(ppos);

  while(LNext(&list, &ppos))
    ShowPointPos(ppos);
}
printf("\n");
```

- 저장된 데이터 삭제
  - xpos가 2인 모든 데이터 삭제
  - LRemove 함수가 주솟값을 반환하기에 동적 할당 메모리의 해제가 가능
  - 리스트 자료구조에 주솟값을 저장 = 주솟값을 실질적인 데이터로 판단
  
```c
compPos.xpos=2; // 기준이 될 Point 구조체 멤버의 변수 설정
compPos.ypos=0;

if(LFirst(&list, &ppos))
{
  if(PointComp(ppos, &compPos)==1) // 만약 ppos가 가리키는 동적할당된 구조체의 멤버변수와 compPos의 xpos와 같다면
  {
    ppos=LRemove(&list);  // 해당 구조체 변수 삭제
    free(ppos); // 동적 할당 메모리 해제
  }

  while(LNext(&list, &ppos)) 
  {
    if(PointComp(ppos, &compPos)==1)
    {
      ppos=LRemove(&list); 
      free(ppos);
    }
  }
}
```

### 배열 기반 리스트의 장점과 단점
- 배열 기반 리스트의 단점(다음 챕터에서 이를 보완하는 새로운 리스트를 배움)
  - 배열의 길이가 초기에 결정되어야 한다. 변경이 불가능하다.
  - 삭제의 과정에서 데이터의 이동(복사)가 매우 빈번히 일어남
- 배열 기반 리스트의 장점
  - 데이터 참조가 쉽다. 인덱스 값 기준으로 어디든 한 번에 참조 가능
