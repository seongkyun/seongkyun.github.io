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
	*pdata = plist->arr[plist->curPosition];  // 값의 반환은 매개변수로, 함수의 반환은 성공여부를 
	return TRUE;
}
```
