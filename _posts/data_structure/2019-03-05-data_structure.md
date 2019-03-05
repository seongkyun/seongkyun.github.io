---
layout: post
title: CH2. 재귀(Recursion)
category: data_structure
tags: [data structure, 자료구조]
comments: true
---

# CH2. 재귀(Recursion)

## 2-1 함수의 재귀적 호출의 이해
- C언어에서의 재귀 시점이 아닌 자료구조의 시점에서 학습하는 재귀의 적용

### 재귀합수의 기본적인 이해
- 탈출 조건이 성립하여야만 함수에서 탈출 할 수 없음
- 탈출 조건이 성립하지 않는 구조가 될 경우 무한 loop에 빠지게 됨

### 재귀함수 디자인 사례
- n의 팩토리얼(n!)을 구하는 함수를 작성하면 다음과 같이 됨

```c
int Factorial(int n)
{
  if (n==0)
    return 0;
  else
    return n * Factorial(n-1);
}
```

## 2-2 재귀의 활용
- 재귀를 효율적으로 활용하기 위한 사고의 전환을 위한 소단원

### 피보나치 수열: Fibonacci Sequence
- 피보나치 수열은 앞의 수 두개를 더하여 다음 수를 만들어가는 수열임
  - 처음 두 수는 0, 1로 주어짐
  - 0, 1, 1, 2, 3, 5, 8, 13, ...
- 위의 특징을 활용하여 피보나치 수열 함수를 작성하면 아래와 같이 됨

```c
int Fibo(int n)
{
  if (n==1)
    return 0;
  else if (n==2)
    return 1;
  else
    return Fibo(n-1) + Fibo(n-2);
}
```

- 재귀적으로 정의된 함수의 호출 순서를 완벽히 나열하려고 할 필요는 없음
  - 더 복잡해질 뿐..

### 이진 탐색 알고리즘의 재귀적 구현
- Chapter 1에서 다룬 이진 탐색 알고리즘을 재귀함수 기반으로 재구현
  - 알고리즘의 논리 자체가 수의 비교라는 반복적 흐름을 갖고있으므로 가능
- 반복 패턴은 아래와 같이 정의됨
  - 탐색 범위의 중앙에 목표 값이 저장되었는지 확인
  - 저장되지 않았다면 탐색 범위를 반으로 줄여서 다시 탐색 시작
  - 탐색 범위의 시작위치를 의미하는 first가 탐색 범위의 끝을 의미하는 last보다 커지는 경우 탐색 종료
- 위의 내용을 기반으로 이진 탐색 알고리즘을 재귀적으로 구현하면 다음과 같다

```c
int BSearchRecur(int ar[], int first, int last, int target)
{
  if(first > last)
    return -1
  mid = (first + last) / 2;
  if(ar[mid] == target)
    return mid;
  else if(target < mid)
    return BSearchRecur(ar, first, mid-1, target);
  else if(target > mid)
    return BSearchRecur(ar, mid+1, last, target);
}
```

- 하지만 재귀함수의 성능은 일반적으로 구현하는 반복문의 경우보다 떨어짐(더 느리고 복잡함)

## 하노이 타워: The Tower of Hanoi
- 재귀함수의 힘을 보여주는 대표적인 예로 꼽힘

### 하노이 타워 문제의 이해
- 3개 원반의 경우
  - 세 개의 크기가 다른 원반이 있으며, 작은 원반, 중간 원반, 큰 원반이 존재
  - 원반은 반드시 상대적으로 큰 원반이 아래, 작은 원반이 위에 존재해야 함
  - 원반은 한 번에 하나씩 옮길 수 있음
  - 총 세 개의 기둥 A, B, C가 존재
  - A에 순서대로 꽂혀있는 원반을 모두 C로 옮기는 경우
    - 총 7번만의 과정으로 쉽게 원반을 옮길 수 있음
- 원반이 많아질 경우에도 동일하게 처리하면 됨

### 하노이 타워의 반복패턴 연구
- 4개의 원반인 경우
  - 네 개의 크기가 다른 원반이 있으며, 1, 2, 3, 4 순서대로 점점 커짐
  - A 기둥의 원반을 B, C 기둥을 이용해 C 기둥으로 모두 옮기는 문제
  - B 기둥에 1, 2, 3을 옮긴 후, C 기둥으로 4를 옮기고 나머지는 3개의 원반인 경우와 동일하게 진행
  - B 기둥에 1, 2, 3을 옮기는 과정도 3개 원반인 경우와 동일
- 막대 A에 꽂혀있는 원반 n개를 막대 C로 옮기는 과정
  - 1단계: 작은 원반 1 ~ n-1 을(n-1개) A에서 B로 이동
  - 2단계: 큰 원반 n 을 A에서 C로 이동
  - 3단계: 작은 원반 1 ~ n-1 을(n-1개) B에서 C로 이동
- 함수의 구성은 num 개의 원반을 by를 거쳐 from에서 to로 이동
  - `void HanoiTowerMove(int num, char from, char by, char to)`
  - 하지만 연산의 구성에서 n-1 의 원반이 필요하므로 n = 1 일 때의 예외 처리가 필요(__탈출조건__)

```c
//frmo에 꽂혀있는 num개의 원반을 by를 거쳐서 to로 이동
void HanoiTowerMove(int num, char from, char by, char to)
{
  if(num == 1) // 이동할 원반의 개수가 1개인 경우
    printf("원반1을 %c에서 %c로 이동 \n", from, to);
  else // 이동할 원반의 개수가 2개 이상일 경우
  {
    HanoiTowerMove(num-1, from, to, by); // 작은 원반 n-1개를 A에서 C를 거쳐 B로 이동
  }
}
```

- 2단계 __큰 원반 n을 A에서 C로 이동__ 와 3단계 __작은 원반 n-1개를 B에서 C로 이동__ 을 추가하면 아래와 같다.

```c
void HanoiTowerMove(int num, char from, char by, char to)
{
  if(num == 1)
    printf("원반1을 %c에서 %c로 이동 \n", from, to);
  else
  {
    HanoiTowerMove(num-1, from, to, by); // 작은 원반 n-1개를 A에서 C를 거쳐 B로 이동
    printf("원반%d을(를) %c에서 %c로 이동 \n", num, from, to); // 큰 원반(n)을 A에서 C로 이동
    HanoiTowerMove(num-1, by, from, to); // 작은 원반 n-1개를 B에서 C로 이동
  }
}
```

- 이를 기반으로 코드를 완성시키면 아래와 같음

```c
#pragma warning(disable:4996)
#include<stdio.h>

void HanoiTowerMove(int num, char from, char by, char to)
{
	if (num == 1)
		printf("원반1을 %c에서 %c로 이동 \n", from, to);
	else
	{
		HanoiTowerMove(num - 1, from, to, by);
		printf("원반%d을(를) %c에서 %c로 이동 \n", num, from, to);
		HanoiTowerMove(num - 1, by, from, to);
	}
}
int main(void)
{
	HanoiTowerMove(5, 'A', 'B', 'C');
	getchar();
	return 0;
}
```

- 재귀함수를 이용하여 복잡한 알고리즘을 간단하게 해결 



