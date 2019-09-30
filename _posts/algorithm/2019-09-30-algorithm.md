---
layout: post
title: 알고리즘 기초-DP 문제풀이 1
category: algorithm
tags: [algorithm]
comments: true
---

# 알고리즘 기초-DP 문제풀이 1
- DP는 문제를 많이 풀면서 감을 잡는게 제일 중요하다.

### 1463 1로 만들기
- 정수 X에 사용 가능한 연산은 다음의 3가지
  1. X가 3으로 나누어 떨어지면 3으로 나눈다.
  2. X가 2로 나누어 떨어지면 2로 나눈다.
  3. 1을 뺀다.
- 어떤 정수 N에 대해 위 세개 연산을 적절히 사용해 1을 만들 때, 그 횟수의 최솟값을 출력해라.

- 1번 조건: X->X/3
- 2번 조건: X->X/2
- 3번 조건: X->X-1
  - 이 경우, 3으로 최대한 나누는 것, 나머지 2로 나누는 것, 나머지 1씩 빼는 식으로 풀 경우 틀리게 된다.
  - 반례: 10
    - 오답: 10 - /2 1회 -> 5 - -1 1회 -> 4 - /2 2회 -> 1 : 총 4회 연산
    - 정답: 10 - -1 1회 -> 9 - /3 2회 -> 1 : 총 3회 연산

- 문제 정의: N을 1로 만드는 최소 연산
  - D[N]: 숫자 N을 1로 만드는 최소 연산 횟수
- 문제 나누기
  1. X->X/3: D[N/3] + 1
    - 3으로 나누어 떨어지는 경우, 나누게 되면 3으로 나누고 (+1) 나머지 숫자인 N/3을 1로 만드는 횟수
  2. X->X/2: D[N/2] + 1
    - 2로 나누어 떨어지는 경우, 나누게 되면 2로 나누고 (+1) 나머지 숫자인 N/2를 1로 만드는 횟수
  3. X->X-1: D[N-1] + 1
    - 1을 빼는 경우, 1을 빼고 (+1) 나머지 숫자인 N-1을 1로 만드는 횟수
- 따라서
  - D[N] = min(D[N/3]+1, D[N/2]+1, D[N-1]+1) = min(D[N/3], D[N/2], D[N-1]) + 1
- __여기서 가장 작은 단위는 예외 처리를 해줘야 함__

- 시간복잡도
  - 다이나믹은 모든 문제를 1번씩 푸는 방식(브루트포스와 유사)
  - 전체 문제의 개수 x 1문제를 푸는 시간
    - N x O(1) = O(N)

```c
#pragma warning(disable:4996)
#include <iostream>
#include <cstdio>

using namespace std;

int memo[1000001];

int calc_TD(int n) // Top-Down 방식으로 n을 1로 만드는 최소연산 횟수
{
	if (n == 1) // 가장 작은 단위 예외 처리
		return 0; // 1을 1로 만드는건 0번 필요하므로 0 반환
	if (memo[n] > 0) // 값이 있다면 값을 return
		return memo[n];
	memo[n] = calc_TD(n - 1) + 1; // -1연산
	if (n % 2 == 0) // n을 2로 나눌 수 있다면
	{// /2 연산
		int tmp = calc_TD(n / 2) + 1; // n/2를 최소로 1로 만들고 1번 더한 수
		if (memo[n] > tmp)
			memo[n] = tmp; // 작은 값으로 초기화
	}
	if (n % 3 == 0) // n을 3으로 나눌 수 있다면
	{// /3 연산
		int tmp = calc_TD(n / 3) + 1; // n/3을 최소로 1로 만들고 1번 더한 수
		if (memo[n] > tmp)
			memo[n] = tmp; // 작은 값으로 초기화
	}
	return memo[n];
}

int calc_BU(int n) // Bottom-Up 방식으로 n을 1로 만드는 최소연산 횟수
{
	memo[1] = 0; // 최소 단위 초기화(1->1은 0번만에)
	for (int i = 2; i <= n; i++)
	{
		memo[i] = memo[i - 1] + 1; // -1
		if (i % 2 == 0 && memo[i] > memo[i / 2] + 1)
		{ // /2
			memo[i] = memo[i / 2] + 1;
		}
		if (i % 3 == 0 && memo[i] > memo[i / 3] + 1)
		{// /3
			memo[i] = memo[i / 3] + 1;
		}
	}
	return memo[n];
}

int main()
{
	int n;
	scanf("%d", &n);
	int result = calc_BU(n);
	printf("%d\n", result);
	//system("pause");
	return 0;
}
```
