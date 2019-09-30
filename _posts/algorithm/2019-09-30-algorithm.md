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

### 11726 2xn 타일링
- 2xn 크기의 직사각형을 1x2, 2x1 타일로 채우는 방법의 수를 구하는 프로그램을 작성한다.
	- 출력은 2xn 크기의 직사각형을 채우는 방법의 수를 10,007로 나눈 나머지를 출력
- 두 종류의 타일을 마지막에 어떤 걸 썼는지에 따라 달라진다.
	- 마지막에 1x2 1개를 썼을 때와(ㅣ 모양), 2x1 2개를 썼을 때 (= 모양)의 경우가 차이 남
- D[N]: 2xN 크기의 직사각형을 채우는 방법의 수
	- 마지막에 1x2 1개를 썼을 때 (+1) + D[N-1]
		- 마지막 세로 1칸을 빼면 전체 길이 N에서 N-1이 되므로
  - 마지막에 2x1 2개를 썼을 때 (+1) + D[N-2]
		- 마지막에 가로 2칸짜리 위아래로 하나씩 빼면 전체 길이 N에서 N-2가 되므로
- 따라서
	- D[N] = D[N-1] + D[N-2]

- D[N]을 정의하고, 경우의 수를 나눠 그 경우에 대해 정의한다.

```c
#pragma warning(disable:4996)
#include <iostream>
#include <cstdio>

using namespace std;

int memo[1001];

int calc_TD(int n) // TD 방식으로 n길이의 직사각형을 채우는 방법의 수
{
	memo[0] = 1; // 최소 단위 정의
	if (n == 1) // 최소 단위 정의
		memo[n] = 1;
	if (memo[n] > 0) // 해당 길이(n)의 직사각형을 채우는 방법의 수가 정해졌으면
		return memo[n]; // 그대로 반환한다(중복 계산 필요 없음)
	
	memo[n] = (calc_TD(n - 1) + calc_TD(n - 2)) % 10007;
	// 아니라면, 정의에 의해 값을 채우고 반환한다.

	return memo[n];
}

int calc_BU(int n) // BU 방식으로 n길이의 직사각형을 채우는 방법의 수
{
	memo[0] = 1; // 최소 단위 정의
	memo[1] = 1; // 최소 단위 정의
	for (int i = 2; i <= n; i++)
	{ // 2부터 n 길이의 직사각형에 대해
		memo[i] = (memo[i - 1] + memo[i - 2]) % 10007; // 정의대로 값을 채워나간다.
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

### 11727 2xn 타일링 2
- 위의 문제에서 타일의 종류가 1개 추가된 경우
- 동일하게
	- D[N]: N 길이의 직사각형을 채우는 방법의 수
- 하지만 타일 종류가 추가되었으므로
	- 2x2 타일로 마지막을 채우는 경우, 2x2 1개(+1) + D[N-2]
		- 가로 2칸이 줄어 N-2 길이의 직사각형을 채우는 방법의 수
- 따라서
	- D[N] = D[N-1] + D[N-2] + D[N-2]

```c
#pragma warning(disable:4996)
#include <iostream>
#include <cstdio>

using namespace std;

int memo[1001];

int calc_TD(int n) // TD 방식으로 n 길이의 정사각형을 채우기
{
	memo[0] = 1;
	if (n == 1)
		memo[n] = 1;
	if (memo[n] > 0)
		return memo[n];
	
	memo[n] = (calc_TD(n - 1) + 2 * calc_TD(n - 2)) % 10007;

	return memo[n];
}

int calc_BU(int n) // BU 방식으로 n 길이의 정사각형을 채우기
{
	memo[0] = 1;
	memo[1] = 1;
	for (int i = 2; i <= n; i++)
	{
		memo[i] = (memo[i - 1] + 2 * memo[i - 2]) % 10007;
	}
	return memo[n];
}

int main()
{
	int n;
	scanf("%d", &n);
	int result = calc_TD(n);
	printf("%d\n", result);
	//system("pause");
	return 0;
}
```

### 9095 1, 2, 3 더하기
- 정수 N을 1, 2, 3으로 나타내는 방법의 수를 센다.
- D[N]: 1, 2, 3의 합으로 N을 나타내는 방법의 수
	- 경우의 수 1: 마지막 숫자가 1일 때
		- 마지막 1로 하고, D[N-1]을 1, 2, 3의 조합 합으로 나타내는 방법의 수
	- 경우의 수 2: 마지막 숫자가 2일 때
		- 마지막 2로 하고, D[N-2]을 1, 2, 3의 조합 합으로 나타내는 방법의 수
	- 경우의 수 3: 마지막 숫자가 3일 때
		- 마지막 3로 하고, D[N-3]을 1, 2, 3의 조합 합으로 나타내는 방법의 수
- D[N] = D[N-1] + D[N-2] + D[N-3]
- 초기값의로 D[0] = 1로 설정
	- 0을 1, 2, 3의 합으로 나타낼 수 있는 경우는 모두 안 쓰는 경우로 1가지로 셈.

```c
#pragma warning(disable:4996)
#include <iostream>
#include <cstdio>
#include <vector>

using namespace std;

int memo[12];

int calc_TD(int n) // TD 방식으로 n을 1, 2, 3으로 나타내는 방법의 수
{
	if (n == 0)
	{// 최소 경우의 수 정의
		memo[n] = 1;
		return memo[n];
	}
	else if (n == 1)
	{// n=1일 때
		memo[n] = calc_TD(n - 1);
	}
	else if (n == 2)
	{// n=2일 때
		memo[n] = calc_TD(n - 1) + calc_TD(n - 2);
	}
	else
	{// 나머지 일반적인 경우
		memo[n] = calc_TD(n - 1) + calc_TD(n - 2) + calc_TD(n - 3);
	}
	
	return memo[n];
}

int calc_BU(int n) // BU 방식으로 n을 1, 2, 3으로 나타내는 경우의 수
{
	memo[0] = 1; // 최소 경우의 수 정의
	
	for (int i = 1; i <= n; i++)
	{ // 1부터 숫자 n까지
		if (i - 1 >= 0)
		{ // 1번째부터 n까지
			memo[i] += memo[i - 1];
		} 
		if (i - 2 >= 0)
		{ // 2번째부터 n까지
			memo[i] += memo[i - 2];
		}
		if (i - 3 >= 0)
		{ // 3번째부터 n까지
			memo[i] += memo[i - 3];
		}
	}

	return memo[n]; // 배열을 완성하고 리턴
}

int main()
{
	int n, c;
	scanf("%d", &c);
	calc_TD(11);

	while (c--)
	{
		scanf("%d", &n);
		printf("%d\n", memo[n]);
	}
	//system("pause");
	return 0;
}
```

### 15988 1, 2, 3 더하기 3
- 정수 N을 1, 2, 3의 합으로 나타내는 방법의 수를 구한다.
- N은 양수이며, 1,000,000보다 작거나 같다
- 각 테스트 케이스의 방법의 수를 1,000,000,009로 나눈 나머지를 출력

- 출력의 범위를 고려해야 한다.

```c
#pragma warning(disable:4996)
#include <iostream>
#include <cstdio>
#include <vector>

using namespace std;

long long memo[1000001];
long long mod = 1000000009;

int calc_BU(int n)
{
	memo[0] = 1;

	for (int i = 1; i <= n; i++)
	{
		if (i - 1 >= 0)
		{
			memo[i] += memo[i - 1];
		}
		if (i - 2 >= 0)
		{
			memo[i] += memo[i - 2];
		}
		if (i - 3 >= 0)
		{
			memo[i] += memo[i - 3];
		}
		memo[i] %= mod;
	}

	return memo[n];
}

int main()
{
	int c;
	long long n;
	scanf("%d", &c);
	calc_BU(1000001);

	while (c--)
	{
		scanf("%lld", &n);
		printf("%lld\n", memo[n]);
	}
	//system("pause");
	return 0;
}
```

