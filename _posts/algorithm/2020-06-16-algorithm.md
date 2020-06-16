---
layout: post
title: 알고리즘 기초-DP 문제풀이 3
category: algorithm
tags: [algorithm]
comments: true
---

# 알고리즘 기초-DP 문제풀이 3
### 11053 가장 긴 증가하는 부분 수열
- LIS 문제!(__중요함__)
  - Longest Increasing Subsequence
- 수열 A가 주어졌을 때, 가장 긴 증가하는 부분 수열을 구하는 문제
- D[N]: N길이의 수열에서 가장 긴 증가하는 부분 수열의 길이

<center>
<figure>
<img src="/assets/post_img/algorithm/2019-10-03-algorithm/fig1.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 위와 같은 경우, i번째 현재 수인 A[i]가 이전의 작은 숫자인 A[j] 보다 커야만 증가하는 모양이므로,
  - 증가하는 모양일 경우 최대 길이를 D[i]에 저장한다.

```c
#pragma warning(disable:4996)
#include <iostream>
#include <cstdio>
#include <vector>

using namespace std;

int d[1001]; // 길이 저장용
int a[1001]; // 숫자 저장용

void calc_BU(int n)
{
	d[1] = 1; // 최소 단위 초기화
	for (int i = 2; i <= n; i++)
	{
		d[i] = 1; // 우선 현재 숫자만 부분 수열이라 가정하고 i번째 길이를 1로 초기화하고
		for (int j = 1; j < i; j++) // i번째 숫자 왼쪽부분으로 j 인덱스로 탐색을 하는데
		{
			if (a[i] > a[j] && d[i] < d[j] + 1)
			{ // 증가하는 수열이므로 현재 숫자인 a[i]가 이전 숫자인 a[j]보다 크고, 현재 길이 d[i]가 이전 길이+현재 숫자 포함 길이(1) 보다 짧으면
				d[i] = d[j] + 1; // 현재 길이 d[i]를 이전 길이 d[j] + 현재 숫자 포함(1) 값으로 초기화
			}
		}
	}
}

int main()
{
	int n;
	scanf("%d", &n);
	for (int i = 1; i <= n; i++)
	{
		scanf("%d", &a[i]);
	}

	calc_BU(n);

	int ans = -1;
	for (int i = 1; i <= n; i++)
	{ // 저장된 배열 내 숫자들 중 가장 큰 값이 가장 긴 증가하는 부분 수열을 의미한다.
		if (ans < d[i])
			ans = d[i];
	}
	printf("%d\n", ans);

	//system("pause");
	return 0;
}
```

### 14002 가장 긴 증가하는 부분 수열 4
- 앞 문제와 같지만, 진짜 수열 값을 출력해야 한다.
- 배열을 구하는 논리의 흐름은 그대로 가져가되, 가장 긴 숫자를 비교하는 부분에서 몇 번째 인덱스의 영향을 받아 현재의 긴 숫자가 나왔는지를 저장하는 별도의 배열을 추가한다.

<center>
<figure>
<img src="/assets/post_img/algorithm/2019-10-03-algorithm/fig2.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

```c
#pragma warning(disable:4996)
#include <iostream>
#include <cstdio>
#include <vector>

using namespace std;

int d[1001]; // 길이 저장용
int a[1001]; // 숫자 저장용
int idx[1001]; // 몇 번째 숫자를 참고했는지 저장용

void calc_BU(int n)
{
	d[1] = 1;
	for (int i = 2; i <= n; i++)
	{
		d[i] = 1;
		for (int j = 1; j < i; j++)
		{
			if (a[i] > a[j] && d[i] < d[j] + 1)
			{
				d[i] = d[j] + 1;
				idx[i] = j; // 현재 인덱스가 몇 번째 숫자를 참고했는지를 저장한다.
			}
		}
	}
}

int main()
{
	int n;
	scanf("%d", &n);
	for (int i = 1; i <= n; i++)
	{
		scanf("%d", &a[i]);
	}

	calc_BU(n);

	int ans = -1;
	int max_idx;
	for (int i = 1; i <= n; i++)
	{
		if (ans < d[i])
		{
			ans = d[i];
			max_idx = i; // 가장 긴 크기의 인덱스를 받고
		}
			
	}
	printf("%d\n", ans);

	vector<int> nums; // 숫자 저장용
	while (true)
	{
		nums.push_back(a[max_idx]); // 순서대로 넣고
		max_idx = idx[max_idx]; // 인덱스를 다음 인덱스로 초기화
		if (max_idx == 0) // 마지막 인덱스는 필연적으로 0을 가리킴
			break;
	}
	
	for (int i = nums.size()-1; i >= 0; i--)
	{
		printf("%d ", nums[i]); // 역순으로 출력한다. 이는 위에서 뒤쪽부터 nums 벡터에 저장되었기 때문
	}
	printf("\n");

	//system("pause");
	return 0;
}
```

### 11055 가장 큰 증가 부분 수열
- 앞의 흐름과 같지만, 합이 가장 큰 증가 부분 수열을 찾아 출력한다.
  - 길이를 나타내는 1 값 대신 자기 자신 숫자 a[i]를 저장하면 됨

```c
#pragma warning(disable:4996)
#include <iostream>
#include <cstdio>
#include <vector>

using namespace std;

int d[1001]; // 합 저장용
int a[1001]; // 숫자 저장용

void calc_BU(int n)
{
	d[1] = a[1]; // 최소단위 초기화
	for (int i = 2; i <= n; i++)
	{
		d[i] = a[i]; // 우선 현재 합을 현재 숫자로 초기화하고
		for (int j = 1; j < i; j++)
		{ // 현재 숫자 i번째 왼쪽으로 모두 탐색을 한다.
			if (a[i] > a[j] && d[i] < d[j] + a[i])
			{ // 증가하는 숫자이고, 현재 합보다 이전 합+현재 숫자가 더 크다면
				d[i] = d[j] + a[i]; // 해당 값으로 초기화해 최댓값을 저장한다.
			}
		}
	}
}

int main()
{
	int n;
	scanf("%d", &n);
	for (int i = 1; i <= n; i++)
	{
		scanf("%d", &a[i]);
	}

	calc_BU(n);

	int ans = -1;
	for (int i = 1; i <= n; i++)
	{
		if (ans < d[i])
		{ // 최댓값을 저장한다.
			ans = d[i];
		}
			
	}
	printf("%d\n", ans);

	//system("pause");
	return 0;
}
```

### 11722 가장 긴 감소하는 부분 수열
- 수열 A가 주어졌을 때 그 수열의 감소하는 부분 수열 중 가장 긴 것을 구하는 문제

```c
#pragma warning(disable:4996)
#include <iostream>
#include <cstdio>
#include <vector>

using namespace std;

int d[1001]; // 수열 길이 저장용
int a[1001]; // 숫자 저장용

void calc_BU(int n)
{
	d[1] = 1; // 최소 단위 초기화
	for (int i = 2; i <= n; i++)
	{
		d[i] = 1; // 현재 길이 초기화
		for (int j = 1; j < i; j++)
		{ // 현재 숫자 왼쪽의 모든 숫자를 참조
			if (a[i] < a[j] && d[i] < d[j] + 1)
			{ // 감소하는 숫자 중, 현재 길이보다 이전 길이+현재 값 포함했을때가 더 길면
				d[i] = d[j] + 1; // 해당 길이로 초기화한다.
			}
		}
	}
}

int main()
{
	int n;
	scanf("%d", &n);
	for (int i = 1; i <= n; i++)
	{
		scanf("%d", &a[i]);
	}

	calc_BU(n);

	int ans = -1;
	for (int i = 1; i <= n; i++)
	{
		if (ans < d[i])
		{
			ans = d[i]; // 최댓값을 찾아 저장
		}
			
	}
	printf("%d\n", ans);

	//system("pause");
	return 0;
}
```

### 11054 가장 긴 바이토닉 부분 수열
- https://www.acmicpc.net/problem/11054
  - https://www.acmicpc.net/source/share/ae99e012416747aab23985009bfbe7ff

### 1912 연속합
- https://www.acmicpc.net/problem/1912
  - https://www.acmicpc.net/source/share/05e491a5a27046bdaf66b4fb3b3db646
- 중요한 문제!
- Maximal Subarray

### 13398 연속합 2
- https://www.acmicpc.net/problem/13398
  - https://www.acmicpc.net/source/share/211c4a9dffdf4972bfee5e656476c682

### 1699 제곱수의 합
- https://www.acmicpc.net/problem/1699
  - https://www.acmicpc.net/source/share/7c915207ae8d420da2057513e06a9176

### 2225 합분해
- https://www.acmicpc.net/problem/2225
  - https://www.acmicpc.net/source/share/8aebbe6b2c284bf882c55404001eee08

### 13707 합분해 2
- https://www.acmicpc.net/problem/13707
  - https://www.acmicpc.net/source/share/9887579548f34864a3c1e5148ae36a46
