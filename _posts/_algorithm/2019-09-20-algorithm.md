---
layout: post
title: 알고리즘 기초-순열 사용하기
category: algorithm
tags: [algorithm]
comments: true
---

# 알고리즘 기초-순열 사용하기
- 순서가 중요한 경우 사용한다.
  - nCm, nPm 과 같이..
- 모든 가능한 경우를 고려하고, 해당 경우의 순서를 또 고려함
- 크기는 항상 N이 되어야 하고, 겹치는 숫자가 존재 할수도, 안할수도 잇음.
- 첫 순열의 숫자는 항상 오름차순, 마지막 숫자는 항상 내림차순이 되어야 한다.

<center>
<figure>
<img src="/assets/post_img/algorithm/2019-09-20-algorithm/fig1.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 크기가 N인 순열은 총 N! 개가 존재함
- 순열은 사전순으로 나열!
- 다음 순열(Next Permutation)은 사전순으로 다음에 오는 순열을 의미
- __C++ STL 에는 algorithm에 vector나 배열 연산이 가능한 `next_permutation`과 `prev_permutation`이 존재함__
  - 겹치는 요소까지 편리하게 관리해줌! 굉장히 편함
  - 주어지는 배열/벡터 요소에서, 오름차순으로 하나씩 변경하며 가능한 가지의 경우 수를 모드 테스트 해 볼 수 있음
- ex. 1 2 3 4 5 6 7 -> 1 2 3 4 5 7 6 -> ... -> 1 7 6 5 4 3 2 (1 시작의 마지막 경우) -> 2 1 3 4 5 6 7 (2 시작의 처음)
  - 이처럼, 1로 시작의 시작은 오름차순(2 3 4 5 6), 1로 시작의 마지막은 내림차순 (7 6 5 4 3 2) 정렬이 됨
- 정렬 알고리즘의 시간 복잡도는 O(N) 

- 모든 순열 문제에 있어서 `do` `while` 문을 사용하면 굉장히 편리하다!
  - 1. 일단 정렬된(오름차순으로) 순열을 만들고
    - C++ `include<algorithm>` 에서 `sort(vector.begin(), vector.end())` 쓰면 편리..
    - `do` 문에 일단 알고리즘 돌게 동작시키고
    - `while` 문 조건으로 `next_permutation` 을 넣으면 된다.(call by reference 원래 값이 정렬됨)

- 순열 문제를 풀 때는 복잡도를 일단 계산해보는게 좋다.
  - 경우의수가 100만이 넘어가는 경우는 다른 방법을 찾는게 좋다고 한다.

### 백준 10972
- 1부터 N까지의 수로 이루어진 순열이 있다. 이때, 사전순으로 다음에 오는 순열을 구하는 프로그램을 작성하시오.
- 사전 순으로 가장 앞서는 순열은 오름차순으로 이루어진 순열이고, 가장 마지막에 오는 순열은 내림차순으로 이루어진 순열이다.

```c
#pragma warning(disable:4996)
#include<cstdio>
#include<iostream>
#include<algorithm>
// next_permutation(start, end): 배열 다음 순열 출력, 마지막이면 false 반환
#include<vector>

using namespace std;

int main(void) 
{
	int n;
	scanf("%d", &n);
	
	vector<int> nums(n);
	for (int j = 0; j < n; j++)
	{
		scanf("%d", &nums[j]);
	}
	if (next_permutation(nums.begin(), nums.end()))
	{
		for (int j = 0; j < n; j++)
			printf("%d ", nums[j]);
		printf("\n");
	}
	else
		printf("-1");
	return 0;
}
```

### 백준 10973
- 1부터 N까지의 수로 이루어진 순열이 있다. 이때, 사전순으로 바로 이전에 오는 순열을 구하는 프로그램을 작성하시오.
- 사전 순으로 가장 앞서는 순열은 오름차순으로 이루어진 순열이고, 가장 마지막에 오는 순열은 내림차순으로 이루어진 순열이다.

```c
#pragma warning(disable:4996)
#include<cstdio>
#include<iostream>
#include<algorithm>
// next_permutation(start, end): 배열 다음 순열 출력, 마지막이면 false 반환
#include<vector>

using namespace std;

int main(void)
{
	int n;
	scanf("%d", &n);

	vector<int> nums(n);
	for (int j = 0; j < n; j++)
	{
		scanf("%d", &nums[j]);
	}
	if (prev_permutation(nums.begin(), nums.end()))
	{
		for (int j = 0; j < n; j++)
			printf("%d ", nums[j]);
		printf("\n");
	}
	else
		printf("-1\n");
	return 0;
}
```

### 백준 10974 (모든 순열)
- 모든 순열을 찾는 문제는 첫 순열을 만들고, 돌려보면 된다.

```c
#pragma warning(disable:4996)
#include<cstdio>
#include<iostream>
#include<algorithm>
// next_permutation(start, end): 배열 다음 순열 출력, 마지막이면 false 반환
#include<vector>

using namespace std;

int main(void) 
{
	int n;
	scanf("%d", &n);
	vector<int> nums(n);
	for (int j = 0; j < n; j++)
		nums[j] = j + 1;
	do
	{
		for (int j = 0; j < n; j++)
			printf("%d ", nums[j]);
		printf("\n");
	} while (next_permutation(nums.begin(), nums.end()));

	return 0;
}
```

- 팩토리얼의 경우, 10! 까지만 순열을 이용한 문제풀이가 먹힌다.

### 백준 10819
- N개의 정수로 이루어진 배열 A가 주어진다. 이때, 배열에 들어있는 정수의 순서를 적절히 바꿔서 다음 식의 최댓값을 구하는 프로그램을 작성하시오.
  - |A[0] - A[1]| + |A[1] - A[2]| + ... + |A[N-2] - A[N-1]|

```c
#pragma warning(disable:4996)
#include<cstdio>
#include<cstdlib>
#include<iostream>
#include<algorithm>
// next_permutation(start, end): 배열 다음 순열 출력, 마지막이면 false 반환
#include<vector>

using namespace std;

int calculator(vector<int> a, int n)
{
	int result = 0;
	for (int j = 0; j < n - 1; j++)
	{
		result += std::abs(a[j] - a[j + 1]);
	}
	return result;
}

int main(void)
{
	int n;
	scanf("%d", &n);

	vector<int> nums(n);
	for (int j = 0; j < n; j++)
	{
		scanf("%d", &nums[j]);
	}
	sort(nums.begin(), nums.end());
	
	int tmp, ans = -1;
	do
	{
		tmp = calculator(nums, n);
		if (ans < tmp)
			ans = tmp;
	} while (next_permutation(nums.begin(), nums.end()));

	printf("%d \n", ans);
	return 0;
}
```

### 백준 10971 (외판원순회, 중요함!)
- Traveling Salesman Problem(TSP) 문제로, 빈출 및 중요한 문제임
- 1에서 시작해 N까지의 도시를 "경제적"으로 방문하는 방법을 고안하는 알고리즘
- N개의 도시 방문 후 원래 도시로 돌아와야 하며, 동일한 도시 재방문은 불가

- https://www.acmicpc.net/problem/10971

- N의 최댓값이 10이므로 worst case의 시간복잡도는 10!이므로 다 돌려서 확인해봐도 됨

```c
#pragma warning(disable:4996)
#include<cstdio>
#include<cstdlib>
#include<iostream>
#include<algorithm>
// next_permutation(start, end): 배열 다음 순열 출력, 마지막이면 false 반환
#include<vector>

using namespace std;

int main(void)
{
	int c;
	int costs[10][10] = {}; // 최댓값을 고려해 bg 생성
	scanf("%d", &c); // 방문할 도시 갯수 설정

	vector<int> nums(c);
	for (int j = 0; j < c; j++)
	{
		nums[j] = j + 1; // 1, 2, ..., c 만큼의 수열을 만든다. 1에서 출발해서 1로 돌아오기위해 1은 고정하고 뒤 순서(2, 3, ..)만 수정되도록함
	}
	
	for (int j = 0; j < c; j++)
	{
		for (int i = 0; i < c; i++)
		{
			scanf("%d", &costs[j][i]); //금액표 만들기(bg)
		}
	}

	int answer = 2147483647; // int형에 담을 수 있는 최대의 수를 담아놓음(최솟값을 찾기 위한 worst case)

	do
	{
		int sum = 0;
		bool ok = true; // 방문해도 되는지 조건을 따지는 변수
		for (int j = 0; j < c - 1; j++) // 해당 도시(j 번째)에서 다음 도시(j+1번째) 비교를 위해 -1만큼 뺀만큼만 for문을 돈다.
		{
			if (costs[nums[j] - 1][nums[j + 1] - 1] == 0) // 만약 다음 도시를 갈 수 없다면(갈 수 없는경우 0임)
				ok = false; // ok 변수를 false로 설정(불가능한 경우)
			else
				sum += costs[nums[j] - 1][nums[j + 1] - 1]; // 방문 가능하다면, 총 금액에 이동비용을 저장
		}
		if (ok && costs[nums[c - 1] - 1][nums[0] - 1] != 0) // 만약 다음 도시 방문 가능하고, 마지막 도시에서 본 도시로 돌아오는게 가능하다면
		{
			sum += costs[nums[c - 1] - 1][nums[0] - 1]; // 해당 비용을 더함
			if (answer > sum)
				answer = sum; // 최솟값을 answer에 저장
		}
	} while ((next_permutation(nums.begin(), nums.end())) && (nums[0] == 1)); // 다음 경우의 수를 고려하기 위해 수열을 한칸 변경
	printf("%d\n", answer);
	return 0;
}
```

### 백준 6603
- 6 이상의 주어진 수에서 서로다른 6개의 수를 뽑아 오름차순으로 출력함.
- 가능한 경우의 수를 모두 출력해야 함

```c
#pragma warning(disable:4996)
#include<cstdio>
#include<cstdlib>
#include<iostream>
#include<algorithm>
// next_permutation(start, end): 배열 다음 순열 출력, 마지막이면 false 반환
#include<vector>

using namespace std;

int main(void)
{
	int num;
	while(true)
	{
		scanf("%d", &num); // 숫자 갯수 입력
		if (num == 0) break;
		vector<int> nums(num);
		for (int j = 0; j < num; j++)
			scanf("%d", &nums[j]); // 숫자들 입력받음
		
		vector<int> choice(num, 0); // 선택할 수를 정하기 위해 입력받은 숫자만큼 0으로 초기화 된 벡터를 만들고
		for (int j = 0; j < 6; j++)
			choice[j] = 1; // 일단 앞에서 6개를 선택하도록 1로 초기화 (1 1 1 1 1 1 0 0....)
		
		do
		{
			vector<int> answer; // answer를 만든다음(stack으로 쓸 예정)
			for (int j = 0; j < num; j++)
			{
				if (choice[j] == 1)
					answer.push_back(nums[j]); // choice 된 숫자면 answer에 집어넣는다.
			}
			sort(answer.begin(), answer.end()); // 선택된 숫자들을 정렬한 후
			for (int j = 0; j < 6; j++)
				printf("%d ", answer[j]); // 출력
			printf("\n");
		} while (prev_permutation(choice.begin(), choice.end())); // 다음(이전)순열로 넘어간다.
		printf("\n");
	}
	return 0;
}
```

### 14888 연산자 끼워넣기
- 주어진 숫자에 주어진 연산자를 끼워넣어 조합 가능한 최댓값, 최솟값을 찾아 출력하는 문제
- + - \* / 는 각각 0, 1, 2, 3 으로 매핑되어있음

- 연산자용 배열을 만들고, 해당 숫자만큼 매핑된 숫자가 반복되도록 순열을 만들고 `next_permutation`으로 돌리면서 하나하나 실험한다.
- 조합 가능한 경우의 수가 제한적이므로 충분히 이렇게 가능함

```c
#pragma warning(disable:4996)
#include<cstdio>
#include<cstdlib>
#include<iostream>
#include<algorithm>
#include<vector>

using namespace std;

int c(int a, int op, int b) // 연산자 종류에 따라 계산해주는 함수
{
	if (op == 0) // +
		return a + b;
	else if (op == 1) // -
		return a - b;
	else if (op == 2) // *
		return a * b;
	else if (op == 3) // /
		return a / b;
	else
		exit(1);
}

int main(void)
{
	int n;
	scanf("%d", &n);
	
	vector<int> nums(n);
	for (int j = 0; j < n; j++)
		scanf("%d", &nums[j]); // 숫자 입력
	
	vector<int> opnum(4);
	for (int j = 0; j < 4; j++)
		scanf("%d", &opnum[j]); // 연산자 갯수들을 입력
	
	vector<int> ops;
	for (int j = 0; j < 4; j++)
	{
		for (int i = 0; i < opnum[j]; i++)
			ops.push_back(j); // ops 벡터에 해당 연산자 갯수만큼 매핑된 숫자가 반복되도록 하여 수열을 만듦
	}

	long long sum = 0;
	long long min_val = 1000000001;
	long long max_val = -1000000001; // 문제에서 10억단위 연산을 사용한다 했으므로 long long으로 사용

	do
	{
		sum = 0;
		for (int j = 0; j < n - 1; j++) // 계산 시 다음 수를 사용하므로 n-1 까지만 인덱스 참조
		{
			if (j == 0)
				sum = c(nums[j], ops[j], nums[j + 1]); // 첫 번째의 경우 첫 번째 수와 다음 수를 계산
			else
				sum = c(sum, ops[j], nums[j + 1]); // 두 번째부턴 중간 결과값을 사용해야 함
		}

		if (min_val > sum) // 최대 및 최솟값을 저장
			min_val = sum;
		if (max_val < sum)
			max_val = sum;
	} while (next_permutation(ops.begin(), ops.end())); // 다음 순열로 넘어간다.

	printf("%d \n", max_val);
	printf("%d \n", min_val);
	return 0;
}
```

- 진도가 너무 늦는데...


