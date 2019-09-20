---
layout: post
title: 알고리즘 기초-재귀 함수 사용하기
category: algorithm
tags: [algorithm]
comments: true
---

# 알고리즘 기초-재귀 함수 사용하기
- 서로 다른 방법을 만드는 파라미터를 만들어 넣어줘야 한다.
- 재귀 함수를 만들때는
  - 1. 불가능한 경우
  - 2. 정답을 찾은 경우
  - 3. 다음 경우로 가야 할 경우
- 세 경우로 나눈 후,
  - 3. 다음 경우로 가야 할 경우
  - 이 때에 case를 분류하여 따로 재귀적으로 함수를 호출해야 한다.

### 9095 1, 2, 3 더하기
- 정수 4를 1, 2, 3의 합으로 나타내는 방법은 총 7가지가 있다. 합을 나타낼 때는 수를 1개 이상 사용해야 한다.

- 1. 불가능한 경우
  - sum이 goal(합)보다 큰 경우
- 2. 찾은 경우
  - sum == goal
- 3. 다음으로 넘어가야 하는 경우
  - case 1: 1을 더할 때
  - case 2: 2를 더할 때
  - case 3: 3을 더할 때
  - 위 경우를 for문을 이용하여 구현했다.
    - for문 안에서는 재귀된 결과를 return하기 위해 result에 값을 누적해 최종 return한다.

```c
#pragma warning(disable:4996)
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
using namespace std;

int checker(int sum, int goal)
{
	if (sum > goal) return 0; // 불가능한 경우
	if (sum == goal) return 1;  // 가능한 경우
	int result = 0;
	for (int i = 1; i < 4; i++) // 한번 더 가야한다면
		result += checker(sum + i, goal); // 각각 1, 2, 3에 대해 한번 더 재귀적 연산 후 결과를 반환받아 result에 저장
	return result; // 최종적으로 result를 반환한다.
}

int main()
{
	int c, num, result;
	scanf("%d", &c);
	while (c)
	{
		result = 0;
		scanf("%d", &num);
		result = checker(0, num);
		printf("%d\n", result);
		c--;
	}
	return 0;
}
```

### 1759 암호 만들기
- 암호의 길이와 암호 가능 문자들을 입력받아 가능한 경우의 수를 오름차순으로 모두 출력.
  - 단, 자음 2개 모음 1개를 필수로 포함한다.
- 자세한 조건은 https://www.acmicpc.net/problem/1759 참조

- 1. 불가능한 경우
  - index가 주어진 문자열 길이만큼인경우(모든 가능한 문자열을 다 포함했음)
- 2. 찾은 경우
  - 가능한 암호 길이가 주어진 암호 길이와 같고, 자음모음 조건을 충족할 때
- 3. 다음으로 넘어가기
  - 현재 index 번째 문자를 사용하거나
  - 현재 index 번째 문자를 사용하지 않거나

```c
#pragma warning(disable:4996)
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <string>
using namespace std;

bool checker(string pwd) // 자음 모음 조건을 확인하는 함수. 만족하면 true, 아니면 false
{
	int length = pwd.length();
	int mo_cnt = 0;
	int ja_cnt = 0;

	for (int i = 0; i < length; i++)
	{
		if (pwd[i] == 'a' || pwd[i] == 'e' || pwd[i] == 'i' || pwd[i] == 'o' || pwd[i] == 'u')
			mo_cnt++;
		else
			ja_cnt++;
	}
	if (mo_cnt > 0 && ja_cnt > 1)
		return true;
	else
		return false;
}

void maker(int idx, vector<char> cstring, int l, int c, string pwd)
{
	if (pwd.length() == l) // 만약 생성된 비밀번호 길이가 주어진 비밀번호 길이와 같고
	{
		if (checker(pwd)) // 주어진 자모음 조건을 만족하면
		{
			cout << pwd << "\n"; // 결과를 출력하고 종료
			return;
		}
	}
	if (idx == c)	return; // 만약 index가 주어진 문자열 길이만큼 다 돌았다면(가능한 문자가 더 없음) 종료
	maker(idx + 1, cstring, l, c, pwd + cstring[idx]); // 현재 문자 사용하는 경우, pwd에 문자를 추가하고 재귀연산
	maker(idx + 1, cstring, l, c, pwd); // 현재 문자 사용하지 않는 경우, index만 올라가고 재귀연산
}

int main()
{
	int l, c;
	scanf("%d %d", &l, &c);

	vector<char> cstring(c);
	for (int i = 0; i < c; i++)
		cin >> cstring[i];
	sort(cstring.begin(), cstring.end());
	maker(0, cstring, l, c, "");
	return 0;
}
```

### 6603 로또
- 주어진 갯수만큼의 수 중에서 6개의 수를 중복하지 않게 오름차순으로 출력
- https://www.acmicpc.net/problem/6603

- 1. 불가능한 경우
  - index가 주어진 문자 갯수만큼 다 돌았다면 가능한 경우가 더 없으므로 끝
- 2. 찾은 경우
  - index가 6이라면(뽑힌 번호 길이가 6) 완성된 경우
- 3. 더 돌아야 하는 경우
  - 현재 번호를 사용한다면, answer에 현재 문자열을 넣고 index를 추가하여 다시 돌고
  - 현재 번호를 사용하지 않는다면 그냥 index만 증가해서 재귀적으로 돈다.

```c
#pragma warning(disable:4996)
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <string>
using namespace std;

void finder(vector<int> &nums, vector<int> &answer, int idx)
{
	if (answer.size() == 6) // 정답(answer)이 완성되었다면 길이가 6이므로
	{
		for (int j = 0; j < 6; j++)
			printf("%d ", answer[j]); // 정답을 출력하고 종료한다.
		printf("\n");
		return;
	}
	if (idx == nums.size()) return; // 만약, 주어진 문자열을 다 돌았다면 가능한 경우가 더이상 없으므로 종료한다.
	answer.push_back(nums[idx]); // 현재 문자열을 사용한다면 정답에 현재 번호를 넣고 
	finder(nums, answer, idx + 1); // idx 증가 후 함수를 돈다.
	answer.pop_back(); // 이후 사용하지 않을 경우를 위해 추가된 번호를 다시 뽑고(pop)
	finder(nums, answer, idx + 1); // 다시 재귀적으로 구성
}

int main()
{
	int num;

	while (1)
	{
		scanf("%d", &num);
		if (num == 0)	break;
		
		vector<int> nums(num);
		for (int j = 0; j < num; j++)
			cin >> nums[j]; // 문자열의 경우 cin으로 받는게 속편한듯 하다. 한참 고생했다..

		vector<int> answer;
		finder(nums, answer, 0);
        printf("\n");
	}
	return 0;
}
```

### 1182 부분수열의 합
- N개의 정수로 이루어진 수열이 있을 때, 크기가 양수인 부분수열 중에서 그 수열의 원소를 다 더한 값이 S가 되는 경우의 수를 구하는 프로그램을 작성하시오.
- 문자 갯수와 결과값이 주어지고 수 들이 주어졌을 때, 수를 조합하여 더해서 그 값이 결과값이 되는 경우의 수를 찾는 문제

- 1. 불가능한 경우
  - 주어진 숫자들만큼 다 돌았는데 없는 경우(idx == 문자 갯수)
- 2. 찾은 경우
  - 더해진 값이 결과값과 같다면, 단 주어진 숫자를 다 돌아야 한다.
    - 그 이유는 만약 수가 다 돌기도 전에 한번에 결과값과 같은 수가 sum에 쌓이는 오동작을 방지하기 위함이다.
- 3. 더 돌아야 하는 경우
  - 현재 숫자를 사용하는 경우, 숫자를 더한다음 idx를 추가해서 재귀적 구성
  - 현재 숫자를 사용하지 않는다면, 숫자를 더하지 않고 idx만 더해져서 넘어간다.
  
```c
#pragma warning(disable:4996)
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <string>
using namespace std;

void counter(int S, int &result, vector<int> &nums, int sum, int idx)
{
	int N = nums.size();
	if ((idx == N) && (sum == S)) // 주어진 숫자를 다 돌았고, 합 값이 결과 값과 같다면
	{
		result += 1; // 정답을 찾았으므로 result를 1 더하고 return한다.
		return;
	}
	else if (idx == N) return; // 그냥 idx == N 이라면 가능한 경우가 없으므로 끝
	counter(S, result, nums, sum + nums[idx], idx + 1); // 현재 수를 사용하는 경우, sum에 반영하여 재귀
	counter(S, result, nums, sum, idx + 1); // 현재 수를 사용하지 않는 경우 idx만 더해져서 넘어간다.
}

int main()
{
	int N, S;
	scanf("%d", &N);
	scanf("%d", &S);

	vector<int> nums(N);
	for (int j = 0; j < N; j++)
		scanf("%d", &nums[j]);

	int result = 0;
	counter(S, result, nums, 0, 0);
	if (S == 0) result -= 1; // S=0이라면 모든 원소가 0으로 더해지는 공집합분을 빼준다.
	printf("%d\n", result);
	return 0;
}
```
