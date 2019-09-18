---
layout: post
title: 알고리즘 기초-수학
category: algorithm
tags: [algorithm]
comments: true
---

# 알고리즘 기초-수학
## 나머지 연산
- 각 자료형마다 담을 수 있는 숫자의 한계치가 다르므로, 만약 큰 수를 처리해야 할 경우 나머지연산을 잘 이용해야 함
  - C의 경우 `int`, `long long`을 통해 정수형을 저장 할 수 있음
    - `int`: $2^{31} -1$
    - `long long`: $2^{63} -1$
  - 큰 수를 계속 연산해야 하는 경우, 중간중간 나머지연산을 사용해야 함
- 덧셈, 뺄셈, 곱셈에는 분배법칙 성립
- 난수셈의 경우 Modular inverse를 구해야 함
  - 역원을 구해야 해서 복잡하나, 나눗셈의 경우 중요도가 떨어진다.
- 알고리즘 문제에서 __"정답을 ~로 나눈 나머지를 출력하라."__ 라는 말이 있는 이유는 정답이 int나 long long 자료형 범위를 넘어가기 때문임
  - 이런 경우, 중간중간 나눠서 계산해야함
- 음수인 경우는 별도 처리해야함
  - 언어별로 처리방식이 다름
  - Python은 알아서 casting 해주지만, 나머지는 나누는 수를 한번 더하고 다시 나눠 나머지를 구해야 함
  - ex. (6 % 3 - 5 % 3) % 3 의 경우, C나 Java는 결과값에 3을 더한 후 다시 3에 대한 나머지 연산을 해야 올바른 답이 계산됨
- 나누기의 나머지 연산은 페르마의 소정리에 의해 다음이 성립
  - $(a / b) \% c = ( a \times b^{c-2}) \% c$
  - 페르마 소정리의 원리는... 중요도가 떨어진다고한다. (위키피디아 참고)
- $ (a \% c - b \% c) \% c$의 나머지 연산은 다음이 성립
  - $0<=a\%c<=c$, $0<=b\%c<=c$ 이므로
  - $-c<(a\%c-b\%c)<2c$ 를 만족
  - 따라서, $(a\%c-b\%c+c)$는 0보다 큰 값을 갖기 때문에 이 상태에서 다시 c로 나눠줘야 원하는 결과(1과 c 사이의 수)가 나옴

## 최대공약수 (GCD)
- 수학문제와 크게 연관되어있음
  - 기약분수(분모와 분자가 더 이상 나눠지지 않는 꼴) 등을 나타낼 때 필요
- 약수를 구한 후, 거기서 최댓값이 바로 GCD
- 가장 쉬운 방법은 2부터 min(A,B)까지 모든 정수로 나누어 보는 방법
- 최대공약수가 1인 두 수를 서로소(Coprime)라고 함
- 하지만 일일히 나눠보는 방법은 비효율적!
- __유클리드 호제법__ 을 이용하면 빠르게 구현 가능
  - GCD(a, b) -> GCD(b, a%b) -> .... GCD(result, 0)
  - 반복했을 때 오른쪽에 0이 오면 왼쪽의 값(result)가 최대공약수가 됨
  - 재귀적 함수로 쉽게 구현 가능
  - 연산량이 보두 비교했을때 O(N)에서 O(log N)으로 기하급수적으로 줄게 됨
  - 재귀합수를 쓰지 않는 방법도 있긴 하지만.. 개인적으로 재귀함수가 더 친숙하다
  
```c
int gcd(int a, int b)
{
  if (b == 0)
    return a;
  else
    return gcd(b, a%b);
}
```

- 세 수의 최대공약수는 다음과 같이 구할 수 있음
  - GCD(a, b, c) = GCD(GCD(a, b), c)
  - N개의 수에 대해서도 동일하게 적용 가능(분배/결합법칙 성립)

## 최소공배수 (LCM)
- 배수 중 가장 작은 공통 수
- 최대공약수(GCD)를 이용해서 쉽게 구할 수 있음
  - 두 수 a, b의 최대공약수를 g라고 했을 때
  - LCM = a/g \* b/g * g = a\*b/c 성립

### 백준 2609
- 두 개의 자연수를 입력받아 최대 공약수와 최소 공배수를 출력하는 프로그램을 작성하시오.

```c
#pragma warning(disable:4996)
#include<iostream>
using namespace std;
int gcd(int a, int b)
{
	if (b == 0)
	{
		return a;
	}
	else
	{
		gcd(b, a%b);
	}
}
int main()
{
	int a, b;
	cin >> a >> b;
	int g = gcd(a, b);
	cout << g << '\n' << a * b / g << '\n';
	return 0;
}
```

### 백준 1934
- 두 자연수 A와 B에 대해서, A의 배수이면서 B의 배수인 자연수를 A와 B의 공배수라고 한다. 이런 공배수 중에서 가장 작은 수를 최소공배수라고 한다. 예를 들어, 6과 15의 공배수는 30, 60, 90등이 있으며, 최소 공배수는 30이다.
- 두 자연수 A와 B가 주어졌을 때, A와 B의 최소공배수를 구하는 프로그램을 작성하시오.

```c
#pragma warning(disable:4996)
#include<iostream>
using namespace std;
int gcd(int a, int b) //최대공약수 반환
{
	if (b == 0)
	{
		return a;
	}
	else
	{
		gcd(b, a%b);
	}
}
int main()
{
	int a, b, g, cnt;
	cin >> cnt;
	for (int j = 0; j < cnt; j++)
	{
		cin >> a >> b;
		g = gcd(a, b);
		cout << a * b / g << endl;
	}
	return 0;
}
```

### 백준 9613
- n개의 수에 대해 가능한 모든 쌍의 GCD 합을 구하기
  - 합을 구하는 경우, 숫자가 매우 커져 오답이 발생 할 수 있다.
  - 그럴땐 `int` 대신 `long long`을 사용하자.

```c
#pragma warning(disable:4996)
#include<iostream>
using namespace std;
int gcd(int a, int b)
{
	if (b == 0)
		return a;
	else
		return gcd(b, a%b);
}
int main()
{
	int cnt, num;
	long long result = 0; //수가 매우 크므로 결과는 long long으로 표현한다.
	int nums[100];
	cin >> cnt;
	for (int j = 0; j < cnt; j++)
	{
		cin >> num;
		for (int i = 0; i < num; i++)
		{
			cin >> nums[i];
		}
		for (int i = 0; i < num; i++)
		{
			for (int k = 1; k < num - i; k++)
			{
				result += gcd(nums[i], nums[i+k]);
			}
		}
		cout << result << endl;
		result = 0;
	}
	return 0;
}
```

## 소수
- 중요하다!
- 소수는 소수, 약수가 1과 자기 자신밖에 없는 수
- N이 소수가 되기 위해선
  - 2보다 크거가 같고
  - N-1보다 작거나 같은 자연수로
  - 나누어 떨어지면 안된다.
- 쉽게 말해 1과 자기 자신만으로 인수분해시 나눠지는 수.

- 주요 알고리즘
  - 1. 어떤 수 N이 소수인지 아닌지 판별하기
  - 2. N보다 작거나 같은 모든 자연수 중 소수를 찾아내기 (범위 내 모든 소수 찾기)
    - 1을 활용 가능하나, 굉장히 비효율적이고 느린 방법
- 기본 조건을 이용해 구현하면 아래와 같음
  - 2보다 크거가 같고 N-1보다 작거나 같은 자연수로 나누어 떨어지면 안된다.
  - 빅 오가 O(N)으로 굉장히 느림.

```c
bool prime(int n)
{
  if(n < 2)
    return false;
  for (int j = 2; j <= n-1; j++)
  {
    if (n % j == 0)
      return false;
  }
  return true;
}
```

- 더 조건을 가볍게 만들면
  - N이 소수가 되기 위해선 2보다 크거나 같고, N/2보다 작거나 같은 자연수로 나누어 떨어지면 안된다.
  - N의 약수 중에서 가장 큰 것은 N/2보다 작거나 같기 때문임
  - 이 또한 위 코드에서 for문 조건만 `j <= n/2`로 바꾸면 되지만 빅오는 동일함.

- 따라서, 루트를 이용한 가장 가벼운 조건을 만들면
  - N = a\*b 이고, a<=b 조건일 경우(항상 참이 되는 조건)
    - a <= root N, b >= root N 이 성립
    - 두 수 a와 b의 차이가 가장 작은 경우는 root N
  - 즉, root N 까지만 검사를 해도 성립함
    - ex. 24 = (1, 2, 3, 4) root N (6, 8, 12, 24) 로 분해 가능
    - 위 예시에서 왼쪽 영역만 검사하면 자연스럽게 우측 영역은 걸러질 수 있음.
  - 이 경우, 빅오는 O(root N) 으로 굉장히 빠름

```c
bool prime(int n)
{
  if (n < 2)
    return false;
  for (int j = 0; j*j <= n; j++)
  {
    if (n % j == 0)
      return false;
  }
  return true;
}
```

- 컴퓨터 `int`형 근사값을 피하기 위해 루트 대신 `i*i`를 사용함

### 백준 1978
- 주어진 수 N개 중에서 소수가 몇 개인지 찾아서 출력하는 프로그램을 작성하시오.

```c
#pragma warning(disable:4996)
#include<iostream>
using namespace std;
bool prime(int a)
{
	if (a < 2)
		return false;
	for (int j = 2; j*j <= a; j++) // 조건 잘 확인하기! 대/소/=
	{
		if (a%j == 0)
			return false;
	}
	return true;

}
int main()
{
	int cnt, num;
    long long result = 0;
	cin >> cnt;
	for (int j = 0; j < cnt; j++)
	{
		cin >> num;
		result += prime(num);
	}
	cout << result << endl;
	return 0;
}
```

- 하지만 이 또한 연산량이 많은 방법으로, 가장 효율적인 방법은 에라토스테네스의 체를 사용하는 것
  - 에라토스테네스의 체는 범위 내의 모든 소수를 구하는 알고리즘
  - 한번 구해놓고 찾아서 쓰기때문에 효율적임

### 에라토스테네스의 체
- 1부터 N까지 범위 안에 들어가는 모든 소수를 구한다.
  - 1. 2부터 N까지 모든 수를 써놓고
  - 2. 아직 지워지지 않은 수 중에서 가장 작은 수를 찾고
  - 3. 그 수는 소수가 되며
  - 4. 그 수의 배수를 모두 지운다.
  - 이 짓을 N까지 반복.
- 즉, 소수를 판별하기 위
  - 1은 일단 소수가 아니므로 지우고
  - 2의 배수를 모두 지우고 2\*2가 N보다 큰지 판단
  - 3의 배수를 모두 지우고 3\*3이 N보다 큰지 판단
  - 계속 반복!

```c
const int MAXVAL = 1000000; // 임의의 N값
bool deleted[MAXVAL+1]; // 지워질 인덱스

int main()
{
  deleted[0] = true;
  deleted[1] = true; //어차피 둘 다 소수가 아니니까 지워져야 함
  for(int j = 2; j<=MAXVAL; j++) // 제일 작은 소수인 2부터 N까지
  {
    if (deleted[j] == false)  // 만약 해당 인덱스가 지워지지 않았다면
    {
      for (int i = i + i; i <= MAXVAL; i += j)  // 해당 인덱스 배수만큼 지움표시(true)
      {
        deleted[i] = true;
      }
    }
  }
}
```

- 위 과정을 통해 `deleted` 리스트에는 지워져야 할 인덱스가 `true`가 표시
- `false`로 남아있는 인덱스가 바로 N까지의 소수가 됨
- 빅 오는 O(log log N)으로, 매우 빠르다고 한다.

### 백준 1929
- M이상 N이하의 소수를 모두 출력하는 프로그램을 작성하시오.
  - `참고 1`: bool로 초기화된 list는 0으로 초기화된다.
  - `참고 2`: 입력받은 숫자, 혹은 계산된 숫자로 리스트 길이 초기화는 불가능!
    - 동적할당 쓰면 되지만, 코딩테스트에서 동적할당을 쓸 일은 없다.
  - `참고 3`: C++ iostream의 cin, cout, endl은 굉장히 느리므로, stdio.h 의 scanf, printf를 사용하자!
```c
#pragma warning(disable:4996)
#include<iostream>
using namespace std;

const int MAXVAL = 1000000;
bool deleted[MAXVAL + 1]; // false로 초기화, 지워진 index는 true

int main()
{
	deleted[0] = true;
	deleted[1] = true; // 0, 1은 자동으로 소수가 아님

	for (int j = 2; j * j <= MAXVAL; j++) // 2부터 최댓값까지
	{
		if (deleted[j] == false) // j번째 수가 지워지지 않았다면
		{
			for (int i = j + j; i <= MAXVAL; i += j)
			{
				deleted[i] = true; // j의 배수들을 지운다
			}
		}
	}

	int a, b;
	cin >> a >> b;
	for (int j = a; j <= b; j++)
	{
		if (deleted[j] == false)
			cout << j << '\n'; 
		// endl은 시간을 많이 잡아먹으므로 '\n' 사용
	}
	return 0;
}
```

## 골드바흐의 추측
- 1742년, 독일의 아마추어 수학가 크리스티안 골드바흐는 레온하르트 오일러에게 다음과 같은 추측을 제안하는 편지를 보냈다.
- __4보다 큰 모든 짝수는 두 홀수 소수의 합으로 나타낼 수 있다.__
- 예를 들어 8은 3 + 5로 나타낼 수 있고, 3과 5는 모두 홀수인 소수이다. 또, 20 = 3 + 17 = 7 + 13, 42 = 5 + 37 = 11 + 31 = 13 + 29 = 19 + 23 이다.
- 이 추측은 아직도 해결되지 않은 문제이다.
- 백만 이하의 모든 짝수에 대해서, 이 추측을 검증하는 프로그램을 작성하시오.
  - __사실 $10^{18}$__ 이하에서는 참으로 검증되었다.

```c
#pragma warning(disable:4996)
#include<iostream>
#include<stdio.h>

using namespace std;

const int MAXVAL = 1000000;
bool deleted[MAXVAL + 1]; // false로 초기화, 지워진 index는 true

int main()
{
  deleted[0] = true;
  deleted[1] = true;
  
  for (int j = 2; j*j <= MAXVAL; j++) // 2부터 최댓값까지 (조건 j*2도 됨)
  {
    if (deleted[j] == false)  // 만약 j번째 수가 지워지지 않았다면
      for (int i = j + j; i <= MAXVAL; i += j)
        deleted[i] = true;  // j의 배수들을 지워라
  }
  
  int num;
  while(1)
  {
    scanf("%d", &num);
    if (num == 0) // 탈출조건
      break;
    for (int j = 3; j <= num - j; j++) // 가능한 가장 작은 소수는 3이며, 어차피 num-j만큼만 돌아도 중간에 걸려서 끝남. 
    {
      if ((deleted[j] == false) & (deleted[num-j] == false)) // 첫 번째 수와 num-j 번째 수 모두 소수이면 조건을 만족하므로
      {
        printf("%d = %d + %d\n", num, j, num - j); // 결과를 출력하고 해당 for 문을 빠져나옴
        break;
      }
    }
  }
  return 0;
}
```

- 끝!







