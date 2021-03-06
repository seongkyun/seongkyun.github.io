---
layout: post
title: 알고리즘 기초-DP 문제풀이 2
category: algorithm
tags: [algorithm]
comments: true
---

# 알고리즘 기초-DP 문제풀이 2
- DP 문제 풀이 순서
  1. 문제를 D[N]에 관련된 식으로 나타내고
  2. D[N]에 관련된 식을 작은 문제로 쪼개고
    - 일반적으로 N-1 번째와 N번째의 관계로 나눔
  3. 쪼개진 문제로 D[N]의 일반식으로 표현
    - 이 과정에서 범위도 고려해야 함
  4. 최소 단위 항을 찾고
  5. 코드로 구현
  
- DP가 적용 가능한 경우?
  - 어떤 임의의 경우에 대해, 뒤 쪽의 답이 정형화 되어있는 꼴
    - 최소 길이 문제 등(앞에서 서울->부산 갈 때, 중간 거리가 모두 서로 최소가 되는 경우)
- BF 풀이 가능한 모든 문제는 DP로 풀 수 없지만
- DP 풀이 가능한 모든 문제는 BF로 풀 수 있다.
  - 이런 경우, DP로 푸는게 훨씬 효율적이며 time over 되지 않는다.
  
### 15990 1, 2, 3 더하기 5
- 정수 n을 1, 2, 3의 합으로 나타내는 방법의 수를 구하는 문제
  - __단, 같은 수를 두 번 이상 연속해서 사용하면 안된다__
  - 이 조건들을 토대로 문제를 쪼갤 수 있음
- D[n]: n을 1, 2, 3의 합으로 나타내는 방법의 수
  - 추가 조건을 고려-> __같은 수를 두 번 이상 연속해서 사용하면 안된다__
- D[n][k]: 정수 n을 1, 2, 3의 합으로 나타내는 방법의 수를 구하는데, 마지막 수는 k로 끝난다.
  - D[i][1]: 마지막이 1로 끝나는 경우, 앞에는(앞 수는 i-1) 2나 3만 올 수 있음
    - D[i][1] = D[i-1][2] + D[i-1][3]
  - D[i][2]: 마지막이 2로 끝나는 경우, 앞에는(앞 수는 i-2) 1이나 3만 올 수 있음
    - D[i][2] = D[i-2][1] + D[i-2][3]
  - D[i][3]: 마지막이 3으로 끝나는 경우, 앞에는(앞 수는 i-3) 1이나 2만 올 수 있음
    - D[i][3] = D[i-3][1] + D[i-3][2]
- 최소 단위를 구해보면(n은 양수)
  - D[n][1]
    - D[n-1][2] + D[n-1][3] when n>1
    - D[n][1] = 0 when n<1
    - D[1][1] = 1
  - D[n][2]
    - D[n-2][1] + D[n-2][3] when i>2
    - D[n][1] = 0 when n<2
    - D[2][2] = 1
  - D[n][3]
    - D[n-3][1] + D[n-3][2] when i>3
    - D[n][3] = 0 when n<3
    - D[3][3] = 1

- 문제처럼 ~수로 나눈 나머지를 출력한다는 것은 수가 그만큼 커질 수도 있기 때문에 중간중간 나눈 나머지를 저장해주는게 좋다.

```c
#pragma warning(disable:4996)
#include <iostream>
#include <cstdio>
#include <vector>

using namespace std;

long long d[100001][4];
long long mod = 1000000009;

void calc_BU(int n)
{
	
	for (int i = 0; i <= n; i++)
	{
		if (i >= 1) // 1로 끝나는 경우의 조건
		{
			if (i == 1) // 최소단위 초기화
				d[i][1] = 1;
			else // 가능한 경우의 연산
				d[i][1] += d[i - 1][2] + d[i - 1][3];
		}
			
		if (i >= 2) // 2로 끝나는 경우의 조건
		{
			if (i == 2) // 최소단위 초기화
				d[i][2] = 1;
			else // 가능한 경우의 연산
				d[i][2] += d[i - 2][1] + d[i - 2][3];
		}
			
		if (i >= 3) // 2으로 끝나는 경우의 조건
		{
			if (i == 3) // 최소 단위 초기화
				d[i][3] = 1;
			else // 가능한 경우의 연산
				d[i][3] += d[i - 3][1] + d[i - 3][2];
		}
			
		d[i][1] %= mod; // 나눠주는 수로 나눠서 
		d[i][2] %= mod;
		d[i][3] %= mod;
	}	
}

int main()
{
	int c;
	long long n;
	scanf("%d", &c);
	calc_BU(100001);

	while (c--)
	{
		scanf("%lld", &n);
		printf("%lld\n", (d[n][1] + d[n][2] + d[n][3]) % mod); // 세 방법의 수 이므로, 더한 값을 출력하는게 옳다.
	}
	//system("pause");
	return 0;
}
```

### 10844 쉬운 계단 수
- 인접한 자리의 차이가 1이 나는 수를 계단 수라고 한다.
- 길이가 N인 계단 수의 개수를 계산한다.
  - N 길이의 계단 수 N-1 번째 수는 N 번째 수보다 크거나 작은 수가 와야 한다.
  - 즉, N 번째 수가 L 이라고 한다면, N-1 번째 수는 L-1 또는 L+1 이어야 한다.
  - D[N][L]: N자리의 수에서 끝 수가 L인 계단 수의 개수
  - N-1 번째 수가 L-1인 경우? -> D[N-1][L-1]
  - N-1 번째 수가 L+1인 경우? -> D[N-1][L+1]
- 따라서, 점화식은 D[N][K] = D[N-1][L-1] + D[N-1][L+1]
- 여기서, 최소단위 범위를 구해보면
  - N-1은 0보다 크거나 같아야 한다.
  - L+1은 9보다 작거나 같아야 한다.
  - 최소단위는 D[1][1] ~ D[1][9] 는 모두 1이다.
    - 길이가 1이고, 끝이 1 ~ 9로 끝나는 경우는 모두 1가지 씩 이므로

```c
#pragma warning(disable:4996)
#include <iostream>
#include <cstdio>
#include <vector>

using namespace std;

long long d[101][10];
long long mod = 1000000000;

void calc_BU(int n)
{
	for (int i = 1; i <= 9; i++)
		d[1][i] = 1; // 최소 단위 초기화

	for (int i = 2; i <= n; i++)
	{ // 전체 2부터 n까지 숫자들 중(1은 앞에서 초기화)
		for (int j = 0; j <= 9; j++)
		{ // 끝 수가 0부터 9까지일 때
			if (j - 1 >= 0) // 작은 수(j-1)가 0 이상일 경우
				d[i][j] += d[i - 1][j - 1];
			if (j + 1 <= 9) // 큰 수(j+1)이 9 이하일 경우
				d[i][j] += d[i - 1][j + 1];
			d[i][j] %= mod; // clipping 막기 위해 중간에 나눈 나머지를 저장한다.
		}
	}
	
}

int main()
{
	int n;
	scanf("%d", &n);
	calc_BU(n);

	long long ans = 0;
	for (int i = 0; i <= 9; i++)
		ans += d[n][i]; // 문제에선 가능한 모든 경우의 수를 물어봤으므로, 0으로 끝날 때 부터 9로 끝날때까지 경우의 수를 모두 더한게 답이다.
	printf("%lld\n", ans%mod); // 정답을 clipping 처리해서 출력
	//system("pause");
	return 0;
}
```

### 11057 오르막 수
- 오르막 수는 수의 자리가 오름차순을 이루는 수를 말한다.
  - __인접한 수가 같아도 오름차순이다.__
- 길이가 N인 오르막 수의 개수를 10,007로 나눈 나머지를 출력한다.

- 점화식: D[i][j]= 길이가 i이고, 마지막 숫자가 j인 오르막 수의 개수
  - 마지막 N번째 수가 L이면, 이전 수인 N-1번째 수는 1부터 L까지 올 수 있다.
    - D[N][L] = D[N-1][0] + D[N-1][1] + ... + D[N-1][L]
  - D[i][j] = D[i-1][0] + D[i-1][1] + ... + D[i-1][j]
- 최소 단위: D[1][j] = 1
  - 길이가 1이고, 마지막 숫자가 j(0~9)인 오르막수는 자기 자신 1개이므로

```c
#pragma warning(disable:4996)
#include <iostream>
#include <cstdio>
#include <vector>

using namespace std;

int d[1001][10];
int mod = 10007;

void calc_BU(int n)
{
	for (int i = 0; i <= 9; i++)
		d[1][i] = 1; // 최소 단위 초기화

	for (int i = 2; i <= n; i++)
	{ // 길이 2인 경우부터 n인 경우까지
		for (int j = 0; j <= 9; j++)
		{ // 마지막 수가 0부터 9까지일 때
			int sum = 0;
			for (int k = 0; k <= j; k++)
			{ // i-1 길이 수는 0부터 k까지 (최대 가능한 경우가 k=j일때이므로)
				sum += d[i - 1][k]; // 모든 경우의 수를 더하고
			}
			d[i][j] = (sum) % mod; //나눈 나머지 저장
		}
	}

}

int main()
{
	int n;
	scanf("%d", &n);
	calc_BU(n);

	long long ans = 0;
	for (int i = 0; i <= 9; i++)
		ans += d[n][i]; // 길이 n인 수의 오르막 수 가능한 경우는 마지막 수 모두를 더해줘야 함 (0부터 9까지)
	printf("%lld\n", ans%mod);
	//system("pause");
	return 0;
}
```

### 2193 이친수
- 이진수 중 아래의 두 조건을 만족하는 수를 이친수로 정의
  1. 이친수는 0으로 시작하지 않는다.
  2. 이친수에서는 1이 두 번 연속으로 나타나지 않는다. (11을 갖고 있지 않음)
- 주어지는 N 자리의 이친수 개수를 출력

- 위의 두 조건을 고려해보면 마지막 N번째가 1로 끝나는 경우와 0으로 끝나는 경우를 나눌 수 있음
  1. N번째 수가 0으로 끝나는 경우
    - N-1번째 수는 0이 오나 1이 오나 상관이 없다.
    - 이 경우의 수는 D[N-1]
  2. N번째 수가 1로 끝나는 경우
    - N-1번째는 무조건 0이 와야 한다.
    - 이렇게 되면 N-2번째는 0이나 1이 와도 상관이 없다.
    - 이 경우의 수는 D[N-2]
- 따라서, 점화식은 다음과 같음
  - D[N]: N 자리의 이친수 개수
  - D[N] = D[N-1] + D[N-2]
- 최소 단위를 구하면
  - D[0] = 0
    - 0자리의 이친수 개수는 0개이므로
  - D[1] = 1
    - 1자리의 이친수 개수는 1인 경우 하나이므로 1

```c
#pragma warning(disable:4996)
#include <iostream>
#include <cstdio>
#include <vector>

using namespace std;

long long d[91];

void calc_BU(int n)
{
	d[1] = 1; // 최소 단위 초기화
	for (int i = 2; i <= n; i++)
	{ // 가능한 경우에 대해
		d[i] = d[i - 1] + d[i - 2]; // 점화식의 관계대로 정의
	}
}

int main()
{
	int n;
	scanf("%d", &n);

	calc_BU(n);

	printf("%lld\n", d[n]); // clipping 위해 long long 씀

	//system("pause");
	return 0;
}
```

### 9465 스티커
- https://www.acmicpc.net/problem/9465
- 2xN 사각형 스티커맵이 주어질 때, 조건을 만족하는 스티커 점수의 최댓값을 출력
  - 스티커를 뜯으면 상하좌우 모두 같이 뜯긴다.

- D[N]: N 길이 사각형에서 점수의 최댓값
  - 마지막 N 번째 칸이 상하 순서대로 OX, XO, XX 로 위만 뜯기고, 아래만 뜯기고, 아무것도 뜯기지 않는 3가지 경우로 나눌 수 있음
  - 각각 0은 OX, 1은 XO, 2는 XX로 매핑
  - D[N][0] = max(D[N-1][1], D[N-1][2]) + N번째 위(OX)뜯은 점수
    - N 번째에는 윗 스티커만 뜯었으므로(OX), N-1번째에는 XO, XX가 올 수 있다.
  - D[N][1] = max(D[N-1][0], D[N-1][2]) + N번째 아래(XO)뜯은 점수
    - N 번째에는 아래 스티커만 뜯었으므로(XO), N-1번째에는 OX, XX가 올 수 있다.
  - D[N][2] = max(D[N-1][0], D[N-1][1], D[N-1][2]) -> N번째에는 스티커를 뜯지 않았다.
    - N 번째에는 스티커를 뜯지 않았으므로(XX), N-1번째에는 OX, XO, XX가 올 수 있다.
    - 또한 N번째 스티커는 뜯지 않았기에 뜯었을 때의 점수를 더할 필요 없다.
- 최소 단위 초기화
  - D[0][0] = D[0][1] = D[0][2] = 0
    - 모두 길이 0인 스티커맵에서 뜯는 경우는 없으므로 0으로 초기화된다.

```c
#pragma warning(disable:4996)
#include <iostream>
#include <cstdio>
#include <algorithm>

using namespace std;

long long d[100001][3]; // 0: ox, 1: xo, 2: xx
long long scores[100001][2]; // 점수 저장용

void calc_BU(int n)
{
	d[0][0] = 0; // 최소 단위 초기화
	d[0][1] = 0;
	d[0][2] = 0;
	for (int i = 1; i <= n; i++)
	{ // 가능한 경우에 대해
		d[i][0] = max(d[i - 1][1], d[i - 1][2]) + scores[i][0]; // 위만 뜯는 경우, 최댓값을 구하고 뜯은 점수를 더한다.
		d[i][1] = max(d[i - 1][0], d[i - 1][2]) + scores[i][1]; // 아래만 뜯는 경우, 최댓값을 구하고 뜯은 점수를 더한다.
		d[i][2] = max(d[i - 1][0], max(d[i - 1][1], d[i - 1][2])); // 뜯지 않은 경우, 최댓값만 구한다.
	}
}

int main()
{
	int c;
	scanf("%d", &c);

	while (c--)
	{
		long long n;
		scanf("%lld", &n);
    
    // 점수를 입력받아 순서대로 저장하고
		for (int i = 1; i <= n; i++)
		{
			scanf("%lld", &scores[i][0]);
		}
		for (int i = 1; i <= n; i++)
		{
			scanf("%lld", &scores[i][1]);
		}

		calc_BU(n); // 계산

		long long ans = 0;
		for (int i = 1; i <= n; i++)
		{ // 전체 맵 중 최댓값이 결과가 되므로, 모든 경우에 대해 최댓값을 구해 ans에 저장한다.
			ans = max(max(ans, d[i][0]), max(d[i][1], d[i][2]));
		}

		printf("%lld\n", ans);
	}

	//system("pause");
	return 0;
}
```

### 2156 포도주 시식
- N 길이의 포도주 잔을 아래의 조건을 만족하며 마실 때, 마실 수 있는 최대의 양
  1. 포도주 잔을 선택하면 남기지 않고 모두 마시며, 마신 후엔 제자리에 둠
  2. 연속으로 놓여있는 3잔을 마실 수 없음

- 문제를 나눠보면
  - N번째 잔이 0잔 연속 마신 경우(N번째 안마심)
  - N번째 잔이 1잔 연속 마신 경우(N번째 마심)
  - N번째 잔이 2잔 연속 마신 경우(N-1, N번째 마심)
- 점화식을 구해보면
  - D[N][연속마신잔]
  - D[N][0] = max(D[N - 1][0], D[N - 1][1], d[N - 1][2])
    - N번째에서 0잔 마신 경우, N-1번째에는 안마셨을 수도, 1잔 연속, 2잔 연속 마셨을 수도 있다.
    - 이 때의 최댓값을 저장
  - D[N][1] = D[N-1][0] + N번째 잔 포도주
    - N번째에서 1잔 마신 경우, N-1잔에선 무조건 안마셨어야 하고, N번째 마신 잔 양을 더하면 됨
  - D[N][2] = D[N-1][1] + N번째 잔 포도주
    - N번째에서 2잔 마신 경우, N-1잔에선 무조건 마셨어야 하고, N번째 마신 잔 양을 더하면 됨
- 최소 단위를 구하면
  - D[1][0]: 1번째에서 안마신 경우이므로 마신 양이 없다. 따라서 0
  - D[1][1]: 1번째에서 1잔연속 마신 경우이므로, 현재 마신 잔을 더한다.
  - D[1][2]: 1번째에서 2잔 연속 마신 경우다. 불가능하다. 0으로 초기화.

```c
#pragma warning(disable:4996)
#include <iostream>
#include <cstdio>
#include <algorithm>

using namespace std;

long long d[10001][3]; // 0: x, 1: 1잔연속, 2: 2잔연속
int podo[10001]; // 포도주 잔에 담긴 양 저장

void calc_BU(int n)
{
	d[1][0] = 0; // 최소 단위 초기화
	d[1][1] = podo[1];
	d[1][2] = 0;
	for (int i = 1; i <= n; i++)
	{ // N번째 잔까지
		d[i][0] = max(d[i - 1][0], max(d[i - 1][1], d[i - 1][2])); // 앞의 정의에 따름
		d[i][1] = d[i - 1][0] + podo[i];
		d[i][2] = d[i - 1][1] + podo[i];
	}
}

int main()
{
	int n;
	scanf("%d", &n);

	for (int i = 1; i <= n; i++)
	{
		scanf("%d", &podo[i]);
	}

	calc_BU(n);

	printf("%lld\n", max(d[n][0], max(d[n][1], d[n][2]))); // 각 N번째가 0, 1, 2경우일 때의 최댓값이 결국 정답이 된다.

	//system("pause");
	return 0;
}
```

- 이 문제는 2차원 말고 1차원으로도 풀 수 있다.
- i번째를 마시지 않은 경우, i-1번째는 상관 없이 아무거나 올 수 있다.
  - 따라서 D[i-1] 로 정의
- i번째를 1연속으로 마신 경우, i-1번째는 무조건 안 마신거다. 그러면 i-2번째엔 마시던, 안마시던 상관 없다.
  - 따라서 D[i-2] + 현재 i번째 마신 양
- i번째를 2연속으로 마신 경우, i번째, i-1번째는 무조건 마신거다. i-2번째에는 무조건 안마셔야 한다.(3잔 연속 x). 그러면 i-3번째에는 마시던, 안마시던 상관 없다.
  - 따라서 D[i-3] + i-1번째 마신 양 + i번째 마신 양

```c
#include <iostream>
using namespace std;
int a[10001];
int d[10001];
int main() {
    int n;
    cin >> n;
    for (int i=1; i<=n; i++) {
        cin >> a[i]; // a엔 포도주 양들이 저장된다.
    }
    d[1] = a[1]; // 최소 단위 초기화(1번째엔 마셨을 때가 무조건 최댓값)
    d[2] = a[1]+a[2]; // 최소 단위 초기화(2번째엔 둘 다 마셨어야 최댓값)
    for (int i=3; i<=n; i++) {
        d[i] = d[i-1];
        if (d[i] < d[i-2] + a[i]) { // 만약 1연속 마신 양이 현재 양보다 크다면
            d[i] = d[i-2] + a[i]; // 큰 값으로 초기화
        }
        if (d[i] < d[i-3] + a[i] + a[i-1]) { // 만약 2연속 마신 양이 현재 양보다 크다면
            d[i] = d[i-3] + a[i] + a[i-1]; // 큰 값으로 초기화
        }
    }
    printf("%d\n",d[n]);
    return 0;
}
```
