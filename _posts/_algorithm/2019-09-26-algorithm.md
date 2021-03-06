---
layout: post
title: 알고리즘 기초-다이나믹 프로그래밍
category: algorithm
tags: [algorithm]
comments: true
---

# 알고리즘 기초-다이나믹 프로그래밍
- 큰 문제를 작은 문제로 나눠서 푸는 알고리즘
  - 큰 문제를 작게 나눠서 풀고, 이를 근거로 다시 큰 문제를 푼다.
- Dynamic Programing이라는 이름은 만든사람이 그냥 멋있어서 붙인 이름.
  - 다이나믹과 관계 없음

- 즉, 문제에는 크기가 있어야 한다.
  - 그래야 큰 문제를 작은 문제들로 나눌 수 있으므로
  - 각각 작은 문제들을 풀고, 그것들을 더해서 원래의 답을 구함
- 다이나믹 프로그래밍 -> 작은 문제가 여러번 나와서 그 정답이 모두 항상 같다.
- 분할 정복 -> 큰 문제를 작게 나눠 풀되, 작은 문제가 중복되지는 않는다.
  - 둘 다 같은 방식

- 다음의 두 가지 속성을 만족해야 다이나믹 프로그래밍으로 문제를 풀 수 있음
  1. Overlapping Subproblem
    - 부분문제가 겹치는 속성
  2. Optimal Substructure
    - 최적 부문 구조. 작은 문제의 정답이 항상 같은 경우.

### Overlapping Subproblem
- 피보나치 수
  - 대표적인 DP 문제
  - $F_0=0$
  - $F_1=1$
  - $F_n=F_{n-1}+F_{n-2}, (n>=2)$
- 재귀적으로 구성되어 있으며 ($F_n=F_{n-1}+F_{n-2}$), 큰 문제 ($F_n$)와 작은 문제들 ($F_{n-1}+F_{n-2}$)로 나뉨
  - 크기는 n으로 정의됨
  - 문제나누기는 n을 n-1, n-2로 나누는것으로 정의됨
- 이제 작은 문제들이 겹치는지 확인해야 한다.

- 정리하자면
  - 큰 문제: N번째 피보나치 수를 구하는 문제
  - 작은 문제: N-1번째 피보나치수를 구하는 문제, N-2번째 피보나치수를 구하는 문제
- 그리고 이 과정이 중복적으로 반복되는 구조

### Optimal Substructure
- 큰 문제와 작은 문제는 상대적임
- 큰 문제와 작은 문제를 같은 방법으로 풀 수 있으며, 문제를 작은 문제로 쪼갤 수 있는 구조
- 즉, 큰 문제의 정답을 작은 문제의 정답에서 구할 수 있음
  - 예를 들어, 서울-부산 경로의 최단거리는 서울-대전-대구-부산라고 할 때
    - 대전-부산의 최단거리는 대전-대구-부산 이 되어야 함
    - 대전-울산-부산이 가장 빠른 경로일 수 없음

- 이러한 Optimal structure를 만족한다면, 문제의 크기에 상관없이 어떤 한 문제의 정답은 일정함
  - 피보나치 수!
  
- 다이나믹 프로그래밍에서 각 문제는 한 번만 풀어야 함
- Optimal Substructure를 만족하기에 __같은 문제는 구할때마다 정답이 같다.__
- 따라서, __정답을 한번 구했으면, 그 정답을 어딘가에 메모해 둔다.__
  - 정답은 매번 동일하므로
- 이런 메모하는 것을 코드의 구현에서는 배열에 저장하는것으로 할 수 있음
- 메모를 한다 해서 영어로 Memorization이라고 한다.
  - 미리 연산해둔 결과를 저장해두면(Memorization) 두 번째 참조부터는 빠르게 가능함

### 피보나치 수

```c
int fibonacci (int n)
{
  if (n <= 1)
  {
    return n;
  }
  else
  {
    return fibonacci(n-1) + fibonacci(n-2)
  }
}
```

- 이 때의 빅-오는 $O(2^N)$

- 만약 위 함수를 이용해서 f(5)를 호출한다면 아래와 같이 겹치는 호출이 발생함

<center>
<figure>
<img src="/assets/post_img/algorithm/2019-09-26-algorithm/del_fig1.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 즉, 한번 답을 구하고 어디에 메모해둔 후 중복 호출시엔 메모된 값을 리턴하면 됨
  - f(3), f(2) 값을 배열에 저장해두고, 다음 참조시엔 저장된 값을 꺼내쓰면 됨

```c
int memo[100];
int fibonacci (int n)
{
  if(n<=1)
  {
    return n;
  }
  else
  {
    if(memo[n] > 0) // 함수 호출 전 확인해서 값이 존재하면 사용
    {
      return memo[n];
    }
    memo[n] = fibonacci(n-1) + fibonacci(n-2);
    return memo[n];
  }
}
```

- 위와같은 프로그래밍 흐름을 다이나믹 프로그래밍이라고 한다.
  - 이 때의 시간복잡도는 O(n)
- 즉, 다이나믹 프로그래밍은 중복되는 작은 문제를 풀고, 앞의 결과를 저장하고 가져와 다음부터 사용하는 것임

- 다이나믹 프로그래밍의 구현에는 두 가지 방법이 있다.
  1. Top-down (재귀함수)
  2. Bottom-up (for문 쓰기)

## Top-down
1. 문제를 작은 문제로 나눈다.
  - fibo(n) = fibo(n-1) + fibo(n-2)
2. 작은 문제를 푼다.
  - fibo(n-1), fibo(n-2) 풀기
3. 작은 문제를 풀었으니 이제 큰 문제를 푼다.
  - 이를 이용해 fibo(n) 구하기
- Top-down 방식의 풀이는 재귀 호출을 이용해 문제를 쉽게 풀 수 있다.

## Bottom-up
- 작은 문제를 다 풀고, 다음 큰 문제, 다음 큰 문제.. 처럼 점점 채워가는 방식
1. 문제를 크기가 작은 문제부터 차례대로 푼다
2. 문제의 크기를 조금씩 크게 만들면서 문제를 점점 푼다.
3. 작은 문제를 풀면서 왔기에 큰 문제는 항상 풀 수 있음
4. 그러다보면 언젠간 풀어야 하는 문제를 풀 수 있음
  - 약간 Greedy 와 비슷한 꼴

- 이를 이용해 피보나치를 구현하면 다음과 같다.

```c
int d[100];
int fibonacci(int n)
{
  d[0] = 0;
  d[1] = 1;
  for (int i=2; i<=n; i++)
  {
    d[i] = d[i-1] + d[i-2];
  }
  return d[n];
}
```

- Top-down, Bottom-up중 자신있는 것을 사용하면 됨
- 서로 못 푸는 case가 존재하지만, 이는 나중에 고려
- 둘 중 뭐가 더 나은지(시간복잡도)는 알 수 없음

## 문제 풀이 전략
- 어떻게 풀 것인가?
- 피보나치처럼 점화식을 만들고 풀어야 함.
  1. 점화식의 정의를 글로 나타내고
    - n 번째 피보나치 수 구하기
    - 그리고 저장공간은 1 차원으로, 변수 개수만큼의 메모 배열을 만듦
  2. 어떻게 문제를 작은 문제로 나누나
    - Top-down 경우엔 재귀 호출 인자의 개수 등도 정해야 함
  3. 어떻게 원래 문제를 풀 수 있는가
    - 문제를 나눈후 수식을 이용해 문제를 표현
    - $F_n=F_{n-1}+F_{n-2}$
