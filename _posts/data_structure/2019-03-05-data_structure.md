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




