---
layout: post
title: CH3. 연결 리스트 (Linked list) 1
category: data_structure
tags: [data structure, 자료구조]
comments: true
---

# CH3. 연결 리스트 (Linked list) 1

## 3-1 추상 자료형: Abstract Data Type

### 컴퓨터 공학에서의 추상 자료형 (Abstract Data Type)
- 추상 자료형은 ADT라고도 불리나 실제 의미상 약간의 차이가 있는 것처럼 느껴질 수 있음
- 이는 실제 의미에서 조금 확장된 의미로 사용되기 때문이며 실제 차이를 보이는것이 아님
  - 즉, 큰 맥락선에선 하나이며 다만 이를 나타내는 형태에서 차이가 나는 것임
- 본 책에서는 자료구조의 관점에서 ADT를 다룸

### 자료구조에서의 추상 자료형
- 지갑에서 동전을 넣고 빼는 등 지갑이 제공하는 기능들의 행위의 묘사
  - 동전을 넣고 빼는 자세한 과정이 생략되어 있음 (지갑을 연다던가..)
- 이처럼 구체적인 기능의 완성과정을 언급하지 않고 순수하게 기능이 무엇인지를 나열한 것을 가리켜 추상 자료형(ADT)라고 함
- 예시를 들기 위해 구조체를 이용하여 지갑을 정의

```c
typedef struct_wallet
{
  int coin100Num;
  int bill5000Num;
} Wallet;
```

- 위처럼 c를 기반으로 구조체를 정의하는 것은 구조체를 기반으로 지갑을 의미하는 Wallet 자료형을 정의하는것
- 하지만 컴퓨터공학 측면에서 위의 구조체 정의만으로 Wallet이라는 자료형의 정의가 완성되는 것이 아님
  - Wallet을 기반으로 하는 연산의 종류를 결정하는 것도 자료형 정의의 일부로 보아야 함
  - 이러한 연산의 종류가 결정되었을 때 자료형의 정의가 완성됨
- 지갑에 대한 예시를 이어 들면 아래와 같음
  - Wallet을 기반으로 제공할 수 있는 기능 관련 연산

```c
int TakeOutMoney(Wallet *pw, int coinNum, int billNum); //돈을 꺼내는 연산
void PutMoney(Wallet *pw, int coinNum, int billNum); // 돈을 넣는 연산
```

- 이렇듯 C언어에서는 구조체에서 필요로 하는 연산을 함수를 이용해 정의
- 만약 Wallet 구조체가 필요로 하는 연산이 위의 두 종류가 다라면, 이로써 Wallet에 대한 자료형의 정의가 완성됨

- 즉, '자료형'의 정의에 '기능, 연산'과 관련된 내용을 명시할 수 있음
