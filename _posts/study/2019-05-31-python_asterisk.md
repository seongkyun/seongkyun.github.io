---
layout: post
title: Python asterisk에 대해
category: study
tags: [Python, Asterisk]
comments: true
---

# Python asterisk에 대해
- 참고 글
  - https://mingrammer.com/understanding-the-asterisk-of-python/
  
- 용도
  - 곱셈 및 거듭제곱 연산으로 사용할 때
  - 리스트형 컨테이너 타입의 데이터를 반복 확장하고자 할 때
  - 가변인자 (Variadic Parameters)를 사용하고자 할 때
    - positional arguments만 받을 때
    - keyword arguments만 받을 때
    - positional arguments와 keyword arguments를 모두 받을 때
  - 컨테이너 타입의 데이터를 Unpacking 할 때

- 파이썬은 타 언어에 비해 비교적 연산자 및 종류가 풍부한 편으로, asterisk(\*)는 곱셈 이상의 여러 의미를 가짐
- 본 글에선 파이썬 코드를 더 파이썬스럽게 쓰기위한 asterisk(\*)에 대한 여러 연산을 살펴봄

## 1. 곱셈 및 거듭제곱 연산으로 사용할 때
- 일반적으로 사용하는 곱셈 연산 및 거듭제곱 연산까지 내장 기능으로 지원

```python
>>> 2 * 3
6
>>> 2 ** 3
8
>>> 1.414 * 1.414
1.9993959999999997
>>> 1.414 ** 1.414
1.6320575353248798
```

## 2. 리스트형 컨테이너 타입의 데이터를 반복 확장하고자 할 때
- 파이썬에선 \*을 숫자형 데이터 뿐만 아니라 리스트형 컨테이너 타입에서 데이터를 반복적으로 확장하기 위해 사용

```python
# 길이 100의 제로값 리스트 초기화
zeros_list = [0] * 100

# 길이 100의 제로값 튜플 선언
zeros_tuple = (0,) * 100

# 리스트 3배 확장 후 연산
vector_list = [[1, 2, 3]]
for i, vector in enumerate(vector_list * 3):
  # i + 1 의 수를 vector_list의 각 숫자(e)에 곱한 후 list를 출력
  print("{0} scalar product of vector: {1}".format((i + 1), [(i + 1) * e for e in vector]))
# 1 scalar product of vector: [1, 2, 3]
# 2 scalar product of vector: [2, 4, 6]
# 3 scalar product of vector: [3, 6, 9]
```

## 3. 가변인자 (Variadic Parameters)를 사용하고자 할 때
- 함수에서 가변인자를 필요로 할 때 (들어오는 인자의 갯수를 모르거나 어떤 인자라도 모두 받아서 처리해야 할 때) 사용
- 파이썬에서는 positional arguments와 keyworkd arguments의 두 가지 종류의 인자가 존재하며, 전자(positional arguments)는 위치에 따라 정해지는 인자, 후자(keyword arguments)는 키워드를 가진(이름을 가진)인자를 의미함
- Variadic positional/Keyword argument를 살피기 전 간단히 두 인자의 차이에 대해 살펴보면 아래와 같음

```python
# 2~4명의 주자로 이루어진 달리기 대회 랭킹을 보여주는 함수
def save_ranking(first, second, third=None, fourth=None):
    rank = {}
    rank[1], rank[2] = first, second
    rank[3] = third if third is not None else 'Nobody'
    rank[4] = fourth if fourth is not None else 'Nobody'
    print(rank)

# positional arguments 2개 전달
save_ranking('ming', 'alice')
# positional arguments 2개와 keyword argument 1개 전달
save_ranking('alice', 'ming', third='mike')
# positional arguments 2개와 keyword arguments 2개 전달 (단, 하나는 positional argument 형태로 전달)
save_ranking('alice', 'ming', 'mike', fourth='jim')
```

- 위 함수는 `first`, `second`라는 두 개의 positional argument와 `third`, `fourth`라는 두 개의 keyword arguments를 받고 있음
- Positional arguments의 경우 생략이 불가능하며 갯수대로 정해진 위치에 인자를 전달해야 함
- Keyworkd arguments`의 경우 함수 선언시 default 값을 설정 할 수 있으며, 만야 ㄱ인자를 생략 할 시 해당 default 값이 인자의 값으로 들어감
  - 생략이 가능한 형태의 인자
- Keyworkd arguments는 positional arguments 전에 선언이 불가하므로 아래와같은 경우 에러 발생

```python
def save_ranking(first, second=None, third, fourth=None):
  ...
# 에러 발생!
```

- 위의 경우 만약 최대 4명의 주자가 아니라 10명 또는 그 이상의 정해지지 않은 주자가 있다고 하면 10개의 인자를 선언하기도 번거로울뿐더러 주자의 수가 미정일 경우 위와 같은 형태로는 처리가 불가능함
- __이 때 사용하는게 바로 가변인자(Variadic arguments)__
- 가변인자는 위에서 설명한 positional arguments와 keyworkd arguments에 모두 사용 가능하며 사용방법은 아래와 같음

### Positional arguments만 받을 때

```python
def save_ranking(*args):
  print(args)

save_ranking('ming', 'alice', 'tom', 'wilson', 'roy')
# ('ming', 'alice', 'tom', 'wilson', 'roy')
```

### Keyword arguments만 받을 때

```python
def save_ranking(**kwargs):
  print(kwargs)

save_ranking(first='ming', second='alice', fourth='wilson', third='tom', fifth='roy')
# {'first': 'ming', 'second': 'alice', 'fourth': 'wilson', 'third': 'tom', 'fifth': 'roy'}
```

### Positional과 keyword arguments를 모두 받을 때

```python
def save_ranking(*args, **kwargs):
    print(args)
    print(kwargs)
    
save_ranking('ming', 'alice', 'tom', fourth='wilson', fifth='roy')
# ('ming', 'alice', 'tom')
# {'fourth': 'wilson', 'fifth': 'roy'}
```

- 위에서 `*args`는 임의의 갯수의 positional arguments를 받음을 의미하며, `**kwargs`는 임의의 갯수의 keyword arguments를 받음을 의미
  - 이 때 `*args`, `**kwargs`형태로 가변인자를 받는것을 __packing__ 이라고 함
- 위의 예시에서 볼 수 있듯이 임의의 갯수와 임의의 키값을 갖는 인자들을 전달하고 있음
  - `*args`: Positional 형태로 전달되는 인자로 __tuple__ 형태로 저장
  - `**kwargs`: Keyword 형태로 전달되는 인자로 __dict__ 형태로 저장
- 앞의 예시와 동일하게 positional 선언 전의 keyword 선언은 에러를 발생시킴

```python
def save_ranking(**kwargs, *args):
  ...
# 에러 발생!
```

- 이러한 가변인자는 매우 일반적으로 사용되므로 오픈소스 코드에서도 쉽게 발견 가능
- 보통은 코드의 일관성을 위해 관례적으로 `*args`와 `**kwargs`를 사용되지만, `*required`나 `**optional`처럼 인자명도 일반변수와 같이 원하는대로 지정이 가능함
  - 단, 오픈소스 프로젝트 진행하고 있고 별 특별한 의미가 없다면 관례적인 표현인 `*args`와 `**kwargs`를 따르는게 좋음

## 4. 컨테이너 타입의 데이터를 unpacking 할 때
- \* 는 컨테이너 타입의 데이터를 unpacking하는 경우에도 사용 가능함
- 이는 3번의 경우와 유사한 원리로 종종 사용할만한 연산방법.
- 가장 쉬운 예로 _list_ 나 _tuple_ 또는 _dict_ 형태의 데이터를 가지고 있고 어떤 함수가 가변인자를 받는 경우에 사용 가능함

```python
from functools import reduce

primes = [2, 3, 5, 7, 11, 13]

def product(*numbers):
    p = reduce(lambda x, y: x * y, numbers)
    return p

product(*primes)
# 30030

product(primes)
# [2, 3, 5, 7, 11, 13]
```

- `product()` 함수가 가변인자를 받고 있기 때문에 리스트의 데이터를 모두 unpacking하여 함수에 전달해야 함
- 이 경우 함수에 값을 전달할 때 `*primes`와 같이 전달하면 `primes` 리스트의 모든 값들이 unpacking되어 `numbers`라는 리스트에 저장됨
- 만약 이를 `primes` 그대로 전달하려면 이 자체가 하나의 값으로 쓰여 `numbers`에는 `primes`라는 원소 하나가 존재하게 됨
  - numbers = [primes] = [[2, 3, 5, 7, 11, 13]]
- _tuple_ 또한 _list_ 와 동일하게 동작하며 _dict_ 의 경우 \* 대신 \*\*을 사용하여 동일한 형태로 사용 가능

```python
headers = {
    'Accept': 'text/plain',
    'Content-Length': 348,
    'Host': 'http://mingrammer.com'
}

def pre_process(**headers):
    content_length = headers['Content-Length']
    print('content length: ', content_length)

    host = headers['Host']
    if 'https' not in host:
        raise ValueError('You must use SSL for http communication')

pre_process(**headers)
# content length:  348
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "<stdin>", line 7, in pre_process
# ValueError: You must use SSL for http communication
```

- 또 다른 형태의 unpacking이 한 가지 더 존재하는데, 이는 ㅎ마수의 인자로써 사용하는게 아닌 리스트나 튜플 데이터를 다른 변수에 가변적으로 unpacking하여 사용하는 형태임

```python
numbers = [1, 2, 3, 4, 5, 6]

# unpacking의 좌변은 리스트 또는 튜플의 형태를 가져야하므로 단일 unpacking의 경우 *a가 아닌 *a,를 사용
*a, = numbers
# a = [1, 2, 3, 4, 5, 6]

*a, b = numbers
# a = [1, 2, 3, 4, 5]
# b = 6

a, *b, = numbers
# a = 1
# b = [2, 3, 4, 5, 6]

a, *b, c = numbers
# a = 1
# b = [2, 3, 4, 5]
# c = 6
```

- 여기서 `*a`, `*b`로 받는 부분들은 우변의 리스트 또는 튜플이 unpacking된 후 다른 변수들에 할당된 값 외의 나머지 값들을 다시 하나의 리스트로 packing함
  - 3번의 가변인자 packing과 동일한 개념

## 결론
- 3번의 내용이 매우 자주 사용되는 중요한 기능이자 자주 헷갈릴 수 있는 부분이기에 숙지 필요

