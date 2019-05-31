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

3. 가변인자 (Variadic Parameters)를 사용하고자 할 때
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




