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


