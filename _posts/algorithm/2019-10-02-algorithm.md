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
  5. 코드로 
