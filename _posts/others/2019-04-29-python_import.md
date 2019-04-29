---
layout: post
title: 파이썬 상위, 하위, 동일 폴더 내 모듈 from, import 하기
category: others
tags: [Ubuntu, Python]
comments: true
---

# 파이썬 상위, 하위, 동일 폴더 내 모듈 from, import 하기
- 참고 글: https://brownbears.tistory.com/296

- 아래와 같은 프로젝트가 존재한다고 가정한다.

```
{%raw%}
project
  -- test
    +-- sub1
      -- __init__.py
      -- a.py
      -- b.py
    +-- sub2
      -- __init__.py
      -- c.py
      -- d.py
    -- e.py
    -- f.py
    -- __init__.py
  -- g.py
{%endraw%}
```

## e.py 에서 다른 모듈 참조(하위 폴더 내 파일, 동일 폴더 내 파일 참조)

```python
# a.py 참조시
from sub1 import a

# f.py 참조시
import f
from . import f
```

## 상위 폴더 내 파일 참조
- `a.py`에서 sub2에 있는 `c.py`를 참조하기

### 부모폴더의 절대경로를 참조 path에 추가
- 모듈의 시작부분 import 위에 아래와 같은 코드를 추가

```python
# a.py 에서 c.py 참조시
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import c
```

- 현재 모듈의 절대경로를 알아내서 상위 폴더 절대경로를 참조 path에 추가하는 방식
- 1 단계의 상위 폴더 경로를 추가할 때 사용
- `a.py`에서 `g.py`를 참조하는 경우 2단계 상위 폴더 경로를 추가해야 하므로 아래와 같음

```python
# a.py에서 g.py 참조시
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))))
import g
```
