---
layout: post
title: 쉘 스크립트(shell script) 작성, 명령어 알아보기
category: others
tags: [shell script, ubuntu]
comments: true
---

# 쉘 스크립트(shell script) 작성, 명령어 알아보기

딥러닝 모델을 학습시키다 보면 중간중간 파라미터를 바꿔줘야 할 때가 생긴다. 이 때 가장 요긴하게 쓰이는것이 쉘 스크립트다.
본 글에서는 쉘 스크립트를 만들고 실행 시키고, 유용한 명령어들을 간단하게 정리한다.

## 쉘 스크립트 작성
- `touch filename.sh`로 쉘 스크립트 파일 생성
- 첫 줄에 `#!/bin/bash`를 적고, 그 밑에줄부터 내용을 쓴다.
  - 스크립트 파일을 bash 쉘로 실행시킨다는 의미.
  - 구지 `#!/bin/bash`을 적지 않아도 실행이 되지만, 그것은 리눅스(우분투)배포판이 기본적으로 bash 쉘로 설정되어 있기 때문.
  - `#!/bin/bash`를 작성하는 것이 쉘 파일임을 미리 알려주는것이므로 다른 쉘간의 오류를 방지 할 수 있음!
  - `#!/bin/bash`, `#!/bin/sh` 둘 다 혼용이 가능함(둘중 하나만 적으면 됨)
- 작성 예시
```
{%raw%}
#!/bin/bash
echo 'Hello, World!'
{%endraw%}
```

## 파일 쓰기/읽기 권한 주기

## 실행 시

## 간단한 명령어들

