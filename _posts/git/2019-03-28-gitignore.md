---
layout: post
title: Gitignore 파일 생성 및 이용
category: git
tags: [github, 지옥에서 온 git]
comments: true
---

# Gitignore 파일 생성 및 이용
- Git으로 프로젝트 관리 시 특정 파일들은 git으로 관리 할 필요가 없는 경우에 사용
  - 자동으로 생성되는 파일 및 weight 파일 등 용량이 큰 파일에 대하여 적용 가능

## Gitignore 파일 만들기
- `touch .gitignore` 로 파일 생성
  - 맨 앞에 "." 이 붙어있어야 함
- `vim .gitignore`로 제외시키고 싶은 파일 또는 폴더, 확장자를 넣어 저장한다.
  - ex 1. weight parameters가 저장된 용량이 큰 trained_nets 폴더를 제외시키고 싶다면
    - `trained_nets` 적은 후 wq로 저장
  - ex 2. \*.log 파일들을 제외시키고 싶다면
    - `*.log` 적은 후 wq로 저장
  - 이렇듯 제외 시키고 싶은 파일 또는 폴더, 확장자명을 추가할 수 있다.
- 다음을 `git add .` 로 모든 파일을 커밋 준비상태로 만든 후 `git push`로 커밋하게 되면 해당 파일들이 제외된 것을 확인 할 수 있다.
