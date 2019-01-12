---
layout: post
title: Github 공부 시작 / 설치 및 환경 만들기 
category: git
tags: [github, 지옥에서 온 git]
comments: true
---

# Github 공부 시작/설치 및 환경 만들기

- 평소에 github를 다루긴 하지만 제대로 공부해서 다루진 않았다.
- 보통은 사용하는 명령어들만 사용해서 별로 불편함을 느끼진 못했다.
  - `git clone address` 라던가 `git submodule update —init —recursive`같은..
- 코드가 복잡해질수록 버전관리의 필요성을 느끼게 된다.
- 따라서 생활코딩의 ‘지옥에서 온 git’이라는 강좌로 공부를 시작해보려 한다.
  - (https://opentutorials.org/course/2708/15242)
## 설치 방법
- `sudo apt-get install git`으로 설치 가능
- 설치 후 커맨드라인에 `git`이라고 친 후 git 관련 명령어가 뜨면 설치 된 상태

## 프로젝트 환경 만들기
- `mkdir folder_name`: 폴더 생성
- `cd folder_name`: 해당 폴더로 이동
- `git init`: 해당 디렉터리를 git의 저장소로 만듦
	- 해당 디렉터리에서 작업을 진행하겠다는것을 git에 알려주는 역할
- 완료 시 .git폴더가 생성 됨
	-  안에는 버전 관련 정보들이 담겨있게 됨

## git이 관리할 대상으로 파일 등록
- 관리 할 파일에 대해서 `git add filename` 으로 git이 파일을 추적하도록 명령함
- `git status`로 프로젝트 폴더의 상태를 확인 할 수 있음
	- 앞으로 해당 파일의 버전이 관리가 됨
