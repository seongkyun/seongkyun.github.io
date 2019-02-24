---
layout: post
title: ubuntu virtualenv 사용방법 정리
category: others
tags: [ubuntu, virtualenv]
comments: true
---

# ubuntu virtualenv 사용방법 정리
- Virutalenv는 우분투 내에서 각종 패키지 및 라이브러리간의 버전의 관리를 쉽게 해 주는 패키지다.
- Github에서 코드를 주워 돌릴 때, 각 코드별로 요구하는 라이브러리 및 디펜던시가 모두 다르고 그에 따른 디버깅을 요하나, 매번 디버깅을 하며 코드를 수정하기엔 비효율적인면이 많다
- Virutalenv는 이를 매우 쉽게 도와주는 툴로, 다른 가상환경만 생성시켜준다면 pytorch 0.3과 0.4등 다른 버전을 동시에 설치하여 관리 할 수 있다.

## 설치방법
- `sudo apt-get install -y python3-pip`로 pip 설치
- `pip3 install virtualenv` 로 virtualenv 설치

## 가상환경 만들기
- 이 글은 python3 기준으로 작성되었으나, 파이썬 버전에 맞게 사용 가능하다.
  - 기본 python 버전에 따라 다르며, python 3.6이나 3.7이 설치된 경우 옵션을 `python3.6` 또는 `python3.7`등으로 변경 가능하다.
- 메인 시스템의 패키지를 그대로 사용하는 가상환경시의 옵션
  - 메인 시스템에서 `pip freeze`로 검색되는 모든 라이브러리를 가져와 동일하게 설치한다.
  - `virtualenv --system-site-packages -p python3 targetDirectory`
  - ex. `virtualenv --system-site-packages -p python3.6 ~/virtualenv/py36`
- 메인 시스템과 별도의 가상환경
  -`virtualenv -p python3 targetDirectory`
  - ex. `virtualenv -p python3.6 ~/virtualenv/py36`
- 파이썬의 옵션으로도 가상환경을 만들 수 있음
  - `python3 -m venv targetDirectory`

## 가상환경 실행
- `source [path]/bin/activate`
- ex. `source ~/virtualenv/py36/bin/activate`

## 가상환경 종료
- `deactivate`

## 가상환경 제거
- 해당 가상환경 폴더를 삭제하면 된다.
- `sudo rm -rf targetDirectory`
- ex. `sudo rm -rf ~/virtualenv/py36`

---

- [참고 글]

https://opentutorials.org/module/2957/17783
