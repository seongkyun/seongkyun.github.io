---
layout: post
title: 우분투 키보드 한글 입력 설정
category: others
tags: [Ubuntu, 한글입력]
comments: true
---

# 우분투 키보드 한글 입력 설정
- 한글 설치
  - `sudo apt-get install fcitx-hangul`로 한글 설치
  - System settings/Language support 들어가서 설치되지 않은 언어팩 모두 설치
  - Keyboard input method system을 ibus에서 fcitx로 변경
  - 재부팅
- 한영 전환 설정
  - AllSettings > Keyboard > Shortcuts Tab > Typing을 선택
  - Switch to Next source, Switch to Previous sourc, Compose Key, Alternative Characters Key를 모두 Disabled로 선택
    - Disabled 선택하기 위해 backspace를 누르면 됨
  - Compose Key의 Disabled를 길게 눌러 Right Alt를 선택
  - Switch to next source는 한영키를 눌러 Multikey를 선택
    - 반드시 Compose Key 설정이 먼저되어야 Multikey가 선택됨
  - 모든 창을 닫고 우측 위에 키보드 모양의 fcitx아이콘을 눌러 Configure Current Input Method를 선택
  - Keyboard-English(US)가 있다면 +를 눌러 Hangul을 추가
    - Only Show Current Language는 체크 해제
    - Korean이 아닌 Hangul을 선택해야 함
  - Global Config tab에서 Trigger Input Method는 한/영키를 눌러 Multikey로 설정(왼쪽 오른쪽 모두)
  - Extrakey for trigger input method는 Disabled로 설정
  - Global Config tab에서 Program > Share State Among Window > All을 선택
