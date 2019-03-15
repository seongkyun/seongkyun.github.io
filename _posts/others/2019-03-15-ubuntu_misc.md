---
layout: post
title: Ubuntu 각종 필요 명령어 메모
category: others
tags: [Ubuntu, 우분투]
comments: true
---

# 프로세스 강제종료
- 코드 실행중 Ctrl+c 로도 프로세스가 중단되지 않는 경우 사용한다
- 1. GPU코드인 경우 nvidia-smi로 해당 프로세스의 PID를 알아낸다.
- 2. `kill -9 PID` 또는 `kill -15 PID`로 프로세스 종료

# 다중 그래픽카드 중 하나만 사용하기
- 여러 그래픽카드가 꽂혀있는 경우, 코드 내에 병렬작업처리 코드가 존재하더라도 아래의 명령어로 해당 그래픽카드만 사용 가능하다
- 새로운 터미널마다 새로 설정 해줘야 한다
- `export CUDA_VISIBLE_DEVICES=GPU_NUMBER`
  - ex. `export CUDA_VISIBLE_DEVICES=0`
- 다중 GPU로 돌아가려면 모든 gpu의 번호를 적어주면 된다
  - ex. `export CUDA_VISIBLE_DEVICES=0,1`

# SCP 파일 전송
- 파일을 전송하는 두 pc에 모두 scp가 깔려있어야 한다.
- 1. 보내려는 파일 또는 폴더가 존재하는 디렉터리로 이동
- 2. `scp -r 보낼폴더,파일명 수신PC_ID@IP_ADDRESS; 수신PC저장위치`
  - ex. `scp -r ./test han@123.456.789.10; /home/user/test`

# SSH 접속
- 커널 상 server pc에 접속할때 사용
- `ssh 접속PC_ID@IP_ADDRESS`
- 다음으로 pw 입력
  - ex. `ssh han@123.456.789.10`

# 우분투 시간 동기화
- grub를 이용한 윈도우/우분투 동시 설치 환경에서 윈도우 시간이 다르게 설정되는것을 막는다.
- `timedatectl set-local-rtc 1`

# Bashrc 오타로 인한 명령어 먹지 않는 경우
- 터미널에 `export PATH=/usr/bin:/bin` 입력
- `vim ./bashrc` 들어가서 오타 다시 
