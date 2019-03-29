---
layout: post
title: PC에 우분투(Ubuntu) CUDA개발환경 세팅하기
category: others
tags: [Ubuntu, 환경 , deep learning]
comments: true
---

# [Ubuntu] PC에 우분투(Ubuntu) CUDA개발환경 세팅하기

PC에 우분투(Ubuntu)를 설치하고, 딥러닝 개발환경을 세팅하는 방법을 정리해봤다.

## Ubuntu 설치(단일 운영체제 부팅)
- 우분투는 각종 오류가 많으므로.. 최신보다는 이전 버전을 설치하는것이 유리하다.(16.04LTS 추천)

1. 우분투 이미지 다운로드('Ubuntu 16.04'라고 치면 바로 뜸)
- [다운로드](http://releases.ubuntu.com/16.04/ubuntu-16.04.5-desktop-amd64.iso)

2. 다운로드 된 이미지를 이용하여 부팅용 USB 제작
- [Universal USB Installer](https://universal-usb-installer.kr.uptodown.com/windows) 사용

3. 완성 된 부팅 USB를 이용하여 설치하고자 하는 PC에 꽂고 부팅
- 이 때, 부팅순서 키(Board 회사마다 다름)를 눌러 USB로 부팅
- 부팅 시에는 Secure boot? Fast boot 옵션이 추가된 USB 말고 일반 USB로 선택해야 오류가 뜨질 않음

4. 설치 완료

## Tty 콘솔 진입 설정
1. Settings 가서 software 다운로드 서버 Korea에서 Main server로 변경
2. `sudo apt-get install vim`
3. `sudo vim /etc/default/grub`
4. insert키 눌러서 편집 모드 진입
5. `GRUB_CMDLINE_LINUX_DEFAULT = "quiet splash"` 를 찾아 `"quite splash nomodeset"`로 수정
6. ESC키 누르고 : 누른 후 wq 입력 후 엔터(vim editor 저장 후 나가기)
7. sudo update-grub
8. sudo reboot

## 파이썬 버전 업그레이드(Python 3.7)
- 나중에 문제가 발생할 수 있기 때문에 처음에 업그레이드 하는게 좋다.
- `sudo apt update`
- `sudo apt install software-properties-common`
- `sudo add-apt-repository ppa:deadsnakes/ppa`
- `sudo apt update`
- `sudo apt install python3.7`
- 설치 완료 후 python3 를 기본으로 설정
  - `sudo update-alternatives --config python`
    - 만약 `update-alternatives: error: no alternatives for python` 라고 등록된 버전이 없다고 뜨면
    - `sudo update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1`
    - `sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.7 2`
  - 다음으로 Python3.7을 default로 하고 엔터를 치면 됨
- 완료 후 `python --version` 으로 현재 버전 확인
- Path 변경이 있을 수 있으므로 `sudo reboot`로 재부팅
- 재부팅 후 pip 이용 위해서 `sudo apt-get install pyhton3-pip` 로 pip 설치
- 만약 pip로 패키지 설치 시 `python3.7 -m pip install package_name`로 해야 python3.7에서 import 가능! 

## 그래픽 드라이버 및 CUDA 설치(CUDA 9.0)
- CUDA를 설치하면 자동으로 그래픽 드라이버를 같이 설치해주므로 그게 오류가 적고 속편하다.

<center>
<figure>
<img src="/assets/post_img/others/2019-01-02-ubuntu_settings/fig1.PNG" alt="views">
</figure>
</center>

1. [링크](https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal)로 들어가 의 옵션에 맞게 세팅 후 "Base Installer" 다운로드

2. `Ctrl + Alt + F1` 동시에 누르기(Virtual terminal 들어가기)
- 아이디랑 비밀번호 입력(Ubuntu 설치 시 생성한 아이디 및 비밀번호)

3. CUDA 및 그래픽 드라이버 설치
- `sudo service lightdm stop` (Kill x server)
- `cd Download` (다운로드 받은 CUDA 설치 파일 존재 위치로 이동)
- `sudo chmod +x CUDA~.run` (실행 권한 주기)
- `sudo ./CUDA~.run` (CUDA 설치)

4. CUDA 설치 시 옵션
- Install NVIDIA Accelerated Graphic driver? Yes
- Install the CUDA 9.0 Toolkit? Yes, Enter location : 엔터키
- Do you want to install a symbolic link? Yes
- Install the CUDA 9.0 Samples? No (Yes도 상관 없으나, 괜히 home 용량만 잡아먹음)
- 설치 완료 후 `sudo reboot`로 재부팅
- 재부팅 후 `nvidia-smi`로 그래픽카드가 잡히면 드라이버 설치 완료

5. CUDA 환경변수 설정
- 재부팅 후 `vim ~/.bashrc`
- 맨 아랫줄에 하기 내용 입력

```
{%raw%}
export PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/lib64:$LD_LIBRARY_PATH
{%endraw%}
```

- 파일을 저장하고 빠져나옴
- `source ~/.bashrc`
- `nvcc --version`으로 CUDA 버전 잡히면 설치 완료

6. 터미널에서 명령어가 듣지 않는 경우
- bashrc 파일 작성 시 오타가 포함될 경우 터미널상의 모든 명령어가 듣지 않는다.
- 이럴 경우, 터미널에서 `export PATH=/usr/bin:/bin`을 입력한 후, 5번의 과정을 다시 수행하면 됨.

## cuDNN 설치
- https://developer.nvidia.com/cudnn 에서 로그인 후 다운로드(version 7.2 이상으로)
- 로그인 후 다운로드 페이지에서 cuDNN Library for Linux 눌러서 다운로드(tgz 파일)
- 다운로드 받은 파일 위치로 이동 후 `tar -zxvf cudnn-~.tgz` 파일 압축풀기
- `cd cuda/include` 이동 후 파일 복사
  - `sudo cp cudnn.h /usr/local/cuda/include`
- `cd ../lib64` 이동 후 파일 복사
  - `sudo cp libcudnn* /usr/local/cuda/lib64`

## ssh 설치(원격 사용을 위해) / pip 설치
1. 우분투 터미널 열기
2. `sudo apt-get install openssh-server pip3`

## Virtualenv 환경 설정
- 우분투에 설치된 각종 패키지의 충돌 등을 방지하기 위해 사용한다.
- Tensorflow/Pytorch 등 각종 툴킷들의 버전 관리에도 편리하다.
- Pytorch(Caffe2)의 경우, Source code로 빌드하는 경우엔 virtualenv 내에서는 불가능하다(Python에서의 해결 불가능한 에러가 발생)

1. `pip3 install virtualenv`

2. 가상환경 생성
- Python 2, 3.5, 3.6등 버전별로 관리가 가능하다.
- Python 2에서 사용 할 경우, 위의 pip 및 virtualenv 설치 시 pip3 대신 pip를 입력하면 된다.
- Python 버전별 가상환경 생성 명령어는 아래와 같다.

```
{%raw%}
# python 2
$ virtualenv directory --python=python2.7

# python 3.5
$ virtualenv directory --python=python3.5

# python 3.6
$ virtualenv directory --python=python3.6
{%endraw%}
```

- 생성 된 가상환경을 제거하고 싶을 경우, 생성된 폴더를 그냥 `sudo rm -rf directory`로 날려버리면 된다.
- 생성 된 가상환경에 설치된 모듈들을 txt파일로 저장하고싶으면 가상환경이 활성화 된 상태에서 아래의 명령어를 이용한다.

```
{%raw%}
(venv) $ pip freeze > requirement.txt
{%endraw%}
```









