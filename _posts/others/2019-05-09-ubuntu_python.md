---
layout: post
title: 우분투에서 파이썬 버전 변경하기
category: others
tags: [Ubuntu, Python]
comments: true
---

# 우분투에서 파이썬 버전 변경하기
- 참고 글: https://codechacha.com/ko/change-python-version/

- 보통 우분투에는 기본적으로 python path가 2.7 버전으로 설정되어 있다.
- Linux의 alternative를 이용하여 python 버전을 쉽게 변경 및 관리가 가능하다.

- Python은 `/usr/bin/python`의 link 파일이고, 이 파일은 `/usr/bin/python2.7`의 link 파일이다. `/usr/bin/python`에는 다양한 버전의 python이 설치된 상태이다.

```
{%raw%}
$ python -V
Python 2.7.14

$ which python
/usr/bin/python

$ ls -al /usr/bin/python
lrwxrwxrwx 1 root root 24  4월 18 19:28 /usr/bin/python -> /usr/bin/python2.7

$ ls /usr/bin/ | grep python
python
python2
python2.7
python3
python3.6
.....
{%endraw%}
```

## 설정 방법
- `--config python` 옵션은 python의 버전을 변경하는 옵션이다.
- `$ sudo update-alternatives --config python` 를 입력하면 python의 버전을 변경 할 수 있다.

```
{%raw%}
$ sudo update-alternatives --config python
update-alternatives: error: no alternatives for python
{%endraw%}
```

- 만약 위와같이 error로 python에 대한 alternative가 설정된것이 없다 뜰 경우, `--install [symbolic link path] python [real path] number` 명령어로 2.7이나 3.6같은 파이썬 버전을 등록해주면 된다.

```
{%raw%}
$ sudo update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1
$ sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2
{%endraw%}
```

- 이 후 다시 `sudo update-alternatives --config python`를 입력하면  설치되어있는 python 버전 선택 메뉴가 등장한다.

```
{%raw%}
$ sudo update-alternatives --config python
There are 2 choices for the alternative python (providing /usr/bin/python).

  Selection    Path                Priority   Status
------------------------------------------------------------
* 0            /usr/bin/python3.6   2         auto mode
  1            /usr/bin/python2.7   1         manual mode
  2            /usr/bin/python3.6   2         manual mode

Press <enter> to keep the current choice[*], or type selection number: 2
{%endraw%}
```

- 원하는 python 버전의 번호를 선택한 후 엔터를 치면 해당 버전이 default path로 설정되게 된다.

```
{%raw%}
$ python --version
Python 3.6.3
{%endraw%}
```

- 위와 같이 `$ python --version`으로 버전 확인이 가능하다.

- 또한 alternative는 python뿐만이 아니라 java와 같은 다양한 프로그램의 버전을 관리하는데 사용이 가능하다.



