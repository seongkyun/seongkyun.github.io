---
layout: post
title: Jupyter Notebook 원격 접속 세팅하기(윈도우10)
category: others
tags: [Jupyter notebook]
comments: true
---

# Jupyter Notebook 원격 접속 세팅하기(윈도우10)
- 윈도우 10 에서 Jupyter Notebook 원격 접속을 세팅한다.
- 일부 환경(직장이나 연구실 등)에서는 Port가 막혀있을 수 있으므로 별도 포트 세팅이 필요하다.

## 노트북 세팅하기
- Anaconda prompt 실행 후 아래 입력
  - `jupyter notebook --generate-config`
  - 노트북 세팅용 `jupyter_notebook_config.py`이 이어지는 디렉터리에 생성된다.
    - 일반적으로 `C:\Users\user_name\.jupyter`에 생성됨
- 암호 설정하기
  - Anaconda prompt에서 `ipython` 입력
  - ipython에서 `from notebook.auth import passwd` 입력
  - `passwd()` 입력 후 노트북용 비밀번호 설정
    - Verify 번호까지 정확하게 입력해야 함
  - 세팅 완료 후 `quit()`으로 ipython 빠져나오기
  - __여기서 출력되는 ssh값을 복사해둔다.__
- 에디터로 `jupyter_notebook_config.py` 열기
  - 파일 내에서 아래와 같이 해당 주석을 찾아서 삭제 후 수정
    - 포트 번호의 경우 맘에드는 복잡한 5자리 임의의 숫자로 설정한다.

```
{%raw%}
...

#c.NotebookApp.ip = 'localhost'
c.NotebookApp.ip = '*'

#c.NotebookApp.password = ''
c.NotebookApp.password = 'sha1:SHA_VALUES'

#c.NotebookApp.password_required = False
c.NotebookApp.password_required = True

#c.NotebookApp.port = 8888
c.NotebookApp.port = YOUR_PORT_NUMBER

{%endraw%}
```

## 방화벽에서 포트 열기
- 윈도우 키 누른 후 방화벽 검색해 "방화벽 상태 확인" 클릭 후 열기
- 방화벽에서 왼쪽 __고급 설정__ 클릭
- 고급 설정에서 __인바운드 규칙__ 클릭
- 인바운드 규칙에서 우측 __새 규칙__ 클릭
  - __포트__ 선택 후 다음 클릭
  - __TCP__ 선택 후 __특정 로컬 포트__ 에 위에서 설정한 포트 번호 5자리(YOUR_PORT_NUMBER) 입력 후 다음 클릭
  - __연결 허용__ 선택 후 다음 클릭
  - __도메인, 개인, 공용__ 모두 선택 후 다음 클릭
  - __이름__ 에 "Remote jupyter connection" 과 같이 알아 볼 수 있게 설정 후 마침 클릭
- 컴퓨터 재시작, 또는 서비스-Remote Desktop Services 재시작 후 사용!

## 사용하기
- Anaconda prompt 실행 후 
  - `jupyter notebook --ip=YOUR_IP` 입력 후 실행
- 원격 디바이스에서 웹 브라우저를 킨 후
  - `YOUR_IP:YOUR_PORT_NUMBER` 입력 후 접근
  - 이 때, 비밀번호를 입력하라고 뜨는데 여기서 비밀번호는 위 ipython에서 설정한 비밀번호를 입력하면 됨
  
