---
layout: post
title: 버전 만들기 / stage area에 대해 알아보기
category: git
tags: [github, 지옥에서 온 git]
comments: true
---

# 버전 만들기 / stage area에 대해 알아보기

## 버전 만들기 (commit)
- 의미있는 변화가 생기는 것을 버전(version)이라고 함.
- 해당 작업이 완결된 상태만을 새로운 버전이라고 할 수 있음.
	- 단순 코드 수정은 새로운 버전으로 볼 수 없음
- 해당 디렉터리의 코드가 자신의 것임을 알려주기 위해 이름 및 메일주소를 정의
	- `git config --global user.name myname`
	- `git config --global user.email mymail@mail.com`
- `git commit` 을 쳐서 자신의 정보가 제대로 등록되었는지 확인
	- committer에 자신의 정보(이름, 메일)가 등록되어있는것 확인 가능
- 밑에는 commit message를 적을 수 있음.
	- 주석 처리 된 부분을 제외한 부분에 메세지를 적어야 함.
- `git log`는 누가, 언제, 무슨 내용(커밋 메세지)을 만들었는지에 대한 정보를 출력
- 다음 버전을 작성 하려면, git 관리 대상으로 등록된 파일을 수정
- 수정 된 파일을 `git status`로 확인 시 해당 파일이 붉은색으로 modified상태로 뜸
- 다음에 `git add filename`로 다시 git에서 그 파일을 다시 등록해줘야 함.
	- 즉, git에 버전 관리를 시작 할때나(새 파일 생성) 새 버전을 생성할때도 `git add filename` 를 또 해줘야함.
	- `git add filename` 없이 그냥 `git commit` 시 커밋 불가
	- Commit 후 커밋 메세지 작성할 수 있는 창이 뜸.
- `git log`를 치면 방금 커밋한 버전을 확인 할 수 있음.
- 커널에서는 vim이 사용이 편하므로, 만약 nano 에디터로 실행 될 경우 `git config --global core.editor “vim”` 을 한 번만 쳐 주면 해결된다.

## Stage area (중요!)
- git은 commit전에 add를 꼭 해야 함
	- 그 이유는 선택적으로 파일을 버전에 포함시키기 위해서임.
- `f1.txt` 파일과 동일한 `f2.txt` 파일을 생성. 
	- `cp f1.txt f2.txt`
- `git status`로 파일의 상태를 확인하면 새로 생성된 `f2.txt` 파일이 tracked되지 않는다 뜸
	- `git add f2.txt`로 새로 생성된 파일을 트랙하도록 설정
- `git commit` 후 커밋 메세지 작성(version 3), 저장
- `git log`로 커밋 된 결과를 확인하면 성공적으로 커밋 된 것을 확인 할 수 있음
- 다음으로 `f1.txt`와 `f2.txt` 의 내용을 모두 다 수정
- 이 상태에서 `git status`로 상태를 확인하면 두 파일이 modified 상태인것을 확인 할 수 있음
- 여기서! `git add f1.txt` 로 `f1.txt`파일만 트래킹 하도록 설정한 후, `git status`로 상태를 확인하면 `f1.txt` 파일만 초록색에 커밋 가능한 상태로 뜸
- `f2.txt`는 커밋이 되지 않을것이라고 뜨며 하며 빨간색 modified 상태
- 즉, `git add filename`로 트래킹을 할 수 있도록 등록 한 파일만 커밋이 되기 때문에 선택적으로 파일을 커밋 할 수 있음.
- 이 상태에서 `git commit`으로 커밋 시 커밋 메세지를 작성 할 수 있고, 작성을 완료 하고 `git log`로 로그 내용을 살펴보면 `f1.txt`파일만 새로운 커밋 메세지로  변경사항이 반영 된 것을 알 수 있음. 
- `git add filename`를 하지 않은 `f2.txt` 파일은 `git status`로 상태 확인 시 커밋 되지 않은것을 알 수 있음.
- 즉, `git add filename`를 해줘야 commit 대기 상태로 들어가 커밋이 가능하게 되는것임. 커밋 대기상태가 아니면 커밋 불가능.
- 이렇게 `git add filename`로 커밋 가능 상태로 만들어주는것을 변경 파일을 ‘stage’에 올리는 것이고, stage에 존재하는 파일만 커밋 가능.
- 정리하자면,
	- Stage: 커밋 대기 상태 파일들이 가는곳
	- Repository는 커밋 된 파일들이 존재하는곳
