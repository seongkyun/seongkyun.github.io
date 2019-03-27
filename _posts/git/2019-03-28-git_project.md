---
layout: post
title: Github에 프로젝트 쉽게 올리기
category: git
tags: [github]
comments: true
---

# Github에 프로젝트 쉽게 올리기
- 업로드 할 폴더로 이동 후 `git init` 입력
  - `.git`이라는 파일이 생성됨
- `.gitignore` 파일을 만들어서 업로드하여 버전 관리를 하지 않을 파일을 설정
- `git add .`으로 해당 프로젝트 폴더 내의 모든 파일을 버전 관리하도록 추가
  - `.gitignore` 파일에서 제외시키는 파일은 자동으로 버전관리에 추가되지 않음
- `git status`로 버전 관리하도록 tracking이 잘 되었는지 확인 가능
  - 초록 글씨로 `new file: filename` 로 표시되어있음
- `git commit -m "commit_message"` 으로 커밋
- github.com 으로 접속하여 로그인 하고, 해당 파일들을 push 해 줄 github 저장소 생성
  - 로그인 후 우측 상단에 "+" 버튼을 누른 후 "New repository" 선택
  - Repository name과 Description 입력(추후 수정 가능)
  - 다음으로 "Create repository" 버튼을 누른 후 https github 주소 복사
    - ex. `https://github.com/seongkyun/pytorch-classifications.git`
- 터미널에서 `git remote add repository_name repository_address` 입력
  - ex. `git remote add origin https://github.com/seongkyun/pytorch-classifications.git`
  - repository_name은 편리하게 설정하면 되나 간편하게 "origin" 으로 설정
- `git push origin master` 로 push
  - github의 계정 정보를 기입하여 로그인 하면 자동으로 push 진행됨
- github 웹에서 업로드된 파일을 변경하게 되면 지역저장소에서 수정 전에 폴더 내에서 `git pull`로 최신 버전(master)으로 동기화 후 작업해야함!
  - 충돌이 발생해서 해결하려면 귀찮아진다..
- 지역저장소에서 작업 후 push 시에는 `git push origin master` 를 치면 된다.

---
- [참고글]

https://medium.com/wasd/github%EC%97%90-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%89%BD%EA%B2%8C-%EC%98%AC%EB%A6%AC%EA%B8%B0-django-1e2c7814a13
