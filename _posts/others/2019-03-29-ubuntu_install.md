---
layout: post
title: 우분투(ubuntu) 설치하기
category: others
tags: [Ubuntu]
comments: true
---

# 우분투(ubuntu) 설치하기
- Grub를 이용한 Windows와의 병렬설치가 아닌 우분투 단독설치방법에 대한 정리

1. 우분투 OS USB로 부팅
2. 우분투 GRUB 진입
  - 진입 후 Try Ubuntu without installing 선택
    - 설치 없이 우분투 사용가능
    - 바탕화면의 Install Ubuntu 16.04LTS 실행
  - Install Ubuntu 눌러도 상관없음
3. 초기언어 설정
  - 한국어 말고 영어로 설정해야 디버깅이 쉬움!
4. Next 누르기
  - 설치 중 업데이트 등 다운로드 여부 구지 체크하지 않아도 됨
5. 설치 형식에서 "Something else" 선택 후 다음
6. 파티션 선택
  - free space 로 잡히는 하드디스크 선택 후 __+__ 버튼을 눌러 할당
    - 만약 free space로 잡히는 파티션이 없다면 "Create new partition" 으로 하드디스크에 파티션 할당
7. Swap 영역 할당
  - 대개 메모리가 충분한경우 구지 필요 없으므로 4096MB로 할당
    - 더 추가하고싶다면 1024의 배수로 할당하면 됨

<center>
<figure>
<img src="/assets/post_img/others/2019-03-29-ubuntu_install/fig_swap.jpg" alt="views">
<figcaption></figcaption>
</figure>
</center>

8. / 영역 생성
  - 패키지들이 설치되는 위치이나, 10GB만 되도 충분하다.  

<center>
<figure>
<img src="/assets/post_img/others/2019-03-29-ubuntu_install/fig_slash.jpg" alt="views">
<figcaption></figcaption>
</figure>
</center>

9. /home 영역 생성
  - 주로 작업하는 영역이므로 용량이 크게 할당되어야 나중에 고생하지 않는다.

<center>
<figure>
<img src="/assets/post_img/others/2019-03-29-ubuntu_install/fig_home.jpg" alt="views">
<figcaption></figcaption>
</figure>
</center>

10. 생성된 영역 확인
  - 각각 swap, /, /home 영역이 생성된 것을 확인하고 Install now(I)로 설치를 시작한다.

<center>
<figure>
<img src="/assets/post_img/others/2019-03-29-ubuntu_install/fig_partitions.jpg" alt="views">
<figcaption></figcaption>
</figure>
</center>

11. 바뀐 점을 디스크에 쓰겟냐는 창이 뜨지만 Next를 눌러 다음으로 진행한다.
12. 지역 및 키보드, 언어 선택
  - Seoul, EN/US 키보드로 기본설정 그대로 하고 Next로 넘어가면 됨
13. 계정 생성
  - 이름 및 컴퓨터 이름, 사용자 이름은 가장 간단하게 해야 나중에 편리하다.  
14. 다음 눌러서 설치 완료하고 재부팅 하면 완료!
