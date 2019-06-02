---
layout: post
title: CH7. 큐 (Queue) 2
category: data_structure
tags: [data structure, 자료구조]
comments: true
---

# CH7. 큐 (Queue) 2

## 7-3 큐의 연결 리스트 기반 구현
- 배열기반과 다르게 설명할 내용이 매우 적음!
  - 단순연결리스트의 복습정도 난이도..
- 본 수업에서는 `F`가 연결리스트의 __머리__ 부분을 가리키고, `R`이 연결리스트의 __꼬리__ 부분을 가리키도록 설정
  - `F`에선 노드가 참조되며 삭제되고, `R`에서는 새 노드가 추가되는 형태
- 연결리스트의 노드 추가에서 중간에서 추가하는 부분 필요 없이 머리에서만 삭제하고 꼬리에서만 추가되는 형태이므로 상대적으로 쉬움!

### 연결 리스트 기반 큐의 헤더 파일
- 일반적인 연결리스트와 크게 다를 것 없음
- 게다가 머리/꼬리에서 각각 삭제와 참조의 역할이 정해져있음
- 큐 구조체의 멤버로는 머리와 꼬리를 가리킬 `front`와 `rear` 포인터 변수만 있으면 됨
- 연결 리스트는 앞에서 공부했던 연결 리스트의 구조체를 그대로 가져다 씀

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-06-03-data_structure/fig1.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

### 연결 리스트 기반 큐의 구현: 초기화
- 새 큐가 선언 될 때 각각 `F`와 `R` 모두 NULL 포인터로 초기화 됨.
- 하지만 큐가 비었는지 판단하기에는 `F`포인터 변수가 NULL인지만을 갖고 비었는지 판단
  - 왜 `R` 포인터변수는 확인하지 않는가?
    - `R` 포인터 변수는 `F` 포인터 변수에 종속적이며 뒤에서 세세하게 다룸.

### 연결 리스트 기반 큐의 구현: enqueue
- Enqueue의 과정은 두 과정으로 나뉨.
  - 큐가 비었을 때와 비어있지 않은 경우.
- 큐가 비어있을 때
  - `F`와 `R` 모두 새 노드를 가리키도록 설정
- 큐가 비어있지 않을 때
  - `F`는 그대로, `R`만 새 노드를 가리키도록 설정
- 논리 흐름 자체가 `F`가 NULL을 가리키는지 아닌지를 기준으로(if문) `R`의 동작이 결정됨
  - `R`이 `F`에 종속적인 이유

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-06-03-data_structure/fig2.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

### 연결 리스트 기반 큐의 구현: dequeue 논리
- Dequeue의 논리 흐름은 다음과 같음
  1. `F`가 가리키던 노드의 주소값을 backup 한 후
  2. `F`가 가리키던 노드의 next 값으로 `F`를 초기화
  3. Backup해 두었던 주소의 노드를 삭제
- 이 과정에서 별도로 `R`에 대한 초기화를 수행하지 않음
- 즉, enqueue와 다르게 노드가 하나가 남건 여러개가 남건 동일한 흐름으로 진행 가능(if문 필요 x)
- 삭제 후 어차피 `R`은 free된 주소를 가리키게 되므로 의미없는 값이 됨

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-06-03-data_structure/fig3.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

### 연결 리스트 기반 큐의 구현: dequeue 정의

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-06-03-data_structure/fig4.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

### 연결 리스트 기반 큐의 실행

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-06-03-data_structure/fig5.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>
