---
layout: post
title: CH5. 연결 리스트 (Linked list) 3-1
category: data_structure
tags: [data structure, 자료구조]
comments: true
---

# CH5. 연결 리스트 (Linked list) 3-1

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-20-data_structure/fig1.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- Chapter5에서는 __원형 연결 리스트__ 와 __양방향 연결 리스트__(노드들이 양방향으로 연결된 꼴)을 다룸
  - 그림상으론 복잡해보이나 본질적으론 일반 연결 리스트와 같음
  - 단순 연결 리스트의 경우 돌아가기가 불가능하나(한방향으로만 이동)
  - 양방향 연결 리스트의 경우 이전으로 돌아가기 가능(하지만 만능은 아님!)
  - 순환 연결 리스트의 경우 원 모양으로 도는 형태
- 단순/양방향/원형 연결 리스트 중 언제 뭘 선택해서 쓰나?
  - 보통 단순연결 리스트로 커버 가능
  - 원형/양방향 연결 리스트의 경우 시스템, 운영체제 단에서 사용하며 다양한 활용 가능성이 있습
    - 메모리를 순환적으로 사용 가능한 구조이므로...
  - 현재 수준에선 직접 응용 보단 이런게 있구나 정도로 이해해도 됨..

## 5-1 원형 연결 리스트
### 원형 연결 리스트의 이해

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-20-data_structure/fig2.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 단순 연결 리스트와 다르게 원형 연결 리스트는 꼬리가 머리를 가리키는 구조
- 단순 연결 리스트에서 꼬리가 머리를 가리키게만 하면 됨
  - 단순 연결 리스트의 마지막 노드는 NULL을 가리킴
  - 원형 연결 리스트의 마지막 노드는 첫 번째 노드를 가리킴

### 원형 연결 리스트의 노드 추가

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-20-data_structure/fig3.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 새 node를 머리 또는 꼬리에 추가 가능
- Head 포인터를 떼어놓고 보면 머리나 꼬리나 같은 구조
  - 모든 노드가 원의 형태를 이루며 연결되어 있기 때문에 사실상 머리와 꼬리의 구분이 없음
- 하지만 노드의 저장 순서 판단을 위해 구별

### 원형 연결 리스트의 대표적인 장점

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-20-data_structure/fig4.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 경우에 따라 머리 또는 꼬리에 추가해야 할 때도 있지만 원형 연결 리스트엔 head나 tail 두개 다 필요하지 않음
  - tail이 head의 역할 커버 가능
- 보통 head의 기능을 tail로 커버 가능하므로 tail 포인터 변수만 둔다.
  - head는 tail->next와 동일함
- 해당 강의에선 위의 기능을 갖는(tail 포인터만 갖는) 변경된 원형 연결 리스트를 구현함
  - 보다 일반적이라고 인식되어짐
  - 또한 원형 연결리스트의 장점을 조금 더 잘 취할 수 있는 구조임

### 변형된 원형 연결 리스트

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-20-data_structure/fig5.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 꼬리를 기리키는 포인터 변수는 tail, 머리를 가리키는 포인터 변수는 tail->next
- 리스트의 꼬리와 머리 주솟값을 쉽게 확인 가능하므로 원형 연결 리스트엔 하나의 포인터 변수만 필요

### 변형된 원형 연결 리스트의 구현범위

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-20-data_structure/fig6.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 조회관련
  - LFirst: 이전과 기능 동일
  - LNext: 원형 연결 리스트를 계속해서 순환하는 형태로 변경 필요
    - 마지막 노드의 next가 첫 번째 노드를 가리키게 하면 됨
- 삭제관련
  - LRemove: 이전에 구현한 연결 리스트와 기능 동일
  - Insert 함수는 특성 확인을 위해 두 개를 정의함
    - LInsert: 새 노드를 꼬리에 추가
    - LInsertfront: 새 노드를 머리에 추가
- 정렬 관련
  - 제거
- 이외의 부분
  - 단일 연결 리스트와 기능 동일
  
### 원형 연결 리스트의 헤더파일과 초기화 함수

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-20-data_structure/fig7.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- node에 대한 구조체 정의는 그대로
- 원형 연결 리스트에 대한 구조체 정의인 `CList`
  - tail은 참조를 위해서, before는 삭제를 위해서 존재
  - 원형연결리스트 정의하는 구조체는 내용만 두고 보변 연결리스트의 종류가 무엇인지 파악하기는 어려움
- 구조체 멤버 초기화 시 모든 멤버를 NULL 또는 0으로 초기화
  - 본인만의 규칙성 있게 초기화 하는것이 나중을 위해서도 좋음..

### 원형 연결 리스트의 구현: 첫 번째 노드 삽입

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-20-data_structure/fig8.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 첫 번째 노드의 경우 꼬리/머리에 추가를 하더라도 원형구조이므로 결과는 동일(전체 구조가 동일)
- 첫 번째 노드의 추가인 경우엔 머리/꼬리 추가 차이가 없음
  - `plist->tail = newNode;`: 새로운 노드의 tail이 새 노드를 카리키도록 (자기 자신을 가리킴)
  - `newNode->next = newNode;`: 새로운 노드의 next가 새 노드를 가리키도록 (자기 자신을 가리킴)
- 첫 번째 노드는 그 자체로 머리이자 꼬리이기 때문에 노드를 앞에 추가하건 뒤에 추가하건 동일함

### 원형 연결 리스트의 구현: 두 번째 이후 노드 머리로 삽입

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-20-data_structure/fig9.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 두 번째 이후 노드의 추가는 머리와 꼬리에서 방법적으로 차이가 존재
  - `newNode->next = plist->tail->next;`: 새 노드의 next가 주어진 리스트의 꼬리의 next가 가리키는 노드를 가리키도록 설정(그림상 1번)
    - tail->next 가 결국 head이므로 plist->head가 가리키는 노드의 주소를 newNode->next에 넣는 것과 동일한 의미
  - `plist->tail->next = newNode;`: 새 노드의 주소를 주어진 리스트의 꼬리의 next로 연결(그림상 1번)
    - 리스트의 마지막 노드가 새 노드(맨 앞 노드)를 가리키도록 설정하는 과정

### 원형 연결 리스트의 구현: 앞과 뒤의 삽입 과정 비교

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-20-data_structure/fig10.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 노드를 머리에 추가한 결과 tail->next가 newNode를 가리키게 됨
- 노드를 꼬리에 추가한 결과 tail이 newNode를 가리키게 됨
  - 뭐든 tail->next가 head라고 이해하면 이해가 쉬움!!
- 실질적인 차이점은 __tail이 누구를 가리키냐가 중요한 차이점임__
  - 즉, tail만 newNode를 가리키도록 하면 꼬리에 추가하는 꼴이 됨

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-20-data_structure/fig11.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 두 번째 이후 노드의 꼬리에 새 노드를 추가하는 부분
  - tail의 위치가 머리에 추가하는 방식과 차이가 있음
- `plist->tail = newNode;`를 추가하여 연결 리스트의 꼬리가 새로 추가된 노드를 가리키도록 하면 됨

### 원형 연결 리스트의 구현: 조회 LFirst

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-20-data_structure/fig12.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- before는 current의 왼쪽 node를 가리키도록 설정되어야 함
- 첫 번째 조회이므로, 우선 주어진 리스트의 before가 주어진 리스트의 tail을 가리키도록 설정
  - 첫 번째 조회이므로 cur은 첫 번째 노드를 가리키고 있어야 하므로 before는 tail이 가리키는 노드의 주소로 초기화됨
- 다음으로 주어진 리스트의 cur이 첫 번째 노드를 가리키도록 설정
  - 주어진 리스트의 tail->next가 결국 head 위치이므로 리스트의 cur 포인터 변수가 head 위치의 노드를 가리키도록 설정

### 원형 연결 리스트의 구현: 조회 LNext

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-20-data_structure/fig13.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 원형 연결 리스트이므로 끝인지 판별하는(tail=NULL) 코드가 필요 없으므로 삭제됨
- LNext에선 참조 후 이동하는 과정에 대한 코드만 보면 됨
  - cur 위치로 before를 당겨주고
  - cur->next 위치로 cur를 당겨줌

### 원형 연결 리스트의 구현: 노드의 삭제(복습)

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-20-data_structure/fig14.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 일반 연결리스트는 dummy node로 삭제 과정을 깔끔하게 함
  - 하지만 원형 연결 리스트에서는 dummy node를 사용하지 않음
  - 또한 삭제할 노드가 tail인지에 따라 연산의 방법이 구분됨
- 하지만 기본적으로 삭제의 핵심연산은 같은 과정을 따름
  - 1: 삭제할 노드의 이전 노드(before가 가리키는 노드)의 next가 삭제할 노드(cur가 가리키는 노드)의 다음 노드(next)를 가리키게 함
    - 현재 cur은 삭제할 노드를 가리키고 있는 상태
    - `plist->before->next = plist->cur->next;`
  - 2: 포인터 변수 cur을 한 칸 뒤로 이동
    - `plist->cur = plist->before;`
  - before는 remove 후 LFirst나 LNext 함수에서 재정의되므로 정의 필요 없음
- 원형 연결 리스트의 삭제도 큰 틀에서 같음

### 원형 연결 리스트의 구현: 노드의 삭제(그림 차이점 비교)

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-20-data_structure/fig15.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 삭제할 노드의 상태와 tail 여부에 따라 삭제 방법이 달라짐
  - case 1. 삭제 노드가 tail이면서 여러개인 경우
    - tail의 위치 조정 필요
  - case 2. 삭제 노드가 tail이면서 하나밖에 없는 경우
    - tail=NULL 초기화 필요

- 삭제할 노드가 tail 노드인가?
  - 맞다면, 마지막 노드인가?
    - 맞다면, 삭제 후 tail=NULL로 초기화
    - 아니라면, 삭제 후 tail을 왼쪽으로 이동시킴
  - 아니라면, 그냥 삭제

- 원형 연결 리스트에서는 더미 노드가 없기 때문에 삭제의 과정이 상황에 따라 달라짐
  - 단일 연결 노드처럼 더미노드를 사용하여 일관되게 구성되어있지 않음

### 원형 연결 리스트 노드 삭제 구현

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-20-data_structure/fig16.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 전체적으로 tail이 가리키는 노드의 주솟값에 대한 설정 후 삭제의 과정을 거침
- `rpos = plist->cur;`을 통해 rpos가 지울 노드를 가리키게 설정
- `if(rpos == plist->tail)`: 만약 지우려는 노드가 tail인 경우
  - `if(plist->tail == plist->tail->next)`: 지우려는 노드가 tail이면서 next가 지가 자신인 경우, 즉 노드가 마지막 노드라면
    - `plist->tail=NULL;`: tail을 NULL 포인터로 초기화
  - 마지막 노드가 아니라면
    - `plist->tail = plist->before`: tail 포인터가 가리키는 노드를 이전 노드로 변경
- 삭제는 단순 연결 리스트와 동일한 방법으로 진행
  - 1. before가 가리키는 노드의 next를 지울 노드(cur)의 next로 초기화
  - 2. cur가 before가 가리키는 노드를 가리키도록 설정
