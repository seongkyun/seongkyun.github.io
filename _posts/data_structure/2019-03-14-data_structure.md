---
layout: post
title: CH4. 연결 리스트 (Linked list) 2-4
category: data_structure
tags: [data structure, 자료구조]
comments: true
---

# CH4. 연결 리스트 (Linked list) 2-3

## 4-3 연결 리스트의 정렬 삽입의 구현
- 리스트에 정렬 기준을 등록
  - 새로운 노드가 정렬 기준에 맞춰 알맞은 자리에 연결되게 됨
  
### 정렬기준 설정
- 정렬기준이 되는 함수를 등록하는 SetSortRule 함수
  - 정렬기준이 되는 함수가 별도로 정의되고, 해당 함수를 함수 포인터를 이용하여 리스트에 등록해주는 역할
- SetSortRule 함수를 통해 전달된 함수정보 저장을 위한 LinkedList의 멤버 comp
  - 함수 포인터 comp에 등록될 함수는 정렬 기준을 정의하는 함수(사용자의 목적 및 필요에 의해 임의대로 정의됨)
- comp에 등록된 정렬기준을 근거로 데이터를 저장하는 SInsert 함수
  - comp의 반환값에 의존하여 데이터를 알맞은 위치에 저장
- SetSortRule 함수가 호출되어 정렬의 기준(함수)이 리스트 멤버 comp에 함수가 등록되면 SInsert 함수 내에서는 comp에 등록된 정렬의 기준을 근거로 데이터를 정렬하여 리스트에 저장(새로운 노드를 기존의 노드 사이에 연결)
  - SInsert 함수는 LInsert 함수 내에서 comp의 정의 여부에 따라 SInsert 또는 FInsert가 호출되어 리스트에 데이터 저장(새로운 노드가 연결됨)
    - SInsert: comp 함수가 정의 된 경우 comp의 규칙에 맞게 새 노드를 기존의 노드 연결 사이에 연결시킴
    - FInsert: comp 함수가 정의되지 않은 경우 새 노드를 기존의 노드 연결의 맨 앞에 연결

### SetSortRule 함수와 멤버 comp

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-14data_structure/fig1.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- SetSortRule 함수내에서 `plist->comp = comp;`로 comp의 주솟값을 갖는, 정렬 기준을 정의한 함수의 주소를 plist의 정렬 기준(comp)으로 등록
- LInsert 함수 내에서 정렬 기준인 comp의 정의 여부에 따라 FInsert 또는 SInsert로 새로운 노드를 생성 후 기존 노드에 연결시킴

### SInsert 함수

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-14data_structure/fig2.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- __필기 스크린샷에 잘못된 부분이 있음__
  - 리스트에 새로운 데이터의 추가는 노드의 앞쪽에 되므로 실제 비교시에는 비교 할 값과 비교 당할 값 두개만 알면 됨
  - 필기에는 비교 할 값과 비교 당할 두 값을 필요하다고 해 놓음(잘못된것!)
  
- 새 노드를 생성
- 새로운 포인터 변수인 pred를 정의하고, 이것을 입력된 리스트의 head가 가리키는 노드로 정의
  - 리스트가 처음 참조되는 경우라면 head는 dummy 노드를 가리키므로 pred 또한 dummy 노드를 가리키게 됨
- 왜 pred 포인터는 입력된 리스트 plist의 head가 가리키는 노드를 가리킬까? (`pred = plist->head;`)
  - 정렬을 정의하는 함수, 즉 comp 함수에는 인자로 (비교 할 숫자, 비교 당할 숫자)가 들어가게 됨
    - 따라서 comp 함수와 수 비교를 통해 비교 할 숫자가 비교 당할 숫자보다 작으면 무조건 비교한 숫자를 갖는 노드의 앞에 노드가 추가됨
  - 하지만 노드를 연결하는 순간에는 연결할 부위의 이전 노드의 주소와 이후 노드의 주소 두 주소를 알아야 연결 할 수 있음!
    - `새 노드->next = 기존 자리 노드->next` , `기존 자리 노드->next = 새 노드 주소` 순서로 노드의 연결이 이루어짐
  - __다음 노드의 주솟값은 현재 노드->next 멤버 변수를 통해 알 수 있지만, 현재 노드 이전 주소의 값은 알 수 없음__
  - 따라서 pred가 이전 노드(그림에서 2가 아닌 dummy부터 시작해야)를 가리켜야 새 노드 연결이 가능해짐
    - pred가 이전 노드를 카리키면 다음 노드는 pred->next 로 정의 가능
    - 그 사이 새 노드 newNode를 `newNode->next = pred->next;`, `pred->next = newNode;` 순서로 연결하여 리스트 완성
