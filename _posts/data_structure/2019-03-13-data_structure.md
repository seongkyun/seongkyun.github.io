---
layout: post
title: CH4. 연결 리스트 (Linked list) 2-3
category: data_structure
tags: [data structure, 자료구조]
comments: true
---

# CH4. 연결 리스트 (Linked list) 2-3

## 4-2 단순 연결 리스트의 ADT와 구현
### 구현 할 더미 노드 기반 연결 리스트
- 연결 리스트의 3대 요소 -> head, tail, current
- 머리에 새 노드를 추가하되 더미 노드가 없는 경우, 첫 번째 노드와 두 번째 이후 노드의 추가 및 삭제 방식이 달라지게 됨
  - 별로 좋지 못한 방식(매 노드마다 일관된 처리 모습을 보이느게 좋음)
- 이 단점을 없애기 위해 dummy node 적용
- dummy node는 실제 유효한 data를 담지 않으며 생성 동시에 초기화 과정에서 달게 됨
- 두 번째 노드부터 유효 data를 입력하게 되므로 일관된 동작 가능하게 됨
  - 코드가 매우 

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-13-data_structure/fig1.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- "DLinkedRead.c"에서 `head = (Node*)malloc(sizeof(Node));`를 통해 간단하게 더미 노드 추가 가능
  - head 포인터가 내용이 빈 메모리 주로를 저장하게만 하면 됨
  - 이를 기반으로 삽입, 삭제, 참조 등의 연산을 하는 코드만 좀 더 세련되게 수정하면 됨
    - 불필요한 분기를 없애는 과정

- head node가 dummy node이므로 두 번째 node부터 첫 번째 유효한 데이터가 입력되게 됨

```c
/**** 데이터를 입력 받는 과정 ****/
while(1)
{
  printf("자연수 입력: ");
  scanf("%d", &readData);
  if(readData < 1)
    break;

  /*** 노드의 추가과정 ***/
  newNode = (Node*)malloc(sizeof(Node));
  newNode->data = readData;
  newNode->next = NULL;

  /*
  if(head == NULL)
    head = newNode;
  else
    tail->next = newNode;
  */
  tail->next = newNode;

  tail = newNode;
}
printf("\n");
```

- 첫 번째 노드는 dummy 노드이므로 첫 번째 데이터가 들어있는 두 번째 node부터 데이터를 출력하게 됨

```c
/**** 입력 받은 데이터의 출력과정 ****/
printf("입력 받은 데이터의 전체출력! \n");
if(head == NULL) 
{
  printf("저장된 자연수가 존재하지 않습니다. \n");
}
else 
{
  cur = head; 
//	printf("%d  ", cur->data);   // 첫 번째 데이터 출력

  while(cur->next != NULL)    // 두 번째 이후의 데이터 출력
  {
    cur = cur->next;
    printf("%d  ", cur->data);
  }
}
printf("\n\n");
```

- 메모리의 해제 또한 동일하게 두 번째 노드 이후로 삭제
- 연결 리스트가 완전히 삭제 되어야 하는 경우 __dummy 노드 또한 삭제가 되어야 함__
- 아래의 코드는 불 필요한 노드에 대한 삭제이므로 dummy node를 삭제하지 않음

```c
/**** 메모리의 해제과정 ****/
if(head == NULL) 
{
  return 0;    // 해제할 노드가 존재하지 않는다.
}
else 
{
  Node * delNode = head;
  Node * delNextNode = head->next;

//	printf("%d을(를) 삭제합니다. \n", head->data);
//	free(delNode);    // 첫 번째 노드의 삭제

  while(delNextNode != NULL)    // 두 번째 이후의 노드 삭제 위한 반복문
  {
    delNode = delNextNode;
    delNextNode = delNextNode->next;

    printf("%d을(를) 삭제합니다. \n", delNode->data);
    free(delNode);    // 두 번째 이후의 노드 삭제
  }
}
```

- __Dummy node의 추가를 통해 코드가 전반적으로 간결해지고 깔끔해짐__

### 정렬 기능 추가된 연결 리스트의 구조체
- 노드 구조체는 기존과 동일
  - 리스트의 일부로써 정의된 구조체
- 구조체 표현을 통해 여러가지 연결 리스트를 관리 할 수 있게 됨
  - 구조체 선언 후 다양한 이름의 다양한 구조체를 관리 가능

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-13-data_structure/fig2.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

### 정렬 기능 추가된 여결 리스트 헤더파일
- `typedef List`를 통해 List로 선언하여 기존 main코드(Chapter3)를 재활용 가능하게 됨

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-13-data_structure/fig3.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

### 더미 노드 연결 리스트 구현: 초기화
- `plist->head = (Node*)malloc(sizeof(Node));`를 통해 더미 노드를 생성

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-13-data_structure/fig4.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

### 더미 노드 연결 리스트 구현: 삽입
- If문을 통해 정렬 기준의 유무를 구별하고, 정렬 기준이 있다면 정렬 기준을 따르고(SInsert) 없다면 앞부분에 추가(FInsert)
- FInsert 함수에서 분기가 사라진 것을 확인 할 수 있음
  - 분기가 적어 가독성이 좋아지고 코드가 깔끔해짐
- 새로운 데이터를 추가함에 있어서 첫 번째 노드/n 번째 노드에 상관없이 동일한 삽입과정을 거친다는 것이 더미 노드 기반 연결 리스트의 장점

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-13-data_structure/fig5.jpg" alt="views">
<img src="/assets/post_img/data_structure/2019-03-13-data_structure/fig6.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

### 더미 노드 연결 리스트 구현: 참조
- LFirst를 통해 첫 번째 유효 데이터의 반환
  - cur가 유효한 첫 번째 node를 가리키게 하고 cur가 가리키는 값을 반환하면 됨
  - before 포인터는 node의 삭제를 진행하기 위해 필요
  - before 포인터는 cur 포인터가 다음 칸으로 옮겨 갈 때 필요(중간에 거쳐가야 주소를 잃어버리지 않음)
    - 기본적으로 cur의 바로 앞의 node를 가리킴
- LNext를 통해 2번째부터의 node의 값을 참조
  - 우선 cur에 before가 가리키던 주소를 넣고, cur에 cur->next 멤버 주소를 집어넣음

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-13-data_structure/fig7.jpg" alt="views">
<img src="/assets/post_img/data_structure/2019-03-13-data_structure/fig8.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

### 더미 노드 연결 리스트 구현: 삭제
- 항상 선 pointer 변경(이동) 후 값을 반환하는 process를 따름
  - cur pointer가 직전 참조된 유효 data를 가리키도록
  - 이를 위해 before가 존재
  - LR은 연속 2번 호출 불가(원칙상)
- LF->LN->...의 함수 호출을 통해 현재 4라는 값이 반환(참조)된 상태
- 삭제 후 cur과 before가 같은 node를 가리키고 있음
  - 그냥 이 상태로 둬도 상관이 없음
    - 이유?: LRemove는 바로 직전 참조 node의 삭제하므로
    - 이로인해 원칙상 LR 연속 2번 호출 불가
    - + LR의 2회 연속 호출은 일반적인 경우가 아님
    - + LF/LN 이후 1회의 LR 호출 가능
- before 포인터는 LF/LN 호출 시 해당 함수 내에서 다시 reset되므로 구지 설정해줄 필요 없음
  - LF/LN에서 `plist->before = plist->cur;` 로 함수 내에서 가장 먼저 초기화됨!
  - LR 이후 LF/LN은 무조건 호출되게 되어있음!
- cur 포인터는 LF/LN 함수 내에서 초기화시 필요하므로 재설정이 필요
  - LF/LN에서 before의 초기화에(`plist->before = plist->cur;`) 사용됨

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-13-data_structure/fig9.jpg" alt="views">
<img src="/assets/post_img/data_structure/2019-03-13-data_structure/fig10.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- LRemove는 backup용으로 별도의 rpos 포인터와 삭제 후 값 반환을 위한 rdata 변수가 필요
- before는 전으로 옮기지 않아도 됨
  - 구지 옮기려면 코드가 불필요하게 길어질 뿐더러 어차피 옮겨도 LF/LN 호출 시 다시 초기화되므로 불필요

### 더미 기반 단순 연결 리스트 한데 묶기
- Chapter3의 함수와 내용 동일
  - main함수 하나도 안바뀌어도 header 파일과 typedef만 바꿔도 그대로 사용 가능
- 하지만 새 노드 추가시 head에 추가되므로 결과가 역순으로 출력
  - 이를 위해 별도의 정렬 삽입 알고리즘을 추가하여 헤더를 추가할 수 있도록 함
  - 다음 챕터에서...
  
<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-13-data_structure/fig11.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>
