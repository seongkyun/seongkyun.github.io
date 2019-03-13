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
- 

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

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-13-data_structure/fig1.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>
