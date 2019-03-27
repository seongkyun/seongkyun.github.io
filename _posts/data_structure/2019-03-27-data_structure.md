---
layout: post
title: CH6. 스택 (Stack) 3
category: data_structure
tags: [data structure, 자료구조]
comments: true
---

# CH6. 스택 (Stack) 3

## 6-4 계산기 프로그램 구현

### 문제 6-2
- 중위 표기법의 후위 표기법으로의 변환
  - 문제 1: 3 + 2 \* 4
  - 문제 2: 2 \* 4 + 3
  - 문제 3: 2 \* 1 + 3 / 2
- 문제 1 풀이
  - 1: \* 4
    - Stack: + <-top
    - 변환수식: 3, 2
  - 2: emtpy
    - Stack: +, \* <-top
    - 변환수식: 3, 2, 4
  - 최종 답: 3, 2, 4, \*, +
- 문제 3 풀이
  - 1: 2 \* 1 + 3 / 2
    - Stack: empty
    - 변환수식: empty
  - 2: + 3 / 2
    - Stack: \* <-top
    - 변환수식: 2, 1
  - 3: / 2
    - Stack: + <-top
    - 변환수식: 2, 1, \*, 3
  - 4: empty
    - Stack: +, / <-top
    - 변환수식: 2, 1, \*, 3, 2
  - 최종 답: 2, 1, \*, 3, 2, /, +

### 문제 6-3
- 중위 표기법의 후위 표기법으로의 변환
  - 문제 : (1 \* 2 + 3) / 4
- 문제 풀이
  - 1: (1 \* 2 + 3) / 4
    - Stack: (, \* <-top
    - 변환수식: 1, 2
  - 2: ) / 4
    - Stack: (, + <-top
    - 변환수식: 1, 2, \*, 3
  - 3: empty
    - Stack: empty
    - 변환수식: 1, 2, \*, 3, +, 4, /
  - 최종 답: 1, 2, \*, 3, +, 4, /
  
### 중위 -> 후위: 프로그램 구현

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-27-data_structure/fig_28.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 중위 표기법으로 입력받은 수식을 후위 표기법으로 변환한다.

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-27-data_structure/fig_29.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 중위 표기법을 후위 표기법으로 변환하기 위해서는 두 개의 Helper function을 사용한다
  - 연산자의 우선 순위를 숫자 형태로 반환하는 `GetOpPrec` 함수
  - 두 연산자간에 비교 우위를 따져주는 `WhoPrecOp` 함수
- `GetOpPrec` 함수는 우선순위가 높을수록 큰 값을 반환한다.
- "("의 경우 연산자 Stack에서 괄호의 시작을 알리는 역할을 하므로 우선순위가 가장 낮게 설정되어야 한다.
  - 괄호의 시작은 곧 Stack에서의 새로운 바닥을 의미하므로 제일 밑에 존재해야 한다.
  - 따라서 +, - 연산자보다 더 낮은 중요도를 갖고있어야 한다.
- 정해진 연산자 외의 경우 처리를 위하여 마지막에 -1을 반환한다.

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-27-data_structure/fig_30.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- `WhoPrecOP` 함수는 두 연산자 op1과 op2를 입력 받고, 우선순위를 비교한다.
- `GetOpPrec` 함수로 연산자를 넘겨 우선순위를 반환받아 각각 op1Prec와 op2Prec로 저장한다.
- 숫자가 클수록 연산의 우선순위가 높으므로 op1의 우선순위가 높다면 1, 아니라면 -1, 같다면 0을 반환한다.
  - 우선순위가 같은 경우에 대해서도 먼저 연산자 Stack에 존재하는 연산자가 우선순위가 높으므로 별도 처리가 필요하다.

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-27-data_structure/fig_31.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- `ConvToRPNExp` 함수를 통해 후위표기법으로 변환해준다.
  - 앞의 `WhoPrecOP`와 `GetOpPrec` 함수를 이용
- 연산자를 쌓을 연산자 스택인 `stack`을 정의하고, 변환된 수식이 저장될 `convExp`를 정의
  - `convExp`는 str형이므로 마지막 '\\n'이 저장되어야 하므로 `(char*)malloc(expLen+1)`로 초기화됨
- 다음으로 `memset`을 통해 `convExp`의 모든 값을 0으로 초기화
- `StackInit`으로 연산자stack의 초기화
  - Stack의 top이 NULL을 가리키도록 함
- for 문 안에서 중위표기법을 후위표기법으로 변환하도록 함
  - 주요 연산과정!
- while문에서는 스택이 비어있는지 판단하고, 스택이 비어있지 않다면 연산자Stack 내부의 모든 남아있는 연산자들을 변환된 수식인 `convExp`에 붙임
- 마지막으로 변환된 수식을 원래 수식인 exp으로 복사하고 convExp를 삭제

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-27-data_structure/fig_32.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 메인 연산인 for문에서는 해당 인덱스 번째 문자를 `tok`으로 받고, 그게 숫자인지 아닌지에 따라 연산을 구분
- tok 문자가 숫자인 경우(`isdigit(tok)이 1을 반환`)
  - 변환수식인 `convExp`에 tok을 가져다 붙임
- tok 문자가 문자(연산자)인 경우
  - Switch 문을 이용하여 올바른 순서대로 연산자Stack에 쌓음

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-27-data_structure/fig_33.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- tok이 "("이라면 괄호의 시작으로 연산자Stack에 새로운 바닥을 깔아야 함
  - 바닥 역할을 하도록 "("을 연산자Stack에 넣고 다음 tok으로 넘어감
- tok이 ")"이라면 괄호의 끝을 의미하므로 연산자Stack에서 차례대로 연산자를 꺼내 연산자Stack의 top이 (popOp)"("일때까지 `convExp`에 붙임
  - 연산자Stack에는 아래의 tok이 연산자인 경우일 때 올바른 순서대로(우선순위) 연산자가 쌓여있음
  - "("이 나온다면 바닥을 의미하므로 break로 빠져나오고 다음 tok으로 넘어감
- tok이 연산자라면(+, -, \*, /)
  - 연산자Stack이 비어있지 않고(`!SIsEmpty(&stack)`) 연산자Stack의 top 연산자가 tok 연산자보다 우선순위가 높은 동안(`WhoPrecOp(SPeek(&stack), tok)>=0)`)
    - 이 과정을 통해 연산자Stack의 top값과 tok의 우선순위 비교를 수행한다.
    - `WhoPrecOp(SPeek(&stack), tok)`에서, 연산자Stack의 top의 중요도가 tok보다 중요하다면 while 문 실행 (`WhoPrecOp(SPeek(&stack), tok) > 0`)
      - tok은 연산자Stack에 쌓일 수 없다.
      - 따라서 연산자Stack의 top 값을 `convExp`에 가져다 붙인다. (`convExp[idx++] = SPop(&stack)`)
    - `WhoPrecOp(SPeek(&stack), tok)`에서, 연산자Stack의 top의 중요도가 tok과 같다면 while 문 실행 (`WhoPrecOp(SPeek(&stack), tok) = 0`)
      - tok은 먼저 연산자Stack에 존재하던 연산자를 빼내고 자신이 쌓여야 한다.
      - 따라서 연산자Stack의 top 값을 `convExp`에 가져다 붙인다. (`convExp[idx++] = SPop(&stack)`)
  - 연산자Stack의 top값과의 비교가 끝났다면, tok을 연산자Stack에 가져다 쌓는다.
    - `SPush(&Sstack, tok)`
    
### 중위 -> 후위: 프로그램 실행

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-27-data_structure/fig_34.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 위 내용을 통합한 프로그램을 실행

### 후위 표기법 수식의 계산

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-27-data_structure/fig_35.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 후위 표기법 수식의 계산은 왼쪽 문자열부터 계산을 한다.
- 첫 번째 연산자가 등장 할 때, 그 바로 앞 2개의 숫자에 대하여 피연산자로 하여 연산을 수행한다.
- 따라서 후기 표기법으로 정리된 수식을 차례대로 Stack에 넣고, 숫자가 나올 경우 push를, 연산자가 나올 경우 pop을 2번 하여 해당 연산자로 연산을 수행한다.
- 연산된 결과는 스택에 다시 넣는다.
- 올바른 수식이 올바르게 후기 표기법으로 변환 된 경우 항상 연산자 앞에는 두 개의 피연산자가 존재한다.

### 후위 표기법 수식 계산 프로그램의 구현

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-27-data_structure/fig_36.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 연산 방법을 정리하자면,
  - 피 연산자는 무조건 스택으로 옮긴다
  - 연산자를 만나면 스택에서 두 개의 피연산자를 꺼내 계산을 한다
  - 계산 결과는 다시 스택에 넣는다

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-27-data_structure/fig_37.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- `EvalRPNExp` 에는 후위표기법으로 변환된 수식이 저장된 메모리주소가 입력된다.
- for문에서 문자를 하나씩 꺼내 tok에 저장하고, 
  - 문자가 숫자라면 (`isdigit(tok)`이 1을 반환) push를 통해 stack에 쌓는다.
  - 아니라면(연산자라면) 2번의 pop을 통해 피연산자를 stack에서 꺼낸 후 연산자의 종류에 따른 연산을 수행한다.
- 연산 과정에서 최종 결과는 다시 stack에 push되어 쌓이게 되므로 마지막엔 stack에서 pop을 통해 최종 결과를 return한다.

### 후위 표기법 수식 계산 프로그램의 실행

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-27-data_structure/fig_38.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

### 계산기 프로그램의 완성

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-27-data_structure/fig_39.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 위의 과정을 포함하여 전체를 연결하는 `InfixCalculator.h`와 `InfixCalculator.c`를 작성한다.

<center>
<figure>
<img src="/assets/post_img/data_structure/2019-03-27-data_structure/fig_40.jpg" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 계산에선 원본 식을 보존하기 위해 `strcpy`를 이용하여 `expcpy`에 원본 식을 복사하여 연산한다.
- `EvalInfixExp` 함수에서 중위 계산식을 후위 계산식으로 변경하여 연산된 최종 수식에 대한  결과를 반환한다.
