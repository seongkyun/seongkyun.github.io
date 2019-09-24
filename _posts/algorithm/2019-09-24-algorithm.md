---
layout: post
title: 알고리즘 기초-BFS
category: algorithm
tags: [algorithm]
comments: true
---

# 알고리즘 기초-BFS
- __가장 중요함!!!!__
- BFS의 목적은 __임의의 정점__ 에서 시작해서, __모든 정점을 한 번씩 방문__ 하는 것!
- 그 중, BFS는 __최단 거리를 구하는 알고리즘__
  - 특정 조건을 만족 할 때, BFS가 최단거리 알고리즘이 됨
    - 특정 조건?
- BFS는 __모든 가중치가 1일 때__ 최단 거리를 구하는 알고리즘!
  - 이유: BFS가 단계별로 움직이기 때문임.
    - BFS는 단계적으로 진행되므로, 모든 가중치가 1일 때 최단거리를 구하는 알고리즘이 된다.
  - 즉, 최단거리 문제의 해결에 사용이 가능하다!
  
### BFS를 이용해 해결 가능한 문제의 조건
- BFS를 이용해 해결 가능한 문제는 아래와 같은 조건을 만족해야 한다.
  - 1. 최소 비용 문제여야 한다.
  - 2. 간선의 가중치가 1이어야 한다.
  - 3. 정점과 간선의 개수가 적어야 한다.
    - 적다는 것은 문제의 조건에 맞춰서 해결할 수 있다는 것을 의미함
    - 시간, 메모리 제한의 해결을 위한것임
- 간선의 가중치가 문제에서 구하라고 하는 최소 비용과 의미가 일치해야 함
  - 간선의 가중치 = 최소 비용
- 즉, 거리의 최소값을 구하는 문제라면?
  - 간선의 가중치 = 거리
- 시간의 최소값을 구하는 문제?
  - 간선의 가중치 = 시간

### 2178 미로 탐색
- (1,1)에서 (N, M)으로 가는 가장 빠른 방법을 찾는 문제
- 앞의 블러드 필 문제들과 동일하게, 이동 가능한 방향의 dx, dy를 만들어 놓고 푼다.
- 해당 방향에 갔을 때, 참조하지 않은 경우 이전 check 값에 1을 더해서 현재 위치까지 오는데 필요한 비용을 계산한다.

```c
#pragma warning (disable:4996)
#include <iostream>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <queue>

using namespace std;
int nodes[100][100]; // 현재 노드 값
int check[100][100]; // 단지 번호 저장
int final_result; // 최종 결과

int dx[] = { -1, 1, 0, 0}; // x 방향 이동
int dy[] = { 0, 0, -1, 1}; // y 방향 이동
	
void bfs(int W, int H, int cur_x, int cur_y)
{
	queue<pair<int, int>> q; // pair 만들기
	q.push(make_pair(cur_x, cur_y)); // 큐 에는 좌표형식으로 수를 담는다.

	int next; // 다음 check숫자를 담을 변수
	check[cur_y][cur_x] = 1; // 현재 시작위치 check를 1로 초기화
	
	while (!q.empty()) // 큐가 빌 때까지
	{
		cur_x = q.front().first; // 현재 좌표를 담고
		cur_y = q.front().second;
		q.pop(); // 큐에서 꺼내고
		next = check[cur_y][cur_x] + 1; // 다음 위치의 check 값을 현재 위치 기준으로 정하고 (분기 되는 부분)
		for (int i = 0; i < 4; i++)
		{
			int nx = cur_x + dx[i]; // 다음 좌표를 만들고
			int ny = cur_y + dy[i];
			if (0 <= nx && nx < W && 0 <= ny && ny < H) // 범위가 넘어가지 않는 선에서
			{
				if (nodes[ny][nx] == 1 && check[ny][nx] == 0) // 다음 노드에 경로가 존재하고, 방문한 적 없으면
				{
					q.push(make_pair(nx, ny)); // 다음 노드를 방문
					check[ny][nx] = next; // 해당 check값은 만들어둔 next로 초기화
					if (nx == W - 1 && ny == H - 1) // 만약 마지막 꼭짓점이면
					{
						final_result = check[ny][nx]; // final reuslt를 초기화함
					}
				}
			}
		}
	}
}

int main()
{
	int N, M; // N=H, M=W
	scanf("%d %d", &N, &M);

	for (int j = 0; j < N; j++)
	{
		for (int i = 0; i < M; i++)
		{
			scanf("%1d", &nodes[j][i]); // 입력 받기
		}
	}

	bfs(M, N, 0, 0); // 함수를 돌고

	printf("%d\n", final_result); // 결과 출력

	system("pause");
	return 0;
}
```

### 7576 토마토


```c
#pragma warning (disable:4996)
#include <iostream>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <queue>

using namespace std;
int nodes[1000][1000]; // 현재 노드 값
int check[1000][1000]; // 단지 번호 저장

int dx[] = { -1, 1, 0, 0}; // 이동 위치
int dy[] = { 0, 0, -1, 1};

queue<pair<int, int>>q;

void rotten(int W, int H, int cur_x, int cur_y, int next)
{
	while (!q.empty())
	{
		cur_x = q.front().first; // 현재 좌표 초기화
		cur_y = q.front().second;
		q.pop(); // 초기화 후 pop
		for (int i = 0; i < 4; i++)
		{
			int nx = cur_x + dx[i]; // 다음 좌표 초기화
			int ny = cur_y + dy[i];
			if (0 <= nx && nx < W && 0 <= ny && ny < H) // 박스 안쪽이라면
			{
				if (nodes[ny][nx] != -1 && check[ny][nx] == 0) // 만약 접근 불가 노드(-1)가 아니고, check 안했다면
				{
					q.push(make_pair(nx, ny)); // 접근하고 check
					check[ny][nx] = check[cur_y][cur_x] + 1; // check 값을 이전 값보다 큰 값으로 초기화
				}
			}
		}
	}

}

int main()
{
	int N, M; // N=H, M=W
	scanf("%d %d", &M, &N);

	for (int j = 0; j < N; j++)
	{
		for (int i = 0; i < M; i++)
		{
			scanf("%d", &nodes[j][i]);


			if (nodes[j][i] == 1) // 토마토가 저장된 노드라면
			{
				q.push(make_pair(i, j)); // 이곳에서 push로 큐에 좌표값 저장
				check[j][i] = 1; // 그리고 check 표시(큐에 넣고 바로 check 해야하므로)
			}
		}
	}

	int next = 1;
	for (int j = 0; j < N; j++)
	{
		for (int i = 0; i < M; i++)
		{
			if (nodes[j][i] == 1)
			{
				rotten(M, N, i, j, next); // 전체 노드들 중 1이 들어간 곳에서 연산 시작
				next++; // 다음 초기화 값 지정
			}
		}
	}

	int maxval = -1;
	bool ok = true;

	for (int j = 0; j < N; j++)
	{
		for (int i = 0; i < M; i++)
		{
			if (check[j][i] >= maxval) maxval = check[j][i]; // 최댓값을 찾고(횟수를 의미)
			if (check[j][i] == 0 && nodes[j][i] != -1) ok = false; // 벽에 가로막힌 case 세고
		}
	}

	if (ok) // 결과 출력
		printf("%d\n", maxval - 1);
	else
		printf("%d\n", -1);

	system("pause");
	return 0;
}
```

### 1697 숨바꼭질
- https://www.acmicpc.net/problem/1697
  - http://codeplus.codes/9116e7f3d4634964a7c5b3f0f88bb332
- 그래프가 아닌 문제를 BFS 최단거리 문제로 적용시켜 풀기
  - 현재 위치를 정점, 위치의 이동을 간선으로 표시.
  - 경우의 수가 매우 많으므로(100,000개) BFS로 푸는 것이 최선이다.
	
- BFS 문제지만, 인접 리스트/행렬을 만들 필요가 없음
  - 문제의 수식으로 다른 정점을 3개만 계산해 보면 됨!
  - 큐에 수빈이(N) 위치를 넣어가며 이동시킴
    - 큐가 빈다 = while 문 빠져나온다 = 수빈이(N)가 동생(K)를 찾았다 -> K=N이 될 때 pop 해야 함
    - 한번 방문한 곳은 다시 방문하지 않아야 효율적이므로 check 한다.

<center>
<figure>
<img src="/assets/post_img/algorithm/2019-09-24-algorithm/fig1.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- N=5, K=17 예시의 위 그림처럼, 가능한 경우를 가지치기 해 가며 따져보면 됨
  - q에는 방문 순서대로, 이미 방문한(checked) 곳은 건너 뛰고 진행
    - q = 5 4 6 10 3 8 7 ... 17
    - 즉, 17을 몇 번만에 방문했는지가 정답 (5-4-8-16-17)

```c
#pragma warning (disable:4996)
#include <iostream>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <queue>

using namespace std;

const int MAX = 200000; // worst case
bool check[MAX + 1];
int time[MAX + 1];

queue<int> q;

int main()
{
	int N, K;
	scanf("%d %d", &N, &K);
	int now;
	check[N] = true; // N의 자리를 check
	q.push(N); // N을 큐에 넣고
	time[N] = 0; // N의 위치에서 시간은 0

	while (!q.empty()) // 큐가 빌 때 까지
	{
		now = q.front(); // 현재 위치를 초기화
		q.pop(); // 큐를 비우고
		if (0 <= now - 1 && check[now - 1] == false) // 이전 위치를 참조하면
		{
			q.push(now - 1); // now-1 위치 방문
			check[now - 1] = true; // now-1 위치 check
			time[now - 1] = time[now] + 1; // 현재위치 걸린시간+1만큼 시간 소요 더됨
		}
		if (now + 1 < MAX && check[now + 1] == false) // 다음 위치를 방문하면
		{
			q.push(now + 1);
			check[now + 1] = true;
			time[now + 1] = time[now] + 1;
		}
		if (now * 2 < MAX && check[now * 2] == false) // 순간이동을 하면
		{
			q.push(now * 2);
			check[now * 2] = true;
			time[now * 2] = time[now] + 1;
		}
	}

	printf("%d\n", time[K]); // K 번째에 N에서 시작했을 때의 소요 시간이 저장됨

	//system("pause");
	return 0;
}
```

### BFS

<center>
<figure>
<img src="/assets/post_img/algorithm/2019-09-24-algorithm/fig2.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 이 경우, A->E의 최단 소요 루트는 위로만 가는 것임
  - A->B, B->C, ..., D->E 이동 간 각자 빠른 길만 선택하면 됨
- 만약, 위 쪽의 길 중 한 번만 선택해야 한다면?
  - 한 번만 위로 빠르게 지나갈 수 있다면?
  
<center>
<figure>
<img src="/assets/post_img/algorithm/2019-09-24-algorithm/fig3.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 같은 B 더라도, 이전에 위로 갔는지, 아래로 갔는지에 따라서 서로 다른 정점이 된다.
  - 의미가 서로 달라짐
  - 즉, __서로 같은 정점은 같은 간선을 갖고 있음을 의미한다.__
- 다음에 선택 가능한 경우의 수의 종류에 따라 서로 다른 정점이 된다.
  - 따라서 위 그림에서 위와 아래 B는 서로 다름
    - 파란 간선을 사용한 횟수를 기준으로 나누므로

<center>
<figure>
<img src="/assets/post_img/algorithm/2019-09-24-algorithm/fig4.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>

- 즉, 할 수 있는 선택이 다르면 위 처럼 다른 정점이 됨
  - C0와 C1은 서로 다른 상태

### 14226 이모티콘
- https://www.acmicpc.net/problem/14226
  - https://www.acmicpc.net/source/share/88648734b074475494ad8253b121cc68
- BFS에서 하나의 정점이 서로 다른 두 개의 정보를 저장하고 있으면 안됨
- 화면에 있는 이모티콘 개수가 5개인 경우
  - 클립보드에 있는 이모티콘의 개수에 따라서 복사하기 연산의 결과가 다름
- 즉, 화면의 이모티콘 개수 S와 클립보드의 이모티콘 개수 C가 중요함
  - 정점을 나타내는 정보가S와 C로 구분됨
- 차 후 try...


