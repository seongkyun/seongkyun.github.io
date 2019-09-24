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


<center>
<figure>
<img src="/assets/post_img/algorithm/2019-09-24-algorithm/fig1.PNG" alt="views">
<figcaption> </figcaption>
</figure>
</center>
