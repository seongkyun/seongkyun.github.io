---
layout: post
title: 알고리즘 기초-BFS2
category: algorithm
tags: [algorithm]
comments: true
---

# 알고리즘 기초-BFS2

### 2206 벽 부수고 이동하기
- check 배열에 방문과 벽 부숨을 다 표현해야 한다.
  - check[j][i][0]: 방문여부
  - check[j][i][1]: 기존 벽 부숨 여부
- 현재 틀린상태로 디버깅중..

```c
#pragma warning(disable:4996)
#include <iostream>
#include <queue>
#include <tuple>

using namespace std;

int check[1000][1000][2]; // [0]: 방문여부, [1]: 벽부숨 여부
int nodes[1000][1000];

int dx[] = { -1,1,0,0 };
int dy[] = { 0,0,-1,1 };

void bfs(int x, int y, int W, int H)
{
	queue <tuple<int, int, int>> q; // x, y, 벽부숨
	int visit = 1;
	check[y][x][0] = visit; // 방문
	int wall = 0;
	check[y][x][1] = wall; // 벽 안부숨
	q.push(make_tuple(0, 0, wall));
	while (!q.empty())
	{
		int cur_x, cur_y, cur_w;
		tie(cur_x, cur_y, cur_w) = q.front();
		q.pop();
		for (int i = 0; i < 4; i++)
		{
			int next_x = cur_x + dx[i];
			int next_y = cur_y + dy[i];
			if (0 <= next_x && next_x < W && 0 <= next_y && next_y < H)
			{ // 범위 안이라면
				if (check[next_y][next_x][0] == 0)
				{ // 다음 방문한적 x
					if (cur_w == 0)
					{// 현재까지 벽 부순적 없다면
						if (nodes[next_y][next_x] == 1)
						{// 다음이 벽이라면
							wall = 1; //벽 부수고 check
							visit = check[cur_y][cur_x][0] + 1;
							check[next_y][next_x][0] = visit;
							check[next_y][next_x][1] = wall;
							q.push(make_tuple(next_x, next_y, wall));
						}
						if (nodes[next_y][next_x] == 0)
						{//다음이 벽이 아니라면
							wall = 0; //벽 부수지 않고 check
							visit = check[cur_y][cur_x][0] + 1;
							check[next_y][next_x][0] = visit;
							check[next_y][next_x][1] = wall;
							q.push(make_tuple(next_x, next_y, wall));
						}
					}
					else
					{// 현재까지 벽 부순적 있다면
						if (nodes[next_y][next_x] == 0)
						{// 다음에 벽이 없다면 부수지 않고 check
							wall = 1; // 이미 부쉈으므로
							visit = check[cur_y][cur_x][0] + 1;
							check[next_y][next_x][0] = visit;
							check[next_y][next_x][1] = wall;
							q.push(make_tuple(next_x, next_y, wall));
						}
					}
				}
				if (next_x == W - 1 && next_y == H - 1)
				{// 다음이 마지막 노드라면
					if (check[next_y][next_x][0] == 1)
					{//방문을 했던 경우라도
						// 새로운 값이 작으면 작은거로 check
						int now = check[cur_y][cur_x][0] + 1; //현재 노드에서 계산된 값이
						int prev = check[next_y][next_x][0]; // 저장되어있던 값보다
						if (now < prev) check[next_y][next_x][0] = now; // 작다면 작은값으로 초기화
					}
				}
			}
		}
	}
}

int main()
{
	int H, W; // N=X, M=Y
	scanf("%d %d", &H, &W);
	for (int j = 0; j < H; j++)
	{
		for (int i = 0; i < W; i++)
		{
			scanf("%1d", &nodes[j][i]);
		}
	}
	
	bfs(0, 0, W, H);
	
	int result = check[H - 1][W - 1][0];
	if (result)	printf("%d\n", result);
	else printf("-1\n");

	system("pause");
	return 0;
}
```

### 3055 탈출
- 고슴도치(S)는 비버(D)에 가야 한다.
- 고슴도치(S)는 1분에 1칸씩 이동한다.
- 물(\*)의 위치는 1분에 1칸씩 상하좌우로 넓어진다.
- 돌(X)의 위치는 변하지 않으며 이동 불가능하다.
- 고슴도치는 비어있는 공간(.)으로만 이동 가능하다.
- 고슴도치(S)가 비버(D)에게 갈 수 있는 가장 짧은 시간을 구해야 한다.
