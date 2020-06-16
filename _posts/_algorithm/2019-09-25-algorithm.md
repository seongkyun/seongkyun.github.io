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
  - check[j][i][0]: 벽을 부순적이 없다면 0차원에 카운팅
  - check[j][i][1]: 벽을 부순적이 있으면 1차원에 카운팅

```c
#pragma warning(disable:4996)
#include <iostream>
#include <queue>
#include <tuple>

using namespace std;

int check[1000][1000][2]; // [0]: 벽 안부쉈을때 , [1]: 벽 부쉈을때
int nodes[1000][1000];

int dx[] = { -1,1,0,0 };
int dy[] = { 0,0,-1,1 };

void bfs(int x, int y, int W, int H)
{
	queue <tuple<int, int, int>> q; // x, y, 벽부수지 않은경우
	check[y][x][0] = 1;//벽 부수지 않고 방문
	q.push(make_tuple(0, 0, 0));
	while (!q.empty())
	{
		int cur_x, cur_y, cur_w;
		tie(cur_x, cur_y, cur_w) = q.front();
		q.pop();

		for (int i = 0; i < 4; i++)
		{
			int next_x = cur_x + dx[i];
			int next_y = cur_y + dy[i];
			int next_w = cur_w;//다음 차원은 벽을 부술지 말지에 따라 
			
			if (0 <= next_x && next_x < W && 0 <= next_y && next_y < H)
			{ // 범위 안이라면

				if (nodes[next_y][next_x] == 0 && check[next_y][next_x][next_w] == 0)
				{//다음이 벽이 아니고, 방문한 적이 없다면
					check[next_y][next_x][next_w] = check[cur_y][cur_x][cur_w] + 1;
					q.push(make_tuple(next_x, next_y, next_w));
				}
				if (cur_w == 0 && nodes[next_y][next_x] == 1 && check[next_y][next_x][1] == 0)
				{//벽 부순적 없고, 다음 노드가 벽이고, 방문한 적이 없다면
					next_w = 1; // 벽 부수기때문에 1차원에 저장
					check[next_y][next_x][next_w] = check[cur_y][cur_x][cur_w] + 1;
					q.push(make_tuple(next_x, next_y, next_w));
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

	int result0 = check[H - 1][W - 1][0];
	int result1 = check[H - 1][W - 1][1];
	int answer;

	if (result0*result1 > 0)
	{//둘 다 계산됐으면 작은값이 답
		if (result0 < result1) answer = result0;
		else answer = result1;
	}
	else if (result0 + result1 == 0)
	{//둘 다 계산 안됐으면 경로 없음
		answer = -1;
	}
	else
	{//둘중 하나만 계산된 경우는 그것만 답이 존재하므로
		if (result0 < result1) answer = result1;
		else answer = result0;
	}
	printf("%d\n", answer);

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

```c
#pragma warning(disable:4996)
#include <iostream>
#include <queue>
#include <tuple>
#include <string>

using namespace std;

int water[50][50];
char nodes[50][50];
int way[50][50];

int dx[] = { -1,1,0,0 };
int dy[] = { 0,0,-1,1 };

// 물 차는 순서지도를 만듦
void flood(int x, int y, int W, int H)
{
	queue<pair<int, int>> q;
	water[y][x] = 1;
	q.push(make_pair(x, y));
	
	while (!q.empty())
	{
		int cur_x = q.front().first;
		int cur_y = q.front().second;
		q.pop();
		for (int i = 0; i < 4; i++)
		{
			int next_x = cur_x + dx[i];
			int next_y = cur_y + dy[i];
			if (0 <= next_x && next_x < W && 0 <= next_y && next_y < H)
			{
				if (water[next_y][next_x] == 0)
				{//물 차는 타이밍 저장
					water[next_y][next_x] = water[cur_y][cur_x] + 1;
					q.push(make_pair(next_x, next_y));
				}
			}
		}
	}
}

//길 찾아가는 방법
void find(int x, int y, int W, int H)
{
	queue<pair<int, int>> q;
	way[y][x] = 1;
	q.push(make_pair(x, y));

	while (!q.empty())
	{
		int cur_x = q.front().first;
		int cur_y = q.front().second;
		q.pop();
		for (int i = 0; i < 4; i++)
		{
			int next_x = cur_x + dx[i];
			int next_y = cur_y + dy[i];
			if (0 <= next_x && next_x < W && 0 <= next_y && next_y < H)
			{//범위 안에서
				if (way[next_y][next_x] == 0 || way[next_y][next_x] == -2)
				{
					int next_check = way[cur_y][cur_x] + 1;//다음 시간 정의
					if (water[next_y][next_x] > next_check || way[next_y][next_x] == -2 || water[next_y][next_x] == 0)
					{// 물 차는 시간이 다음 시간보다 크거나, 비버 집이거나(-2), 물이 없는경우(0)
						way[next_y][next_x] = next_check;
						q.push(make_pair(next_x, next_y));
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

	string inputs;
	int sx, sy, rx, ry;

	for (int j = 0; j < H; j++)
	{
		cin >> inputs;
		for (int i = 0; i < W; i++)
		{
			nodes[j][i] = inputs[i];
			if (inputs[i] == 'X')
			{ // 돌: -1
				water[j][i] = -1;
				way[j][i] = -1;
			}
			else if (inputs[i] == '*')
			{ // 물: 1
				water[j][i] = 1;
				way[j][i] = 1;
			}
			else if (inputs[i] == 'D')
			{ // 비버집: -2
				water[j][i] = -2;
				way[j][i] = -2;
				rx = i;
				ry = j;
			}
			else if (inputs[i] == 'S')
			{ // 시작점
				sx = i;
				sy = j;
			}
		}
	}

	for (int j = 0; j < H; j++)
	{
		for (int i = 0; i < W; i++)
		{
			if (nodes[j][i] == '*')
			{ // 물 위치에서 물 차는 지도를 만든다.
				flood(i, j, W, H);
			}
		}
	}
	
	//지도를 바탕으로 길을 찾아간다.
	find(sx, sy, W, H);

	int answer = way[ry][rx] - 1;//비버집 도착 시간
	if (answer > 0) printf("%d\n", answer);//집을 찾은경우 값은 0보다 항상 크다.
	else printf("KAKTUS\n");

    //system("pause");
	return 0;
}
```
