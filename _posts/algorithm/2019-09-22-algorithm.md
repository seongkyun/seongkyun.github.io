---
layout: post
title: 알고리즘 기초-플러드 필
category: algorithm
tags: [algorithm]
comments: true
---

# 알고리즘 기초-플러드 필
- 어떤 위치위 연결된 모든 위치를 찾는 알고리즘

### 2667 단지번호붙이기
- https://www.acmicpc.net/problem/2667
  - https://www.acmicpc.net/source/share/3e6999a82c774a51b2b70da44e90247f
  
- 0은 집이 없는 곳, 1은 집이 있는 곳 (각 칸에 0 또는 1이 저장)
- 지도를 가지고 연결된 집 모임인 단지를 정의하고, 단지에 번호를 붙임
  - + 각 단지의 크기를 구하기
  - 연결: 좌, 우, 아래, 위로 집이 있는 경우
  
```c
#pragma warning (disable:4996)
#include <iostream>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <queue>

using namespace std;
int nodes[25][25]; // 현재 노드 값
int check[25][25]; // 단지 방문 및 번호저장
int cnts[25 * 25]; // 단지 번호 갯수 저장

int dx[] = { -1, 1, 0, 0 }; // x방향 이동
int dy[] = { 0, 0, -1, 1 }; // y방향 이동

void bfs(int N, int cur_x, int cur_y, int cnt)
{
	queue<pair<int, int>> q;
	q.push(make_pair(cur_x, cur_y)); // 큐에 현재 노드 넣고
	check[cur_y][cur_x] = cnt; // 현재 단지번호로 check
	cnts[cnt]++; // 현재 단지 번호의 갯수를 1 증가
	while (!q.empty()) // 큐가 빌 동안 참조
	{
		cur_x = q.front().first; // 현재 노드 x위치
		cur_y = q.front().second; // 현재 노드 y위치
		q.pop(); // 현재노드 참조 완료했으니 빼고
		for (int i = 0; i < 4; i++) // 상하좌우 이동가능한 모든 경우에 대해
		{
			int nx = cur_x + dx[i]; // 다음 x위치
			int ny = cur_y + dy[i]; // 다음 y위치
			if (0 <= nx && nx < N && 0 <= ny && ny < N) // 다음 위치가 전체 범위를 넘어가지 않는 경우에 대해
			{
				if (nodes[ny][nx] == 1 && check[ny][nx] == 0) // 다음 노드가 건물이 있는 1 값이고, 참조되지 않았다면
				{
					q.push(make_pair(nx, ny)); // 참조
					check[ny][nx] = cnt; // check board를 cnt값으로 초기화하고
					cnts[cnt]++; // 해당 cnt 값을 증가
				}
			}
		}
	}
}

int main()
{
	int N;
	scanf("%d", &N);
	
	for (int j = 0; j < N; j++)
	{
		for (int i = 0; i < N; i++)
		{
			scanf("%1d", &nodes[j][i]); // 핝 자리의 int 값을 받기 위함
		}
	}

	int cnt = 0;
	for (int j = 0; j < N; j++)
	{
		for (int i = 0; i < N; i++)
		{
			if (nodes[j][i] == 1 && check[j][i] == 0) // 참조 된 적 없는 노드라면
			{
				cnt++; // cnt 값을 1 증가시키고
				bfs(N, i, j, cnt); // 참조 시작
			}
		}
	}
	printf("%d\n", cnt);
	vector<int> results;
	for (int i = 1; i <= cnt; i++)
	{
		results.push_back(cnts[i]); // 정답을 위한 cnt 값들을 저장함
	}

	sort(results.begin(), results.end()); // 정렬 후
	for (int i = 0; i < cnt; i++)
	{
		printf("%d\n", results[i]); // 출력
	}

	//system("pause");
	return 0;
}

```


### 4963 섬의 개수
- https://www.acmicpc.net/problem/4963
  - https://www.acmicpc.net/source/share/eeebe2e472d44fc081f1ed68eb0b512d
