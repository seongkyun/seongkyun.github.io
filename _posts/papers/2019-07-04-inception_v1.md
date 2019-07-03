---
layout: post
title: Going Deeper with Convolutions (Inception)
category: papers
tags: [Deep learning]
comments: true
---

# Going Deeper with Convolutions

Original paper: https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf

Authors: Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed

- 참고 글: https://datascienceschool.net/view-notebook/8d34d65bcced42ef84996b5d56321ba9/

# Inception (GoogLeNet)
- 2014년 ILSVRC에서 1등을 한 모델로, Going Deeper with Convolutions 라는 논문에서 Inception이란 이름으로 발표됨
- Inception은 이 버전 이후 v4까지 여러 버전이 발표되었으며, 이번 글에선 Inception v1에 대해서만 다룸
- ILSVRC 2014에서 팀명이 GoogLeNet이므로 구글넷으로 부른다.

## 구글의 가설
- 딥러닝에서 대용량 데이터 학습시 일반적으로 망이 깊고 레이어가 넓은 모델의 성능이 좋다는것이 정설
- 하지만 현실적으로는 네트워크를 크게 만들면 파라미터가 많이 늘어나고, 망이 늘어날때마다 연산량이 exponential하게 많아지며 overfitting, vanishing gradient등의 문제가 발생해 학습이 어려워짐
- 이를 해결하기 위한 방안 중 하나가 Sparse connectivity임.
- 현재까지 사용된 convolution 연산은 densely하게 연결되어있음
- 이를 높은 correlation을 가진 노드들끼리만 연결하도록, 즉 노드들 간의 연결을 sparse하도록 바꾼다면 연산량과 파라미터수가 줄고, 따라서 overfitting 또한 개선될 것이라 생각함
  - Fully connected network에서 사용하는 dropout과 비슷한 기능을 할 것이라고 본 것임
- 하지만 실제로는 dense matrix 연산보다 sparse matrix 연산이 더 큰 computational resource를 사용
  - LeNet때의 CNN은 sparse한 CNN 연산을 사용함.
  - 이후 연산을 병렬처리하기위해 dense connection을 사용했고, 이에 따라 dense matrix 연산기술이 발전함
  - 반면 sparse matrix 연산은 dense matrix 연산만큼 발전하지 못했고, dense matrix연산보다 비효율적이게 됨
- 따라서, 위 목적을 달성하기 위해 sparse connectivity를 사용하는것은 해결방안이 될 수 없었음

- 여기에서, 구글이 고민한것은 어떻게 노드 간의 연결을 줄이면서(sparse connectivity) 행렬 연산은 dense 연산을 하도록 처리하는가였으며, 이 결과가 바로 Inception module임.

## Inception module

<center>
<figure>
<img src="/assets/post_img/papers/2019-07-04-inception_v1/fig1.png" alt="views">
<figcaption></figcaption>
</figure>

- 구글넷이 깊은 망을 만들고도 학습이 가능했던것은 inception module 덕분임
- 위 그림은 inception module의 구조를 나타낸 것으로, 입력값에 대해 4가지 종류의 연산을 수행하고 4개의 결과를 채널 방향으로 concat함
- 이러한 inception module이 모델에 총 9개가 있음

- Inception module의 4가지 연산은 각각
  - 1x1 convolution
  - 1x1 convolution 후, 3x3 convolution
  - 1x1 convolution 후, 5x5 convolution,
  - 3x3 MaxPooling후 1x1 convolution
  - 이 결과들을 Channel-wise concat(feature map을 쌓기)
- 이 중, 1x1 conv 연산은 모호하게 여겨질 수 있으나 핵심 역할을 하며 기능은 아래와 같음
  - 채널의 수를 조절하는 역할. 채널의 수를 조절한다는것은 채널간의 correlation을 연산한다는 의미라고 할 수 있음. 기존의 conv 연산은 3x3커널 연산의 경우 3x3 크기의 지역 정보와 함께 채널 간의 정보 또한 같이 고려하여 하나의 값으로 나타냄. 다르게 말하면 하나의 커널이 2가지의 역할 모두 수행해야 하는것을 의미. 대신, 이전에 1x1 conv를 사용하면 채널 방향으로만 conv 연산을 수행하므로 채널간의 특징을 추출하게 되며, 3x3은 공간방향의 지역 정보에만 집중하여 특징을 추출하게 됨. (역할을 세분화 해줌) 채널간의 관계정보는 1x1 conv에서 사용되는 파라미터들끼리, 이미지의 지역 정보는 3x3 conv에 사용되는 파라미터들끼리 연결된다는점에서 노드간의 연결을 줄였다고 볼 수 있음
  - 1x1 conv 연산으로 이미지의 채널을 줄여준다면 3x3과 5x5 conv 레이어에서의 파라미터 개수를 절약 할 수 잇음. 이 덕분에 망을 기존의 cnn 구조들보다 더욱 깊게 만들고도 파라미터가 그렇게 많지 않음
  
  
