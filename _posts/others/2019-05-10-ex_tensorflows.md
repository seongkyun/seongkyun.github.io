---
layout: post
title: Tensorflow 이전버전 pip 설치 및 CUDA dependencies
category: others
tags: [Ubuntu, tensorflow]
comments: true
---

# Tensorflow 이전버전 pip 설치 및 CUDA dependencies
- 참고 글: https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible

- 어느 라이브러리던지 다 동일하겠지만 tensorflow toolkit은 GPU를 이용한 CUDA 연동시 관련 dependency가 다르면 무조건 오류가 발생한다.
- 게다가 tensorflow가 2.0으로 크게 개편되면서 홈페이지에선 이전 버전에 대한 설치 안내가 싹 사라졌다.
- 본인은 pytorch를 주로 사용하지만 tensorboard의 편의때문에 `tensorboardX`와 더불어 tensorboard를 설치해 사용하고, tensorboard를 사용하기 위해선 tensorflow가 dependency로 필요하게 된다.
- 하지만 현재 pytorch 버전에 맞게 CUDA 9.0과 CUDNN v9.0이 설치되어있지만 최신 tensorflow2.0을 사용하고자 그래픽 드라이버 업데이트(over 410.xx), CUDA 10.0으로 업그레이드를 하는 모험을 할 수 없다.
- 이를 위해 쉽게 pip installer로 이전 버전 tensorflow를 설치하고, 관련 디펜던시에 대해 정리했다.

<center>
<figure>
<img src="/assets/post_img/others/2019-05-10-ex_tensorflows/fig1.PNG" alt="views">
<figcaption></figcaption>
</figure>
</center>
