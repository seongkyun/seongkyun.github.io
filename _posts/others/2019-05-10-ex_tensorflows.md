---
layout: post
title: Tensorflow 이전버전 pip 설치 및 CUDA dependencies
category: others
tags: [Ubuntu, tensorflow]
comments: true
---

# Tensorflow 이전버전 pip 설치 및 CUDA dependencies
- 참고 글: https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible

- Pytorch에서 tensorboard를 사용하기 위해 필요
  - Pytorch에서 tensorboard를 사용 가능하게 해주는 `tensorboardX`는 dependency로 `tensorflow`, `tensorboard`가 필요
  - 설치 순서는 `tensorflow` -> `tensorboardX`를 설치하면 된다.
    - `tensorboard`는 `tensorflow` 설치 시 자동으로 알맞은 버전을 설치한다.

## CUDA 및 cuDNN 버전 확인
- `$ cat /usr/local/cuda/version.txt` 로 CUDA 버전 확인
- `$ cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2` 로 cuDNN 버전 확인

## Linux GPU CUDA dependency
- 반드시 자신의 cuDNN, CUDA에 알맞은 tensorflow 버전을 설치해야 하며, 다르게 될 경우 십중팔구 error가 발생한다.

<center>
<figure>
<img src="/assets/post_img/others/2019-05-10-ex_tensorflows/linux_gpu.png" alt="views">
<figcaption></figcaption>
</figure>
</center>

## 설치
- `$ pip install tensorflow-gpu==version` 을 입력하면 된다.
  - ex. `$ pip install tensorflow-gpu==1.12.0`
  - 설치가 완료되면 자동으로 해당하는 tesnorboard가 설치된다.
- Pytorch에서 tensorboard 이용 시 `$ pip install tensorboardX`로 tensorboardX를 설치한다.
