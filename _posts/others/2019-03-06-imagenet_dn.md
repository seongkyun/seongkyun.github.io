---
layout: post
title: ImageNet LSVRC 2012 데이터셋 다운로드 받기
category: others
tags: [ImageNet, download]
comments: true
---

# ImageNet LSVRC 2012 데이터셋 다운로드 받기

## Training set(138GB)
- `wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar`
- 용량이 큰 만큼 매우 오래 걸림
  - 약 5일정도..
- `nohup wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar` 로 백그라운드 다운로드도 가능

## Validation set(6.3GB)
- `wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar`

## 압축 풀기
- __학습 데이터셋 압축 풀기__
- `mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train`
- `tar -xvf ILSVRC2012_img_train.tar`
- `rm -f ILSVRC2012_img_train.tar` (만약 원본 압축파일을 지우려면)
- `find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done`
- `cd ..`

- __Validation 데이터셋 압축 풀기__
- `mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar`
- `wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash`

- 작업이 완료되면 train 폴더 안에 카테고리별로, val 폴더 안에 카테고리별로 정리되어 들어간다.

## ImageNet training in Pytorch
- https://github.com/pytorch/examples/tree/master/imagenet 참조
