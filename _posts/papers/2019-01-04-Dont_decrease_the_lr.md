---
layout: post
title: Dont't decay the learning rate, increase the batch size
category: papers
tags: [Deep learning]
comments: true
---

# Dont't decay the learning rate, increase the batch size

Original paper: https://arxiv.org/abs/1711.00489

Authors: Samuel L. Smith, Pieter-Jan Kindermans, Chris Ying, Quoc V. Le (Google Brain)

## Practical view of Generalization
- 기존 연구들은 어떻게 해야 generalization이 되는지를 많이 제안했었음.
  - Imagenet challenge에서 제안된 여러 구조들이 generalization이 잘 되는 구조와 hyper parameter setting들을 전부 다 포함
  - Generalization 성능이 좋은 구조와 hyper parameter들을 유지하면서 응용하려면?
- 본 논문에서는 generalization에 크게 영향을 끼치는 learning rate, batch size에 대해 다룸

## Batch size in the Deep learning
- Batch size가 크면 연산이 효율적(빠른 학습 가능)
- Batch size가 작으면 generalization이 잘 됨
- 연산 효율을 좋게 하면서 generalization을 잘 시키는 방법에 대해 본 논문에서는 연구

## Batch size and Generalization
<center>
<figure>
<img src="/assets/post_img/papers/2019-01-03-Dont_decrease_the_lr/fig1.PNG" alt="views">
</figure>
</center>

- Imagenet을 1시간 안에 학습 시키는 논문
  - P.Goyal et al. (2017), "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
- 위 논문에서 주요하게 사용한게 Linear scaling rule
  - Batch size 크기에 비례해서 learning rate를 조절해야한다는 rule
  - Batch size가 2배가 되면, learning rate도 2배가 되어야 함

## Contribution

continued

---
- [참고 글]

https://www.youtube.com/watch?v=jFpO-E4RPhQ

