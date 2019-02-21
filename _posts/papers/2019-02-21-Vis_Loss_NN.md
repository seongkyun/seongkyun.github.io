---
layout: post
title: Visualizing the Loss Landscape of Neural Nets
category: papers
tags: [Deep learning, Mobilenetv2, Linear bottleneck]
comments: true
---

# Visualizing the Loss Landscape of Neural Nets

Original paper: https://arxiv.org/abs/1712.09913

Authors: Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein

## Abstract
- Neural network의 학습은 좋은 minimizer나 highly non-convex loss function을 찾는 능력에 의해 성능이 결정된다. 각 Skip connection을 포함하는 ResNet과 같은 네트워크 구조가 쉽게 학습되고, 학습 파라미터(batch size, learning rate, optimizer)를 잘 설정하며 일반화(generalization)를 잘 시키는 minimizer를 잘 만드는것으로 알려져 있다.
- 하지만, 이러한 차이의 이유와 근본적인 loss landscape에 대한 영향은 잘 알려져 있지 않다.
- 이 논문에선 neural loss function의 구조, 다양한 시각화(visualization)방법을 이용하여 generalization 측면에서의 loss landscape의 효과에 대해서 탐구한다.
- 우선, loss function curvature(곡률)의 시각화를 돕는 filter normalization이라는 방법을 제안하고, 각 loss function간 비교를 한다.
- 다음으로, 다양한 시각화 방법을 이용하여 어떻게 network의 구조가 loss landscape에 영향을 주는지, training 파라미터가 minimizer의 shape에 어떻게 영향을 주는지 탐구한다.

## 1. Introduction
- Neural network의 학습은 high-dimensional non-convex loss function을 최소화시키는것이 필요하다. 일반적인 neural loss function의 학습의 난해함(NP-hard)에도 불구하고[2], data와 label들이 학습 전에 randomized 되어있다 하더라도 간단한 gradient 계산 방법들을 이용해 global minimizer들(zero or near-zero training loss에서 parameter configurations)을 찾을 수 있다[42]. 하지만 이런 경우는 일반적인 경우가 아니며 neural net의 trainability(학습가능성)은 network의 구조, optimizer의 종류, 초기화(initialization) 방법 등등 다양한 고려사항에 의해 결정된다. 이러한 다양한 옵션들의 선택지에도 불구하고 근본적인 loss surface는 명확하지 않다. Loss function의 평가는 학습 데이터셋에서 모든 data point들에 대하여 looping이 필요하므로 비용이 많이들어 불가능하기 때문에 loss function의 평가는 이론적으로만 존재했다.

<center>
<figure>
<img src="/assets/post_img/papers/2019-02-21-Vis_Loss_NN/fig1.PNG" alt="views">
<figcaption>Figure 1. ResNet-56에 대해 skip connection의 유무에 따른 loss surface plot. 논문에서 제안하는 filter normalization scheme에 의해 shrpness, flatness의 비교가 가능하다.</figcaption>
</figure>
</center>

- 시각화는 neural network이 왜 동작하는지에 대한 답변에 도움이 된다. 특히, 왜 highly non-convex neural loss function을 최소화 시킬 수 있는가(에 대한 답변에 대한 도움이 된다). 논문에서는 neural loss function의 실증적 묘사를 위해 논문에서는 고 해상도 시각화 방법을 사용한다. 다음으로 어떻게 서로 다른 네트워크 구조가 loss landscape에 영향을 끼치는지를 탐구한다. 거기에 더해서, neural loss function의 non-convex 구조가 그 네트워크의 학습가능성과 연결되어있는지 탐구하고, 어떻게 neural minimizer의 지형(sharpness.flatness, surrounding landscape)이 일반화(generalization) 특성에 형향을 끼치는지에 대해 탐구한다.
- 논문에선 이러한 연구를 위해 training 과정 중 찾아지는 서로 다른 minima간의 비교를 위해 simple한 "filter normalization" scheme을 제안한다. 다음으로 서로 다른 방법에 의해 찾아지는 minimizer들의 sharpness/flatness를 탐구하기 위해 시각화를 사용하고, 네트워크 구조의 차이(skip connection이 있는지, filter의 개수, 네트워크 깊이 등)가 loss landscape에 영향을 끼치는가에 대한 연구를 수행한다.

### Contributions

## Theoretical Background

## The Basics of Loss Function Visualization

### 1-Dimensional Linear Interpolation

### Contour Plots & Random Directions

## Proposed Visualization: Filter-Wise Normalization

## The Sharp vs Flat Dilemma

### Filter Normalizaed Plots

## What Makes Neural Networks Trainable? Insights on the (Non)Convexity Structure of Loss Surfaces

### Experimental Setup

### The Effect of Network Depth

### Shortcut Connections to the Rescue

### Wide Models vs Thin Models

### Implications for Network Initialization

### Landscape Geometry Affects Generalization

### A note of caution: Are we really seeing convexity?

## Visualizing Optimization Paths

### Why Random Directions Fail: Low-Dimensional Optimization Trajectories

### Effective Trajectory Plotting using PCA Directions

## Conclusion

<center>
<figure>
<img src="/assets/post_img/papers/2019-02-21-Vis_Loss_NN/fig1.png" alt="views">
<figcaption>contents</figcaption>
</figure>
</center>
