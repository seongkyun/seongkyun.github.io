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
- 논문에선 이러한 연구를 위해 training 과정 중 찾아지는 서로 다른 minima간의 비교를 위해 simple한 "filter normalization" scheme을 제안한다. 다음으로 서로 다른 방법에 의해 찾아지는 minimizer들의 sharpness/flatness를 탐구하기 위해 시각화를 사용하고, 네트워크 구조의 차이(skip connection이 있는지, filter의 개수, 네트워크 깊이 등)가 loss landscape에 영향을 끼치는가에 대한 연구를 수행한다. 본 논문의 목표는 어떻게 loss function의 지형이 neural net의 일반화(generalization)에 영향을 끼치는가에 대한 이해다.

### 1.1. Contributions
- 논문에선 의미있는 loss function의 시각화에 대한 방법을 연구한다. 그 다음, 이러한 다양한 시각화 방법을 이용하여 loss landscape의 지형이 어떻게 학습가능성과 generalization error에 영향을 주는지에 대해 탐구한다. 아래에서 본 논문에서 다루는 더 자세한 issue에 대해 설명한다.
  - 논문에서는 다양한 loss function의 시각화 방법들의 단점을 드러내고, 간단한 시각화 방법들의 loss function minimizer의 local geometry(sharpness or flatness)를 정확하게 capture하는것에 대한 실패를 보여준다.
  - 논문에서는 "filter normalization"에 기반하는 간단한 시각화 방법을 제안한다. 논문에서 제안하는 normalization이 사용될 때 Minimizer들의 sharpness가 얼마나 generalization error와 유사한지를 보이고, 서로 다른 네트워크 구조와 학습 방법에 대한 비교를 수행한다.
  - 논문에서는 네트워크가 충분히 깊을 때 neural loss landscape가 거의 convex 상태에서 매우 복잡한(chaotic)상태로 빠르게 전환되는지에 대해 관찰한다. Convex에서 chaotic으로 가는 변환은 generalization error의 급격한 하락과 함게 발생하며, 학습가능성을 매우 낮추게 된다.
  - 논문에서는 skip connection이 어떻게 minimizer를 flat하게 만들고, chaotic하게 되는것을 방지하는지에 대해 관찰한다. 그래고 매우 깊은 네트워크의 학습에는 skip connection이 필수로 필요하게 되는지에 대한 설명을 한다.
  - 논문에서는 local minima 주변의 Hessian의 smallest(most negative) eigen-values를 계산하여 non-convexity를 양적으로 측정한다.
  - 논문에서는 SGD optimization의 궤적(trajectory)들의 시각화에 대한 연구를 한다. 논문에서는 이러한 궤적의 시각화에서 발생하는 어려움을 설명하고, extremely low dimensional space에 있는 optimization trajectory들을 보여준다. 이러한 low dimensionality는 논문의 2-dimensional visualization에서 관찰된 것 같이 loss landscape에 크게 볼록한 영역(large, nearly convex region)이 존재함으로 설명 가능하다.

## 3. The Basics of Loss Function Visualization
- Neural network은 학습에 영상 $\{x_{i}\}$ 과 label $\{y_{i}\}$ 같은 feature vector 뭉치(corpus)가 필요하며, $ L(\theta)=\frac{1}{m} \sum_{i=1}^{m}l(x_{i}, y_{i}\; ;\theta) $ 와 같은 loss function을 최소화시는 과정이 포함되고, 그 과정에서 $ \theta $ 로 정의되는 weight parameter를 $ m $ 개의 샘플을 이용하여 잘 얻어지는지 계산한다. 
Neural net은 많은 파라미터를 포함하며, 따라서 loss function은 very high-dimensional space에 존재하게 된다. 하지만 시각화는 1D(line)나 2D(surface) plot등 low-dimension에서만 가능하며, dimensionality gap을 줄위기 위한 몇 가지 방법들이 존재한다.
- __1-Dimensional Linear Interpolation__
  - Loss function을 plot하기 위한 간단하고 가벼운 방법이다. 두 개의 파라미터 세트인 $\theta$ 와 $\theta'$ 를 설정하고, loss function의 값들을 이러한 두 점을 이어 plot한다.
  - 이 방법은 다른 minima에서의 sharpness와 flatness에 대한 연구에 폭넓게 사용되었으며, batch size에 의한 sharpness의 dependence의 연구에도 사용되었다.
  - 1D linear interpolation 방법은 몇가지 약점이 있다. 우선 1D plot으로는 non-convexities의 시각화가 매우 어렵다. 다음으로, 이 방법은 네트워크의 batch normalization이나 invariance symmetries를 고려하지 않는다. 이러한 이유로 인해 1D interpolation plot으로 생성된 visual sharpness로는 적절한 비교가 불가능하다.
- __Contour Plots & Random Directions__
  - 이 방법을 이용하기 위해서는, 하나의 center point $\theta^{\ast}$ 를 그래프에서 정의하고, 두개의 direction vector $\delta$ 와 $\eta$ 를 정한다. 
  - 다음으로 function을 $ f(\alpha)=L(\theta^{\ast}+\alpha \delta) $ 의 1D line이나 
  $f(\alpha , \beta)=L(\theta^{\ast}+\alpha \delta +\beta \eta)$ (식 1)의 2D surface로 plot한다. 하지만 2D plotting은 연산량이 매우 많고 이러한 방법들은 보통 loss surface의 complex non-convexity를 capture하지 못하는 small region을 저 해상도(low-resolution)로 plot 한다. 이러한 이유로 본 논문에서는 weight space의 large slice에 대해 고 해상도 시각화를 사용하여 네트워크 디자인(설계)가 non-convex structure에 미치는 영향을 시각화 한다.

## 4. Proposed Visualization: Filter-Wise Normalization

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
