---
layout: post
title: CNN의 parameter 개수와 tensor 사이즈 계산하기
category: study
tags: [CNN parameter, tensor size]
comments: true
---

# CNN의 parameter 개수와 tensor 사이즈 계산하기
- 이번 글에서는 네트워크의 텐서 사이즈와 파라미터의 갯수를 계산하는 공식에 대해 다루려 한다.
- 아래의 AlexNet을 이용하여 예시를 든다.

<center>
<figure>
<img src="/assets/post_img/study/2019-01-25-num_of_parameters/fig1.png" alt="views">
<figcaption>Alexnet의 구조</figcaption>
</figure>
</center>

- AlexNet의 구조
  - Input: 227*227*3 크기의 컬러 이미지. 논문의 224*224 사이즈는 오타임
  - Conv-1: 11*11 크기의 커널 96개, stride=4, padding=0
  - MaxPool-1: stride 2, 3*3 max pooling layer
  - Conv-2: 5*5 크기의 커널 256개, stride=1, padding=2
  - MaxPool-2: stride 2, 3*3 max pooling layer
  - Conv-3: 3*3 크기의 커널 384개, stride=1, padding=1
  - Conv-4: 3*3 크기의 커널 384개, stride=1, padding=1
  - Conv-5: 3*3 크기의 커널 256개, stride=1, padding=1
  - Maxpool-3: stride 2, 3*3 max pooling layer
  - FC-1: 4096개의 fully connected layer
  - FC-2: 4096개의 fully connected layer
  - FC-3: 1000개의 fully connected layer

- 본 글에서는 AlexNet을 이용하여 아래의 내용을 알아본다.
  - 각 단계에서 텐서의 크기를 계산하는 방법
  - 네트워크에서 총 파라미터 개수를 계산하는 방법

## Convolution layer의 output tensor size
- 각각 기호를 아래와 같이 정의
  - $O$: Size(width) of output image
  - $I$: Size(width) of input image
  - $K$: Size(width) of kernels used in the Conv layer
  - $N$: Number of kernels
  - $S$: Stride of the convolution operation
  - $P$: Padding size
- $O$(Size(width) of output image)는 다음과 같이 정의 됨

$$O=\frac{I-K+2P}{S}+1$$

- 출력 이미지의 채널 수는 커널의 갯수($N$)와 같음

### Example on AlexNet
- AlexNet의 입력 이미지 크기는 227*227*3
- 첫 번째 conv layer(Conv-1)는 11*11*3 크기의 커널 96개, stride=4, padding=0

$$O=\frac{227-11+2*0}{4}+1=55$$

- 따라서, Conv-1 의 출력 출력 tensor size는 $55\times 55\times 96$임.
  - 각 커널 당 하나의 채널을 나타내므로, 3채널(RGB) 이미지에 대해 3배가 곱해져 총 $55\times 55\times 96\times 3$이 됨.
  - Conv-2, 3, 4, 5도 동일한 방법으로 계산 가능
  


---
- [참고 글]

https://www.learnopencv.com/number-of-parameters-and-tensor-sizes-in-convolutional-neural-network/
