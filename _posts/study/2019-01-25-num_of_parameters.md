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

## MaxPool layer의 output tensor size
- 각각 기호를 아래와 같이 정의
  - $O$: Size(width) of output image
  - $I$: Size(width) of input image
  - $S$: Stride of the convolution operation
  - $P_{s}$: Pooling size
- $O$(Size(width) of output image)는 다음과 같이 정의 됨

$$O=\frac{I-P_{s}}{s}+1$$

- Convolution layer와는 다르게 출력의 채널 수는 입력의 개수와 동일
- Conv layer의 $O$ 수식에서 커널 크기($K$)를 $P_{s}$로 대체하고 $P=0$으로 설정하면 동일한 식이 됨

### Example on AlexNet
- MaxPool-1은 stride 2, 사이즈는 3*3, 이전 단(Conv-1)의 출력 크기는 $55\times 55\times 96$임

$$O=\frac{55-3}{2}+1=27$$

- 따라서 출력의 크기는 $27\times 27\times 96$
- MaxPool-2, 3도 동일한 방법으로 계산

## Fully Connected layer의 output tensor size
- FC layer는 layer의 뉴런 수와 동일한 길의의 벡터를 출력
- AlexNet summary
  - AlexNet에서 입력은 크기 227x227x3의 이미지
  - Conv-1의 출력은 MaxPool-1을 거치며 55x55x96에서 27x27x96으로 변환됨
  - Conv-2 이후에는 size가 27x27x256에서 MaxPool-2을 거치며 13x13x256으로 변경됨
  - Conv-3은 크기를 13x13x384로 변환
  - Conv-4는 크기가 유지됨
  - Conv-5는 크기를 27x27x256으로 변환함
  - 마지막으로 MaxPool-3는 크기를 6x6x256으로 줄임
  - 이 이미지는 크기 4096x1 크기의 벡터로 변환되는 FC-1에 feed됨
  - FC-2는 크기를 유지
  - FC-3 은 size를 1000x1로 변환

## Convolution layer의 parameter 갯수
- CNN의 각 layer는 weight parameter와 bias parameter가 존재.
- 전체 네트워크의 parameter 수는 각 conv layer 파라미터 수의 합

- 각각 기호를 아래와 같이 정의
  - $W_{c}$: Number of weights of the Conv layer
  - $B_{c}$: Number of biases of the Conv layer
  - $P_{c}$: Number of parameters of the Conv layer
  - $K$: Size(width) of kernels used in the Conv layer
  - $N$: Number of kernels
  - $C$: Number of channels of the input image
  
$$W_{c}=K^{2}\times C\times N \\ B_{c}=N \\P_{c}=W_{c}+B_{c}$$
  
- Conv layer에서 모든 커널의 깊이는 항상 입력 이미지의 채널 수와 같음
- 따라서 모든 커널에는 $K^{2}\times C$개의 parameter들이 있으며, 그러한 커널들이 $N$개 존재

### Example on AlexNet
- AlexNet의 Conv-1에 대해
  - 입력 이미지의 채널 수 $C=3$
  - Kernel size $K=11$
  - 전체 커널 개수 $N=96$
- 따라서 파라미터의 갯수는 아래와 같이 정의됨

$$W_{c}=11^{2}\times 3\times 96=34,848 \\ B_{c}=96 \\P_{c}=94,848+96=34,944$$

- Conv-2/3/4/5도 동일한 방법으로 각각 614,656/885,120/1,327,488/884,992개의 parameter를 갖는것을 계산 가능
- AlexNet conv layer의 parameter 개수는 3,747,200개
- FC layer의 parameter 수가 더해지지 않았으므로 전체 네트워크의 parameter 개수가 아님
- Conv layer의 장점은 weight parameter가 공유되므로 FC layer에 비해 매개변수가 훨씬 작다는 장점이 있음

## MaxPool layer의 parameter 갯수
- Pooling, stride, padding은 hyper parameter임(계산 X)

## Fully Connnected layer의 parameter 갯수
- CNN에는 두 종류의 FC layer가 존재
  - 마지막 Conv layer의 바로 뒤에 붙는 FC layer
  - 다른 FC layer에 연결되는 FC layer

### Case1: FC layer connected to a Conv layer
- 각각의 기호를 아래와 같이 정의
  - $W_{cf}$: Number of weights of a FC layer which is connected to a Conv layer
  - $B_{cf}$: Number of biases of a FC layer which is connected to a Conv layer
  - $P_{cf}$: Number of parameters of a FC layer which is connected to a Conv layer
  - $O$: Size(width) of th output image of the previous Conv layer
  - $N$: Number of kernels in the previous Conv layer
  - $F$: Number of neurons in the FC Layer
  
  $$W_{cf}=O^{2}\times N\times F \\B_{cf}=F \\P_{cf}=W_{cf}+B_{cf}$$

### Example on AlexNet
- Conv layer의 마지막단에 바로 붙는 FC-1 layer에 대해, $O=6$, $N=256$, $F=4096$임

$$W_{cf}=6^{2}\times 256\times 4096=37,748,736 \\B_{cf}=4096 \\P_{cf}=W_{cf}+B_{cf}=37,752,832$$

- 이 수는 모든 Conv layer의 pameter 갯수들보다 많은 수(그만큼 FC layer에는 많은 파라미터들이 필요)

### Case2: FC layer connected to a FC Layer
- 각각의 기호를 아래와 같이 정의
  - $W_{ff}$: Number of weights of a FC layer which is connected to a FC layer
  - $B_{ff}$: Number of biases of a FC layer which is connected to a FC layer
  - $P_{ff}$: Number of parameters of a FC layer which is connected to a FC layer
  - $F$: Number of neurons in th FC layer
  - $F_{-1}$: Number of neurons in the previous FC layer
  
$$W_{ff}=F_{-1}\times F \\ B_{ff}=F \\ P_{ff}=W_{ff}+B_{ff}$$
  
- 위의 식에서, $F_{-1}\times F$는 이전 FC layer의 neuron과 현재 FC layer의 neuron 사이의 총 연결 가중치의 개수.
- Bias parameter의 개수는 뉴런의 개수($F$)와 같음

### Example on AlexNet
- 마지막 FC layer인 FC-3은 $F_{-1}=4096,\; F=1000$이므로

$$W_{ff}=4096\times 1000=4,096,000 \\ B_{ff}=1,000 \\ P_{ff}=W_{ff}+B_{ff}=4,097,000$$

- FC-2 layer의 parameter 개수도 동일한 방법으로 16,781,312개로 계산 됨

# AlexNet의 총 parameter 개수 및 tensor size
- AlexNet의 전체 parameter 수는 5개의 convolution layer와 3개의 FC layer에서 계산되는 parameter 개수들의 합
  - __62,378,344__ 개.
- 자세한 parameter 및 tensor size는 

<center>
<figure>
<img src="/assets/post_img/study/2019-01-25-num_of_parameters/fig2.png" alt="views">
</figure>
</center>


---
- [참고 글]

https://www.learnopencv.com/number-of-parameters-and-tensor-sizes-in-convolutional-neural-network/
