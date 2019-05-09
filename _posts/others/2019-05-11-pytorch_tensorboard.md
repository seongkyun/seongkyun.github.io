---
layout: post
title: Pytorch에서 tensorboard로 loss plot하기
category: others
tags: [Pytorch, tensorboard, tensorflow]
comments: true
---

# Pytorch에서 tensorboard로 loss plot하기
- 참고 글: https://pythonkim.tistory.com/39

<center>
<figure>
<img src="/assets/post_img/others/2019-05-11-pytorch_tensorboard/fig1.png" alt="views">
<figcaption></figcaption>
</figure>
</center>

- Tensorboard는 tensorflow에서 제공하는 툴로, log를 그래프로 시각화하여 보여주는 도구다.

## 설치
- Pytorch에서 tensorboard로 loss plot을 하기 위해서는 `tensorboardX`가 필수로 설치되어 있어야 한다.
  - 설치: `pip install tensorboardX`
  - tensorboardX를 사용하기 위해선 tensorboard가 필요하며, tensorboard는 tensorflow가 필요하다.
  - tensorflow를 설치하면 알맞는 버전의 tensorboard가 자동으로 설치된다.
    - 설치: `pip install tensorflow-gpu==version`
    - version에는 자신의 CUDA와 cuDNN 버전에 알맞는 버전의 tensorflow version을 넣으면 됨(본인은 `tensorflow-gpu==1.12.0`)

## 코드 작업
- Pytorch 코드 내에서 별도의 작업이 필요하다.
- 학습 코드 상단부에 summaryWriter를 정의해준다.

```python
from tensorboardX import SummaryWriter
summary = SummaryWriter()
...
```

- 모델 학습과정에서 tensorboard에 plot 할 값들에 대한 x축으로 사용할 값(변수)이 필요하다.
  - 전체에 대한 iteration이 담기면 되므로, 본인은 total iteration이 담기는 `iteration` 변수를 설정하였음
  - 전체 학습 loop에 대해 tensorboard에서 x축으로 사용할 변수(`iteration`)는 epoch이 변해도 초기화되지않고 이어서 커져야 함
- 변수 설정 후, 학습 loop에서 매 iteration마다 값을 update하면 실효성도 떨어지고 너무 데이터가 많아진다.
  - 따라서 적절하게 (10이나 20iteration마다) 값을 update한다.
- 또한, pytorch tensor에 담긴 loss값은 값의 형태로 전달되도록 `tensor_name.item()` 멤버를 이용하여 리턴받는다.

```python
from tensorboardX import SummaryWriter
summary = SummaryWriter()

...

for iteration in range(start_iter, max_iter):
  
  ...

  out = net(input)
  loss_a, loss_b = criterion(out)
  loss = loss_a, loss_b
  loss.backward()
  optimizer.step()
  if iteration % 10 == 0: # 매 10 iteration마다
    summary.add_scalar('loss/loss_a', loss_a.item(), iteration)
    summary.add_scalar('loss/loss_b', loss_b.item(), iteration)
    summary.add_scalar('learning_rate', lr, iteration)
    summary.add_scalar('loss/loss', {"loss_a": loss_a.item(),
                                    "loss_b": loss_b.item(),
                                    "loss": loss.item()}, iteration)

...

```

## Tensorboard 실행

