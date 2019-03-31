---
layout: post
title: Face detection model 성능 비교(WIDERFace)
category: study
tags: [Face detection]
comments: true
---

# Face detection model 성능 비교(WIDERFace)
- 원글: https://medium.com/nodeflux/performance-showdown-of-publicly-available-face-detection-model-7c725747094a
- Github: https://github.com/nodefluxio/face-detector-benchmark
- Detection 방법 설명 1: https://medium.com/nodeflux/the-evolution-of-computer-vision-techniques-on-face-detection-part-1-7fb5896aaac0
- Detection 방법 설명 2: https://medium.com/nodeflux/the-evolution-of-computer-vision-techniques-on-face-detection-part-2-4af3b22df7c2

---
- 딥러닝을 사용하지 않는 모델과 사용하는 모델의 성능 차이의 비교
  - 딥러닝 모델은 정확하나 computation cost가 크므로 필요에 따라 올바른 모델을 선택해야 함
- Detection 방법 설명 1과 2글에서 다루는 방법들에 대한 성능 비교
  - [OpenCV Haar Cascades Clasifier](https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html)
  - [DLib Histogram of Oriented Gradients (HOG)](http://dlib.net/face_detector.py.html)
  - [DLib Convolutional Neural Network (CNN)](http://dlib.net/cnn_face_detector.py.html)
  - [Multi-task Cascaded CNN (MTCNN) — Tensorflow](https://github.com/kpzhang93/MTCNN_face_detection_alignment)
  - [Mobilenet-SSD Face Detector — Tensorflow](https://github.com/yeephycho/tensorflow-face-detection)

- 위의 모델들의 WIDER Face dataset에 대한 정확도/속도의 비교

<center>
<figure>
<img src="/assets/post_img/study/2019-03-25-face_detection/fig1.jpeg" alt="views">
<figcaption>WIDER Face dataset variations</figcaption>
</figure>
</center>

## Performance Metrics
- 각각 face detection 모델에 대한 성능을 측정하며, 성능은 accuracy와 complexity를 측정

### Accuracy
- Object detection과 마찬가지로 average IoU를 측정함(mean Average Precision, mAP)
  - Jaccard overlap으로 정의되는 겹쳐지는 부분에 대한 비율을 측정하여 True Positive 여부를 결정
  - 그 값이 1에 가까울수록 모델이 객체의 위치를 정확하게 추론한 것임
  - 계산되는 IoU의 평균값을 계산함
  
<center>
<figure>
<img src="/assets/post_img/study/2019-03-25-face_detection/fig2.png" alt="views">
<figcaption>IoU formula</figcaption>
</figure>
</center>

- __Mean averaged precision (mAP)__ 는 object detector 모델이 얼마나 정확하게 해당 class를 갖는 객체의 위치를 검출하였는가를 측정함. Face detection의 경우 테스트셋에 대하여 face의 위치로 정의된 좌표(Ground Truth, GT)에 얼마나 올바르게 모델이 추론결과 박스를 그렸는지를 측정한다. 일반적인 mAP의 계산은 아래와 같다.

$$Precision=\frac{True\; Positive}{True\; Positive+False\; Positive}$$

- True Positive는 모델에 의해 예측된 위치가 face의 위치를 정확하게 예측한 경우의 횟수를 의미한다. False Positive는 모델에 의해 예측된 위치가 잘못 예측된 경우의 횟수를 의미한다.
- Object detection의 경우, Jaccard overlap이 일정한 threshold 값을 넘었을 때 올바르게 예측된것으로 간주하며, 보통은 0.5, 0.75, 0.95등으로 다양한 기준으로 True Positive 여부를 판단하지만, 보통은 0.5 기준으로 하며 이를 mAP@0.5로 표현한다.

### Complexity
- 정확하게 추론하는 모델의 경우 complexity가 높아 많은 computation cost를 요구한다. 보통은 complexity가 높을수록 processing time(inference time)이 많이 걸리므로 실시간성이 떨어질 수 있다.
- 이 글에선 model의 complexity를 CPU, GPU, RAM resource의 usage를 기준으로 판단한다. 또한단일 1080p 이미지가 입력으로 들어갔을 때의 inference time을 측정한다. Inference time은 이미지가 입력되고 최종 출력물이 출력될 때 까지의 시간을 기준으로 한다.

## Benchmarked Dataset
- 사용한 데이터셋은 WIDER Face Dataset이며, 32,203개의 이미지에 393,703개의 얼굴 레이블이 존재한다. 하지만 해당 데이터셋은 일반적인 얼굴 외에도 다양한 pose나 scale, occlusion을 갖는다. 따라서 일반적인 detection model에 대한 합리적인 평가가 가능하다.
- 하지만 dataset에서 test를 제외한 train과 validation 데이터셋만 이용하였으모, 모호한(invalid) GT에 대해선 제외처리하였다. 또한 15\*15 픽셀 미만의 크기를 갖는 얼굴에 대해서도 유의미한 정보가 아니라 판단하여 제외시켰다. 따라서 총 16,106개의 이미지에 대해 98,871개의 얼굴정보를 갖고 실험을 진행했다.

## Experiment and Result

- [Github 주소](https://github.com/nodefluxio/face-detector-benchmark/blob/master/benchmark-result.txt) 에서 직접 실험이 가능하다. 실험에선 아래의 5가지 알고리즘에 대한 실험을 진행했다.
  - Model 1: [OpenCV Haar Cascades Clasifier](https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html)
  - Model 2: [DLib Histogram of Oriented Gradients (HOG)](http://dlib.net/face_detector.py.html)
  - Model 3: [DLib Convolutional Neural Network (CNN)](http://dlib.net/cnn_face_detector.py.html)
  - Model 4: [Multi-task Cascaded CNN (MTCNN) — Tensorflow](https://github.com/kpzhang93/MTCNN_face_detection_alignment)
  - Model 5: [Mobilenet-SSD Face Detector — Tensorflow](https://github.com/yeephycho/tensorflow-face-detection)
- Benchmark에 사용된 컴퓨터 사양은 아래와 같다.
  - CPU: Intel Core i7-7700HQ (quadcore)
  - GPU: Nvidia Geforce GTX1060
  - RAM: 16GB
- 전체적인 성능은 mAP와 inference time/prediction processing time으로 측정함.

- __Model 1, 2는 CPU, Model 3, 4, 5는 GPU로 돌린 실험 결과 (Inference time)__

<center>
<figure>
<img src="/assets/post_img/study/2019-03-25-face_detection/fig3.jpeg" alt="views">
<figcaption></figcaption>
</figure>
</center>

- __모든 모델을 CPU에서 돌린 결과 (Inference time)__

<center>
<figure>
<img src="/assets/post_img/study/2019-03-25-face_detection/fig4.jpeg" alt="views">
<figcaption></figcaption>
</figure>
</center>

- __Model 1, 2는 CPU, Model 3, 4, 5는 GPU로 돌린 실험 결과 (Resource usage)__

<center>
<figure>
<img src="/assets/post_img/study/2019-03-25-face_detection/fig5.jpeg" alt="views">
<figcaption></figcaption>
</figure>
</center>

- __모든 모델을 CPU에서 돌린 결과 (Resource usage)__

<center>
<figure>
<img src="/assets/post_img/study/2019-03-25-face_detection/fig6.jpeg" alt="views">
<figcaption></figcaption>
</figure>
</center>

- __전체 모델에 대한 수치적 실험 결과__
  - IoU threshold: 0.5
- OpenCV Haar Cascade Face Detector
  - Average IOU = 0.219
  - mAP = 0.307
  - Inferencing time (On CPU) : 0.159 s
  - Resource Usage (On CPU)
    - Memory Usage : 414.453 MiB
    - CPU Utilization : 680-730%
- DLib HOG Face Detector
  - Average IOU = 0.253
  - mAP = 0.365
  - Inferencing time (On CPU) : 0.239 s
  - Resource Usage (On CPU):
    - Memory Usage : 270.777 MiB
    - CPU Utilization : 99-100%
- DLib CNN MMOD Face Detector
  - Average IOU = 0.286
  - mAP = 0.416
  - Inferencing time (On GPU) : 0.111 s
  - Inferencing time (On CPU) : 4.534 s
  - Resource Usage (On GPU):
    - Memory Usage : 1171.367 MiB
    - GPU Memory Usage : 1037 MiB
    - GPU Core Utilization : 75-90%
    - CPU Utilization : 99-100%
  - Resource Usage (On CPU):
    - Memory Usage : 588.898 MiB
    - CPU Utilization : 250-450%
- Tensorflow MTCNN Face Detector
  - Average IOU = 0.417
  - mAP = 0.517 
  - Inferencing time (On GPU) : 0.699 s
  - Inferencing time (On CPU) : 1.979 s
  - Resource Usage (On GPU):
    - Memory Usage : 2074.180 MiB
    - GPU Memory Usage : 5004 MiB
    - GPU Core Utilization : 10-40%
    - CPU Utilization : 111-120%
  - Resource Usage (On CPU):
    - Memory Usage : 790.129 MiB
    - CPU Utilization : 500-600%
- Tensorflow Mobilenet SSD Face Detector
  - Average IOU = 0.598
  - mAP = 0.751
  - Inferencing time (On GPU) : 0.0238 s
  - Inferencing time (On CPU) : 0.1650 s
  - Resource Usage (On GPU):
    - Memory Usage : 1967.676 MiB
    - GPU Memory Usage : 502 MiB
    - GPU Core Utilization : 47-58%
    - CPU Utilization : 140-150%
  - Resource Usage (On CPU):
    - Memory Usage : 536.270 MiB
    - CPU Utilization : 670-700%
