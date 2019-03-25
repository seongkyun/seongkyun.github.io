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
<img src="/assets/post_img/study/2019-03-25-faec_detection/fig1.jpeg" alt="views">
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
<img src="/assets/post_img/study/2019-03-25-faec_detection/fig2.png" alt="views">
<figcaption>IoU formula</figcaption>
</figure>
</center>

- __Mean averaged precision (mAP)__ 는 object detector 모델이 얼마나 정확하게 해당 class를 갖는 객체의 위치를 검출하였는가를 측정함. Face detection의 경우 테스트셋에 대하여 face의 위치로 정의된 좌표(Ground Truth, GT)에 얼마나 올바르게 모델이 추론결과 박스를 그렸는지를 측정한다. 일반적인 mAP의 계산은 아래와 같다.

$$Precision=\frac{True\; Positive}{True\; Positive+False\; Positive}$$


---

- [참고 글]

https://medium.com/nodeflux/performance-showdown-of-publicly-available-face-detection-model-7c725747094a
