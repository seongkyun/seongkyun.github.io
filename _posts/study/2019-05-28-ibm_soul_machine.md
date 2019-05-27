---
layout: post
title: IBM Watson Soul Machine
category: study
tags: [IBM Watson, Soul Machine, Background Survey]
comments: true
---

# [Background Survey] IBM Watson Soul Machine

## IBM Watson
- 참고 글
  - http://www.seminartoday.net/news/articleView.html?idxno=10131
  - https://www.ibm.com/watson/kr-ko/products.html
- __IBM Watson API:__ IBM에서 제공하는 다양한 AI API에 기본으로 탑재되는 인공지능. Watson API는 IBM Watson의 Cognitive 기술을 담아 애플리케이션이나 솔루션을 개발 할 수 있도록 모듈화 한 서비스. IBM 클라우드 플랫폼인 '블루믹스' 상에서 이용 가능하며, 제공하는 IBM Watson의 API는 아래와 같음.
  - AI Assistant: 다양한 대화 기술을 애플리케이션에 통합
    - Watson Assistant: 간편한 도구 및 대화 트리로 챗봇 개발 (한국어 지원)
  - Knowledge: 가속화된 데이터 최적화 기능을 통해 통찰력 확보
    - Discovery: 데이터에서 필요한 정보를 찾아서 정답을 추론하고 trand(추세)를 모니터링하여 드러나지 않는 패턴을 찾음 (한국어 지원)
    - Discovery News: 사전에 보강된 뉴스 콘텐츠에 실시간으로 엑세스 (한국어 지원)
    - Natural Language Understanding: 고급 텍스트 분석을 위한 자연어 처리 (한국어 지원)
    - Knowledge Studio: 비정형 텍스트에서 의미 있는 정보를 찾고 통찰력을 갖도록 Watson을 학습시킴 (한국어 지원)
  - Vision: 콘텐츠를 식별하고 태깅한 후, 이미지에서 발견된 세부 정보를 분석하고 추론
    - Visual Recognition: 머신 러닝 기술을 사용하여 시각적 콘텐츠를 태깅하구 분류 (한국어 지원)
  - Speech: 텍스트와 음성을 변환할 수 있으며, 음성모델을 사용자가 지정할 수 있음
    - Language Translator: 텍스트를 한 언어에서 다른 언어로 번역 (한국어 지원)
    - Natural Language Classifier: 자연어를 해석하고 분류 (한국어 지원)
  - Enpathy: 어조, 성격 및 감정 상태를 이해
    - Personality Insights: 텍스트를 통해 성격 특성을 예측 (한국어 지원)
    - Tone Analyzer: 텍스트에 나타난 감정과 커뮤니케이션 스타일을 이해
- 위 기능들 중 필요한 기능들간의 조합이 가능하며, 다양한 데모 및 예제들이 제공됨

## Soul Machine
- 참고 글
  - https://m.blog.naver.com/PostView.nhn?blogId=ibm_korea&logNo=221330091341&proxyReferer=https%3A%2F%2Fwww.google.com%2F
  - https://www.ibm.com/case-studies/soul-machines-hybrid-cloud-ai-chatbot
- __소울 머신__ 은 뉴질랜드에 본사를 둔 회사로, IBM Watson의 Watson Assistant 기반의 챗봇에 사람의 얼굴 표정과 목소리로 사용자의 감정까지 고려하는 인공지능 아바타를 제작.
- 사용자의 표정이나 목소리로 분석된 감정을 판단하여 슬픈 얼굴 표정이나 목소리로 위로를 전하고 답변하는 등의 기능 수행.
- 소울머신은 IBM Watson 기술을 이용해 감정적으로 고객의 요구 및 질문에 반응하는 인공 인물을 제작하는 서비스를 제공
- 소울머신 적용시 40%이상의 human factor 감소, 지속적인 학습으로 정확도의 점차적 향상이 가능
- IBM Watson API가 적용된 플랫폼에 대해 맞춤형 아바타 제작 및 적용시 8-12주의 시간 소모
- 클라우딩 서비스 기반의 IBM Watson Assistant와 소울머신 플랫폼을 통합하여 구현되며, 아래와 같이 동작
  - 사용자가 소울머신의 가상 인공 인물에게 질문
  - 소울머신 플랫폼에서 입력된 고객의 음성 오디오 스트림을 Watson Assistant API로 전송(Cloud)
  - Watson이 __전송된 오디오를 텍스트로 변환__
  - Watson이 해당 질문에 연관된 답변을 __사전에 정해진 규칙을 따르도록 알맞은 답변을 생성__
  - 생성된 답변을 소울머신 플랫폼에 전달
  - Watson이 답변을 생성하는 동안 __소울머신 플랫폼은 고객의 목소리 톤과 미세한 얼굴 표정 등 감지된 시청각 자료를 분석__
  - 최종적으로 Watson이 만든 답변과 소울머신의 시청각 자료 분석 결과를 토대로 사용자의 질문에 대한 알맞은 목소리 톤과 얼굴 표정으로 결과 출력(대답)
- 2017년 호주 정부의 장애은 서비스 국가표준인 NDIS에 적용하기 위한 Nadia를 개발하여 적용

<center>
<figure>
<img src="/assets/post_img/study/2019-05-28-ibm_soul_machine/fig1.png" alt="views">
<figcaption></figcaption>
</figure>
</center>
