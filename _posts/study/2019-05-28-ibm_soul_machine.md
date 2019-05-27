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
  - https://www.scmp.com/lifestyle/gadgets/article/2188050/5g-networking-makes-artificial-humans-and-real-time-holograms
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

## Soul Machine Timeline
- 2017년 __호주 정부의 장애인 서비스 국가표준인 NDIS__ 에 소울머신과 IBM Watson 기반의 __Nadia__ 인공지능 아바타 제작 및 적용
  - https://www.cmo.com.au/article/648607/going-beyond-chatbots-avatars-next-stage-ai-cx/
- 2017년 __Air Newzealand__ 에 소울머신과 IBM Watson 기반의 __Sophie__ 인공지능 아바타 제작 및 적용하여 항공편 예약 등의 서비스를 제공
  - https://www.soulmachines.com/news/2017/09/26/soul-machines-latest-project-with-air-new-zealand-shows-the-potential-of-digital-humans-in-customer-service/
- 2017년 __Autodesk__ 와 제휴하여 소울머신과 IBM Watson 기반의 __Ava__ 인공지능 아바타를 제작 및 적용하여 24/7 서비스 질문 및 답변 적용
  - https://www.soulmachines.com/news/2017/11/15/hot-off-the-press-soul-machines-partners-with-autodesk-to-launch-ava-at-autodesk-university-2017-2/
- 2018년 __Royal Bank of Scotland__ 에 소울머신과 IBM Watson 기반의 __Cora__ 인공지능 아바타를 제작 및 적용
  - https://www.soulmachines.com/news/2018/02/26/press-cora-is-causing-quite-a-stir-across-global-media/
- 2018년 __NatWest bank UK__ 에 소울머신과 IBM Watson 기반의 __Cora__ 인공지능 아바타 적용
  - https://www.soulmachines.com/news/2018/02/23/press-natwest-begins-testing-ai-driven-digital-human-in-banking-first/
- 2018년 __Daimler Financial Services__ 에 소울머신과 IBM Watson 기반의 __Sarah__ 인공지능 아바타를 제작 및 적용
  - https://www.soulmachines.com/news/2018/02/26/press-new-partnership-with-daimler-financial-services-2/
- 2018년 __IBM Watson IoT platform__ 에 소울머신과 IBM Watson 기반의 __INES__ 인공지능 아바타를 제작 및 적용
  - https://www.soulmachines.com/news/2018/04/26/press-a-new-security-digital-human-who-can-answer-cybersecurity-questions/
- 2018년 __ANZ (Australia and New Zealand Banking Group)__ 에 소울머신과 IBM Watson 기반의 __Jamie__ 인공지능 아바타를 제작 및 적용
  - https://www.soulmachines.com/news/2018/07/09/hot-off-the-press-introducing-jamie-anzs-new-digital-assistant/
- 2018년 __가상 인공지능 아기 Baby X__ 제안
  - https://www.soulmachines.com/news/2018/08/29/press-nigel-latta-launches-a-curious-mind-series-featuring-babyx/
- 2018년 __오클랜드 에너지 회사 Vector__ 에 소울머신과 IBM Watson 기반의 __Will__ 인공지능 교육용 아바타를 제작하여 어린아이들에게 학습용 자료 배포
  - https://www.soulmachines.com/news/2018/09/18/press-worlds-first-digital-teacher-starts-work-teaching-kids-about-energy/
- 2019년 __Bank ABC (Arab Banking Corporation)__ 에 소울머신과 IBM Watson 기반의 __Fatema__ 인공지능 아바타를 제작 및 적용
  - https://www.soulmachines.com/news/2019/02/25/press-virtual-banks-to-dominate-future-banking-sector/
- 2019년 미국 Verizon 통신사는 소울머신의 인공지능 아바타와 자사 기술로 인공지능 아바타 'Lia'를 개발하고, 이에 5G를 접목시켜 가상현실(VR) 서비스를 제공할 예정
  - https://www.scmp.com/lifestyle/gadgets/article/2188050/5g-networking-makes-artificial-humans-and-real-time-holograms
  - https://www.soulmachines.com/news/2019/03/12/press-5g-will-help-bring-digital-humans-to-life/
