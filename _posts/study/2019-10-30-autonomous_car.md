---
layout: post
title: 자율주행 자동차에 대해서
category: study
tags: [자율주행, Autonomous driving]
comments: true
---

# 자율주행 자동차에 대해서
- 자율주행 자동차란?
  - 자동차관리법 제 2조 제 1호의 3항에 따라 "승객의 조작 없이 자동차 스스로 운행이 가능한 자동차"를 의미함
  - 즉, 운전자의 핸들, 가속페달, 브레이크 조절이 필요없이 정밀지도와 각종 센서를 이용해 상황을 파악하고 스스로 목적지까지 움직여주는 자동차

### 자율주행의 필요성
- 인적요인으로 발생하는 교통사고(90% 이상)를 방지
  - 음주, 과속, 피로누적 등등..
  - 전 세계적으로 매년 125만명의 사람이 운전자 과실로 인한 교통사고로 사망
  - 자율주행을 적용해 이러한 인적요인을 제거
- 안전한 차간거리 유지, 적절한 속도 관리로 정체를 발생시킬 수 있는 요소를 차단해 교통의 효율화 가능
- 고령 운전자 및 장애인을 위해 필요
- 운전의 효율화
- 정리하자면
  - 안전성: 인적요인에 의한 사고 감소
  - 효율성 및 친환경성: 교통체증 완화, 운전의 효율화를 통한 환경오염 감소
  - 편의성: 사용자의 편의성 향상
  - 사회적 수용성: 노약자의 이동성 향상
  - 접근성: 도시 주요부로의 접근성 향상

## 자율주행 기술
### 자율주행 단계
- 자동차기술자협회(SAE)의 5단계와 미국도로교통안전국(NHTSA)의 4단계가 존재
- SAE 기준
  - 0단계: 자동제어 장치 없이 일반적으로 사람이 운전
  - 1단계: Foot off. 자동긴급제동장치(AEB), 어뎁티브 크루즈 컨트롤(ACC), 핸들 조향 지원 및 자동보조시스템 등의 적용
  - 2단계: Hands off. 핸들 방향 조종 및 가감속 등 하나 이상의 첨단운전자보조시스템(ADAS) 적용
  - 3단계: Eye off. 자율주행으로 정의되는 단계. 가속, 주행, 제동이 모두 자동으로 수행되나 특정 상황시 운전자의 개입이 필요한 조건부 자동화 단계
  - 4단계: Head off. 시내주행을 포함한 도로 환경에서 운전자의 개입이나 모니터링이 필요 없는 자율주행. 고도의 자동화 단계
  - 5단계: Driver off. 완벽한 자율주행 수준. 모든 환경하에 운전자의 개입 없이 자율주행이 가능한 완전 자동화 단계

### 자율주행차의 주요 요소
- 주요 기술로는 __환경 인식, 위치인식 및 맵핑, 판단, 제어, HCI__ 로 나뉨
  - 환경 인식: 레이더, 라이다, 카메라, 초음파 센서 등의 각종 센서로 정적 장애물(가로등, 건물, 전봇대 등)과 동적 장애물(차량, 보행자 등), 도로 표식(차선, 신호, 횡단보도 등)을 인식하고, V2X통신을 통해 도로 인프라 및 주변 차량과 주행정보를 교환
  - 위치인식 및 맵핑: GPS, INS, Encoder, 기타 맵핑을 위한 센서를 사용. 자동차의 절대/상대 위치를 계산
  - 판단: 목적지 이동, 장애물 회피 경로 계획, 주행상황별(신호등 처리, 차선유지 및 변경, 회전, 추월, 유턴, 급정지, 끼어들기, 주정차 등) 행동을 스스로 판단
  - 제어: 운전자가 지정한 경로대로 주행하기 위해 차량의 조향, 속도 변경, 기어 등의 엑츄에이터 제어
  - HCI: Human Computer Interaction. 주로 HMI(Human Machine Interaction)으로 정의됨. 운전자에게 경고 또는 정보를 제공하고 운전자의 명령을 입력받음
- 자율주행의 메커니즘
  - 주행 주변 환경에 대한 각종 정보를 센서들을 이용해 수집하는 단계
  - 센서를 통해 수집된 정보들을 처리하고 차량을 통제하는 알고리즘 단계
  - 알고리즘을 실시간으로 처리하는 전산처리 단계
- 시스템적 요소는 인지, 판단, 제어로 나뉨
  - 인지: 자동차의 센서들이 차량 주변에 대한 데이터를 수집
    - 경로탐색: 고정밀 디지털 지도
    - 센서: 카메라, 레이더, 라이다 등
    - V2X: V2V, V2I 등의 통신
  - 판단: 수집된 데이터를 소프트웨어 알고리즘으로 분석/해석
    - 경로탐색: 차량의 절대적, 상대적 좌표, 3D 지도정보
    - 센서: 차량 주변의 장애물, 도로 표식, 신호
    - V2X:도로의 인프라 및 주변차량 정보
    - 위의 정보들을 취합해 목적지까지의 주행 경로 설정, 돌발상황에 대한 판단과 주행 전략 등을 결정
  - 제어: 앞 단계에서 취합된 정보들을 바탕으로 운행 계획에 따라 자동차의 동작을 제어
    - 차량 제어
    - 조향 조절, 엔진 가/감속
    - 운전자에게 경고 및 정보 제공
- 자율주행차 주요 기술 구성
  - 자율주행 차량의 센싱
    - 레이더, 라이다, 레이저 센서
    - HD급의 vision camera
    - 적외선/자외선 센서
    - 환경 센서
  - 자율주행 소프트웨어  
    - 주행상황 인지 소프트웨어
    - 센싱 융합처리 드라이빙 컴퓨팅
    - 인공지능(AI)기반 컴퓨팅
    - 클라우드 기반 컴퓨팅
  - 통신보안
    - 차량 무선통신 시스템
    - 협력주행 네트워킹
    - 이동통신망 연동 네트워킹
    - V2V, V2I(infra) 정보보호 및 보안
    - 통신 이상징후 탐지
  - 안전운전
    - 교통약자 안전운전 지원 HMI
    - 운전자 케어 UI/UX
    - 다중 차량 무선충전
    - 인포테인먼트
- 자율주행 소프트웨어 구조
  - Cloud map, GPS, INS/IMU, Vehicle State, Lidar 등을 이용한 Localization (Map matching)
  - Lidar, Camera 등의 센서에서 얻어진 정보들을 이용해 Object detecion, tracking, Lane detection 등을 수행
  - 위의 결과들과 RNDF, Traffic light information을 토대로 Navigation 
    - Global/local path planning, Path following, Speed planning
  - Navigation 결과를 VCU(Vehicle Control Unit)로 전달
    - Steering, Brake, Acceleration

### 자율주행 주요 기술
- 컴퓨터 플랫폼: Nvidia Drive PX2, Xavier, Mobileye EyeQ, TI TDA 3X 등의 인공지능 플랫폼
- V2X 통신: C-ITS, 5G, WAVE(도로 인프라 및 주변차량 정보 수집/취합)
- 보안: 전장 플랫폼 보안, 내/외부 네트워크 보안, KMS(암복호화 키 관리 시스템), AFW(어플리케이션 방화벽)
- AI: 강화학습, 딥러닝 학습
  - 인지: 차선, 보행자, 차량 검출등의 Object detection
  - 판단: 의도 판단, 경로 예측
  - 제어: 사용자 개인화 기반 최적 제어

### 자율주행 차량의 센서
- 주로 인지를 위해 사용
- 핵심센서는 다음과 같음
  - 카메라: 대상 물체에 대한 형태 정보를 제공하며 차선, 표지판, 신호등, 차량, 보행자 등을 탐지하기 위한 정보를 센싱
  - 라이다: 레이저를 이용해 현재위치부터 목표물까지의 거리, 방향, 속도, 온도 등의 특성을 감지하며 전파에 가까운 성질을 가진 레이터 광선을 이용하여 활용가능범위가 매우 넓음
  - 레이다: 전자기파를 발사해 반사 신호 분석을 통해 거리, 높이, 방향, 속도 등 주변정보를 습득
- 센서들은 외적인 요인들로 인해 클리닝 문제, 내구성 문제 등이 발생 할 수 있기 때문에 해당 문제 또한 중요함
  - 정상적인 자율주행을 위해선 최적 상태의 센서가 유지되어야 함

<center>
<figure>
<img src="/assets/post_img/study/2019-10-30-autonomous_car/fig1.jpg" alt="views">
<figcaption>현대자동차 넥쏘에 적용 센서들</figcaption>
</figure>
</center>

### 카메라