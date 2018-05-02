---
layout:     post
title:      Wide & Deep Learning for Recommender Systems
date:       2018-04-28 00:00:00
author:     Google Inc.
categories: Data-Science
---  
  
  
**Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra,Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil,Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, Hemal Shah의 [*Wide & Deep Learning for Recommender Systems*](https://arxiv.org/pdf/1606.07792v1.pdf)을 번역했습니다.**
  
  
- - -
  
## 초록
  
비선형 변수 변환을 적용한 일반화 선형 모형은 입력값이 희소한 대규모 회귀 분석 및 분류 문제에 널리 사용된다. *광범위한* 교차곱 변수 변환을 통한 변수의 교호 작용 기억은 효과적이고 해석하긴 쉽지만 일반화를 위해서 피쳐 엔지니어링에 더 많은 노력이 필요하다. 적은 피쳐 엔지니어링으로 *심층* 신경망은 희소한 변수에 대해 학습한 저차원 임베딩을 통해 눈에 보이지 않는 변수 조합에 대한 일반화를 보다 잘 할 수 있다. 그러나 임베딩을 통한 심층 신경망은 사용자 - 항목 교호 작용이 희소하고 계수가 높을 때 지나치게 일반화되어 크게 관련없는 항목을 추천할 수 있다. 이 논문은 추천 시스템에 기억과 일반화 이점을 결합하기 위해 Wide & Deep 학습, 즉 함께 훈련시킨 광범위한 선형 모형 및 심층 신경망를 제시한다. Google은 10억 명이 넘는 활성 사용자와 100만 개가 넘는 앱을 보유한 상용 모바일 앱 스토어 Google Play에 시스템을 구축하고 평가했다. 온라인 실험 결과에 따르면 Wide & Deep은 광범위한 선형 모형만 사용한 것과 심층 신경망만 사용한 것 대비해서 앱 가입을 크게 증가시켰다. 우린 또한 TensorFlow에 오픈소스를 구현해놨다.
  
## 1. 개론
  
추천 시스템은 사용자 및 맥락 정보의 집합이 입력 질의이고 품목마다 순위가 매겨진 목록이 출력인 검색 질의 시스템 일종으로 볼 수 있다. 추천 업무는 질의문이 주어졌을 때 데이터베이스에서 관련 품목을 찾고 클릭 또는 구매 같은 특정 목표에 기반하여 품목의 순위를 매기는 것이다.
  
추천 검색 시스템의 한 가지 난점은 일반적인 검색 순위 문제와 마찬가지로 *기억*과 *일반화*를 모두 이뤄내는 것이다. 기억은 동시에 빈발하는 품목 또는 특성을 학습하고 과거 내역에서 이용가능한 상관 관계를 뽑아내 대략적인 정의를 내린다. 한편, 일반화는 상관 관계의 이행성(transtivity)에 기반하고 결코 또는 거의 발생하지 않은 새로운 변수 조합을 탐구한다. 기억에 근거한 추천은 보통 사용자가 이미 행동을 취했던 품목과 직접적으로 관련되어 있다. 기억과 비교할 때, 일반화는 추천 품목의 다양성을 향상시키는 경향이 있다. 이 글에서는 Google Play 스토어 앱 추천 문제에 초점을 맞추지만 일반적인 추천 시스템에 적용해볼 수 있다.
  
기업 내 거대 규모의 온라인 추천 및 순위 시스템에선 로지스틱 회귀 같은 일반화된 선형 모형이 간단하고 확장 가능하며 해석하기 쉽기 때문에 널리 사용된다. 종종 one-hot 인코딩을 사용하여 이진화한 희소 변수에 대해 모형 훈련을 진행한다. 예를 들자면 이진값 변수 "user_installed_app = netflix"는 사용자가 Netflix를 설치한 경우 값 1을 가진다. 기억은 AND("user_installed_app = netflix, impression_app = pandora")와 같이 교차 기능 제품 변환을 사용하여 효과적으로 얻을 수 있습니다. 사용자가 Netflix를 설치 한 다음 나중에 Pandora로 표시하면 값은 1입니다. 이는 피처 쌍의 동시 발생이 대상 레이블과 어떻게 관련되는지 설명합니다. 일반화는 덜 세분화 된 기능을 사용하여 추가 할 수 있습니다 (예 : AND (user_installed_category = video, impression_category = music)하지만 수동 피쳐 엔지니어링이 종종 필요합니다. 제품 간 변환의 한 가지 한계는 교육 데이터에 나타나지 않은 쿼리 항목 기능 쌍을 일반화하지 않는다는 것입니다.
