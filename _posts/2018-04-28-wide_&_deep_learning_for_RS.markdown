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
  
비선형 변수 변환을 적용한 일반화 선형 모형은 입력값이 희소한 대규모 회귀 분석 및 분류 문제에 널리 사용된다. *광범위한* 교차곱 변수 변환을 통한 변수의 교호 작용 기억은 효과적이고 해석하기 쉽지만 일반화를 위해선 피쳐 엔지니어링에 더 많은 노력이 필요하다. 적은 피쳐 엔지니어링으로 *심층* 신경망은 희소한 변수에 대해 학습한 저차원 임베딩을 통해 눈에 보이지 않는 변수 조합을 일반화를 보다 잘 할 수 있다. 그러나 임베딩을 통한 심층 신경망은 사용자 - 항목 교호 작용이 희소하고 계수가 높을 때 지나치게 일반화하여 크게 관련없는 항목을 추천할 수 있다. 이 논문은 추천 시스템에 기억과 일반화 이점을 결합하기 위해 Wide & Deep 학습, 즉 공동으로 훈련한 광범위한 선형 모형 및 심층 신경망를 제시한다. Google은 10억 명이 넘는 활성 사용자와 100만 개가 넘는 앱을 보유한 상용 모바일 앱 스토어 Google Play에 시스템을 구축하고 평가했다. 온라인 실험 결과에 따르면 Wide & Deep은 광범위한 선형 모형만 사용한 것과 심층 신경망만 사용한 것 대비해서 앱 가입을 크게 증가시켰다. 또한 TensorFlow에 오픈소스로 구현해놨다.
  
## 1. 개론
