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
  
비선형 변수 변환이 적용된 일반화 선형 모형은 입력값이 희소한 대규모 회귀 분석 및 분류 문제에 널리 사용된다. *광범위한* 교차곱 변수 변환을 통한 변수의 교호 작용 기억은 효과적이고 해석하기 쉽지만 일반화를 위해선 피쳐 엔지니어링에 더 많은 노력이 필요하다. 적은 피쳐 엔지니어링으로 *심층* 신경망은 희소한 변수에 대해 학습된 저차원 임베딩으로 눈에 보이지 않는 변수 조합을 보다 잘 일반화할 수 있다. 그러나 임베딩을 통한 심층 신경망은 사용자 - 항목 상호 작용이 희소하고 상위에 있을 때 덜 관련성이있는 항목을 지나치게 일반화하고 추천 할 수 있습니다.
