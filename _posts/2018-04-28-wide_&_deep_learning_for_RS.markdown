---
layout:     post
title:      Wide & Deep Learning for Recommender Systems
date:       2018-04-28 00:00:00
author:     Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, Hemal Shah
categories: Data-Science
---  
  
  
**Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra,Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil,Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, Hemal Shah의 [*Wide & Deep Learning for Recommender Systems
*](https://arxiv.org/pdf/1606.07792v1.pdf)을 번역했습니다.**
  
  
- - -
  
## 초록

비선형 피처 변환이 적용된 일반 선형 모델 대규모 회귀 분석 및 분류에 널리 사용됩니다. 희소 한 입력에 대한 문제. 지형지 물의 암기 다양한 제품 간 기능을 통한 상호 작용 변환은 효과적이고 해석 가능하며, 일반화 더 많은 기능 엔지니어링 노력이 필요합니다. 적은 돈으로 기능 공학, 심 신경 네트워크가 더 잘 일반화 될 수 있습니다. 저 차원의 특징 조합을 보이지 않게하는 방법 스파 스 기능에 대해 학습 된 밀집된 임베딩. 하나, 내장 된 심 신경 네트워크는 지나치게 일반화 될 수있다. 사용자 - 항목 상호 작용이있을 때 덜 관련성이 높은 항목을 추천합니다. 드문 드문하고 높은 순위입니다. 이 논문에서는 넓고 깊은 학습 - 공동으로 훈련 된 넓은 선형 모델 그리고 깊은 신경 네트워크 - 암기의 이점을 결합 추천 시스템을위한 일반화. 우리 Google Play에서 시스템을 제작 및 평가했으며, 10 억 이상의 활성 모바일 앱 스토어 사용자 및 백만 이상의 앱 온라인 실험 결과 Wide & Deep이 앱 구매를 크게 늘린 것으로 나타났습니다. 와이드 전용 및 딥 전용 모델과 비교할 때 우리 TensorFlow에서 우리의 구현을 오픈 소스로 제공했습니다.
