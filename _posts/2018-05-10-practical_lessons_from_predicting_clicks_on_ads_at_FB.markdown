---
layout:     post
title:      Practical Lessons from Predicting Clicks on Ads at Facebook
date:       2018-05-10 00:00:00
author:     Facebook
categories: Data-Science
---  
  
  
**Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yanxin Shi, Antoine Atallah, Ralf Herbrich, Stuart Bowers, Joaquin Quiñonero Candela의 [*Practical Lessons from Predicting Clicks on Ads at Facebook*](http://quinonero.net/Publications/predicting-clicks-facebook.pdf)을 번역했습니다.**
  
  
- - -
  
## 초록
  
온라인 광고에서 광고주는 광고 클릭 같이 측정가능한 사용자 응답에 대해 입찰하고 대금을 지불한다. 그러므로 클릭 예측 시스템은 많은 온라인 광고 시스템의 핵심이다. 7억 5천만명 이상의 일일 활동중인 사용자와 1 백만 이상의 적극적인 광고주로 인해 페이스 북 광고에 대한 클릭을 예측하는 것은 어려운 기계 학습 작업입니다. 이 논문에서는 결정 트리와 로지스틱 회귀를 결합한 모델을 소개하며,이 중 하나를 3 % 이상 능가하며 전반적인 시스템 성능에 상당한 영향을 미칩니다. 그런 다음 몇 가지 기본 매개 변수가 시스템의 최종 예측 성능에 미치는 영향을 조사합니다. 당연한 일이지만, 가장 중요한 것은 적합한 기능을 보유하는 것입니다. 사용자 또는 광고에 대한 이력 정보를 캡처하는 기능은 다른 유형의 기능을 지배합니다. 올바른 기능과 올바른 모델 (의사 결정 트리 및 로지스틱 회귀)을 얻은 후에는 다른 요소가 작은 역할을합니다 (규모가 작은 경우에도 확장은 중요 함). 데이터 신선도, 학습률 스키마 및 데이터 샘플링에 대한 최적의 처리 방법을 선택하면 모델을 약간 향상시킬 수 있습니다. 그러나 값 비싼 기능을 추가하거나 처음부터 올바른 모델을 선택하는 것보다 훨씬 어렵습니다.
  
## 1. 개론
