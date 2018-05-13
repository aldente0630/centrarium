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
  
온라인 광고에서 광고주는 광고 클릭 같이 측정가능한 사용자 응답에 대해 입찰하여 대금을 지불한다. 그러므로 클릭 예측 시스템은 온라인 광고 시스템 대부분의 핵심이다. 7억 5천만명 이상 일일 활동중인 사용자와 1백만명 이상 유효 광고주를 고려할 때 페이스북 광고 클릭을 예측하는 일은 상당히 어려운 기계학습 작업이다. 이 논문은 결정 트리와 로지스틱 회귀를 결합한 모형을 소개하며 해당 모형은 전반적인 시스템 성능에 큰 영향을 미치면서 결정 트리나 로지스틱 회귀 하나만 사용한 것보다 예측 성능을 3% 이상 향상시켰다. 다음으로 몇 가지 기본 매개 변수가 시스템 최종 예측 성능에 미치는 영향을 조사했다. 당연한 말이지만 가장 중요한 건 적절한 변수를 찾는 것이다. 사용자 본인 또는 광고 이력 정보에 관한 변수는 다른 유형의 변수보다 훨씬 중요하다. 올바른 변수와 올바른 모형(의사 결정 트리 및 로지스틱 회귀)이 얻어진 다음에야 다른 요소들이 작게 기여한다(기여도가 작더라도 범위가 큰 시스템을 고려할 때 중요하다). 데이터 신선도, 학습률 스키마와 데이터 샘플링 방법을 최적화 처리해서 모형 성능을 다소 향상시킬 수 있으나 매우 값진 변수를 추가하거나 올바른 모형을 처음부터 선택하는 것보다 그 정도는 훨씬 못하다.
  
## 1. 개론
  
디지털 광고 업계는 수십억 달러 규모이며 매년 급격히 커지고 있다. 대부분의 온라인 광고 플랫폼에서는 광고를 동적으로 할당하며 사용자 관심도에 따라 조정한다. 사용자에 대한 광고 후보의 예상 효용을 계산하는 일에 기계학습이 중점적인 역할을 하며 이런 방식으로 시장 효율성이 높아진다.
