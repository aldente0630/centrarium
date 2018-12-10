---
layout: post
title: 해석가능한 XGBoost 기계학습
date: 2018-12-10 00:00:00
author: Scott Lundberg
categories: Data-Science
---  
  
  
**Scott Lundberg의 [*Interpretable Machine Learning with XGBoost*](https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27)를 번역했습니다.**
  
  
- - -

기계 학습 모형을 잘못 해석할 때의 위험성 그리고 올바르게 해석할 때의 가치에 관한 이야기다. 그래디언트 부스팅 머신이나 랜덤 포레스트 같은 앙상블 트리 모형의 굳건한 정확도를 확인했다면, 또 결과를 해석해야 한다면 유익하고 도움이 될 것이다.
  
은행에서 고객의 재정 상태를 예측하는 업무가 있다고 상상해보자. 모형이 정확할수록 은행은 많은 돈을 벌겠지만 예측값이 대출 신청에 사용될 터이니 해당 예측을 한 합법적 이유를 설명해야 한다. 여러 모형을 실험한 결과 XGBoost가 구현한 그래디언트 부스팅 트리가 정확도를 가장 높다는 걸 알았다. 불행하게도 XGBoost가 예측을 내린 이유를 설명하는 것이 어려워 보이므로 선형 모델로 후퇴하거나 XGBoost 모델을 해석하는 방법을 결정해야합니다. 데이터 과학자가 정확성을 포기하기를 원하지 않기 때문에 우리는 후자를 시도하고 복잡한 XGBoost 모델 (1,247 깊이 6 나무가 생기기도 함)을 해석하기로 결정합니다.
