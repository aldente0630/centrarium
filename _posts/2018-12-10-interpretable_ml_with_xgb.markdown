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
  
은행에서 고객의 재정 상태를 예측하는 업무가 있다고 상상해보자. 모형이 정확할수록 은행은 많은 돈을 벌겠지만 예측값이 대출 신청에 사용될 터이니 해당 예측을 한 합법적 이유를 설명해야 한다. 여러 모형을 실험한 결과 XGBoost가 구현한 그래디언트 부스팅 트리 정확도가 가장 높다는걸 알았다. XGBoost가 예측한 이유를 설명하긴 까다로워 보이므로 선형 모형으로 돌아가거나 XGBoost 모형을 해석할 수 있는 방안을 고심해봐야한다. 데이터 과학자라면 정확도를 포기하고 싶지 않을 것이기에 후자를 시도하며 복잡한 XGBoost 모형(깊이가 6인 1,247개의 트리)을 해석해보기로 결정했다.
  
## 고전적인 전역 변수 중요도 측정
  
첫 번째 확실한 선택은 파이썬 XGBoost 인터페이스에서 plot_importance () 메소드를 사용해 보는 것이다. 데이터셋에서 각 변수의 중요도를 나타내는 매력적이고 단순한 막대 차트를 제공한다(본문을 재현하는 코드는 [주피터 노트북](https://slundberg.github.io/shap/notebooks/Census+income+classification+with+XGBoost.html)에 있음)
