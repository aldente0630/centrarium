---
layout: post
title: LightGBM 고효율 그래디언트 부스팅 결정 트리
date: 2018-06-29 00:00:00
author: Microsoft Research, Peking University, Microsoft Redmond
categories: Data-Science
---  
  
  
**Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu의 [*LightGBM: A Highly Efficient Gradient Boosting Decision Tree*](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)을 번역했습니다.**
  
  
- - -
  
## 초록
그래디언트 부스팅 결정 트리(GBDT)는 널리 사용되는 기계 학습 알고리즘이며 XGBoost와 pGBRT 같이 효율적으로 구현해놓은 것들이 몇 가지 있다. 해당 구현은 엔지니어링의 많은 요소를 최적화시켰지만 고차원 변수에 데이터 크기가 큰 경우 효율성과 확장성은 여전히 불만족스럽다. 주된 이유로 각 변수마다 가능한 모든 분할점에 대해 정보 획득을 평가하려면 데이터 개체 모두를 스캔해야하는데 이에 많은 시간이 소요된다는 점이다. 이 문제를 해결하기 위해 *기울기 기반 표본추출*(GOSS)과 *배타적 변수 묶음*(EFB)이라는 새로운 기술 두 가지를 제안한다. GOSS를 통해 데이터 개체 중 기울기가 작은 상당 부분을 제외시키고 나머지만 사용하여 정보를 얻을 수 있다. 기울기가 큰 데이터 개체가 정보 획득 계산에 더 중요한 역할을 하기에 GOSS는 훨씬 작은 크기의 데이터로 정보 획득을 매우 정확하게 추정할 수 있다. EFB를 통해 변수 개수를 줄이기 위해 상호 배타적 변수들(예컨대, 0이 아닌 값을 동시에 갖는 일이 거의 없는 변수들)을 묶는다. 배타적 변수의 최적 묶음을 찾는 일은 NP-hard지만 탐욕 알고리즘을 통해 매우 좋은 근사 비율을 얻을 수 있다. 따라서 분할점 결정 정확도를 크게 훼손시키지 않으면서 변수 개수를 효과적으로 줄일 수 있다. GOSS와 EFB을 적용하여 GBDT 새롭게 구현한 걸 *LightGBM*라고 부르겠다. 여러 공용 데이터셋에 대한 실험을 통해 LightGBM은 기존 GBDT 훈련 과정을 최대 20배 이상 빠르게 하면서 정확도는 거의 동일하게 달성한다는 점을 보여준다.

(번역 중)
