---
layout: post
title: LightGBM: 고효율 그래디언트 부스팅 결정 트리
date: 2018-06-29 00:00:00
author: Microsoft Research, Peking University, Microsoft Redmond
categories: Data-Science
---
  
  
**Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu의 [*LightGBM: A Highly Efficient Gradient Boosting Decision Tree*](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)을 번역했습니다.**
  

- - -
    
## 초록
Gradient Boosting Decision Tree (GBDT)는 널리 사용되는 기계 학습 알고리즘이며 XGBoost 및 pGBRT와 같은 몇 가지 효과적인 구현이 있습니다. 이러한 구현에서 많은 엔지니어링 최적화가 채택되었지만 피처 차원이 높고 데이터 크기가 클 경우 효율성과 확장 성은 여전히 ​​만족스럽지 않습니다. 주된 이유는 각 기능에 대해 모든 가능한 스플릿 포인트의 정보 획득을 평가하기 위해 모든 데이터 인스턴스를 스캔해야하므로 매우 많은 시간이 소요됩니다. 이 문제를 해결하기 위해 GOSS (Gradient-Based Sampling)와 EFB (Exclusive Feature Bundling)라는 두 가지 새로운 기술을 제안합니다. GOSS를 사용하면 작은 그라디언트가있는 상당 비율의 데이터 인스턴스를 제외하고 나머지를 사용하여
정보 이득. 큰 기울기를 가진 데이터 인스턴스가 정보 게인 계산에 더 중요한 역할을하기 때문에 GOSS는 훨씬 작은 데이터 크기로 정보 게인을 매우 정확하게 추정 할 수 있음을 증명합니다. EFB를 사용하면 피쳐 수를 줄이기 위해 상호 배타적 인 피쳐 (예 : 0이 아닌 값은 거의 사용하지 않음)를 묶습니다. 우리는 배타적 피처의 최적 번들링을 찾는 것이 NP 하드이지만, 그리 디 알고리즘은 아주 좋은 근사 비율을 얻을 수 있다는 것을 증명합니다. 따라서 분할 점 결정의 정확성을 크게 손상시키지 않으면 서 피처 수를 효과적으로 줄일 수 있습니다. 우리는 GOSS 및 EFB LightGBM을 사용하여 새로운 GBDT 구현을 호출합니다. 여러 개의 공용 데이터 세트에 대한 실험을 통해 LightGBM은 기존 GBDT의 교육 과정을 최대 20 배 이상 가속하면서 거의 동일한 정확도를 달성한다는 것을 보여줍니다.
