---
layout: post
title: DeepFM, CTR 예측을 위한 신경망 기반 팩토라이제이션 머신
date: 2019-12-29 00:00:00
author: Harbin Institute of Technology, Huawei
categories: Data-Science
---  
  
  
**Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, Xiuqiang He의 [*DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*](https://arxiv.org/pdf/1703.04247.pdf)을 번역했습니다.**
  
  
- - -
  
# 초록
  
추천 시스템의 CTR을 극대화하려면 사용자 행동에 담긴, 변수 간 복잡한 상호 작용을 학습하는 것이 중요하다. 장족의 발전이 있었지만 기존 방법들은 여전히 저차 또는 고차 상호 작용에 대한 편향이 크거나 전문적인 특성 공학 작업을 필요로 한다. 본 논문은 변수 간 저차와 고차 상호 작용 모두에 중점을 두는 종단 간 학습 모형을 도출해낸다. 제안한 모형 DeepFM은 팩토라이제이션 머신의 추천 성능과 신경망 구조를 통해 변수 학습하는 심층 학습 기법을 결합한다. Google의 Wide & Deep 최신 모형과 비교할 때 DeepFM은 "넓은"과 "깊은" 부분이 입력을 공유하며 원천 변수 외에 특성 공학이 따로 필요없다. 기존 CTR 예측 모형 대비 DeepFM 효과와 효율성을 입증하기 위해 벤치 마크 데이터와 상용 데이터 모두를 이용해 종합 실험을 수행했다.
  
# 1. 서론
  
CTR(Click-Through Rate) 예측은 사용자가 추천 품목을 클릭할 확률을 추정해야 하는 추천 시스템에서 중요하다. 많은 추천 시스템에서 목표는 클릭 수를 최대화하는 것이므로 사용자에게 반환되는 항목의 예상 CTR을 기준으로 순위를 지정할 수 있습니다. 온라인 광고와 같은 다른 응용 시나리오에서는 수익을 향상시키는 것도 중요하므로 순위 전략은 모든 후보자에 대해 CTR × bid로 조정될 수 있습니다. 여기서 "bid"는 품목이 사용자가 클릭합니다. 두 경우 모두 키가 CTR을 정확하게 추정하는 것이 확실합니다.
