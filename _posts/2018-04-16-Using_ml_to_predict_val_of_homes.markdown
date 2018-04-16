---
layout: post
title:  Using Machine Learning to Predict Value of Homes on Airbnb
date:   2018-04-16 00:00:00
author: Robert Chang
categories: Data-Science
---  
  
  
**Robert Chang의 [*Using Machine Learning to Predict Value of Homes on Airbnb*](https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d)을 번역했습니다.**
  
  
- - -

## 서론
  
  데이터 제품은 항상 Airbnb 서비스의 중요한 부분이다. 그러나 우리는 오랫동안 데이터 제품를 만드는데 비용이 많이 든다는 사실을 인지해왔다. 예를 들어 개인화 검색 순위는 손님이 집을 더 쉽게 찾도록 도와주며 스마트 가격 정책은 주인이 수요와 공급에 따라 더 경쟁력있는 가격을 설정할 수 있게끔 도와준다. 그러나 이 프로젝트들은 각각 데이터 과학 및 공학만을 위한 시간과 노력을 필요로 했다.
  
  최근 Airbnb 기계 학습 인프라의 발전으로 새로운 기계 학습 모형을 제품으로 배포하는 비용이 크게 절감되었다. 예를 들어 ML Infra팀은 고품질의, 검증되고, 재사용 가능한 변수들을 사용자가 모형에 활용할 수 있게 범용 변수 저장소를 구축했다. 데이터 과학자는 여러 AutoML 도구를 작업 흐름에 통합시켜 모형 선택 및 성능 벤치 마크를 가속화했다. 또한 ML Infra팀은 Jupyter 노트북을 Airflow 파이프 라인으로 변환하는 새로운 프레임워크를 만들었다.
  
  이 문서에서는 LTV 모델링, 즉, Airbnb에 올라온 집 가치를 예측하는 특정 활용 사례를 통해 이런 도구들이 어떻게 함께 동작하여 모델링 절차를 빠르게 하고 전체적인 개발 비용을 낮추는지 설명하겠다.
    
## LTV란 무엇인가?
  
  전자 상거래 및 마켓플레이스 기업에서 인기있는 개념인 고객 생애 가치(LTV)는 사용자마다 고정 시간 동안 발생할 예상 가치를 뜻하며 대개 달러 단위로 측정한다.
  
  Spotify와 Netflix 등 전자 상거래 회사는 LTV를 구독료 설정 등 가격 결정에 자주 사용한다. Airbnb 같은 마켓플레이스 기업에서 사용자 LTV를 알면 다양한 마케팅 채널에 예산을 효율적으로 할당하고 키워드 기반 온라인 마케팅에 대해 보다 정확한 입찰 가격을 계산하고 목록 세그먼트를 더 잘 만들 수 있다.
  
  과거 데이터를 사용해서 기존 목록 [과거 가치를 계산](https://medium.com/swlh/diligence-at-social-capital-part-3-cohorts-and-revenue-ltv-ab65a07464e1)할 수 있지만 새 목록 LTV를 예측하기 위해 기계 학습을 사용하여 한 걸음 더 나아갔다.
    
## LTV 모델링을 위한 기계학습 작업 흐름
 
  데이터 과학자는 일반적으로 피쳐 엔지니어링, 프로토타이핑 및 모형 선택 같은 기계학습 관련 작업에 익숙하다. 그러나 모형 프로토타입을 제품에 사용하려면 종종 데이터 과학자가 익숙하지 않은 데이터 엔지니어링 기술들의 직교 집합이 필요하다.

!(https://cdn-images-1.medium.com/max/1600/1*zT1gNPErRqizxlngxXCtBA.png)
