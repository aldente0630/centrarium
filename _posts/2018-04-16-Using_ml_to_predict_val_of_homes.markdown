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
