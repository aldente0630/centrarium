---
layout: post
title: 디스플레이 광고를 위한 단순하고 확장 가능한 응답 예측
date: 2018-08-26 00:00:00
author: Criteo, Microsoft, LinkedIn
categories: Data-Science
---  
  
  
**Olivier Chapelle, Eren Manavoglu, Romer Rosales의 [*Simple and Scalable Response Prediction for Display Advertising*](http://people.csail.mit.edu/romer/papers/TISTRespPredAds.pdf)를 번역했습니다.**
  
  
- - -
    
## 초록
  
클릭률과 전환율은 디스플레이 광고에서 두 가지 핵심 예측 과제이다. 본 논문에서는 디스플레이 광고의 세부 사항을 다루기 위해 특별히 설계한 로지스틱 회귀 기반 기계 학습 프레임워크를 제시한다. 결과 시스템에는 다음과 같은 특징이 있다. 구현과 배포가 쉽다. 확장성이 뛰어나다 (테라바이트 단위의 데이터를 훈련시켰다). 최첨단 정확도를 갖춘 모형을 제공한다.
  
## 1. 
