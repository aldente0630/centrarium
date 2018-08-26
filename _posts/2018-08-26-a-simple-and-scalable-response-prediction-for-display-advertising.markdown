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
  
클릭률 및 전환율은 디스플레이 광고에서 두 가지 핵심 예측 과제입니다. 본 백서에서는 디스플레이 광고의 세부 사항을 다루기 위해 특별히 설계된 로지스틱 회귀를 기반으로하는 기계 학습 프레임 워크를 제시합니다. 결과 시스템에는 다음과 같은 특징이 있습니다. 구현 및 배포가 쉽습니다. 확장 성이 뛰어납니다 (테라 바이트 단위의 데이터를 교육했습니다). 최첨단 정확도를 갖춘 모델을 제공합니다.
  
## 1. 
