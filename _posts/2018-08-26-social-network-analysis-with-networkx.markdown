---
layout: post
title: NetworkX를 이용한 사회 연결망 분석
date: 2018-08-26 00:00:00
author: Manojit Nandi
categories: Data-Science
---  
  
  
**Manojit Nandi의 [*Social Network Analysis with NetworkX*](https://blog.dominodatalab.com/social-network-analysis-with-networkx)를 번역했습니다.**
  
  
- - -


실제 문제의 많은 유형은 데이터의 레코드 간 종속성을 포함합니다. 예를 들어, 사회 학자들은 사람들이 동료들의 행동에 어떻게 영향을 미치는지 이해하려고 열심입니다. 생물 학자들은 단백질이 어떻게 다른 단백질의 작용을 조절하는지 배우고 싶어한다. 의존성과 관련된 문제는 종종 그래프로 모델링 할 수 있으며, 과학자들은 네트워크 분석이라고 불리는이 질문에 대답하기위한 방법을 개발했습니다.
