---
layout: post
title: NetworkX를 이용한 사회 연결망 분석
date: 2018-08-26 00:00:00
author: Manojit Nandi
categories: Data-Science
---  
  
  
**Manojit Nandi의 [*Social Network Analysis with NetworkX*](https://blog.dominodatalab.com/social-network-analysis-with-networkx)를 번역했습니다.**
  
  
- - -


현실 속 문제의 많은 유형에서 데이터 기록 사이에 종속성이 존재한다. 예를 들어 사회학자는 사람이 친구의 행동에 영향을 어떻게 미치는지 이해하고 싶어한다. 생물학자는 단백질이 다른 단백질의 작용을 어떻게 조절하는지 알고 싶어한다. 의존성과 관련된 문제는 종종 그래프로 모형화할 수 있고 과학자들은 이 문제를 해결하기 위해 네트워크 분석이라는 방법론을 개발했다.
  
이 글은 파이썬 라이브러리 [NetworkX](https://networkx.github.io)를 사용하여 네트워크 데이터를 처리하고 네트워크 분석으로 흥미로운 문제를 푸는 방법에 관해 설명한다. 아래 예제 모두가 담긴, 멋진 IPython Notebook을 [Domino](https://app.dominodatalab.com/u/LeJit/FacebookNetwork/browse?)에서 찾을 수 있다.
