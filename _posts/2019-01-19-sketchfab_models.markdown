---
layout: post
title: 암시적 행렬 분해(전통적인 ALS 방법) 소개와 LightFM을 이용한 순위 학습
date: 2019-01-19 00:00:00
author: Ethan Rosenthal
categories: Data-Science
---  
  
  
**Ethan Rosenthal의 [*Intro to Implicit Matrix Factorization: Classic ALS with Sketchfab Models 외 1편*](https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1)을 번역했습니다.**
  
  
- - -
  
지난 글에서 웹사이트 [Sketchfab](https://sketchfab.com)로부터 암시적 피드백 데이터를 수집하는 방법에 대해 설명했다. 그리고 이 데이터를 사용해 추천 시스템을 실제 구현해보겠다고 이야기했다. 자, 이제 만들어보자!
  
암시적 피드백 추천기를 조사 할 때 가장 좋은 곳은 Koren et al.의 고전적 논문 "암시 적 피드백 데이터 세트를위한 협업 필터링"에 설명 된 모델입니다. (경고 : pdf 링크). 나는이 모델에 대한 문헌 및 기계 학습 라이브러리에서 많은 이름을 보았다. 나는 자주 사용되는 이름 인 경향이있는 WRED (Weighted Regularized Matrix Factorization)라고 부를 것입니다. WRMF는 암묵적 행렬 인수 분해의 고전적 암석과 같습니다. 가장 유행이 아닐지도 모르지만 절대로 스타일을 벗어날 수는 없습니다. 그리고 내가 그것을 사용할 때마다 나는 내가 나가는 것을 좋아할 것을 확신합니다. 특히,이 모델은 합리적인 직관적 인 의미를 지니 며, 확장 가능하며 가장 중요한 것은 조정하기가 쉽다는 것입니다. 확률 적 구배 하강 모델보다 훨씬 적은 수의 하이퍼 파라미터가 있습니다.

(번역 중)
