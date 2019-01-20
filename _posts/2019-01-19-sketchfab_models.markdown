---
layout: post
title: 암시적 행렬 분해(고전적인 ALS 방법) 소개와 LightFM을 이용한 순위 학습
date: 2019-01-19 00:00:00
author: Ethan Rosenthal
categories: Data-Science
---  
  
  
**Ethan Rosenthal의 [*Intro to Implicit Matrix Factorization: Classic ALS with Sketchfab Models 외 1편*](https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1)을 번역했습니다.**
  
  
- - -
  
지난 글에서 웹사이트 [Sketchfab](https://sketchfab.com)로부터 암시적 피드백 데이터를 수집하는 방법에 대해 설명했다. 그리고 이 데이터를 사용해 추천 시스템을 실제 구현해보겠다고 이야기했다. 자, 이제 만들어보자!
  
암시적 피드백을 위한 추천 모형을 살펴볼 때 Koren 등이 저술한 고전 논문 ["암시적 피드백 데이터셋을 위한 협업 필터링"](http://yifanhu.net/PUB/cf.pdf)(경고: pdf 링크)에서 설명한 모형으로부터 시작하는게 좋아보인다. 이 모형은 각종 문헌과 기계학습 라이브러리마다 다양하게 불리운다. 여기선 꽤 자주 사용되는 이름 중 하나인 *가중치가 적용된 제약적 행렬 분해*(WRMF)라고 부르겠다. WRMF는 암시적 행렬 분해에 있어 클래식 락 음악 같은 존재이다. 최신 유행은 아닐지도 몰라도 스타일에서 크게 벗어나지 않는다. 난 이 모형을 사용할 때마다 문제를 잘 해결하고 있다는 확신이 든다. 특히 이 모형은 합리적이며 직관적인 의미를 가지고 확장 가능하며 가장 좋은 점은 조정하기 편리하다는 것이다. 확률적 경사 하강 방법의 모형보다 훨씬 적은 수의 하이퍼 파라미터만 갖는다.
  
[명시적 피드백 행렬 분해](https://www.ethanrosenthal.com/2016/01/09/explicit-matrix-factorization-sgd-als)에 대한 과거 게시물을 떠올려보면 다음과 같은 손실 함수(편향 없는)가 있었다.
  
$$L_{exp} = \sum_{u, i \in S}(r_{ui} - \mathbf{x}^T_u \cdot \mathbf{y}_i)^2 + \lambda_{x} \sum_u {\lVert \mathbf{x}_u \rVert}^2 + \lambda_{y} \sum_i {\lVert \mathbf{y}_i \rVert}^2$$
  
여기서 \\(r_{ui}\\)는 사용자-품목 *점수* 행렬의 요소이고 \\(\mathbf{x}_u (\mathbf{y}_i)\\)는 사용자 \\(u\\)(품목 \\(i\\))의 잠재 요인이며 \\(S\\)는 고객-품목 점수의 전체 집합이다.  
WRMF는 이 손실 함수를 단순 수정 한 것이다.
  
$$L_{WRMF} = \sum_{u, i}c_{ui}(p_{ui} - \mathbf{x}^T_u \cdot \mathbf{y}_i)^2 + \lambda_{x} \sum_u {\lVert \mathbf{x}_u \rVert}^2 + \lambda_{y} \sum_i {\lVert \mathbf{y}_i \rVert}^2$$

여기서는 \\(S\\)의 요소만 합하는 대신 행렬 전체에 대해 합한다. 암시적 피드백이기 때문에 점수가 존재하지않음을 기억하라. 대신 항목에 대한 사용자의 선호도를 가지고 있다. WRMF 손실 함수에서 점수 행렬 \\(r_{ui}\\)이 선호도 행렬 \\(p_{ui}\\)로 바뀌었다. 사용자가 항목과 전혀 상호 작용하지 않았다면 \\(p_{ui} = 1\\), 그렇지 않으면 \\(p_{ui} = 0\\)이라고 가정하자.  
  
손실 함수 중 새롭게 나타난 또 다른 항은 \\(c_{ui}\\)이다. 이걸 신뢰도 행렬라고 부르며 사용자 \\(u\\)가 품목 \\(i\\)에 실제로 선호도 \\(p_{ui}\\)를 갖는지 얼마나 신뢰할 수 있는지에 관해 대략적으로 설명한다.

(번역 중)
