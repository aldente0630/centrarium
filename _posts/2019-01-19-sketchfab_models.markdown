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
  
암시적 피드백을 위한 추천 모형을 살펴볼 때 Koren 등이 저술한 고전 논문 ["암시적 피드백 데이터셋을 위한 협업 필터링"](http://yifanhu.net/PUB/cf.pdf)(경고: pdf 링크)에서 설명한 모형으로부터 시작하는게 좋아보인다. 이 모형은 각종 문헌과 기계학습 라이브러리마다 다양하게 불려진다. 여기선 꽤 자주 사용되는 이름인 '가중치가 적용된 제약적 행렬 분해'(WRMF)라고 부르겠다. WRMF는 암시적 행렬 분해에 있어 클래식 락 음악 같은 존재이다. 최신 유행은 아닐지도 몰라도 스타일에서 크게 벗어나지 않는다. 그리고 내가 그것을 사용할 때마다 나는 내가 나가는 것을 좋아할 것을 확신합니다. 특히,이 모델은 합리적인 직관적 인 의미를 지니 며, 확장 가능하며 가장 중요한 것은 조정하기가 쉽다는 것입니다. 확률 적 구배 하강 모델보다 훨씬 적은 수의 하이퍼 파라미터가 있습니다.

(번역 중)
