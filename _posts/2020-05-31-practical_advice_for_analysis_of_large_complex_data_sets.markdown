---
layout: post
title: 크고 복잡한 데이터셋 분석을 위한 실무 지침
date: 2020-05-31 00:00:00
author: Patrick Riley
categories: Data-Science
---  
  
  
**Patrick Riley의 [*Practical advice for analysis of large, complex data sets*](http://www.unofficialgoogledatascience.com/2016/10/practical-advice-for-analysis-of-large.html)을 번역했습니다.**
  
  
- - -

  나는 Google 검색 로그에 관한 데이터 과학 팀을 몇 년 간 이끌었다. 우리 팀은 복잡한 결과에 대해 이유를 찾고, 행동 로그를 통해 새로운 현상을 관측하고, 다른 사람이 수행한 분석을 검증하고, 사용자 행동에 대한 지표를 해석해달라는 요청을 종종 받았다. 어떤 팀원들은 수준 높은 데이터 분석을 쉽고 능숙하게 해냈다. 이런 엔지니어와 분석가들은 흔히 "주의 깊고" "꼼꼼하다고" 이야기된다. 이 형용사는 실제로 무얼 의미할까? 이 수식어를 얻으려면 어떤 행동을 해야할까?
  
  이 질문에 대한 대답을 문서로 정리한 뒤 '좋은 데이터 분석'이라는 낙천적이고 단순한 제목으로 Google 사내에 공유했다. 놀랍게도 이 문서는 내가 지난 11 년 동안 Google에서 수행한 그 어떤 일보다 더 많은 사람들이 관심을 갖고 읽었다. 주요 내용을 업데이트한 지 4년이 지났음에도 확인할 때마다 문서를 열어 본 Google 직원이 여러 명인 것으로 드러났다.
