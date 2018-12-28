---
layout: post
title: 하이브에서의 조인 유형
date: 2018-12-28 00:00:00
author: Weidong Zhou
categories: Data-Engineering
---  
  
  
**Weidong Zhou의 [*Join Type in Hive: Common Join 외 3편*](https://weidongzhou.wordpress.com/2017/06/06/join-type-in-hive-common-join)을 번역했습니다.**
  
  
- - -

# 1. 일반 조인
  
하이브 쿼리의 성능 튜닝 중 하나의 영역은 실행 중에 조인 유형에주의를 기울여야합니다. 오라클의 조인 유형과 마찬가지로 다양한 유형의 실행 시간이 크게 다를 수 있습니다. 다음 몇 블로그에서 하이브의 조인 유형에 대해 논의 할 것입니다. 첫 번째 조인 유형은 **공통 조인**입니다.
