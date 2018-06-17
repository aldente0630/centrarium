---
layout: post
title: 아파치 에어플로우로 작업흐름 개발해보기
date: 2018-06-17 00:00:00
author: Michal Karzynski
categories: Data-Science
---  
  
  
**Michal Karzynski의 [*Get Started Developing Workflows with Apache Airflow*](http://michal.karzynski.pl/blog/2017/03/19/developing-workflows-with-apache-airflow)을 번역했습니다.**
  
  
- - -
  
[Apache Airflow](https://airflow.apache.org)는 복잡한 계산을 요하는 작업흐름과 데이터 처리 파이프라인을 조정하기 위해 만든 오픈소스 도구이다. 길이가 긴 스크립트 실행을 cron으로 돌리거나 빅데이터 처리 배치 작업을 정기적으로 수행하려고 할 때 Airflow가 도움이 될 수 있다. 이 포스트는 Airflow를 이용해 파이프라인 구현을 시도해보려는 사람들을 위한 입문 튜토리얼이다.
