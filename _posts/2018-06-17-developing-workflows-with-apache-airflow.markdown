---
layout: post
title: 아파치 에어플로우로 작업흐름 개발해보기
date: 2018-06-17 00:00:00
author: Michal Karzynski
categories: Data-Science
---  
  
  
**Michal Karzynski의 [*Get Started Developing Workflows with Apache Airflow*](http://michal.karzynski.pl/blog/2017/03/19/developing-workflows-with-apache-airflow)을 번역했습니다.**
  
  
- - -
  
Apache Airflow는 복잡한 계산 워크 플로우와 데이터 처리 파이프 라인을 조정하기위한 오픈 소스 도구입니다. 더 긴 스크립트를 실행하는 cron 작업을 실행하거나 큰 데이터 처리 일괄 처리 작업 일정을 유지하면 Airflow가 도움이 될 수 있습니다. 이 기사에서는 기류가있는 파이프 라인 작성을 시작하려는 사람들을위한 입문서 자습서를 제공합니다.
