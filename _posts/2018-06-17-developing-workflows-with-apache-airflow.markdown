---
layout: post
title: 아파치 에어플로우로 작업흐름 개발해보기
date: 2018-06-17 00:00:00
author: Michal Karzynski
categories: Data-Engineering
---  
  
  
**Michal Karzynski의 [*Get Started Developing Workflows with Apache Airflow*](http://michal.karzynski.pl/blog/2017/03/19/developing-workflows-with-apache-airflow)을 번역했습니다.**
  
  
- - -
  
[Apache Airflow](https://airflow.apache.org)는 복잡한 계산을 요하는 작업흐름과 데이터 처리 파이프라인을 조정하기 위해 만든 오픈소스 도구이다. 길이가 긴 스크립트 실행을 cron으로 돌리거나 빅데이터 처리 배치 작업을 정기적으로 수행하려고 할 때 Airflow가 도움이 될 수 있다. 이 포스트는 Airflow를 이용해 파이프라인 구현을 시도해보려는 사람들을 위한 입문 튜토리얼이다.

Airflow 상의 작업흐름은 방향성 비순환 그래프(DAG)로 설계한다. 즉, 작업흐름을 짤 때 어떻게 작업을 독립적으로 실행가능한 형태로 나눌 수 있을까 생각해봐야한다. 그 다음에야 작업을 그래프로 결합하여 전체적인 논리 흐름에 맞게 합칠 수 있다.

![에어플로우 파이프라인 DAG 예시](https://aldente0630.github.io/assets/developing_workflows_with_apache_airflow1.PNG)  
  
그래프 모양이 작업흐름의 전반적인 로직을 결정한다. Airflow DAG는 여러 분기를 포함할 수 있고 작업흐름 실행 시 건너뛸 지점과 중단할 지점을 결정할 수 있다.
  
오류가 발생하면 각 작업을 여러 번 다시 시도 할 수 있기 때문에 매우 탄력적 인 디자인을 만듭니다. 공기 흐름을 완전히 멈추고 실행중인 워크 플로가 마지막 미완성 작업을 다시 시작하여 다시 시작됩니다.
  
(번역 중)
