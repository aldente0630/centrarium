---
layout: post
title: 아파치 에어플로우로 작업흐름 개발해보기
date: 2018-06-17 00:00:00
author: Michal Karzynski
categories: Data-Engineering
---  
  
  
**Michal Karzynski의 [*Get Started Developing Workflows with Apache Airflow*](http://michal.karzynski.pl/blog/2017/03/19/developing-workflows-with-apache-airflow)을 번역했습니다.**
  
  
- - -
  
[Apache Airflow](https://airflow.apache.org)는 복잡한 계산을 요하는 작업흐름과 데이터 처리 파이프라인을 조정하기 위해 만든 오픈소스 도구이다. 길이가 긴 스크립트 실행을 cron으로 돌리거나 빅데이터 처리 배치 작업을 정기적으로 수행하려고 할 때 Airflow가 도움이 될 수 있다. 이 포스트는 Airflow를 이용하여 파이프라인을  구현해보려고 시도하는 이를 위한 입문 튜토리얼이다.

Airflow 상의 작업흐름은 방향성 비순환 그래프(DAG)로 설계된다. 즉, 작업흐름을 짤 때 그것이 어떻게 독립적으로 실행가능한 태스크들로 나뉠 수 있을까 생각해봐야한다. 그 다음에야 각 태스크를 그래프로 결합하여 전체적인 논리 흐름에 맞게 합칠 수 있다.

![에어플로우 파이프라인 DAG 예시](https://aldente0630.github.io/assets/developing_workflows_with_apache_airflow1.PNG)  
  
그래프 모양이 작업흐름의 전반적인 로직을 결정한다. Airflow DAG는 여러 분기를 포함할 수 있고 작업흐름 실행 시 건너뛸 지점과 중단할 지점을 결정할 수 있다.
  
Airflow는 각 태스크에서 오류가 발생할 때마다 여러 번 재실행하기에 매우 회복성 높은 설계를 이끌어낸다. Airflow를 완전히 멈췄다가 미완료 태스크를 재시작하면서 실행 중이던 업무 흐름으로 되돌아갈 수 있다.
  
> Airflow 오퍼레이터를 설계할 때 한 번 넘게 실행될 수 있음을 염두에 둬야한다. 각 태스크는 [멱등](https://ko.wikipedia.org/wiki/멱등법칙)이어야한다. 즉, 의도하지 않은 결과를 발생시키지 않고 여러 번 수행될 수 있어야한다.
  
### Airflow 명명법
    
다음은 Airflow 업무흐름을 설계할 때 사용하는 몇 가지 용어에 관한 간략한 개요이다.
  
* Airflow **DAG**는 **태스크**로 구성된다.
* 각 태스크는 **오퍼레이터** 클래스를 인스턴스화하여 만든다. 구성된 오퍼레이터 인스턴스는 다음과 같이 태스크가 된다. `my_task = MyOperator(...)`
* DAG가 시작되면 Airflow는 데이터베이스에 **DAG 런** 항목을 만든다.
* 특정 DAG 런 맥락에서 태스크를 실행하면 **태스크 인스턴스**가 만들어진다.
* `AIRFLOW_HOME`은 DAG 정의 파일과 Airflow 플러그인을 저장하는 디렉토리이다.

| 언제? | DAG | 태스크 | 다른 태스크 관련 정보 |
|:---------|:---------|:---------|:---------|
| 정의했을 때 | DAG | 태스크 | [get_flat_realtives](https://airflow.apache.org/_modules/airflow/models.html#BaseOperator.get_flat_relatives)|
| 실행했을 때 | DAG 런 | 태스크 인스턴스 | [xcom_pull](https://airflow.incubator.apache.org/concepts.html#xcoms)|
| 기본 클래스 | DAG | BaseOperator | |

Airflow 문서는 여러 [개념들](https://airflow.apache.org/concepts.html)에 대해 상세한 정보를 제공한다.

## 선행요건
  
Airflow는 파이썬으로 짜여있다. 컴퓨터에 파이썬이 이미 깔려있다고 가정하겠다. 참고로 난 파이썬3를 사용하고 있다(지금이 2017년이기 때문이지, 이봐들 어서!). 그러나 Airflow는 파이썬2도 지원한다. Virtualenv 또한 깔려있다고 가정하겠다.
```bash
$ python3 --version
Python 3.6.0
$ virtualenv --version
15.1.0
```
  
## Airflow 설치하기

본 튜토리얼을 위한 작업 공간 디렉토리를 만들고 그 안에 파이썬3 Virtualenv 디렉토리를 만들자.
```bash
$ cd /path/to/my/airflow/workspace
$ virtualenv -p `which python3` venv
$ source venv/bin/activate
(venv) $
```
  
이제 Airflow 1.8을 설치해보자.
```bash
(venv) $ pip install airflow==1.8.0
```
  
이제 DAG 정의 파일과 Airflow 플러그인이 저장되는 `AIRFLOW_HOME` 디렉토리를 만들어야한다. 디렉토리가 만들어지면 `AIRFLOW_HOME` 환경 변수를 설정하자.
```bash
(venv) $ cd /path/to/my/airflow/workspace
(venv) $ mkdir airflow_home
(venv) $ export AIRFLOW_HOME=`pwd`/airflow_home
```

이제 Airflow 명령을 실행할 수 있다. 다음 명령어 실행을 시도해보자.
```bash
(venv) $ airflow version
  ____________       _____________
 ____    |__( )_________  __/__  /________      __
____  /| |_  /__  ___/_  /_ __  /_  __ \_ | /| / /
___  ___ |  / _  /   _  __/ _  / / /_/ /_ |/ |/ /
 _/_/  |_/_/  /_/    /_/    /_/  \____/____/|__/
   v1.8.0rc5+apache.incubating
 ```

`airflow version` 명령을 실행시키면 Airflow는 `AIRFLOW_HOME`에 기본 구성 파일 airflow.cfg를 만든다.
```bash
airflow_home
├── airflow.cfg
└── unittests.cfg
 ```
   
본 튜토리얼은 `airflow.cfg`에 저장한 환경설정 기본값을 사용한다. 하지만 Airflow 설정을 조정하고 싶은 경우 해당 파일을 변경하라. Airflow [환경설정](https://airflow.apache.org/configuration.html)에 관한 자세한 내용은 문서를 참조하라.

### Airflow DB 초기화하기
  
다음 단계는 Airflow SQLite 데이터베이스를 만들고 초기화하는 명령을 실행하는 것이다.
```bash
(venv) $ airflow initdb
 ```

데이터베이스는 기본적으로 `airflow.db`에 작성된다.
```bash
airflow_home
├── airflow.cfg
├── airflow.db        <- Airflow SQLite DB
└── unittests.cfg
 ```
 > SQLite는 로컬 테스트와 개발 용도로 사용해도 괜찮지만 동시 액세스를 지원하지 않기 때문에 프로덕션 환경에서는 Postgres나 MySQL 같이 보다 강력한 데이터베이스 솔루션을 사용하는 편이 좋을 것이다.
 
### Airflow 웹 서버 시작하기
  
Airflow UI는 Flask 웹 응용 프로그램 형태로 제공된다. 다음 명령을 실행해서 시작할 수 있다.
```bash
(venv) $ airflow webserver
 ``` 
  
이제 브라우저가 Airflow가 시작된 호스트의 포트 8080으로 이동하여 Airflow UI를 방문 할 수 있습니다 (예 : http : // localhost : 8080 / admin /
  
> Airflow에는 DAG 예제 몇 가지가 있다. 이 예제는 dags_folder에 적어도 하나의 DAG 정의 파일이 있어야 작동합니다. airflow.cfg의 load_examples 설정을 변경하여 예제 DAG를 숨길 수 있습니다.
  
## Airflow DAG 처음 만들어보기

(번역 중)
