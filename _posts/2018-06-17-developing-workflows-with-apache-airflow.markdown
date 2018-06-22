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
  
그래프 모양이 작업흐름의 전반적인 논리 구조를 결정한다. Airflow DAG는 여러 분기를 포함할 수 있고 작업흐름 실행 시 건너뛸 지점과 중단할 지점을 결정할 수 있다.
  
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
  
이제 브라우저에서 Airflow가 구동된 호스트 포트 `8080`으로 이동하여 Airflow UI를 방문할 수 있다. (예컨대 http://localhost:8080/admin/)
  
> Airflow상에 DAG 예제 몇 가지가 있다. 이 예제들은 `dags_folder`에 적어도 DAG 정의 파일이 한개 이상 있어야 작동한다. `airflow.cfg`의 `load_examples` 설정을 변경하여 DAG 예제를 숨길 수 있다.
  
## 처음 만들어보는 Airflow DAG 
  
좋다, 모든 게 준비됐다면 코드를 작성해보자. Hello World 작업흐름을 만들어볼 것이다. 이 작업흐름은 "Hello world!" 로그 찍는 일만 할거다. 
  
DAG 정의 파일이 `AIRFLOW_HOME/dags`에 저장되게 `dags_folder`를 만들어라. 이 디렉토리에 `hello_world.py` 파일을 만들자.
```bash
airflow_home
├── airflow.cfg
├── airflow.db
├── dags                <- Your DAGs directory
│   └── hello_world.py  <- Your DAG definition file
└── unittests.cfg
 ``` 
  
`dags/hello_world.py`에 다음 코드를 추가하자.
```python
from datetime import datetime
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator

def print_hello():
    return 'Hello world!'

dag = DAG('hello_world', description='Simple tutorial DAG',
          schedule_interval='0 12 * * *',
          start_date=datetime(2017, 3, 20), catchup=False)

dummy_operator = DummyOperator(task_id='dummy_task', retries=3, dag=dag)

hello_operator = PythonOperator(task_id='hello_task', python_callable=print_hello, dag=dag)

dummy_operator >> hello_operator
```
  
이 파일은 아무 일도 하지 않는 `DummyOperator`와 태스크가 실행될 때 `print_hello` 함수를 호출하는 `PythonOperator` 오퍼레이터 두 개만 가지고 간단한 DAG를 생성한다.

## DAG 돌려보기
  
DAG를 돌리려면 두번째 터미널을 열고 다음 명령을 실행해서 Airflow 스케줄러를 구동시키자.
```bash
$ cd /path/to/my/airflow/workspace
$ export AIRFLOW_HOME=`pwd`/airflow_home
$ source venv/bin/activate
(venv) $ airflow scheduler
```
  
> 스케줄러는 실행시킬 태스크를 이그제큐터로 보낸다. Airflow는 스케줄러에 의해 자동 시작되는 이그제큐터로 `SequentialExecutor`를 기본적으로 사용한다. 프로덕션 단계에서는 `CeleryExecutor` 같이 보다 탄탄한 이그제큐터를 사용해보고 싶을 것이다.
  
브라우저에서 Airflow UI를 새로고침하면 Airflow UI에 `hello_world` DAG가 표시된다.

![Airflow UI에서의 Hello World DAG](https://aldente0630.github.io/assets/developing_workflows_with_apache_airflow2.PNG)

DAG를 구동시키려면 먼저 작업흐름을 켜고(화살표 1) **DAG 동작** 버튼(화살표 2)을 클릭하고 마지막으로 **그래프 보기**(화살표 3)를 클릭해서 실행 진행률을 확인할 수 있다.

![Hello World DAG 런 - 그래프 보기](https://aldente0630.github.io/assets/developing_workflows_with_apache_airflow3.PNG)
  
두 개의 태스크 모두 **성공** 상태가 될 때까지 그래프 보기를 새로고침해볼 수 있다. 작업이 완료되면 `hello_task`를 클릭하여 들어가서 **로그 보기**를 클릭해볼 수 있다. 모든 것이 예상대로 동작하면 로그에 여러 줄이 표시되고 그 중 다음 같은 내용이 표시된다.
```python
[2017-03-19 13:49:58,789] {base_task_runner.py:95} INFO - Subtask: --------------------------------------------------------------------------------
[2017-03-19 13:49:58,789] {base_task_runner.py:95} INFO - Subtask: Starting attempt 1 of 1
[2017-03-19 13:49:58,789] {base_task_runner.py:95} INFO - Subtask: --------------------------------------------------------------------------------
[2017-03-19 13:49:58,790] {base_task_runner.py:95} INFO - Subtask: 
[2017-03-19 13:49:58,800] {base_task_runner.py:95} INFO - Subtask: [2017-03-19 13:49:58,800] {models.py:1342} INFO - Executing <Task(PythonOperator): hello_task> on 2017-03-19 13:49:44.775843
[2017-03-19 13:49:58,818] {base_task_runner.py:95} INFO - Subtask: [2017-03-19 13:49:58,818] {python_operator.py:81} INFO - Done. Returned value was: Hello world!
```
  
이 단계의 코드는 GitHub의 [해당 커밋](https://github.com/postrational/airflow_tutorial/tree/f91257e88ce2c0d30b032e92dc004c06754376fd/airflow_home)을 통해 받을 수 있다.

## 처음 만들어보는 Airflow 오퍼레이터 
  
나만의 Airflow 오퍼레이터를 만들어보자. 오퍼레이터는 작업흐름의 논리 상 단일 작업을 수행하는 원자 단위이다. 오퍼레이터는 파이썬 클래스(`BaseOperator`의 하위 클래스)로 작성되며 `__init__` 함수를 통해 환경설정이 구성되고 태스크 인스턴스를 실행할 때 `execute` 메서드를 통해 호출된다.
  
`execute` 메서드가 반환하는 값은 `return_value` 키 아래 Xcom 메시지로 저장된다. 나중에 이 주제를 다룰 것이다.

또한 `execute` 메서드는 `airflow.exceptions` 중 하나인 `AirflowSkipException`을 발생시킬 수 있다. 이 경우 태스크 인스턴스는 건너 뛴 상태로 전환된다.
  
다른 형태의 예외가 발생하면 `retiries` 최대 횟수에 도달할 때까지 작업을 재시도한다.
  
> execute 메소드는 여러 번 재시도될 수 있으므로 [멱등](https://ko.wikipedia.org/wiki/멱등법칙)이어야한다.
  
`plugins/my_operators.py`라는 Airflow 플러그인 파일을 작성하여 첫 연산자를 만들어 볼 거다. 먼저 `airflow_home/plugins` 디렉토리를 만든 다음 `my_operators.py` 파일에 다음 내용을 적어 추가해라.
```python
import logging

from airflow.models import BaseOperator
from airflow.plugins_manager import AirflowPlugin
from airflow.utils.decorators import apply_defaults

log = logging.getLogger(__name__)

class MyFirstOperator(BaseOperator):

    @apply_defaults
    def __init__(self, my_operator_param, *args, **kwargs):
        self.operator_param = my_operator_param
        super(MyFirstOperator, self).__init__(*args, **kwargs)

    def execute(self, context):
        log.info("Hello World!")
        log.info('operator_param: %s', self.operator_param)

class MyFirstPlugin(AirflowPlugin):
    name = "my_first_plugin"
    operators = [MyFirstOperator]
 ```
   
이 파일에서는 `MyFirstOperator`라는 새 연산자를 정의한다. `execute` 메소드는 매우 간단하다. "Hello World!"와 매개 변수값 하나 로그 남기는 일만 한다. 매개 변수는 `__init__` 함수에서 설정한다.
  
또한 `MyFirstPlugin`이라는 Airflow 플러그인을 정의하고 있다. `airflow_home/plugins` 디렉토리에 플러그인을 정의한 파일을 저장함으로써 플러그인이 제공하는 기능과 그것이 정의한 모든 오퍼레이터를 Airflow가 가져다 쓸 수 있다. 이 오퍼레이터를 `from airflow.operators import MyFirstOperator`를 써서 불러올 수 있다.
  
[Airflow 플러그인](https://airflow.apache.org/plugins.html)은 문서에서 더 자세한 정보를 얻을 수 있다.

> `PYTHONPATH`가 사용자 정의 모듈을 저장한 디렉토리를 포함하도록 설정됐는지 확인해라.
  
이제 오퍼레이터를 테스트할 새 DAG를 만들어야한다. `dags/test_operators.py` 파일을 만들고 다음 내용으로 채우자.
```python
from datetime import datetime
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators import MyFirstOperator

dag = DAG('my_test_dag', description='Another tutorial DAG',
          schedule_interval='0 12 * * *',
          start_date=datetime(2017, 3, 20), catchup=False)

dummy_task = DummyOperator(task_id='dummy_task', dag=dag)

operator_task = MyFirstOperator(my_operator_param='This is a test.',
                                task_id='my_first_operator_task', dag=dag)

dummy_task >> operator_task
```
  
`DummyOperator` 태스크를 이용해서 `my_test_dag`라는 간단한 DAG를 만들었고 새로 만든 `MyFirstOperator`를 이용해 또 다른 태스크를 만들었다. DAG 정의 중 `my_operator_param`에 대한 환경설정 값 전달하는 방법에 유의해라.
  
이 단계에서 소스 트리는 다음과 같다.
```bash
airflow_home
├── airflow.cfg
├── airflow.db
├── dags
│   └── hello_world.py
│   └── test_operators.py  <- Second DAG definition file
├── plugins
│   └── my_operators.py    <- Your plugin file
└── unittests.cfg
```
  
이 단계의 코드는 GitHub의 [해당 커밋](https://github.com/postrational/airflow_tutorial/tree/fc918909763eba0a1671ecda4629b4ffec45c441/airflow_home)을 통해 받을 수 있다.
  
새로운 오퍼레이터를 테스트해보려면 Airflow 웹 서버와 스케줄러를 중지시킨 후(CTRL-C) 재시작해야 한다. 그런 다음 Airflow UI로 돌아가 `my_test_dag` DAG를 켜고 실행시켜라. `my_first_operator_task` 로그를 살펴봐라.

## Airflow 오퍼레이터 디버깅하기
  
DAG 런을 동작시키고 모든 업스트림 작업이 완료 될 때까지 기다렸다가 새 운영자를 다시 시도해야하는 경우 디버깅이 빨리 지루할 수 있습니다. 고맙게도 Airflow에는 `airflow test` 명령이 있습니다.이 명령을 사용하여 특정 DAG 실행 상황에서 단일 운영자를 수동으로 시작할 수 있습니다.
  
이 명령은 dag 이름, 작업 이름 및 특정 DAG 실행과 관련된 날짜의 3 가지 인수를 사용합니다.
```bash
(venv) $ airflow test my_test_dag my_first_operator_task 2017-03-18T18:00:00.0
```
  
이 명령을 사용하여 작업자 코드를 조정하면서 필요한만큼 여러 번 작업을 다시 시작할 수 있습니다.
  
> 특정 DAG 실행에서 작업을 테스트하려는 경우 실패한 작업 인스턴스의 로그에서 필요한 날짜 값을 찾을 수 있습니다.
  
### IPython으로 Airflow 오퍼레이터 디버깅하기
  
운영자 코드를 디버깅하는 데 사용할 수있는 멋진 트릭이 있습니다. veny에 IPython을 설치 한 경우 :
```bash
(venv) $ pip install ipython
```
  
그런 다음 코드에 IPython의 `embed()` 명령을 배치 할 수 있습니다 (예 : 연산자의 `execute` 메소드).
```python
ef execute(self, context):
    log.info("Hello World!")

    from IPython import embed; embed()

    log.info('operator_param: %s', self.operator_param)
```
  
이제 `airflow test` 명령을 다시 실행하면 다음과 같습니다.
```bash
(venv) $ airflow test my_test_dag my_first_operator_task 2017-03-18T18:00:00.0
```
  
작업은 실행되지만 실행이 멈추고 IPython 셸에 드롭됩니다.이 쉘에서 코드에서 `embed()`을 배치 한 위치를 탐색 할 수 있습니다.
```python
In [1]: context
Out[1]:
{'END_DATE': '2017-03-18',
 'conf': <module 'airflow.configuration' from '/path/to/my/airflow/workspace/venv/lib/python3.6/site-packages/airflow/configuration.py'>,
 'dag': <DAG: my_test_dag>,
 'dag_run': None,
 'ds': '2017-03-18',
 'ds_nodash': '20170318',
 'end_date': '2017-03-18',
 'execution_date': datetime.datetime(2017, 3, 18, 18, 0),
 'latest_date': '2017-03-18',
 'macros': <module 'airflow.macros' from '/path/to/my/airflow/workspace/venv/lib/python3.6/site-packages/airflow/macros/__init__.py'>,
 'next_execution_date': datetime.datetime(2017, 3, 19, 12, 0),
 'params': {},
 'prev_execution_date': datetime.datetime(2017, 3, 18, 12, 0),
 'run_id': None,
 'tables': None,
 'task': <Task(MyFirstOperator): my_first_operator_task>,
 'task_instance': <TaskInstance: my_test_dag.my_first_operator_task 2017-03-18 18:00:00 [running]>,
 'task_instance_key_str': 'my_test_dag__my_first_operator_task__20170318',
 'test_mode': True,
 'ti': <TaskInstance: my_test_dag.my_first_operator_task 2017-03-18 18:00:00 [running]>,
 'tomorrow_ds': '2017-03-19',
 'tomorrow_ds_nodash': '20170319',
 'ts': '2017-03-18T18:00:00',
 'ts_nodash': '20170318T180000',
 'var': {'json': None, 'value': None},
 'yesterday_ds': '2017-03-17',
 'yesterday_ds_nodash': '20170317'}

In [2]: self.operator_param
Out[2]: 'This is a test.'
```
  
물론 파이썬의 인터랙티브 디버거 인 `pdb`(`import pdb; pdb.set_trace()`) 나 IPython의 향상된 버전 인 `ipdb`(`import ipdb; ipdb.set_trace()`)를 사용할 수도 있습니다. 또는 `airflow test` 기반 실행 구성을 사용하여 PyCharm과 같은 IDE에서 중단 점을 설정할 수도 있습니다.
  
이 단계의 코드는 GitHub의 [해당 커밋](https://github.com/postrational/airflow_tutorial/tree/45fe1a53d1306ad4e385dc7e85d8e606f860f750/airflow_home)을 통해 받을 수 있다.
  
(번역 중)
