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
  
하이브 쿼리 성능 튜닝에서 신경써야할 부분 중 하나는 쿼리 실행 시 이뤄지는 조인 유형이다. 오라클의 조인 유형과 마찬가지로 여러 다른 유형에 따라 실행 시간이 크게 달라질 수 있다. 몇 번의 연재에 걸쳐 하이브의 조인 유형에 대해 논의할 것이다. 조인 유형의 첫 번째는 **일반 조인**이다.
  
**일반 조인**은 하이브의 기본적인 조인 유형으로 **셔플 조인**, **분산 조인** 또는 **정렬 병합 조인**이라고도한다. 조인 작업 동안 두 테이블의 모든 행을 조인 키 기반으로 전체 노드에 분산시킨다. 이 과정을 통해 조인 키가 동일한 값들은 동일한 노드에서 작업이 종료된다. 해당 조인 작업은 맵 / 리듀스의 온전한 주기를 갖는다.
  
![그림1](https://aldente0630.github.io/assets/join_type_in_hive1.jpg)
  
**작동 원리**
1. 맵 단계에서 맵퍼는 테이블을 읽고 조인 컬럼 값을 키로 정해 출력한다. 키 - 값 쌍을 중간 파일에 기록한다.
2. 셔플 단계에서 이러한 쌍을 정렬하고 병합한다. 동일한 키의 모든 행을 동일한 리듀스 인스턴스로 전송한다.
3. 리듀스 단계에서 리듀서는 정렬한 데이터를 가져와 조인을 수행한다.

**일반 조인**의 장점은 테이블 크기와 상관없이 작동한다는 점이다. 그러나 셔플은 비용이 매우 큰 작업이기에 자원을 많이 잡아먹는다. 데이터에서 소수의 조인 키가 차지하는 비율이 매우 클 경우 해당 리듀서에 과부하가 걸리게된다. 대다수의 리듀서에서 조인 작업이 완료됐지만 일부 리듀서가 계속 실행되는 식으로 문제의 증상이 나타난다. 쿼리의 총 실행 시간은 실행 시간이 가장 긴 리듀서가 결정한다. 분명히 이건 전형적인 **데이터 비대칭** 문제이다. 이어지는 내용에서 이러한 비대칭 문제을 다루는 특수 조인에 관해 논할 것이다.

**조인 유형을 식별하는 방법**  
**EXPLAIN** 명령을 사용하면 **리듀스 오퍼레이터 트리** 바로 밑에 **조인 오퍼레이터**가 표시된다.
  
# 2. 맵 조인
  
마지막 블로그에서 Hive : Common Join의 기본 조인 유형에 대해 설명했습니다. 이 블로그에서는 **지도 조인 (Auto Map Join)** 또는 **지도 측 조인 (Map Side Join)** 또는 **브로드 캐스트 조인 (Broadcast Join)** 이라고도합니다.

**공통 조인** 또는 **정렬 병합 조인**의 주요 문제점 중 하나는 데이터 셔플 링에 너무 많은 활동 지출입니다. Hive 쿼리 속도를 높이기 위해 **Map Join**을 사용할 수 있습니다. 조인의 테이블 중 하나가 작은 테이블이고 메모리에로드 될 수 있으면 맵 조인을 사용할 수 있습니다.
  
![그림2](https://aldente0630.github.io/assets/join_type_in_hive2.jpg)
  
**맵 결합**의 첫 번째 단계는 원래 맵 축소 작업 전에 맵 축소 태스크를 작성하는 것입니다. 이 map / reduce 작업은 작은 테이블의 데이터를 HDFS에서 읽어 와서 메모리 내 해시 테이블에 저장 한 다음 해시 테이블 파일에 저장합니다. 그런 다음 원본 조인 맵 축소 작업이 시작되면 해시 테이블 파일을 [Hadoop 분산 캐시](https://hadoop.apache.org/docs/r1.2.1/api/org/apache/hadoop/filecache/DistributedCache.html)로 이동합니다. 그러면 Hadoop 분산 캐시가 각 매퍼의 로컬 디스크에 파일을 채 웁니다. 따라서 모든 매퍼는이 해시 테이블 파일을 메모리에로드 한 다음 맵 스테이지에서 조인을 수행 할 수 있습니다. 예를 들어 큰 테이블 A와 작은 테이블 B가있는 조인의 경우 테이블 A의 모든 매퍼에 대해 테이블 B가 완전히 읽혀집니다. 더 작은 테이블이 메모리에로드 된 후 MapReduce 작업의 맵 구문에서 조인이 수행되면 축소 기가 필요하지 않고 축소 단계가 건너 뜁니다. 지도 조인은 일반 기본 조인보다 빠르게 수행됩니다.

**매개 변수**

* **Map Join**의 가장 중요한 매개 변수는 **hive.auto.convert.join**입니다. **true**로 설정해야합니다.
* 조인 할 때, 작은 테이블의 결정은 매개 변수 **hive.mapjoin.smalltable.filesize**에 의해 제어됩니다. 기본적으로 25MB입니다.
* 세 개 이상의 테이블이 조인에 포함되면 Hive는 모든 테이블의 크기가 더 작은 것으로 가정 한 3 개 이상의지도 쪽 조인을 생성합니다. 조인 속도를 더 높이려면 n-1 테이블의 크기가 기본값 인 10MB보다 작은 경우 세 개 이상의지도 측 조인을 단일 맵 측 조인으로 결합 할 수 있습니다. 이를 위해서는 **hive.auto.convert.join.noconditionaltask** 매개 변수를 **true**로 설정하고 매개 변수 **hive.auto.convert.join.noconditionaltask.size**를 지정해야합니다.

**제한**

전체 외부 조인은 절대로 **맵 조인**으로 변환되지 않습니다.
오른쪽 테이블의 크기가 25MB보다 작 으면 왼쪽 외부 조인을 맵 조인으로 변환 할 수 있습니다. 오른쪽 외부 조인이 작동하지 않습니다.

**조인을 식별하는 방법**
**EXPLAIN** 명령을 사용할 때 **Map Operator Tree** 바로 아래에 **Join Join Operator**가 표시됩니다.

**다른**  
  
힌트를 사용하여 맵 결합을 사용하여 쿼리를 지정할 수 있습니다. 아래 예제는 힌트에 넣은 테이블이 더 작고 테이블 B를 수동으로 캐시하도록합니다.
  
```sql
Select /*+ MAPJOIN(b) */ a.key, a.value from a join b on a.key = b.key
```
  
**예제**  
  
```sql
hive> set hive.auto.convert.join=true;
hive> set hive.auto.convert.join.noconditionaltask=true;
hive> set hive.auto.convert.join.noconditionaltask.size=20971520
hive> set hive.auto.convert.join.use.nonstaged=true;
hive> set hive.mapjoin.smalltable.filesize = 30000000; 
```

(번역 중)
  

