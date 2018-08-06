---
layout: post
title: 판다스 코드 속도 최적화를 위한 초보자 안내서
date: 2018-08-05 00:00:00
author: Sofia Heisler
categories: Data-Science
---  
  
  
**Sofia Heisler의 [*A Beginner's Guide to Optimizing Pandas Code for Speed*](https://engineering.upside.com/a-beginners-guide-to-optimizing-pandas-code-for-speed-c09ef2c6a4d6)를 번역했습니다.**
  
  
- - -
  
파이썬으로 데이터 분석을 했다면 Wes McKinney가 작성한 환상적인 분석 라이브러리 [판다스](http://pandas.pydata.org/pandas-docs/stable)를 아마 사용해봤을거다. 판다스는 데이터프레임 분석 기능을 파이썬에 부여함으로써 파이썬을 R이나 SAS 같은 기존 분석 도구와 어깨를 나란히 하게 만들었다.
  
불행히도 초기 판다스는 "느리다"는 불쾌한 평판을 얻었다. 물론 판다스 코드는 완전 최적화시킨 C 코드 계산 속도에 도달하지 못할 것이다. 그러나 좋은 소식은 잘 작성한 판다스 코드는 응용 프로그램 대부분 에서 *충분히 빠르다*는 것이다. 그리고 판다스의 속도 면에서 부족한 점을 강력하고 사용하기 쉬운 기능으로 만회할 수 있다는 것이다.
  
이 게시물에서 판다스 데이터프레임에 함수를 적용하는 몇 가지 방법론의 효율성을 가장 느린 속도부터 가장 빠른 속도까지 나열하여 검토하겠다.  
1 . 색인를 사용하여 데이터프레임 행을 반복하는 방법  
2 . `iterrows()`를 사용한 반복  
3 . `apply()`를 사용한 반복  
4 . 판다스 시리즈를 사용한 벡터화  
5 . NumPy 배열을 사용한 벡터화  
  
예제 함수로 [Haversine](https://en.wikipedia.org/wiki/Haversine_formula)(또는 Great Circle) 거리 수식을 사용했다. 이 함수는 두 점의 위도와 경도를 취하여 지구 곡률을 조정하고 그 사이의 직선 거리를 계산한다. 함수는 다음과 같다.
  
```python
import numpy as np

# Haversine 기본 거리 공식을 정의함
def haversine(lat1, lon1, lat2, lon2):
    MILES = 3959
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    total_miles = MILES * c
    return total_miles
```
  
실제 데이터 상에서 함수를 시험해보기 위해 [익스피디아 개발자 사이트](https://developer.ean.com)에서 제공한 뉴욕 주 내 모든 호텔 좌표가 포함된 데이터셋을 사용했다. 각 호텔과 표본 좌표(NYC의 [Brooklyn Superhero Supply Store](https://www.superherosupplies.com)라는 환상적인 소상점) 사이 거리를 계산했다.
  
이 블로그에서 사용한 함수를 포함해 데이터셋 및 주피터 노트북을 [여기서](https://github.com/s-heisler/pycon2017-optimizing-pandas) 다운로드 받을 수 있다.
  
이 게시물은 [여기서](https://www.youtube.com/watch?v=HN5d490_KKk) 볼 수 있는 파이콘 토크를 기반으로 한다.
  
* * *
  
## 판다스에서 단순 반복, 절대 하지 말아야 할 일
  
먼저 판다스 데이터 구조의 기본부터 빠르게 살펴보겠다. 기본적인 판다스 구조는 **데이터프레임**과 **시리즈**라는 두 가지 형태로 제공된다. 데이터프레임은 축 레이블이 있는 2차원 **배열**이다. 즉, 데이터프레임은 열에는 열 이름이, 행에는 색인 레이블이 붙어있는 행렬이다. 판다스 데이터프레임의 단일 열 또는 행은 판다스 시리즈 즉, 축 레이블이 있는 1차원 배열이다.
  
나와 함께 일했던 판다스 초보자 거의 대부분은 데이터프레임 행을 하나씩 반복하며 사용자 지정 함수를 적용하려고 했다. 이 접근법의 장점은 다른 반복 가능한 파이썬 객체와 상호 작용하는 방식, 예컨대 리스트나 튜플에 대해 반복하는 방식과 동일하다는 점이다. 반대로 단점은 판다스의 단순 반복이 아무 짝에 쓸모 없는 가장 느린 방법이라는 점이다. 아래에서 논의할 접근법과 달리 판다스의 단순 반복은 내장된 최적화를 이용하지 않기 때문에 극히 비효율적(그리고 종종 읽기도 쉽지 않다)이다.
  
예를 들어 다음과 같이 작성할 수 있다.
  
```python
# 모든 행을 수동으로 반복하며 거리의 시리즈를 반환하는 함수를 정의함
def haversine_looping(df):
    distance_list = []
    for i in range(0, len(df)):
        d = haversine(40.671, -73.985, df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        distance_list.append(d)
    return distance_list
 ```
    
(번역 중)