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
  
나와 함께 일했던 판다스 초보자 거의 대부분은 데이터프레임 행을 하나씩 반복하며 사용자 지정 함수를 적용하려고 했다. 이 접근법의 장점은 다른 반복 가능한 파이썬 객체와 상호 작용하는 방식, 예컨대 리스트나 튜플에 대해 반복하는 방식과 동일하다는 점이다. 반대로 단점은 판다스의 단순 반복이 아무 짝에 쓸데없는 가장 느린 방법이라는 점이다. 아래에서 논의할 접근법과 달리 판다스의 단순 반복은 내장된 최적화를 이용하지 않기 때문에 극히 비효율적(그리고 종종 읽기도 쉽지 않다)이다.
  
예를 들어 다음과 같이 작성할 수 있다.
  
```python
# 모든 행을 수동으로 반복하며 일련의 거리를 반환하는 함수를 정의함
def haversine_looping(df):
    distance_list = []
    for i in range(0, len(df)):
        d = haversine(40.671, -73.985, df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        distance_list.append(d)
    return distance_list
 ```
   
 위 함수를 실행하는 데 걸리는 시간을 알기 위해 `%timeit` 명령을 사용했다. `%timeit`은 [주피터 노트북](http://jupyter.org) 용으로 특별히 제작한 "[매직](http://ipython.readthedocs.io/en/stable/interactive/magics.html)" 명령이다. (모든 매직 명령은 명령을 한 줄에 적용할 경우 `%` 기호로 시작하고 전체 주피터 셀에 적용하려면 `%%`로 시작한다.) `%timeit`은 함수를 여러 번 실행하고 얻은 실행 시간 평균과 표준 편차를 인쇄한다. 물론 `%timeit`으로 얻은 실행 시간은 함수를 실행시키는 시스템마다 모두 똑같진 않다. 그럼에도 동일한 시스템, 동일한 데이터셋에 대해 여러 다른 함수의 실행 시간을 비교하는데 유용할 벤치마크 도구로 제공된다.
  
 ```python
 %%timeit

# Haversine 반복 함수 실행하기
df['distance'] = haversine_looping(df)
```
  
그러면 다음과 같은 결과를 반환한다.
  
```bash
645 ms ± 31 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```
  
이 단순 반복 함수는 약 645ms가 걸리고 표준 편차는 31ms이다. 이게 빠른 것처럼 보일 수도 있지만 약 1,600개 행을 처리하는데 필요한 함수임을 고려할 때 실제로 느린 것이다. 이 불행한 상태를 어떻게 개선시킬 수 있을지 살펴보겠다.
  
* * *
  
## iterrows()를 사용한 반복
  
반복문을 돌려야할 때 iterrows() 메소드를 사용하는 건 행을 반복하기 위한 더 좋은 방법이다. iterrows()는 데이터프레임의 행을 반복하며 행 자체를 포함하는 객체에 덧붙여 각 행의 색인을 반환하는 제너레이터다. iterrows()는 판다스 데이터프레임과 함께 작동하게끔 최적화되어 있으며 표준 함수 대부분을 실행하는 데 가장 효율적인 방법은 아니지만(나중에 자세히 설명) 단순 반복보다는 상당히 개선되었다. 예제의 경우 iterrows()는 행을 수동으로 반복하는 것보다 거의 똑같은 문제를 약 4배 빠르게 해결한다.
  
```python
%%timeit

# 반복을 통해 행에 적용되는 Haversine 함수
haversine_series = []
for index, row in df.iterrows():
    haversine_series.append(haversine(40.671, -73.985, row['latitude'], row['longitude']))
df['distance'] = haversine_series
```
  
```bash
166 ms ± 2.42 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```
  
* * *
  
## apply 메소드를 사용한 더 나은 반복
  
`iterrows()`보다 더 좋은 옵션은 데이터프레임의 특정 축(행 또는 열을 의미)을 따라 함수를 적용하는 [`apply()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.apply.html) 메소드를 사용하는 것이다. `apply()`는 본질적으로 행을 반복하지만 `iterrows()`보다 훨씬 효율적이다. 예를 들어 Cython에서 이터레이터를 사용하는 것과 같이 내부 최적화를 다양하게 활용하면 훨씬 효율적이다.  
  
익명의 람다 함수를 사용하여 Haversine 함수를 각 행에 적용하면 각 행의 특정 셀을 함수 입력값으로 지정할 수 있다. 람다 함수는 판다스가 행(축 = 1)과 열(축 = 0) 중 어디로 함수를 적용할지 정할 수 있게끔 마지막에 축 매개 변수를 포함한다.
  
```python
%%timeit

# Haversine 함수를 시간 재며 어플라이함
df['distance'] = df.apply(lambda row: haversine(40.671, -73.985, row['latitude'], row['longitude']), axis=1)
```
  
```bash
90.6 ms ± 7.55 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```
  
`iterrows()`를 `apply()`로 바꾸면 함수 실행 시간이 반으로 줄어든다!  
  
실제로 함수 내 어느 부분이 얼마만큼 실행 시간 걸리는지 알기 위해 [라인 프로파일러](https://github.com/rkern/line_profiler) 도구(주피터에서 `% lprun magic` 명령)를 실행할 수 있다.  

```python
# 라인 프로파일러와 함께 행에 어플라이한 Haversine
%lprun -f haversine df.apply(lambda row: haversine(40.671, -73.985, row['latitude'], row['longitude']), axis=1)
```
  
(번역 중)
