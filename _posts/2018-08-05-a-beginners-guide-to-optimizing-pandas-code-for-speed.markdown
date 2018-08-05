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
  
예제 함수로 [Haversine](https://en.wikipedia.org/wiki/Haversine_formula)(또는 Great Circle) 거리 수식을 사용하겠다. 이 함수는 두 점의 위도와 경도를 취하여 지구 곡률을 조정하고 그 사이의 직선 거리를 계산한다. 함수는 다음과 같다.
  
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
  
(번역 중)
