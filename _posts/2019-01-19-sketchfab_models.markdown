---
layout: post
title: 암시적 행렬 분해(고전적인 ALS 방법) 소개와 LightFM을 이용한 순위 학습
date: 2019-01-19 00:00:00
author: Ethan Rosenthal
categories: Data-Science
---  
  
  
**Ethan Rosenthal의 [*Intro to Implicit Matrix Factorization: Classic ALS with Sketchfab Models 외 1편*](https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1)을 번역했습니다.**
  
  
- - -
  
지난 글에서 웹사이트 [Sketchfab](https://sketchfab.com)로부터 암시적 피드백 데이터를 수집하는 방법에 대해 설명했다. 그리고 이 데이터를 사용해 추천 시스템을 실제 구현해보겠다고 이야기했다. 자, 이제 만들어보자!
  
암시적 피드백을 위한 추천 모형을 살펴볼 때 Koren 등이 저술한 고전 논문 ["암시적 피드백 데이터셋을 위한 협업 필터링"](http://yifanhu.net/PUB/cf.pdf)(경고: pdf 링크)에서 설명한 모형으로부터 시작하는게 좋아보인다. 이 모형은 각종 문헌과 기계학습 라이브러리마다 다양하게 불리운다. 여기선 꽤 자주 사용되는 이름 중 하나인 *가중치가 적용된 제약적 행렬 분해*(WRMF)라고 부르겠다. WRMF는 암시적 행렬 분해에 있어 클래식 락 음악 같은 존재이다. 최신 유행은 아닐지도 몰라도 스타일에서 크게 벗어나지 않는다. 난 이 모형을 사용할 때마다 문제를 잘 해결하고 있다는 확신이 든다. 특히 이 모형은 합리적이며 직관적인 의미를 가지고 확장 가능하며 가장 좋은 점은 조정하기 편리하다는 것이다. 확률적 경사 하강 방법의 모형보다 훨씬 적은 수의 하이퍼 파라미터만 갖는다.
  
[명시적 피드백 행렬 분해](https://www.ethanrosenthal.com/2016/01/09/explicit-matrix-factorization-sgd-als)에 대한 과거 게시물을 떠올려보면 다음과 같은 손실 함수(편향 없는)가 있었다.
  
$$L_{exp} = \sum_{u, i \in S}(r_{ui} - \mathbf{x}^T_u \cdot \mathbf{y}_i)^2 + \lambda_{x} \sum_u {\lVert \mathbf{x}_u \rVert}^2 + \lambda_{y} \sum_i {\lVert \mathbf{y}_i \rVert}^2$$
  
여기서 \\(r_{ui}\\)는 사용자-품목 *점수* 행렬의 요소이고 \\(\mathbf{x}_u (\mathbf{y}_i)\\)는 사용자 \\(u\\)(품목 \\(i\\))의 잠재 요인이며 \\(S\\)는 고객-품목 점수의 전체 집합이다.  
WRMF는 이 손실 함수를 단순 수정 한 것이다.
  
$$L_{WRMF} = \sum_{u, i}c_{ui}(p_{ui} - \mathbf{x}^T_u \cdot \mathbf{y}_i)^2 + \lambda_{x} \sum_u {\lVert \mathbf{x}_u \rVert}^2 + \lambda_{y} \sum_i {\lVert \mathbf{y}_i \rVert}^2$$

여기서는 \\(S\\)의 요소만 합하는 대신 행렬 전체에 대해 합한다. 암시적 피드백이기 때문에 점수가 존재하지않음을 기억하라. 대신 항목에 대한 사용자의 선호도를 가지고 있다. WRMF 손실 함수에서 점수 행렬 \\(r_{ui}\\)이 선호도 행렬 \\(p_{ui}\\)로 바뀌었다. 사용자가 항목과 전혀 상호 작용하지 않았다면 \\(p_{ui} = 1\\), 그렇지 않으면 \\(p_{ui} = 0\\)이라고 가정하자.  
  
손실 함수 중 새롭게 나타난 또 다른 항은 \\(c_{ui}\\)이다. 이를 신뢰도 행렬라고 부르며 사용자 \\(u\\)가 품목 \\(i\\)에 대해 실제 선호도 \\(p_{ui}\\)를 갖는다는 사실을 얼마나 신뢰할 수 있는지 대략 설명하는 역할을 한다. 논문 중 저자가 고려하는 신뢰도 공식 중 하나는 상호 작용 횟수에 대한 선형 함수이다. 즉, 사용자가 웹사이트에서 어떤 품목을 클릭한 횟수가 \\(d_{ui}\\) 라면
  
$$c_{ui} = 1 + \alpha d_{ui}$$
  
이다. 여기서 \\(\alpha\\)는 교차검증을 통해 정해지는 하이퍼 파라미터이다. Sketchfab 데이터 사례는 이진값인 "좋아요"만 있으므로 \\(d_{ui} \in 0, 1\\)이다.
  
다시 돌아가면 WRMF는 어떤 품목과 상호 작용한 적 없는 사용자가 해당 품목을 *좋아하지* 않는다고 가정하진 않는다. WRMF는 해당 사용자가 해당 품목에 대해 부정적인 선호도를 가지고 있다고 가정하지만 신뢰도라는 하이퍼 파라미터를 통해 그 가정을 얼마나 신뢰할지 선택할 수 있다.
  
자, 이제 예전 명시적 행렬 분해 게시물처럼 이 알고리즘을 최적화하는 방법에 관한 전체적인 전개를 Latex 떡칠로 적어볼 수 있지만 다른 이들이 이미 여러 번 끝내놨다. 다음은 위대한 StackOverflow의 [답변](https://math.stackexchange.com/questions/1072451/analytic-solution-for-matrix-factorization-using-alternating-least-squares/1073170#1073170)이다. Dirac 표기법으로 전개하는 내용이 마음에 든다면 Sudeep Das [게시물](http://datamusing.info/blog/2015/01/07/implicit-feedback-and-collaborative-filtering)을 확인해라.

# WRMF 라이브러리
  
WRMF를 구현한 오픈 소스 코드는 많은 곳에서 찾을 수 있다. 교차 최소 자승법은 손실 함수를 최적화하는 가장 보편적인 방법이다. 이 방법은 확률적 경사 하강법보다 조정하기가 덜 까다롭고 모형은 [처치 곤란 병렬](https://en.wikipedia.org/wiki/Embarrassingly_parallel)로 돌릴 수 있다.
  
가장 처음 봤던 해당 알고리즘의 코드는 Chris Johnson [저장소](https://github.com/MrChrisJohnson/implicit-mf)의 것이다. 이 코드는 파이썬으로 작성되었고 희소 행렬을 멋지게 사용하여 일반적인 작업을 완료한다. Thierry Bertin-Mahieux는 이 코드를 가져 와서 파이썬 멀티 프로세싱 라이브러리를 사용하여 [병렬 처리](https://github.com/tbertinmahieux/implicit-mf)했다. 이는 정확도의 손실없이 상당한 속도 향상을 가져왔다.
  
Quora의 사람들은 [qmf](https://github.com/quora/qmf)라고 불리는 라이브러리를 가지고 나왔다. 병렬 처리된 qmf는 C ++로 짜여있다. 난 사용해보지 않았지만 아마 파이썬 멀티 프로세싱 버전보다 빠를 것이다. 마지막으로 Ben Frederickson은 순수 Cython으로 병렬 코드를 작성해 [이곳에](https://github.com/benfred/implicit) 올려놓았다. 이건 성능적인 측면에서 다른 파이썬 버전들을 납작하게 눌러버렸고 심지어 qmf보다 다소 [빠르다](https://github.com/benfred/implicit/tree/master/benchmarks)(좀 이상하지만).
  
나는 이 게시물을 위해 Ben의 라이브러리를 사용하기로 했다. 왜냐면 (1) 파이썬으로 계속 개발할 수 있고 (2) 매우 빠르기 때문이다. 라이브러리를 포크한 뒤 격자 탐색과 학습 곡선 계산을 손쉽게 수행하기 위해 알고리즘을 감싸는 조그만 클래스를 작성했다. 어떤 테스트도 해보지 않아 사용자가 위험을 직접 감수해야겠지만 [여기](https://github.com/EthanRosenthal/implicit) 내 포크를 자유롭게 체크 아웃해서 써도 된다. :)
  
# 데이터 주무르기
  
여기까지 하고 WRMF 모형을 훈련시켜서 Sketchfab 모델을 추천해보자!
  
첫 번째 단계는 데이터를 불러와서 "사용자 수" 곱하기 "항목 수" 크기의 상호 작용 행렬로 변환하는 것이다. 데이터의 각 행은 사용자가 Sketchfab 웹사이트에서 "좋아하는" 모델을 나타내며 현재 csv 형태로 저장되었다. 첫 번째 열은 모델 이름이고 두 번째 열은 고유한 모델 ID(`mid`)이며 세 번째 열은 익명화된 사용자 ID(`uid`)이다.
  
```python
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import pickle
import csv
import implicit
import itertools
import copy
plt.style.use('ggplot')
```
  
```python
df = pd.read_csv('../data/model_likes_anon.psv',
                 sep='|', quoting=csv.QUOTE_MINIMAL,
                 quotechar='\\')
df.head()
```
  
| | **modelname** | **mid** | **uid** |
|:--|:-----------------------------------|:---------------------------------|:---------------------------------|
| 0 | 3D fanart Noel From Sora no Method | 5dcebcfaedbd4e7b8a27bd1ae55f1ac3 | 7ac1b40648fff523d7220a5d07b04d9b |
| 1 | 3D fanart Noel From Sora no Method | 5dcebcfaedbd4e7b8a27bd1ae55f1ac3 | 2b4ad286afe3369d39f1bb7aa2528bc7 |
| 2 | 3D fanart Noel From Sora no Method | 5dcebcfaedbd4e7b8a27bd1ae55f1ac3 | 1bf0993ebab175a896ac8003bed91b4b |
| 3 | 3D fanart Noel From Sora no Method | 5dcebcfaedbd4e7b8a27bd1ae55f1ac3 | 6484211de8b9a023a7d9ab1641d22e7c |
| 4 | 3D fanart Noel From Sora no Method | 5dcebcfaedbd4e7b8a27bd1ae55f1ac3 | 1109ee298494fbd192e27878432c718a |
  
```python
print('중복 행 수: ' + str(df.duplicated().sum()))
print('이상하네 - 그냥 버리자')
df.drop_duplicates(inplace=True)
```

```bash
중복 행 수 155
이상하네 - 그냥 버리자
```
  
```python
df = df[['uid', 'mid']]
df.head()
```

| | **uid** | **mid** |
|:--|:---------------------------------|:---------------------------------|
| 0 | 7ac1b40648fff523d7220a5d07b04d9b | 5dcebcfaedbd4e7b8a27bd1ae55f1ac3 |
| 1 | 2b4ad286afe3369d39f1bb7aa2528bc7 | 5dcebcfaedbd4e7b8a27bd1ae55f1ac3 |
| 2 | 1bf0993ebab175a896ac8003bed91b4b | 5dcebcfaedbd4e7b8a27bd1ae55f1ac3 |
| 3 | 6484211de8b9a023a7d9ab1641d22e7c | 5dcebcfaedbd4e7b8a27bd1ae55f1ac3 |
| 4 | 1109ee298494fbd192e27878432c718a | 5dcebcfaedbd4e7b8a27bd1ae55f1ac3 |
  
```python
n_users = df.uid.unique().shape[0]
n_items = df.mid.unique().shape[0]

print('사용자 수: {}'.format(n_users))
print('모델 개수: {}'.format(n_items))
print('희소 정도: {:4.3f}%'.format(float(df.shape[0]) / float(n_users*n_items) * 100))
```
  
```bash
사용자 수: 62583
모델 개수: 28806
희소 정도: 0.035%
```
  
암시적 추천은 데이터가 희소해도 성능이 좋지만 상호 작용 행렬을 좀 더 밀집되게 만들면 훨씬 좋아질 수 있다. 좋아요가 적어도 5번 이상 있는 모델만 데이터를 수집했다. 그러나 사용자 모두가 좋아요를 적어도 5번 이상 한 건 아닐 수 있다. 좋아요를 한 모델이 5개 미만인 사용자를 날려버리자. 해당 사용자를 날려버리면 일부 모델이 또 좋아요 5개 미만으로 떨어질 수 있으므로 수렴할 때까지 사용자와 모델 날리는 작업을 앞뒤로 반복하자.
  
```python
def threshold_likes(df, uid_min, mid_min):
    n_users = df.uid.unique().shape[0]
    n_items = df.mid.unique().shape[0]
    sparsity = float(df.shape[0]) / float(n_users*n_items) * 100
    print('최초 좋아요 정보')
    print('사용자 수: {}'.format(n_users))
    print('모델 개수: {}'.format(n_items))
    print('희소 정도: {:4.3f}%'.format(sparsity))
    
    done = False
    while not done:
        starting_shape = df.shape[0]
        mid_counts = df.groupby('uid').mid.count()
        df = df[~df.uid.isin(mid_counts[mid_counts < mid_min].index.tolist())]
        uid_counts = df.groupby('mid').uid.count()
        df = df[~df.mid.isin(uid_counts[uid_counts < uid_min].index.tolist())]
        ending_shape = df.shape[0]
        if starting_shape == ending_shape:
            done = True
    
    assert(df.groupby('uid').mid.count().min() >= mid_min)
    assert(df.groupby('mid').uid.count().min() >= uid_min)
    
    n_users = df.uid.unique().shape[0]
    n_items = df.mid.unique().shape[0]
    sparsity = float(df.shape[0]) / float(n_users*n_items) * 100
    print('최종 좋아요 정보')
    print('사용자 수: {}'.format(n_users))
    print('모델 개수: {}'.format(n_items))
    print('희소 정도: {:4.3f}%'.format(sparsity))
    return df
```
  
```python
df_lim = threshold_likes(df, 5, 5)
```
  
```bash
최초 좋아요 정보
사용자 수: 62583
모델 개수: 28806
희소 정도: 0.035%
최종 좋아요 정보
사용자 수: 15274
모델 개수: 25655
희소 정도: 0.140%
```
  
좋은, 우리는 괜찮은 추천을 만들기에 적합해야 0.1 % 이상입니다. 이제 우리는 상호 작용 또는 "좋아하는"행렬에 대해 각각의 uid와 mid를 각각 행과 열로 매핑해야합니다. 이것은 파이썬 사전으로 간단하게 할 수 있습니다.
  
```python
# 맵핑 만들기
mid_to_idx = {}
idx_to_mid = {}
for (idx, mid) in enumerate(df_lim.mid.unique().tolist()):
    mid_to_idx[mid] = idx
    idx_to_mid[idx] = mid
    
uid_to_idx = {}
idx_to_uid = {}
for (idx, uid) in enumerate(df_lim.uid.unique().tolist()):
    uid_to_idx[uid] = idx
    idx_to_uid[idx] = uid
```
  
마지막 단계는 실제로 행렬을 만드는 것입니다. 너무 많은 메모리를 차지하지 않도록 스파 스 행렬을 사용합니다. 스파 스 매트릭스는 여러 형태로 제공되기 때문에 까다 롭습니다. 그리고 이들 사이에는 거대한 성능상의 상충 관계가 있습니다. 아래는 좋아요 매트릭스를 구축하는 아주 느린 방법입니다. 나는 %% timeit을 실행하려했지만 끝내기를 기다리는 지루함을 느꼈다.
  
```python
# # 이건 실행하지마!
# num_users = df_lim.uid.unique().shape[0]
# num_items = df_lim.mid.unique().shape[0]
# likes = sparse.csr_matrix((num_users, num_items), dtype=np.float64)
# for row in df_lim.itertuples():
#     likes[uid_to_idx[uid], mid_to_idx[row.mid]] = 1.0
```
  
양자 택일로, 아래는 우리가 50 만 명의 행렬을 만들고 있다고 생각하면 꽤 빠르다.
  
```python
def map_ids(row, mapper):
    return mapper[row]
```
  
```python
%%timeit
I = df_lim.uid.apply(map_ids, args=[uid_to_idx]).as_matrix()
J = df_lim.mid.apply(map_ids, args=[mid_to_idx]).as_matrix()
V = np.ones(I.shape[0])
likes = sparse.coo_matrix((V, (I, J)), dtype=np.float64)
likes = likes.tocsr()
```
  
```bash
1 loop, best of 3: 876 ms per loop
```
  
```bash
I = df_lim.uid.apply(map_ids, args=[uid_to_idx]).as_matrix()
J = df_lim.mid.apply(map_ids, args=[mid_to_idx]).as_matrix()
V = np.ones(I.shape[0])
likes = sparse.coo_matrix((V, (I, J)), dtype=np.float64)
likes = likes.tocsr()
```
  
# 교차 검증: 
  
(번역 중)
