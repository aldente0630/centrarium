---
layout: post
title: 암시적 행렬 분해(고전적인 ALS 방법) 소개와 LightFM을 이용한 순위 학습
date: 2019-01-19 00:00:00
author: Ethan Rosenthal
categories: Data-Science
---  
  
  
**Ethan Rosenthal의 [*Intro to Implicit Matrix Factorization: Classic ALS with Sketchfab Models 외 1편*](https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1)을 번역했습니다.**
  
  
- - -

# 암시적 행렬 분해 소개: Sketchfab 모델에 적용한 고전적인 ALS 방법
  
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

## WRMF 라이브러리
  
WRMF를 구현한 오픈 소스 코드는 많은 곳에서 찾을 수 있다. 교차 최소 자승법은 손실 함수를 최적화하는 가장 보편적인 방법이다. 이 방법은 확률적 경사 하강법보다 조정하기가 덜 까다롭고 모형은 [처치 곤란 병렬](https://en.wikipedia.org/wiki/Embarrassingly_parallel)로 돌릴 수 있다.
  
가장 처음 봤던 해당 알고리즘의 코드는 Chris Johnson [저장소](https://github.com/MrChrisJohnson/implicit-mf)의 것이다. 이 코드는 파이썬으로 작성되었고 희소 행렬을 멋지게 사용하여 일반적인 작업을 완료한다. Thierry Bertin-Mahieux는 이 코드를 가져 와서 파이썬 멀티 프로세싱 라이브러리를 사용하여 [병렬 처리](https://github.com/tbertinmahieux/implicit-mf)했다. 이는 정확도의 손실없이 상당한 속도 향상을 가져왔다.
  
Quora의 사람들은 [qmf](https://github.com/quora/qmf)라고 불리는 라이브러리를 가지고 나왔다. 병렬 처리된 qmf는 C++로 짜여있다. 난 사용해보지 않았지만 아마 파이썬 멀티 프로세싱 버전보다 빠를 것이다. 마지막으로 Ben Frederickson은 순수 Cython으로 병렬 코드를 작성해 [이곳에](https://github.com/benfred/implicit) 올려놓았다. 이건 성능적인 측면에서 다른 파이썬 버전들을 납작하게 눌러버렸고 심지어 qmf보다 다소 [빠르다](https://github.com/benfred/implicit/tree/master/benchmarks)(좀 이상하지만).
  
나는 이 게시물을 위해 Ben의 라이브러리를 사용하기로 했다. 왜냐면 (1) 파이썬으로 계속 개발할 수 있고 (2) 매우 빠르기 때문이다. 라이브러리를 포크한 뒤 격자 탐색과 학습 곡선 계산을 손쉽게 수행하기 위해 알고리즘을 감싸는 조그만 클래스를 작성했다. 어떤 테스트도 해보지 않아 사용자가 위험을 직접 감수해야겠지만 [여기](https://github.com/EthanRosenthal/implicit) 내 포크를 자유롭게 체크 아웃해서 써도 된다. :)
  
## 데이터 주무르기
  
여기까지 하고 WRMF 모형을 훈련시켜서 Sketchfab 모델을 추천해보자!
  
첫 번째 단계는 데이터를 불러와서 "사용자 수" 곱하기 "항목 수" 크기의 상호 작용 행렬로 변환하는 것이다. 데이터의 각 행은 사용자가 Sketchfab 웹사이트에서 "좋아요"를 누른 모델을 나타내며 현재 csv 형태로 저장되었다. 첫 번째 열은 모델 이름이고 두 번째 열은 고유한 모델 ID(`mid`)이며 세 번째 열은 익명화된 사용자 ID(`uid`)이다.
  
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
  
암시적 피드백 데이터가 희소해도 추천 성능은 괜찮게 나올테지만 상호작용 행렬을 밀집되게 만들면 더 좋아질 수 있다. 좋아요가 최소 5번 이상 있는 모델만 데이터를 수집했다. 그러나 사용자 전부가 좋아요를 최소 5번 이상 누른 건 아닐 수 있다. 좋아요를 누른 모델이 5개 미만인 사용자를 날려버리자. 해당 사용자를 날려버리면서 일부 모델이 좋아요 5개 미만으로 다시 떨어질 수 있다. 수렴할 때까지 사용자와 모델 날리는 작업을 교대로 반복하자.
  
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
  
좋다, 희소 정도가 0.1% 이상이므로 괜찮은 추천을 하기에 적당하다. 이제 상호작용 또는 "좋아요" 행렬을 위해 각각의 `uid`와 `mid`를 상응하는 행과 열로 매핑해야한다. 이 작업은 파이썬 딕셔너리로 간단하게 할 수 있다.
  
```python
# 매핑 만들기
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
  
마지막 단계는 실제로 행렬을 만드는 작업이다. 메모리를 너무 많이 차지하지 않도록 희소 행렬을 사용한다. 희소 행렬은 여러 형태로 제공되기 때문에 좀 까다롭다. 그리고 이들 간에 성능 상 어마어마한 트레이드 오프가 있다. 아래는 좋아요 행렬을 구축하는 아주 느린 방법이다. `%%timeit`을 돌렸지만 실행 완료를 기다리자니 지루해졌다.
  
```python
# # 이건 실행하지말자!
# num_users = df_lim.uid.unique().shape[0]
# num_items = df_lim.mid.unique().shape[0]
# likes = sparse.csr_matrix((num_users, num_items), dtype=np.float64)
# for row in df_lim.itertuples():
#     likes[uid_to_idx[uid], mid_to_idx[row.mid]] = 1.0
```
  
그 대신 아래 것은 50만 명의 행렬을 만들고 있다고 생각하면 꽤 빠르다.
  
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
  
```python
I = df_lim.uid.apply(map_ids, args=[uid_to_idx]).as_matrix()
J = df_lim.mid.apply(map_ids, args=[mid_to_idx]).as_matrix()
V = np.ones(I.shape[0])
likes = sparse.coo_matrix((V, (I, J)), dtype=np.float64)
likes = likes.tocsr()
```
  
## 교차 검증: 데이터 분할
  
자, 좋아요 행렬을 훈련과 시험 행렬로 분할할 필요가 있다. 이를 다소 교묘하게(이렇게 한 단어로 끝낼 문제일까?) 진행했다. 최적화 측정 단위로 precision@k를 이용할 생각이다. k는 5 정도가 좋을 것 같다. 그러나 일부 사용자의 경우 훈련 쪽에서 시험 쪽으로 5개 품목을 이동시키면 훈련셋에 데이터가 남아 있지 않을 수 있다(사람마다 좋아요가 최소 5개임을 기억하자). 따라서 train_test_split은 데이터 일부를 시험셋으로 이동시키기 전에 좋아요가 적어도 2\*k(이 경우 10개) 이상인 사람들을 찾는다. 교차 검증은 좋아요가 많은 사용자 쪽으로 편향될 것이 분명하다. 그래도 가보자.
  
```python
def train_test_split(ratings, split_count, fraction=None):
    """
    추천 데이터를 훈련과 시험셋으로 분할하기
    
    파라미터
    ------
    ratings: scipy.sparse 행렬
        고객과 품목 간의 상호작용.
    split_count: 정수
        훈련셋에서 시험셋으로 이동시킬 고객 당 고객-품목-상호작용 갯수.
    fractions: 부동소숫점
        상호작용 일부를 시험셋으로 분할시킬 사용자 비율. 만약 None이면 사용자 전체를 고려한다.
    """
    # 참고: 아래 작업을 하기 위한 가장 빠른 방법은 아닐 것이다.
    train = ratings.copy().tocoo()
    test = sparse.lil_matrix(train.shape)
    
    if fraction:
        try:
            user_index = np.random.choice(
                np.where(np.bincount(train.row) >= split_count * 2)[0], 
                replace=False,
                size=np.int32(np.floor(fraction * train.shape[0]))
            ).tolist()
        except:
            print(('상호작용 개수가 {} 넘는 사용자로'
                  '{} 비율 채우기 어려움')\
                  .format(2*k, fraction))
            raise
    else:
        user_index = range(train.shape[0])
        
    train = train.tolil()

    for user in user_index:
        test_ratings = np.random.choice(ratings.getrow(user).indices, 
                                        size=split_count, 
                                        replace=False)
        train[user, test_ratings] = 0.
        # 점수의 경우 지금은 단지 1.0이다.
        test[user, test_ratings] = ratings[user, test_ratings]
   
    
    # 시험셋과 훈련셋은 절대 겹치지 말아야한다.
    assert(train.multiply(test).nnz == 0)
    return train.tocsr(), test.tocsr(), user_index
```
   
```python
train, test, user_index = train_test_split(likes, 5, fraction=0.2)
```
  
## 교차 검증: 격자 탐색
  
이제 데이터가 훈련 및 시험 행렬로 분할되었으므로 거대한 격자 탐색을 실행하여 하이퍼 매개 변수를 최적화하자. 최적화시킬 매개 변수는 4개 있다.
  
1.`num_factors`: 잠재적 요인의 개수 또는 모형이 갖는 차원의 정도.  
2.`regularization`: 사용자 및 품목 요인에 대한 정규화 척도.  
3.`alpha`: 신뢰도 척도 항목.  
4.`iterations`: 교대 최소 자승법를 통한 최적화 수행 시 반복 횟수.  
  
평균 제곱 오차(MSE)와 k까지의 정밀도(p@k)를 따라가며 확인할 생각이지만 둘 중 후자에 주로 신경을 쓸 것이다. 측정 단위 계산을 돕고 훈련 로그를 멋지게 출력하기 위해 몇 가지 함수를 아래에 작성했다. 여러 다른 하이퍼 파라미터 조합에 대해 일련의 학습 곡선(즉, 훈련 과정의 각 단계마다 성능 측정 단위로 평가)을 계산할 것이다. scikit-learn에 감사한다. 오픈소스이기에 기본적으로 GridSearchCV 코드를 베껴서 만들었다.
  
```python
from sklearn.metrics import mean_squared_error
def calculate_mse(model, ratings, user_index=None):
    preds = model.predict_for_customers()
    if user_index:
        return mean_squared_error(ratings[user_index, :].toarray().ravel(),
                                  preds[user_index, :].ravel())
    
    return mean_squared_error(ratings.toarray().ravel(),
                              preds.ravel())
```
  
```python
def precision_at_k(model, ratings, k=5, user_index=None):
    if not user_index:
        user_index = range(ratings.shape[0])
    ratings = ratings.tocsr()
    precisions = []
    # 참고: 아래 코드는 대량의 데이터셋인 경우 실행이 불가능할 수 있다.
    predictions = model.predict_for_customers()
    for user in user_index:
        # 대량의 데이터 셋인 경우 아래와 같이 예측을 행 단위로 계산해라.
        # predictions = np.array([model.predict(row, i) for i in xrange(ratings.shape[1])])
        top_k = np.argsort(-predictions[user, :])[:k]
        labels = ratings.getrow(user).indices
        precision = float(len(set(top_k) & set(labels))) / float(k)
        precisions.append(precision)
    return np.mean(precisions)        
```
  
```python
def print_log(row, header=False, spacing=12):
    top = ''
    middle = ''
    bottom = ''
    for r in row:
        top += '+{}'.format('-'*spacing)
        if isinstance(r, str):
            middle += '| {0:^{1}} '.format(r, spacing-2)
        elif isinstance(r, int):
            middle += '| {0:^{1}} '.format(r, spacing-2)
        elif isinstance(r, float):
            middle += '| {0:^{1}.5f} '.format(r, spacing-2)
        bottom += '+{}'.format('='*spacing)
    top += '+'
    middle += '|'
    bottom += '+'
    if header:
        print(top)
        print(middle)
        print(bottom)
    else:
        print(middle)
        print(top)    
```
  
```python
def learning_curve(model, train, test, epochs, k=5, user_index=None):
    if not user_index:
        user_index = range(train.shape[0])
    prev_epoch = 0
    train_precision = []
    train_mse = []
    test_precision = []
    test_mse = []
    
    headers = ['epochs', 'p@k train', 'p@k test',
               'mse train', 'mse test']
    print_log(headers, header=True)
    
    for epoch in epochs:
        model.iterations = epoch - prev_epoch
        if not hasattr(model, 'user_vectors'):
            model.fit(train)
        else:
            model.fit_partial(train)
        train_mse.append(calculate_mse(model, train, user_index))
        train_precision.append(precision_at_k(model, train, k, user_index))
        test_mse.append(calculate_mse(model, test, user_index))
        test_precision.append(precision_at_k(model, test, k, user_index))
        row = [epoch, train_precision[-1], test_precision[-1],
               train_mse[-1], test_mse[-1]]
        print_log(row)
        prev_epoch = epoch
    return model, train_precision, train_mse, test_precision, test_mse
```
  
```python
def grid_search_learning_curve(base_model, train, test, param_grid,
                               user_index=None, patk=5, epochs=range(2, 40, 2)):
    """
    sklearn 격자 탐색을 보고 "영감을 얻었음"(훔쳤음)
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_search.py
    """
    curves = []
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        this_model = copy.deepcopy(base_model)
        print_line = []
        for k, v in params.items():
            setattr(this_model, k, v)
            print_line.append((k, v))

        print(' | '.join('{}: {}'.format(k, v) for (k, v) in print_line))
        _, train_patk, train_mse, test_patk, test_mse = learning_curve(this_model, train, test,
                                                                epochs, k=patk, user_index=user_index)
        curves.append({'params': params,
                       'patk': {'train': train_patk, 'test': test_patk},
                       'mse': {'train': train_mse, 'test': test_mse}})
    return curves  
```
  
아래 파라미터 격자가 엄청 거대하기 때문에 6년 된 4-코어 i5로 돌리는데 2일이나 걸렸다. 성능 측정 함수는 실제 훈련 과정보다 조금 더 느린 것으로 나타났다. 이 함수들은 간단히 병렬화될 수 있으며 나중에 작업해 볼 생각이다.

```python
param_grid = {'num_factors': [10, 20, 40, 80, 120],
              'regularization': [0.0, 1e-5, 1e-3, 1e-1, 1e1, 1e2],
              'alpha': [1, 10, 50, 100, 500, 1000]}
```
  
```python
base_model = implicit.ALS()
```
  
```python
curves = grid_search_learning_curve(base_model, train, test,
                                    param_grid,
                                    user_index=user_index,
                                    patk=5)
```
  
훈련 로그는 말도 안되게 길긴 한데 여기를 클릭해서 확인할 수 있다. 다음은 최고 결과를 수행한 출력이다.
  
```bash
alpha: 50 | num_factors: 40 | regularization: 0.1
+------------+------------+------------+------------+------------+
|   epochs   | p@k train  |  p@k test  | mse train  |  mse test  |
+============+============+============+============+============+
|     2      |  0.33988   |  0.02541   |  0.01333   |  0.01403   |
+------------+------------+------------+------------+------------+
|     4      |  0.31395   |  0.03916   |  0.01296   |  0.01377   |
+------------+------------+------------+------------+------------+
|     6      |  0.30085   |  0.04231   |  0.01288   |  0.01372   |
+------------+------------+------------+------------+------------+
|     8      |  0.29175   |  0.04231   |  0.01285   |  0.01370   |
+------------+------------+------------+------------+------------+
|     10     |  0.28638   |  0.04407   |  0.01284   |  0.01370   |
+------------+------------+------------+------------+------------+
|     12     |  0.28684   |  0.04492   |  0.01284   |  0.01371   |
+------------+------------+------------+------------+------------+
|     14     |  0.28533   |  0.04571   |  0.01285   |  0.01371   |
+------------+------------+------------+------------+------------+
|     16     |  0.28389   |  0.04689   |  0.01285   |  0.01372   |
+------------+------------+------------+------------+------------+
|     18     |  0.28454   |  0.04695   |  0.01286   |  0.01373   |
+------------+------------+------------+------------+------------+
|     20     |  0.28454   |  0.04728   |  0.01287   |  0.01374   |
+------------+------------+------------+------------+------------+
|     22     |  0.28409   |  0.04761   |  0.01288   |  0.01376   |
+------------+------------+------------+------------+------------+
|     24     |  0.28251   |  0.04689   |  0.01289   |  0.01377   |
+------------+------------+------------+------------+------------+
|     26     |  0.28186   |  0.04656   |  0.01290   |  0.01378   |
+------------+------------+------------+------------+------------+
|     28     |  0.28199   |  0.04676   |  0.01291   |  0.01379   |
+------------+------------+------------+------------+------------+
|     30     |  0.28127   |  0.04669   |  0.01292   |  0.01380   |
+------------+------------+------------+------------+------------+
|     32     |  0.28173   |  0.04650   |  0.01292   |  0.01381   |
+------------+------------+------------+------------+------------+
|     34     |  0.28153   |  0.04650   |  0.01293   |  0.01382   |
+------------+------------+------------+------------+------------+
|     36     |  0.28166   |  0.04604   |  0.01294   |  0.01382   |
+------------+------------+------------+------------+------------+
|     38     |  0.28153   |  0.04637   |  0.01295   |  0.01383   |
+------------+------------+------------+------------+------------+
```
  
최고 결과 수행 시 학습 곡선이 어떻게 생겼는지 살펴보도록 하자.
  
```python
best_curves = sorted(curves, key=lambda x: max(x['patk']['test']), reverse=True)
```
  
```python
print(best_curves[0]['params'])
max_score = max(best_curves[0]['patk']['test'])
print(max_score)
iterations = range(2, 40, 2)[best_curves[0]['patk']['test'].index(max_score)]
print('Epoch: {}'.format(iterations))
```
  
```python
print(best_curves[0]['params'])
max_score = max(best_curves[0]['patk']['test'])
print(max_score)
iterations = range(2, 40, 2)[best_curves[0]['patk']['test'].index(max_score)]
print('Epoch: {}'.format(iterations))
```
  
```bash
{'alpha': 50, 'num_factors': 40, 'regularization': 0.1}
0.0476096922069
Epoch: 22
```
  
```python
import seaborn as sns
sns.set_style('white')
fig, ax = plt.subplots()
sns.despine(fig);
plt.plot(epochs, best_curves[0]['patk']['test']);
plt.xlabel('Epochs', fontsize=24);
plt.ylabel('Test p@k', fontsize=24);
plt.xticks(fontsize=18);
plt.yticks(fontsize=18);
plt.title('Best learning curve', fontsize=30);
```
  
![그림1](https://aldente0630.github.io/assets/sketchfab_models1.png)  
  
곡선이 약간 들쭉날쭉하지만 최고 이폭인 22를 지나면 곡선이 유의하게 감소하지는 않는다. 즉, 조기 종료 사용에 너무 조심스럽지 않아도 된다(p@k가 신경써야할 유일한 측정 단위라면).
  
모든 학습 곡선을 그려볼 수 있으며 하이퍼 파라미터 차이가 *확연한* 성능 차이를 가져옴을 알 수 있다.
  
```python
all_test_patks = [x['patk']['test'] for x in best_curves]
```
  
```python
fig, ax = plt.subplots(figsize=(8, 10));
sns.despine(fig);
epochs = range(2, 40, 2)
totes = len(all_test_patks)
for i, test_patk in enumerate(all_test_patks):
    ax.plot(epochs, test_patk,
             alpha=1/(.1*i+1),
             c=sns.color_palette()[0]);
    
plt.xlabel('Epochs', fontsize=24);
plt.ylabel('Test p@k', fontsize=24);
plt.xticks(fontsize=18);
plt.yticks(fontsize=18);
plt.title('Grid-search p@k traces', fontsize=30);
```
  
![그림2](https://aldente0630.github.io/assets/sketchfab_models2.png)  
  
## 스케치 추천하기
  
모든 과정 끝에 최적 하이퍼 파라미터를 마침내 얻었다. 이제 더욱 세밀한 격자 탐색을 수행하거나 사용자와 품목 정규화 효과의 비율을 변화시키는 정도에 따라 바뀌는 결과를 살펴볼 수 있다. 그러나 2일을 또 기다리고 싶지는 않았다...
  
최적 하이퍼 파라미터를 사용하여 *모든* 데이터로 WRMF 모형을 훈련시키고 품목 대 품목 추천을 시각화해보자. 사용자 대 사용자 추천은 시각화하거나 얼마나 정확한지 감을 잡기가 다소 어렵다.
  
```python
params = best_curves[0]['params']
params['iterations'] = range(2, 40, 2)[best_curves[0]['patk']['test'].index(max_score)]
bestALS = implicit.ALS(**params)
```
  
```python
bestALS.fit(likes)
```
  
품목 대 품목 추천을 얻기 위해 `ALS` 클래스에 `predict_for_items`라는 작은 메서드를 만들었다. 이건 본질적으로 품목 벡터 모든 조합 간의 내적이다. `norm=True`(기본값)로 하면 이 내적은 각 품목 벡터의 길이로 정규화되어 코사인 유사도와 같게 된다. 이는 유사한 두 품목이 내재된 또는 잠재 공간 안에서 얼마나 유사한지 알려준다.
  
```python
def predict_for_items(self, norm=True):
  """모든 품목에 대한 품목 추천"""
  pred = self.item_vectors.dot(self.item_vectors.T)
  if norm:
      norms = np.array([np.sqrt(np.diagonal(pred))])
      pred = pred / norms / norms.T
  return pred
```
  
```python
item_similarities = bestALS.predict_for_items()
```
  
이제 일부 모델들과 그와 연관된 추천들을 시각화해서 추천 모형이 얼마나 잘 동작하는지 느껴보도록 하자. 모델의 섬네일을 가져 오기 위해 Sketchfab API에 간단히 질의하면 된다. 아래는 품목 유사도, 색인 그리고 색인-`mid` 매퍼를 사용하여 추천의 섬네일 URL 목록을 반환하는 도우미 함수이다. 첫 번째 추천은 코사인 유사도가 1인 관계로 항상 본 모델 자신임을 유의하라.
  
```python
import requests
def get_thumbnails(sim, idx, idx_to_mid, N=10):
    row = sim[idx, :]
    thumbs = []
    for x in np.argsort(-row)[:N]:
        response = requests.get('https://sketchfab.com/i/models/{}'.format(idx_to_mid[x])).json()
        thumb = [x['url'] for x in response['thumbnails']['images'] if x['width'] == 200 and x['height']==200]
        if not thumb:
            print('no thumbnail')
        else:
            thumb = thumb[0]
        thumbs.append(thumb)
    return thumbs
```
  
```python
thumbs = get_thumbnails(item_similarities, 0, idx_to_mid)
```
  
```python
print(thumbs[0])
```
  
```bash
https://dg5bepmjyhz9h.cloudfront.net/urls/5dcebcfaedbd4e7b8a27bd1ae55f1ac3/dist/thumbnails/a59f9de0148e4986a181483f47826fe0/200x200.jpeg
```
    
이제 HTML 및 핵심 IPython 기능을 사용하여 이미지를 표시할 수 있다.
  
```python
from IPython.display import display, HTML

def display_thumbs(thumbs, N=5):
    thumb_html = "<img style='width: 160px; margin: 0px; \
                  border: 1px solid black;' src='{}' />"
    images = ''
    display(HTML('<font size=5>'+'Input Model'+'</font>'))
    display(HTML(thumb_html.format(thumbs[0])))
    display(HTML('<font size=5>'+'Similar Models'+'</font>'))

    for url in thumbs[1:N+1]:
        images += thumb_html.format(url)
    display(HTML(images))
```
  
```python
# 색인 임의로 고르기
rand_model = np.random.randint(0, len(idx_to_mid))
display_thumbs(get_thumbnails(item_similarities, rand_model, idx_to_mid))
```
  
입력 모델
  
![그림3](https://aldente0630.github.io/assets/sketchfab_models3.png)  
  
유사 모델
  
![그림4](https://aldente0630.github.io/assets/sketchfab_models4.png)  
  
```python
# 또 다른 색인 임의로 고르기
rand_model = np.random.randint(0, len(idx_to_mid))
display_thumbs(get_thumbnails(item_similarities, rand_model, idx_to_mid))
```
  
입력 모델
  
![그림5](https://aldente0630.github.io/assets/sketchfab_models5.png)  
   
유사 모델
  
![그림6](https://aldente0630.github.io/assets/sketchfab_models6.png)  
  
```python
# 행운을 위해 하나 더
rand_model = np.random.randint(0, len(idx_to_mid))
display_thumbs(get_thumbnails(item_similarities, rand_model, idx_to_mid))
```
입력 모델
  
유사 모델
  
![그림8](https://aldente0630.github.io/assets/sketchfab_models8.png)  
  
추천이 완벽하진 않지만(위의 경찰차+녹색 괴물 참조) 추천 모형이 유사도를 학습한 건 분명해 보인다.
  
한 걸음 물러서서 잠시 생각해보자.
  
알고리즘은 이 모델들이 어떤 모습인지, 어떤 태그가 붙어있을지, 또는 그린이가 누구인지 아무것도 모른다. 이 알고리즘은 단지 어떤 사용자가 어떤 모델을 좋아했는지 알 뿐이다. 이런 점을 생각하면 꽤 놀랍다.
  
## 그 다음은?
  
오늘은 암시적 MF계의 클래식 락 음악인 가중치가 부여된 제약적 행렬 분해를 배웠다. 다음 번에는 순위 학습이라고 암시적 피드백 모형을 최적화하는 또 다른 방법에 대해 알아볼 것이다. 순위 학습 모형을 사용하면 모델과 사용자에 대한 추가 정보(예: 모델에 할당한 카테고리 및 태그)를 포함시킬 수 있다. 그 후에 이미지와 사전 학습된 신경망을 사용하는 비지도 추천이 이러한 방법과 어떻게 비교될 수 있는지 살펴보고 마지막으로 이 추천들을 최종 사용자에게 제공할 플라스크 앱을 제작할 것이다.
  
계속 지켜봐주길!
  
# LightFM을 이용한 Sketchfab 모델 순위 학습
  
암시적 행렬 분해를 소개하는 마지막 글이며 재미있는 것들을 다룰 것이다. 암시적 행렬 분해를 위한 또 다른 방법, 순위 학습을 살펴본 다음 라이브러리 [LightFM](http://lyst.github.io/lightfm/docs/home.html)을 사용하여 부가 정보를 추천 모형에 통합시킬 것이다. 다음으로 하이퍼 파라미터 교차 검증에 대해 [scikit-optimize](https://scikit-optimize.github.io)를 이용하여 격자 탐색보다 더 똑똑하게 해낼 것이다. 마지막으로 사용자 및 품목과 동일한 공간에 부가 정보를 내재화시켜 사용자 대 품목 그리고 품목 대 품목의 단순 추천을 넘어설 것이다. 가자!

## 과거로부터의 교훈
  
Birchbox에서 일하기 시작했을 때 암시적 피드백 행렬 분해 추천 시스템에 풍부한 고객 정보를 통합시킬 수있는 방법을 모색해야만 했다. 나는 내가 무엇을 하고 있는지 전혀 몰랐다. 그래서 구글 검색을 많이 했다. 추천 시스템에는 주로 두 가지 패러다임, 즉 인구 통계학적 고객 데이터가 있고 다른 유사 고객을 찾기 위해 이 데이터를 사용하는 방식의 내용 기반 접근과 각 사용자가 상호작용 품목들을 평가한 내용이 데이터로 있는 "점수 기반" 접근이 있기에 이는 어려운 문제였다. 나는 두 접근 방식이 결합하기 원했다.
  
조사 자료의 고전, [추천 시스템을 위한 행렬 분해 기법](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)(pdf 링크)을 읽으니 "부가적인 입력 원천" 절이 있었다. 여기에는 소위 "부가 정보"를 추천 시스템에 통합시키는 접근법이 포함되어 있다. 아이디어는 상대적으로 단순했다.(행렬 분해에 대해 머리를 부여잡게 만드는 초기 장벽에 비하면 단순하다.) 정규 행렬 분해 모형에서 사용자 \\(u\\)는 사용자의 잠재 요인(이에 대한 배경 설명은 이전 게시물들을 봐라)을 담고 있는 단일 벡터 \\(\mathbf{x}_u\\)로 표현된다. 자, 해당 사용자에 대한 인구 통계학적 정보가 다음과 같이 있다.
  
| **변수** | **값** |
|:---------|:---------|
| 성별 | 여성 |
| 연령대 | 25-34 |
| 소득 구간 | $65-79K |
  
이 변수들 각각을 "속성-공간" \\(A\\)로 원-핫-인코딩할 수 있고 각각의 속성 \\(a\\)가 잠재 벡터 \\(\mathbf{s}_a\\)를 갖는다고 가정하자. 마지막으로 "총" 사용자 벡터는 원래 사용자 벡터 \\(\mathbf{x}_u\\)에 연관된 속성 벡터들을 더한 것이라고 가정하자. 만약 \\(N(u)\\)가 사용자 \\(u\\)에 속하는 속성 집합을 나타낸다면 총 사용자 벡터는 아래와 같다.
    
$$\mathbf{x}_u + \sum_{a \in N(u)} \mathbf{s}_a$$
   
품목의 부가 정보에 대한 가정으로 동일한 집합을 만들 수 있다. 이제 추천을 통해 더 나은 결과를 얻을 수 있을 뿐더러 벡터 그리고 부가 정보 벡터 간의 유사도를 결과적으로 학습할 수 있다. 이에 대한 일반적인 개요는 Dia & Co의 [기술 블로그](https://making.dia.com)에 쓴 [게시물](https://making.dia.com/embedding-everything-for-anything2anything-recommendations-fca7f58f53ff)을 참조하라.

좋다, 접근법은 명확하다. 아마도 마지막 게시물의 암시적 피드백 목표 함수에 이것을 더하고 해를 구할 것이다. 맞을까? 글쎄, 수리적으로 풀어봤지만 불행히도 이건 기존 방식을 확장하여 풀 수 없었다. 부가 정보 벡터에 대한 집합을 사용하면 마지막 게시물의 교대 최소 자승법(ALS)이 3 방향 교대 문제가 된다. ALS 최적화에는 데이터 희소성을 이용하여 계산을 확장하는 특별한 속임수가 있다. 아아, 이 속임수는 ALS 단계에서 부가 정보 벡터를 위한 해를 구할 경우 사용할 수 없다.

이제 어떻게해야 할까?
  
## 순위 학습 - BPR
  
마지막 게시물의 목적 함수에 대한 ALS 나 SGD (Stochastic Gradient descent)가 아닌 암시 적 피드백 행렬 인수 분해 문제를 최적화하는 또 다른 방법이 있음이 밝혀졌습니다. 이 최적화 방법은 일반적으로 [Learning to Rank](https://en.wikipedia.org/wiki/Learning_to_rank)라는 이름으로 시작되며 정보 검색 이론에서 시작되었습니다.
  
묵시적인 피드백으로 순위를 매기기 위해 Learning을 사용하는 고전적인 방법은 종이 BPR에 포함되었습니다. Steffen Rendle이 처음 쓴 Implicit Feedback에서 [Bayesian Personalized Ranking](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf)(pdf 링크)은 암시적이고 인수 분해 된 모든 점에서 나쁜 점이었습니다. 이 아이디어는 양수와 음수를 샘플링하고 쌍 비교를 실행하는 것에 중점을 둡니다. 이 예에서 데이터 세트는 사용자가 웹 사이트에서 다양한 항목을 클릭 한 횟수로 구성됩니다. BPR은 다음과 같이 진행됩니다 (단순화 된 형태로).
  
1.무작위로 사용자 \\(u\\)를 선택한 다음 사용자가 클릭 한 임의 항목 \\(i\\)를 선택하십시오. 이것은 우리의 *긍정적인* 항목입니다.  
2.사용자가 항목 \\(i\\)보다 *적은* 시간에 클릭 한 항목 \\(j\\)를 임의로 선택합니다 (여기에는 클릭 한 적이없는 항목이 포함됨). 이것은 우리의 *부정적인* 항목입니다.  
3.사용자 \\(u\\)와 긍정적 인 항목 \\(i\\)에 대해 "점수", \\(p_{ui}\\)를 예측하려는 방정식을 사용하십시오. 행렬 인수 분해의 경우, 이것은 \\( \mathbf{x}_u \cdot \mathbf{y}_i \\)일 수 있습니다.  
4.사용자 \\(u\\) 및 부정적 항목 \\(j\\), \\(p_{uj}\\)의 점수를 예측합니다.  
5.양수와 음수 점수의 차이점을 찾으십시오. \\(x_{uij} = p_{ui} - p_{uj}\\)  
6.이 차이를 시그 모이 드를 통해 전달하고 확률 적 구배 강하 (SGD)를 통해 모든 모델 매개 변수를 업데이트하기위한 가중치로 사용합니다.  
  
이 방법은 내가 처음 읽었을 때 나에게 급진적 인 것처럼 보였다. 특히, 우리는 우리가 예측하고있는 점수의 실제 가치에 대해 신경 쓰지 않습니다. 우리가 신경 쓰는 것은 사용자가 클릭 한 항목보다 더 자주 클릭 한 항목보다 순위가 높은 항목입니다. 따라서 우리 모델은 "순위를 매기는 법"을 배우게됩니다. :)
  
이 모델은 샘플링 기반 접근법을 사용하기 때문에 저자는 다른 느린 방법과 비교하여 상당히 빠르고 확장 가능할 수 있음을 보여줍니다. 또한, 저자들은 BPR이 바람직한 특성 일 수있는 ROC 곡선 (AUC)에서 면적을 직접 최적화한다고 주장한다. 가장 중요한 것은 나를 위해, 계산 속도를 날려 버리지 않고 쉽게 부가 정보를 추가 할 수 있습니다.    
  
## 순위 학습 - WARP
  
BPR의 가까운 친척은 [WSABIE](http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf)에서 처음 소개 된 Weighted Approximate-Rank Pairwise loss (WARP 손실)입니다 : Weston 외의 대규모 어휘 이미지 주석 확장. al. WARP는 BPR과 매우 유사합니다. 사용자에 대해 긍정적이고 부정적인 항목을 샘플링하고, 둘 다 예측하고 차이를 가져옵니다. BPR에서는이 차이를 가중치로 사용하여 SGD 업데이트를 수행합니다. WARP에서 잘못된 예측을 한 경우에만 SGD 업데이트를 실행합니다. 즉, 제외 항목이 양수 항목보다 높은 점수를 예측합니다. 잘못 예측하지 않으면 잘못된 예측을하거나 잘못된 값을 얻을 때까지 부정적인 항목을 계속 그립니다.
  
WARP 논문의 저자는이 프로세스가 BPR의 AUC 최적화에서 정밀도 최적화로 이동한다고 주장합니다. 이 옵션은 대부분의 추천 시스템과 관련이있을 것으로 보이며 일반적으로 BPR 손실보다 WARP 손실을 최적화하는 것이 좋습니다. WARP는 두 개의 하이퍼 매개 변수를 도입합니다. 하나는 SGD 업데이트를 구현하는 예측이 얼마나 틀린지를 결정하는 마진입니다. 이 문서에서 마진은 1이므로 SGD 업데이트를 구현하기 위해 \\(p_{uj} > p_{ui} + 1\\)을 추측해야합니다. 다른 하이퍼 매개 변수는 포기하고 다음 사용자로 이동하기 전에 부정확 한 샘플을 그리려는 의도가있는 횟수를 결정하는 컷오프입니다.
  
이 샘플링 방법을 사용하면 교육을 시작할 때 훈련되지 않은 모델에 대한 예측이 잘못되기 쉽기 때문에 WARP가 빠르게 실행됩니다. 그러나 훈련을 마친 후에 WARP는 예측이 잘못되어 업데이트 할 때까지 많은 항목을 샘플링하고 예측해야하기 때문에 느리게 실행됩니다. 그래도 모델이 잘못 예측하게 만드는 것이 어렵다면 긍정적 인 문제와 같을 것입니다.
  
## LightFM
  
내 Birchbox 이야기로 돌아 가기. 나는 원래 순수한 numpy와 scipy에서 부차적 인 정보를 가진 BPR을 구현했다. 이것은 매우 느린 것으로 판명되었고, 그 즈음에 Lyst의 사람들이 LightFM 패키지를 공개했습니다. LightFM은 Cython으로 작성되었으며 HOGWILD SGD를 통해 병렬화됩니다. 이것은 내 코드를 물 밖으로 날려 버렸고 행복하게 LightFM으로 전환했습니다.
  
LightFM은 위와 동일한 방법을 사용하여 부수 정보를 통합합니다. 총 "사용자 벡터"는 사용자의 관련 부가 정보 벡터 (사용자 "기능"이라고 함)를 합한 것으로 가정하고 유사하게 취급합니다. 위의 예제에서 우리는 \\(\mathbf{x}_u\\)와 \\(\mathbf{s}_a\\)라는 두 종류의 잠복 벡터를 가지고 있다고 가정했습니다. LightFM은 모든 것을 부수적 인 정보 또는 기능으로 취급합니다. 사용자 \\(u\\)에 대해 특정 사용자 벡터를 갖기 위해서는 해당 사용자에 대해 단일 기능으로 한 - 핫 - 인코딩해야합니다.


(번역 중)
