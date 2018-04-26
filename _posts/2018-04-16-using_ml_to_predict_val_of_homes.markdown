---
layout:     post
title:      Using Machine Learning to Predict Value of Homes on Airbnb
date:       2018-04-16 00:00:00
author:     Robert Chang
categories: Data-Science
---  
  
  
**Robert Chang의 [*Using Machine Learning to Predict Value of Homes on Airbnb*](https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d)을 번역했습니다.**
  
  
- - -
  
## 서론
  
데이터 제품은 항상 Airbnb 서비스의 중요한 부분이다. 그러나 우리는 오랫동안 데이터 제품를 만드는데 비용이 많이 든다는 사실을 인지해왔다. 예를 들어 개인화 검색 순위는 게스트가 집을 더 쉽게 찾도록 도와주며 스마트 가격 정책은 호스트가 수요와 공급에 따라 더 경쟁력있는 가격을 설정할 수 있게끔 도와준다. 그러나 이 프로젝트들은 각각 데이터 과학과 공학만을 위한 시간 및 노력을 필요로 했다.
  
최근 Airbnb 기계 학습 인프라의 발전으로 새로운 기계 학습 모형을 제품으로 배포하는 비용이 크게 절감되었다. 예를 들어 ML Infra팀은 고품질의, 검증되고, 재사용 가능한 변수들을 사용자가 모형에 활용할 수 있게 범용 변수 저장소를 구축했다. 데이터 과학자는 여러 AutoML 도구를 작업 흐름에 통합시켜 모형 선택 및 성능 벤치마크를 가속화했다. 또한 ML Infra팀은 Jupyter 노트북을 Airflow 파이프라인으로 변환하는 새로운 프레임워크를 만들었다.
  
이 문서에서는 LTV 모델링, 즉, Airbnb에 올라온 집 가치를 예측하는 특정 활용 사례를 통해 이런 도구들이 어떻게 함께 동작하며 모델링 절차를 빠르게 하고 전체적인 개발 비용을 낮추는지 설명하겠다.
    
## LTV란 무엇인가?
  
전자 상거래 및 마켓플레이스 기업에서 인기있는 개념인 고객 생애 가치(LTV)는 사용자 당 고정 시간 동안 발생할 추정 가치를 뜻하며 대개 달러 단위로 측정된다.
  
Spotify와 Netflix 등 전자 상거래 회사는 LTV를 구독료 설정 등 가격 결정에 자주 사용한다. Airbnb 같은 마켓플레이스 기업에서 사용자 LTV를 알면 다양한 마케팅 채널에 예산을 효율적으로 할당하고 키워드 기반 온라인 마케팅에 대해 보다 정확한 입찰 가격을 계산하고 숙소 세그먼트를 더 잘 만들 수 있다.
  
과거 데이터를 사용해서 기존 숙소 [과거 가치를 계산](https://medium.com/swlh/diligence-at-social-capital-part-3-cohorts-and-revenue-ltv-ab65a07464e1)할 수 있지만 새 숙소 LTV를 예측하기 위해 기계 학습을 사용하여 한 걸음 더 나아갔다.
  
## LTV 모델링을 위한 기계학습 작업 흐름
  
데이터 과학자는 일반적으로 피쳐 엔지니어링, 프로토타이핑 및 모형 선택 같은 기계학습 관련 작업에 익숙하다. 그러나 모형 프로토타입을 제품으로 사용하려면 종종 데이터 과학자가 익숙하지 않은 데이터 엔지니어링 기술의 직교 집합이 필요하다.
  
![](https://aldente0630.github.io/assets/using_machine_learning_to_predict_value_of_homes_on_airbnb1.png)
  
다행히 Airbnb에는 ML 모형 제품화 뒷편으로 엔지니어링 작업을 일반화시켜주는 기계 학습 도구가 있다. 사실 이런 놀라운 도구가 없었다면 모형을 제품화하기 어려웠을거다. 게시물 나머지 부분에서 각 과제를 해결하는데 사용한 도구를 함께 제시할 것이며 네 가지 주제를 다룰 것이다.
  
* **피쳐 엔지니어링:** 관계있는 변수를 정의
* **프로토타이핑과 훈련:** 모형 프로토타입을 훈련
* **모형 선택과 평가:** 모형을 선택하고 조정
* **제품화:** 선택한 모형 프로토타입을 제품으로 바꿈
  
## 피쳐 엔지니어링
>*사용한 도구: Airbnb 내부 변수 저장소 — Zipline*
  
지도 기계 학습 프로젝트 첫 번째 단계 중 하나는 선택한 결과 변수와 상관성이 있을 관련 변수들을 정의하는 것이다. 이 과정을 피쳐 엔지니어링이라고한다. 예를 들어 LTV 예측할 때 다음 180 달력일 중 숙소를 이용할 수 있는 일자 비율 또는 동일한 시장에서 비교가능한 숙소 대비 현 숙소의 가격을 계산할 수 있다.
  
Airbnb에서 피쳐 엔지니어링은 종종 Hive 쿼리를 작성하여 처음부터 변수 만드는 걸 의미한다. 그러나 특정 도메인 지식과 비즈니스 로직이 필요하기 때문에 이 작업은 지루하고 시간이 오래 걸린다. 따라서 변수 파이프라인은 쉽게 공유하거나 재사용할 수 없는 경우가 많다. 이 작업을 보다 확장 가능하게 하기 위해 우리는 호스트, 게스트, 숙소 또는 시장 수준과 같이 다양한 세분화 수준에서 변수를 제공하는 훈련용 변수 저장소 Zipline을 개발했다.
  
이 내부 도구의 크라우드소스적 특징으로 인해 데이터 과학자는 다른 사람들이 과거 프로젝트에서 준비했던 다양한, 고품질의, 검증된 변수들을 사용할 수 있다. 원하는 변수를 사용할 수 없는 경우 사용자는 다음과 같은 변수 구성 파일을 이용하여 자신만의 변수를 만들 수 있다.
  
 ```json
source: {
  type: hive
  query:"""
    SELECT
        id_listing as listing
      , dim_city as city
      , dim_country as country
      , dim_is_active as is_active
      , CONCAT(ds, '23:59:59.999') as ts
    FROM
      core_data.dim_listings
    WHERE
      ds BETWEEN '{{ start_date }}' AND '{{ end_date }}'
  """
  dependencies: [core_data.dim_listings]
  is_snapshot: true
  start_date: 2010-01-01
}
features: {
  city: "City in which the listing is located."
  country: "Country in which the listing is located."
  is_active: "If the listing is active as of the date partition."
}
 ```
  
훈련 데이터를 구성하는데 다중형 변수가 필요한 경우 Zipline은 지능형 키값 조인을 자동으로 수행하고 뒤편으로 훈련 데이터를 채워넣는다. 숙소 LTV 모형을 위해 기존 Zipline 변수를 사용했고 나만의 변수를 일부 추가했다. 결론적으로 모형에는 다음을 포함한 변수 150개 이상이 사용되었다.
  
* **장소:** 국가, 시장, 이웃과 다양한 지리학적 변수
* **가격:** 야간 요금, 청소 비용, 유사한 숙소 대비 가격 점수
* **이용가능성:** 숙박 가능한 총 일수, 직접 예약을 막아놨던 일수 백분율
* **예약가능성:** 예약 횟수 또는 지난 X일간 예약되었던 날 수
* **품질:** 리뷰 점수, 리뷰 개수와 어메니티

![훈련 데이터 예시](https://aldente0630.github.io/assets/using_machine_learning_to_predict_value_of_homes_on_airbnb2.png)
  
예측 변수와 결과 변수가 정의되면 과거 데이터로부터 학습하게끔 모형 훈련을 시작해볼 수 있다.
  
## 프로토타이핑과 훈련
>*사용한 도구: Python 기계학습 라이브러리 — [scikit-learn](http://scikit-learn.org/stable/)*
  
위 훈련 데이터셋 예제처럼 모형 적합하기 전에 데이터 처리를 종종 해줘야한다.
  
* **결측값 대체:** 데이터 결측이 있는지 또 데이터 결측이 무작위로 발생했는지 확인해야한다. 무작위로 발생하지 않다면 왜 발생했는지 조사하고 근본 원인을 이해해야한다. 무작위로 발생했다면 결측값 대체를 수행한다.
  
* **범주형 변수 인코딩:** 문자열 값에 모형을 적합시킬 수 없기 때문에 종종 모형에선 범주를 그대로 사용할 수 없다. 범주 수가 적으면 [one-hot 인코딩](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) 사용을 고려할 수 있다. 그러나 차원 수가 높으면 [서수 인코딩](https://www.kaggle.com/general/16927)을 사용하여 각 범주의 빈도 수를 인코딩하는게 좋다.
  
이 단계에서는 사용할 변수 조합 중 무엇이 가장 적합한지 모르기 때문에 신속한 반복을 가능하게 하는 코드를 작성하는게 중요하다. [Scikit-Learn](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) 및 [Spark](https://spark.apache.org/docs/latest/ml-pipeline.html) 같은 오픈 소스 도구에서 일반적으로 사용할 수 있는 파이프라인 구조는 프로토타이핑을 위한 매우 편리한 도구다. 파이프라인을 사용하여 데이터 과학자는 변수 변환 방법과 훈련시킬 모형을 서술하는 고수준의 청사진을 지정할 수 있다. 좀 더 구체화하기 위해 LTV 모형 파이프라인 코드 스니펫은 아래와 같다.
  
```python
transforms = []

transforms.append(
    ('select_binary', ColumnSelector(features=binary))
)

transforms.append(
    ('numeric', ExtendedPipeline([
        ('select', ColumnSelector(features=numeric)),
        ('impute', Imputer(missing_values='NaN', strategy='mean', axis=0)),
    ]))
)

for field in categorical:
    transforms.append(
        (field, ExtendedPipeline([
            ('select', ColumnSelector(features=[field])),
            ('encode', OrdinalEncoder(min_support=10))
            ])
        )
    )
    
features = FeatureUnion(transforms)
```
  
고수준에서 파이프라인을 사용하여 다양한 유형의 변수에 대해 유형이 불리언, 범주형 또는 수치형인지에 따라 데이터 변환을 지정한다. [FeatureUnion](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html)은 변수를 컬럼 단위로 결합하여 최종 훈련 데이터셋을 생성한다.
  
프로토타입을 파이프라인을 통해 작성하면 [데이터 변환자](http://scikit-learn.org/stable/data_transforms.html)를 사용하여 지루한 데이터 변환을 추상화시킬 수 있다는 이점이 있다. 총괄하자면 이러한 변환을 통해 훈련 및 평가 과정에서 일관성있게 데이터가 변환되므로 프로토타입을 제품화할 때 데이터 변환에 관한 보편적인 문제를 해결할 수 있다.
  
또한 파이프라인은 모형 적합과 데이터 변환을 분리한다. 위 코드에 나와있지 않지만 데이터 과학자는 모형 적합 [추정자](http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)를 마지막에 지정하는 단계를 추가할 수 있다. 데이터 과학자는 표본 외 오차를 개선하기 위해 여러 추정자를 탐색하며 최적 모형을 선택하는 모형 선택 과정을 수행할 수 있다.
  
## 모형 선택 수행
>*사용한 도구: 다양한 [AutoML](https://medium.com/airbnb-engineering/automated-machine-learning-a-paradigm-shift-that-accelerates-data-scientist-productivity-airbnb-f1f8a10d61f8) 프레임워크*
  
이전 구문에서 언급했듯이 우리는 어떤 모형 후보가 제품화에 가장 적합한지 결정해야한다. 이러한 결정을 내리기 위해서는 모형 해석 가능성과 모형 복잡도 간의 절충점을 고려해야한다. 예를 들어, 희소 선형 모형은 해석하기 쉽지만 데이터를 일반화하기에는 복잡도가 충분하지않다. 트리 기반 모형은 비선형 패턴을 잡아낼만큼 충분히 유연하지만 해석하기가 어렵다. 이를 [편의 - 분산 절충](http://scott.fortmann-roe.com/docs/BiasVariance.html)이라고 한다.
  
![James, Witten, Hastie와 Tibshirani가 쓴 Introduction to Statistical Learning with R에서 참조한 그림](https://aldente0630.github.io/assets/using_machine_learning_to_predict_value_of_homes_on_airbnb3.png)
  
보험 또는 신용 심사와 같은 적용 사례에서 모형은 의도적으로 특정 고객을 차별하는 걸 피하는게 중요하기 때문에 모형을 해석할 수 있어야한다. 그러나 이미지 분류 같은 적용 사례에서는 해석 가능한 모형보다 성능이 뛰어난 분류기를 갖는게 훨씬 더 중요하다.
  
모형 선택에 많은 시간이 소요될 수 있으므로 다양한 [AutoML](https://medium.com/airbnb-engineering/automated-machine-learning-a-paradigm-shift-that-accelerates-data-scientist-productivity-airbnb-f1f8a10d61f8) 도구를 사용하여 수행 속도를 향상시키는 방법을 실험했다. 다양한 모형을 탐색하여 어떤 유형의 모형이 성능이 보통 우수한지 발견했다. 예를 들어, [eXtreme gradient boosted trees(XGBoost)](https://github.com/dmlc/xgboost)는 평균 응답 모형, 능선 회귀 모형 및 단일 의사 결정 트리와 같은 벤치마크 모형보다 월등히 뛰어나다는 것을 알게되었다.
  
![RMSE 비교를 통해 모형 선택 수행이 가능하다](https://aldente0630.github.io/assets/using_machine_learning_to_predict_value_of_homes_on_airbnb4.png)
  
주된 목표는 숙소 가치를 예측하는 것이었기 때문에 해석 가능성보단 유연성을 고려한 XGBoost를 최종 모형으로 사용하는 것에 이견이 없었다.
  
## 모형 프로토타입을 제품화시키기
>*사용한 도구: Airbnb의 노트북 변환 프레임워크 — ML Automator*
  
이전에 언급했듯이 제품화 파이프라인을 구축하는건 로컬 랩탑에서 프로토타입을 만드는 것과 상당히 다르다. 예를 들어 주기적으로 재훈련을 어떻게 수행할 수 있을까? 얼마나 많은 수의 예제를 효율적으로 채점합니까? 우리는 시간이 지남에 따라 모델 성능을 모니터링하는 파이프 라인을 어떻게 구축합니까?
  
Airbnb에서 우리는 Jupyter 노트북을 [Airflow](https://medium.com/airbnb-engineering/airflow-a-workflow-management-platform-46318b977fd8) 기계 학습 파이프 라인으로 자동 변환하는 **ML Automator**라는 프레임 워크를 구축했습니다. 이 프레임 워크는 이미 Python으로 프로토 타입을 작성하는 데 익숙한 데이터 과학자를 위해 특별히 설계되었으며 제한된 데이터 엔지니어링 경험을 바탕으로 자신의 모델을 프로덕션으로 가져 가고자합니다.
  
![A simplified overview of the ML Automator Framework (photo credit: Aaron Keys)](https://aldente0630.github.io/assets/using_machine_learning_to_predict_value_of_homes_on_airbnb5.png)

* 첫째, 프레임 워크는 사용자가 노트북에 모델 구성을 지정해야합니다. 이 모델 구성의 목적은 프레임 워크에 교육 테이블의 위치, 교육을 위해 할당 할 계산 리소스의 수 및 점수 계산 방법을 알려주는 것입니다.

* 또한 데이터 과학자는 특정 *적합성* 및 *변형* 기능을 작성해야합니다. fit 함수는 학습이 정확하게 수행되는 방법을 지정하며, 변환 함수는 분산 스코어링을위한 Python UDF로 래핑됩니다 (필요한 경우).
  
다음은 LTV 모델에서 적합 함수 및 변형 함수가 정의되는 방법을 보여주는 코드 스 니펫입니다. fit 함수는 프레임 워크에 XGBoost 모델이 훈련되고 이전에 정의한 파이프 라인에 따라 데이터 변환이 수행된다는 사실을 알려줍니다.
  
```python
def fit(X_train, y_train):
    import multiprocessing
    from ml_helpers.sklearn_extensions import DenseMatrixConverter
    from ml_helpers.data import split_records
    from xgboost import XGBRegressor

    global model
    
    model = {}
    n_subset = N_EXAMPLES
    X_subset = {k: v[:n_subset] for k, v in X_train.iteritems()}
    model['transformations'] = ExtendedPipeline([
                ('features', features),
                ('densify', DenseMatrixConverter()),
            ]).fit(X_subset)
    
    # apply transforms in parallel
    Xt = model['transformations'].transform_parallel(X_train)
    
    # fit the model in parallel
    model['regressor'] = XGBRegressor().fit(Xt, y_train)
        
def transform(X):
    # return dictionary
    global model
    Xt = model['transformations'].transform(X)
    return {'score': model['regressor'].predict(Xt)}
```
  
노트북이 병합되면 ML Automator는 숙련 된 모델을 [Python UDF](https://www.florianwilhelm.info/2016/10/python_udf_in_hive/)로 래핑하고 아래의 것과 같은 [Airflow](https://airflow.incubator.apache.org/) 파이프 라인을 만듭니다. 데이터 직렬화, 주기적 재 학습 스케줄링 및 분산 스코어링과 같은 데이터 엔지니어링 작업은 모두이 일괄 처리 작업의 일부로 캡슐화됩니다. 결과적으로이 프레임 워크는 데이터 과학자와 함께 모델을 생산에 투입하는 전담 데이터 엔지니어가있는 것처럼 데이터 과학자를위한 모델 개발 비용을 크게 절감합니다!

![A graph view of our LTV Airflow DAG, running in production](https://aldente0630.github.io/assets/using_machine_learning_to_predict_value_of_homes_on_airbnb6.png)

**참고**: 생산 단계를 넘어서 시간이 지남에 따라 모델 성능을 추적하거나 모델링을 위해 탄성 계산 환경을 활용하는 것과 같은 다른 주제가 있습니다. 여기서는이 게시물에서 다루지 않을 것입니다. 안심하십시오, 이들은 개발중인 모든 활동 영역입니다.

## 배운 교훈과 앞을 향한 교훈

지난 몇 달간, 데이터 과학자들은 ML Infra와 매우 밀접하게 제휴를 맺었으며이 협력을 통해 많은 훌륭한 패턴과 아이디어가 생겨났습니다. 사실, 우리는 이러한 도구가 Airbnb에서 기계 학습 모델을 개발하는 방법에 대한 새로운 패러다임을 열 것이라고 믿습니다.

* **첫째, 모델 개발 비용이 현저하게 낮습니다:** 기능 엔지니어링을위한 Zipline, 모델 프로토 타이핑을위한 파이프 라인, 모델 선택 및 벤치마킹을위한 AutoML, 그리고 최종적으로 생산 자동화를위한 ML Automator와 같은 개별 툴의 서로 다른 강점을 결합하여 개발주기를 대폭 단축했습니다.

* **둘째, 노트북 구동 디자인은 진입 장벽을 줄여줍니다:** 프레임 워크에 익숙하지 않은 데이터 과학자는 실제 사례가 과다하게 등장하는 즉시 액세스 할 수 있습니다. 프로덕션 환경에서 사용되는 노트북은 정확하고, 자체 문서화되며, 최신으로 보장됩니다. 이 디자인은 신규 사용자의 강력한 입양을 유도합니다.

* **결과적으로 팀은 ML 제품 아이디어에 더 많은 투자를 할 의향이 있습니다:** 이 글을 쓰는 시점에서 우리는 유사한 접근 방법을 통해 ML 제품 아이디어를 탐색하는 여러 다른 팀이 있습니다 : 목록 검사 대기열 우선 순위 지정, cohosts를 추가하고, 낮은 품질의 목록을 자동으로 표시합니다.

우리는이 프레임 워크의 미래와 함께 가져온 새로운 패러다임에 대해 매우 흥분하고 있습니다. 프로토 타입 제작과 생산 간 격차를 줄임으로써 우리는 데이터 과학자와 엔지니어가 종단 간 기계 학습 프로젝트를 추구하고 제품을 더 효과적으로 만들 수있게되었습니다.
