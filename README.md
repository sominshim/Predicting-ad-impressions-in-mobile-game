# Predicting-ad-impressions-in-mobile-game

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2455f923-8fce-41fa-8a6a-b95584ee2f4e/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2455f923-8fce-41fa-8a6a-b95584ee2f4e/Untitled.png)

**Figure 1** Overview design of the proposed system development

# 1. Target Selection

⇒ tutorial event 에서 selection 진행

tutorial data 보간

- event_params_ad_id, user_properties_ad_id 합집합
⇒ event_params_ad_id 가 user_properties_ad_id를 포함함. 1row 빼고
- user_pseudo_id

**[target 대상]**

- tutorial을 모두 완료한 user
: tutorial_id = 1,2,3,4 포함 & 'event_previous_timestamp' == np.nan
- ~~(위 내용에 포함됨)~~ 신~~규 user 
: 우리가 받은 데이터기간 전에 처음 접속한 user : ad_id로 groupby한 후, tutorial의 ga_session_number/tutorial_id가 1이 없는 user~~
- day7 까지 데이터가 있는 user
: 첫 접속일이 데이터의 마지막날의 일주일 전보다 나중인 user 제거
ex) 0101 ~ 0130 → 0101 ~ 0123 ⇒ 0124에 들어온 user가 day 7 이 있을 수가 없기 때문.

~~ad_id not null - stage 연속적이지 않음 → ad_id is null - stage 매칭
ad_id not null stage 빠진 개수 = ad_id is null 개수 혹시 같으면,,??!! 두둥탁
 ⇒ knn으로 ㅋㅋㅋ~~

재설치자 (ad_id는 같지만 user_pseudo_id는 다른 user) 

재설치시 보통 user_level 이 reset 됨. 근데 reset되지 않은 user_level은 ~~ga_session_number 로 보간할 수 있음.~~

(방법 : ga_session_number가 1로 reset되는 부분부터 reset 
⇒ ga_session_number 도 항상 reset되는 것은 아님..

- 재설치 후 pseudo_id가 바뀌지 않는 경우 존재 ⇒ 얼마나 존재하는지 보자

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6ee52968-8e07-4d2b-aaa6-0b7f9832fb18/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6ee52968-8e07-4d2b-aaa6-0b7f9832fb18/Untitled.png)

재설치자 → day 지속. 즉 고려하지 않음

# 2. Split train/test

1. tutorial 로 ad_id별 day_0 추출
2. ads 의 date와 day_0로 day계산
3. day 2-7 누적 imps 값이 1 이상인 사람 = 1 / 0인 사람 = 0
→ 이 기준으로 비율맞춰 train, test split 된 total ad_id, train(1)/test(0) dataframe 생성하고 공유

# 3. Preprocessing

train에 포함된 ad_id를 가지고

각 event에서 ['device_is_limited_ad_tracking'] == 'True'인 ad_id 제거

## Step1) 엑셀시트에 나와있는 컬럼 추출

[https://drive.google.com/file/d/1w0gzVq-K8vcYnn73HCQB3zS2C8Cum_-2/view](https://drive.google.com/file/d/1w0gzVq-K8vcYnn73HCQB3zS2C8Cum_-2/view)

## Step2) 연속형, 카테고리형 데이터 파악

- 데이터 변형

    ex) 실제 데이터는 int형이지만 본질적 의미는 category형 data일 경우 바꿔서 분석

    - code

        본질적 의미 파악은 각각 해야함

        ```python
        #int str으로 변환
        event_df['column_name'] = event_df['column_name'].astype(str)
        ```

- 통계치 확인
    - 개별 feature가 어떤 분포를 보이는지
    - numeric feature :
    - count(total, null 등), mean, std, min, max 등
    - 정규분포를 보이는지 (normality), 치우쳤는지 (skewness), 특정 값에 몰려있는지 (low variance). 
       분포 모양에 따라서 transformation이 필요할 수 있음.
    - categorical feature :
    - count(total, null 등), unique value count, mode, frequency table 등
    - 한 두가지 카테고리에 집중되어 있는지 (Imbalanced), 고르게 퍼져있는지. 
       고르게 퍼진 경우가 더 많은 정보를 제공할 가능성이 높음.

## Step3) **NaN 값 처리**

데이터 세트의 값 중 몇 퍼센트가 누락되었는지 확인

```python
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
```

1. Drop missing values
    - Nan값이 아닌 Nan 의미의 값 → Nan으로 채우기
    ex) 999, 'x'

        ```python
        missing_values=['??','na','X','999999']
        df = df.replace(missing_values, np.NaN)
        ```

    - drop column (NaN > 50%)

        ```python
        NAN = [(c, df[c].isna().mean()*100) for c in df]
        NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])
        NAN = NAN[NAN.percentage > 50]
        NAN.sort_values("percentage", ascending=False)
        ```

    - 한쪽에 치우쳐져있을 경우 feature제거
    숫자형 → 왜도 standard,/ minmax / normalize/ log 변환/... 등 함수로 구현해서 적용

        카테고리형 → 분산 threshold=0.5 ? 
        [https://colab.research.google.com/drive/1TNPqL-d8a_lQx-ypSCxb3-pUXgSGdN7C?usp=sharing](https://colab.research.google.com/drive/1TNPqL-d8a_lQx-ypSCxb3-pUXgSGdN7C?usp=sharing)

    - 

2. Filling in missing values

    [누락 데이터(Missing value)를 처리하는 7가지 방법 / Data Imputation](https://dining-developer.tistory.com/19)

    - ad_id 
    user_pseudo_id 가 같은 ad_id로 대체 → drop
    - numeric feature
    → fill mean/median value
        - Mean - 이상치가 없을 경우, 평균이 이상치에 영향을 많이 받기 때문 @
        - Median - 이상치가 많을 경우

        ```python
        # df["Math"] = df["Math"].astype("float64")
        m = round(df["Math"].mean(), 2)
        df["Math"].fillna(m, inplace=True)

        import numpy as np
        from sklearn.impute import SimpleImputer
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(X)
        ```

        [sklearn.impute 사용법](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)

        참고) 데이터가 학습 데이터와 테스트 데이터로 나누어진 경우는 반드시 fit() 함수와 transform() 함수를 따로 사용

        - 💡Question

            ```python
            transformer.fit(x_train) #SimpleImputer 모델에 x_train 데이터 적용 (평균값 계산)
            x_train = transformer.transform(x_train) 
            x_test = transformer.transform(x_test)
            ```

        한쪽에 치우쳐져있을 경우 feature제거
        숫자형 → 왜도 standard,/ minmax / normalize/ log 변환/... 등 함수로 구현해서 적용

    - categorical feature
        - 빈도 높은 값으로 채우기 Mode

            ```python
            imp1 = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
            ```

        - knn모델 이용하여 예측
            - code

                ```python
                # categorical feature의 경우, mode 또는 classification model을 활용 (ex. KNN)
                df['col'].fillna(df['col'].mode()[0])

                # mode 계산 (2가지 방법은 동일함)
                mode = df[col].value_counts().index[0]
                mode = df[col].mode()[0]

                # KNN model 활용
                # model에 데이터를 넣기 위해서는 categorical value를 각각 경우에 맞는 numeric value로 변환해줘야함 (label encoding). 자세한 encoding 방법에 대해서는 아래 2.2.8에서 다룸.
                from sklearn.preprocessing import LabelEncoder

                encoder_X = LabelEncoder()
                encoder_y = LabelEncoder()

                # missing value가 전혀 없는 깨끗한 데이터셋을 발라내서 KNN 알고리즘을 학습시켜야 함
                train_X = train_X.apply(encoder_X.fit_transform)
                train_y = encoder_y.fit_transform(train_y)

                from sklearn.neighbors import KNeighborsClassifier

                knn = KNeighborsClassifier(n_neighbors=5).fit(train_X, train_y)

                # KNN 예측값으로 missing value 대체하기
                # col은 missing value가 존재하여 대체가 필요한 column, ind_cols는 missing value가 없고 유사도를 판단하기 위한 기준 column들.
                df_null_only = df[df.isnull().any(axis=1)]
                df_null_only[col] = knn.predict(df_null_only[ind_cols].apply(lambda x: encoder.fit_transform(x)))

                # 처리된 결과를 기존 데이터와 합차기
                df_final = pd.concat([train_X, df_null_only], axis=0)

                # 실제 categorical value를 확인할 수 있음.
                LabelEncoder.inverse_transform(df_null_only[col])
                ```

## Step4) Outlier 처리

1. outlier 제거

    → ad_id target_df에 outlier columns(0/1) 만들어서 제거 대상표시

2. outlier 대체

## Step5) feature engineering

[feature engineering](https://www.notion.so/feature-engineering-941f6f6e38d1481ebd5e5b3f0047d36b)

1. feature extraction 새로운 feature 생성
ex) feature들의 조합으로, time관련 등,,
2. day 계산 → groupby ad_id ga_session_id ['timestamp'].min() → session_start_date
session_start_date - day0 date (tutorial)
3. day0, 1 에 해당하는 feature 만들기 (집계)
4. Target: ads → day 2~7 누적합

---

추가 feature 생성

# 4. feature selection

1. correlation → 1차 feature selection

    ---

    merge event

    ---

2. embedded method → 2차 feature selection

# 5. Modeling

# 6. Evaluate model

# 7. Develop Model