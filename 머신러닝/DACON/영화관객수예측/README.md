



## 데이터 분석

### 데이터

- title : 영화의 제목
- distributor : 배급사
- genre : 장르
- release_time : 개봉일
- time : 상영시간(분)
- screening_rat : 상영등급
- director : 감독이름
- dir_prev_bfnum : 해당 감독이 이 영화를 만들기 전 제작에 참여한 영화에서의 평균 관객수(단 관객수가 알려지지 않은 영화 제외)
- dir_prev_num : 해당 감독이 이 영화를 만들기 전 제작에 참여한 영화의 개수(단 관객수가 알려지지 않은 영화 제외)
- num_staff : 스텝수
- num_actor : 주연배우수
- box_off_num : 관객수



### 분석

- head()를 통해 데이터를 보자

![image-20220124102224089](README.assets/image-20220124102224089.png)

- 제목은 제외하자.

- 결측치 : dir_prev_bfnum이 train에서는 330개, test에서는 136개의 결측치가 존재
- train

![image-20220124103039467](README.assets/image-20220124103039467.png)

- test

![image-20220124103051614](README.assets/image-20220124103051614.png)

- 원핫 인코딩 - distributor, genre, release_time, screening_rat, director
  - 주의점: distributor와 director, release_time을 인트형으로 바꿔서 하면 test데이터에 train데이터에는 없는 데이터가 들어있어서 결측치 문제가 발생한다. 그에 따른 방안을 생각해보자.

- 상관관계

![image-20220124110933110](README.assets/image-20220124110933110.png)

- 분포도

![image-20220124110858789](README.assets/image-20220124110858789.png)

- 장르별 영화 관객수 평균값

![image-20220124150247604](README.assets/image-20220124150247604.png)

- 배급사별 영화 관객수 중위값 기준

![image-20220124150322273](README.assets/image-20220124150322273.png)

- 순차적 코딩

![image-20220124150340193](README.assets/image-20220124150340193.png)

- 상영등급은 더미 변수로 만든다.

```python
train_input = pd.get_dummies(columns = ['screening_rat'], data = train_input)
```



### 본문 내용

```python
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_validate, KFold
from xgboost import XGBRegressor


if __name__ == "__main__":
    train_csv = ""
    test_csv = ""
    if len(sys.argv) == 3:
        train_csv = sys.argv[1]
        test_csv = sys.argv[2]
    train, test = read_csv(train_csv, test_csv)


    # 1. 데이터 분석 및 전처리
    pre = PreProcessor(train, test)
    # 1-1) 영화 배급사 전처리
    train, test = pre.pre_distributor()
    pre.train, pre.test = train, test

    # 1-2) 오브젝트형을 int형으로 전환하자.
    # 장르는 장르별 영화 관객수 평균값으로 작은 값부터 넣어주고
    # 배급사는 배급사별 영화 관객수 중위값 기준으로 하자.
    # print(train.groupby('genre').box_off_num.mean().sort_values())
    train, test = pre.object_to_int()
    pre.train, pre.test = train, test
    # print(train.info())

    train_input, train_target, test_input = pre.feature_data()
    # print(test_input.info())
    # print(train_input.info())

    # e표현 없애기
    np.set_printoptions(precision=6, suppress=True)
    # 2. 모델링 및 예측, 검사
    lr = ModelFactory(train_input, train_target, test_input)
    rf_test_target, rf_test_validation = lr.random_forest()
    # print(rf_test_target)
    # hg_test_target, hg_test_validation = lr.hist_gradient()
    # print(hg_test_target)
    lgbm_test_target, lgbm_test_validation = lr.hist_gradient()
    xgbm_test_target, xgbm_test_validation = lr.xgbm()
    print(xgbm_test_target)
    print(xgbm_test_validation)
    test_target = (rf_test_target + lgbm_test_target + xgbm_test_target) / 3
    print(pd.DataFrame(test_target).round())
    # # 3. 파일로 저장
    generate_submission('submission.csv', test_target, 'predict.csv')
```



### 데이터 불러오기

```python
def read_csv(train_csv, test_csv):
    return pd.read_csv(train_csv).drop(['title', 'director', 'release_time'], axis=1), pd.read_csv(test_csv).drop(['title', 'director', 'release_time'], axis=1)
```





### 데이터 전처리

```python
def get_dis(x):
    if 'CJ' in x or 'CGV' in x:
        return 'CJ'
    elif '쇼박스' in x:
        return '쇼박스'
    elif 'SK' in x:
        return 'SK'
    elif '리틀빅픽' in x:
        return '리틀빅픽처스'
    elif '스폰지' in x:
        return '스폰지'
    elif '싸이더스' in x:
        return '싸이더스'
    elif '에이원' in x:
        return '에이원'
    elif '마인스' in x:
        return '마인스'
    elif '마운틴픽' in x:
        return '마운틴픽처스'
    elif '디씨드' in x:
        return '디씨드'
    elif '드림팩트' in x:
        return '드림팩트'
    elif '메가박스' in x:
        return '메가박스'
    elif '마운틴' in x:
        return '마운틴'
    else:
        return x

class PreProcessor:
    def __init__(self, train, test):
        self.train = train
        self.test = test


    # 영화 배급사 전처리
    def pre_distributor(self):
        train['distributor'] = self.train.distributor.str.replace("(주)", '')
        test['distributor'] = self.test.distributor.str.replace("(주)", '')
        train['distributor'] = [re.sub(r'[^0-9a-zA-Z가-힣]', '', x) for x in train.distributor]
        test['distributor'] = [re.sub(r'[^0-9a-zA-Z가-힣]', '', x) for x in test.distributor]
        train['distributor'] = train.distributor.apply(get_dis)
        test['distributor'] = test.distributor.apply(get_dis)
        return train, test


    # 오브젝트 -> 인트형 전환
    def object_to_int(self):
        # 컬럼의 카테고리 별로 box_off_num의 평균이 낮은 순서부터 책정했음.
        train['genre_rank'] = self.train.genre.map({'뮤지컬': 1, '다큐멘터리': 2, '서스펜스': 3, '애니메이션': 4, '멜로/로맨스': 5,
                                               '미스터리': 6, '공포': 7, '드라마': 8, '코미디': 9, 'SF': 10, '액션': 11, '느와르': 12})
        test['genre_rank'] = self.test.genre.map({'뮤지컬': 1, '다큐멘터리': 2, '서스펜스': 3, '애니메이션': 4, '멜로/로맨스': 5,
                                             '미스터리': 6, '공포': 7, '드라마': 8, '코미디': 9, 'SF': 10, '액션': 11, '느와르': 12})
        # num_rank 생성
        # 배급사별 영화 관객수 중위값 기준으로 배급사 랭크 인코딩
        tr_nm_rank = self.train.groupby('distributor').box_off_num.median().reset_index(name='num_rank').sort_values(by='num_rank')
        tr_nm_rank['num_rank'] = [i + 1 for i in range(tr_nm_rank.shape[0])]
        train_data = pd.merge(train, tr_nm_rank, how='left')
        test_data = pd.merge(test, tr_nm_rank, how='left')
        test_data.fillna(0, inplace=True)
        return train_data, test_data

    def feature_data(self):
        train_target = np.log1p(self.train.box_off_num)
        train_input = self.train[['num_rank', 'time', 'num_staff', 'num_actor', 'genre_rank', 'screening_rat']]
        train_input = pd.get_dummies(columns=['screening_rat'], data=train_input)
        train_input['num_actor'] = np.log1p(train_input['num_actor'])
        test_input = self.test[['num_rank', 'time', 'num_staff', 'num_actor', 'genre_rank', 'screening_rat']]
        test_input = pd.get_dummies(columns=['screening_rat'], data=test_input)
        test_input['num_actor'] = np.log1p(test_input['num_actor'])
        return train_input, train_target, test_input
```



### 데이터 분석 클래스

```python
class DataAnalyzer:
    def __init__(self, train, test):
        self.train = train
        self.test = test

    plt.figure(figsize=(10, 10))
    # 분포도
    def hist(self):
        self.train.hist(bins=100)
        plt.show()

    # 상관관계
    def correlation(self):
        sns.heatmap(data=self.train.corr(), annot=True)
        plt.show()

    # 각 피쳐와 관객수 사이의 분포 그래프
    def distribute_between(self, feature):
        for i in range(0, 7):
            plt.subplot(3, 3, i+1)
            sns.barplot(x=feature, y=self.train.columns[i], data=self.train)
        plt.tight_layout()
        plt.show()
```





### 모델 클래스

```python
class ModelFactory:

    def __init__(self, train_input, train_target, test_input):
        # train_input : 훈련 입력 데이터
        # train_target : 훈련 정답 데이터
        self.train_input = train_input
        self.train_target = train_target
        self.test_input = test_input

    # 히스토그램 그래디언트 부스팅 학습 모델
    def hist_gradient(self):
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        hg = HistGradientBoostingRegressor()
        rmse_list = []
        hg_pred = np.zeros((test.shape[0]))
        for tr_idx, val_idx in kf.split(self.train_input, self.train_target):
            tr_x, tr_y = self.train_input.iloc[tr_idx], self.train_target.iloc[tr_idx]
            val_x, val_y = self.train_input.iloc[val_idx], self.train_target.iloc[val_idx]

            hg.fit(tr_x, tr_y)

            pred = np.expm1([0 if x < 0 else x for x in hg.predict(val_x)])
            sub_pred = np.expm1([0 if x < 0 else x for x in hg.predict(self.test_input)])
            rmse = np.sqrt(mean_squared_error(val_y, pred))

            rmse_list.append(rmse)

            hg_pred += (sub_pred / 10)
        return hg_pred, rmse_list

    # 랜덤 포레스트 학습 모델
    def random_forest(self):
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        rf = RandomForestRegressor(n_estimators=50, max_depth=16)
        rmse_list = []
        rf_pred = np.zeros((test.shape[0]))
        for tr_idx, val_idx in kf.split(self.train_input, self.train_target):
            tr_x, tr_y = self.train_input.iloc[tr_idx], self.train_target.iloc[tr_idx]
            val_x, val_y = self.train_input.iloc[val_idx], self.train_target.iloc[val_idx]

            rf.fit(tr_x, tr_y)

            pred = np.expm1([0 if x < 0 else x for x in rf.predict(val_x)])
            sub_pred = np.expm1([0 if x < 0 else x for x in rf.predict(self.test_input)])
            rmse = np.sqrt(mean_squared_error(val_y, pred))

            rmse_list.append(rmse)

            rf_pred += (sub_pred / 10)
        return rf_pred, rmse_list

    # LightGBM 모델
    def light_gbm(self):
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        lgbm = LGBMRegressor(n_estimators=50, max_depth=16 )
        rmse_list = []
        lgbm_pred = np.zeros((test.shape[0]))
        for tr_idx, val_idx in kf.split(self.train_input, self.train_target):
            tr_x, tr_y = self.train_input.iloc[tr_idx], self.train_target.iloc[tr_idx]
            val_x, val_y = self.train_input.iloc[val_idx], self.train_target.iloc[val_idx]

            lgbm.fit(tr_x,
                     tr_y,
                     eval_set=[(tr_x, tr_y), (val_x, val_y)],
                     eval_metric="rmse")

            pred = np.expm1([0 if x < 0 else x for x in lgbm.predict(val_x)])
            sub_pred = np.expm1([0 if x < 0 else x for x in lgbm.predict(self.test_input)])
            rmse = np.sqrt(mean_squared_error(val_y, pred))

            rmse_list.append(rmse)

            lgbm_pred += (sub_pred / 10)
        return lgbm_pred, rmse_list

    def xgbm(self):
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        xg = XGBRegressor(n_estimators=50, max_depth=16)
        rmse_list = []
        xg_pred = np.zeros((test.shape[0]))
        for tr_idx, val_idx in kf.split(self.train_input, self.train_target):
            tr_x, tr_y = self.train_input.iloc[tr_idx], self.train_target.iloc[tr_idx]
            val_x, val_y = self.train_input.iloc[val_idx], self.train_target.iloc[val_idx]

            xg.fit(tr_x, tr_y)

            pred = np.expm1([0 if x < 0 else x for x in xg.predict(val_x)])
            sub_pred = np.expm1([0 if x < 0 else x for x in xg.predict(self.test_input)])
            rmse = np.sqrt(mean_squared_error(val_y, pred))

            rmse_list.append(rmse)

            xg_pred += (sub_pred / 10)
        return xg_pred, rmse_list
```







### 모델 예측값 파일 생성 함수

```python
def generate_submission(submission_path ,test_target, save_path):
    submission = pd.read_csv(submission_path)
    submission['box_off_num'] = test_target
    submission.to_csv(save_path, index=False)
```



