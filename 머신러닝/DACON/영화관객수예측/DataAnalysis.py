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



def read_csv(train_csv, test_csv):
    return pd.read_csv(train_csv).drop(['title', 'director', 'release_time'], axis=1), pd.read_csv(test_csv).drop(['title', 'director', 'release_time'], axis=1)

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

# 학습 모델 클래스
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
        rf = RandomForestRegressor()
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
        lgbm = LGBMRegressor()
        rmse_list = []
        lgbm_pred = np.zeros((test.shape[0]))
        for tr_idx, val_idx in kf.split(self.train_input, self.train_target):
            tr_x, tr_y = self.train_input.iloc[tr_idx], self.train_target.iloc[tr_idx]
            val_x, val_y = self.train_input.iloc[val_idx], self.train_target.iloc[val_idx]

            lgbm.fit(tr_x, tr_y)

            pred = np.expm1([0 if x < 0 else x for x in lgbm.predict(val_x)])
            sub_pred = np.expm1([0 if x < 0 else x for x in lgbm.predict(self.test_input)])
            rmse = np.sqrt(mean_squared_error(val_y, pred))

            rmse_list.append(rmse)

            lgbm_pred += (sub_pred / 10)
        return lgbm_pred, rmse_list

    def xgbm(self):
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        xg = LGBMRegressor()
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


# 교차 검증 함수
def cross_validation(model, train_input, train_target):
    # model : 데이터 학습 모델
    # train_input : 학습 입력 데이터
    # train_target : 학습 정답 데이터
    # kf = KFold(n_splits=10, shuffle=True, random_state=42)
    return cross_validate(model, train_input, train_target,
                          cv= 5,
                          return_train_score=True,
                          scoring='neg_mean_squared_error',
                          n_jobs=-1)

# 속성값 중요도 검사 함수
def feature_importance(model, train_input, train_target):
    # model : 데이터 학습 모델
    # train_input : 학습 입력 데이터
    # train_target : 학습 정답 데이터
    return permutation_importance(model, train_input, train_target,
                                  n_repeats=1, n_jobs=-1)

# 예측값 반환 함수
def predict_data(model, test_input):
    # model : 학습된 모델
    # test_input : 테스트 입력 데이터
    return model.predict(test_input)

# 모델의 test 예측값 파일 생성 함수
def generate_submission(submission_path ,test_target, save_path):
    submission = pd.read_csv(submission_path)
    submission['box_off_num'] = test_target
    submission.to_csv(save_path, index=False)

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
