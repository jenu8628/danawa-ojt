import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMClassifier

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, RandomizedSearchCV
from sklearn.inspection import permutation_importance

def read_csv(train_csv, test_csv):
    return pd.read_csv(train_csv), pd.read_csv(test_csv)


class DataAnlysis:
    # 데이터 분석, 그래프 클래스
    plt.figure(figsize=(12, 12))

    def __init__(self, train, test):
        self.train = train
        self.test = test

    def correlation(self):
        # 상관관계 그래프 생성
        sns.heatmap(data=self.train.corr(), annot=True)
        plt.show()


class PreProcessor:
    # 전처리 클래스
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def object_to_int(self, *features):
        # object를 int형으로 바꾼다.
        # female: 0, male: 1
        # C: 0, Q: 1, S: 2
        enc = LabelEncoder()
        for feature in features:
            enc.fit(self.train[feature])
            train[feature] = enc.transform(self.train[feature])
            test[feature] = enc.transform(self.test[feature])
        return train.drop('Cabin', axis=1), test.drop('Cabin', axis=1)

    def fill_value(self):
        # 평균으로 결측치 채우기
        train = self.train.fillna(round(self.train.mean()))
        test = self.test.fillna(round(self.test.mean()))
        return train, test

    def del_nul(self):
        # 빈값제거
        train = self.train.dropna(axis=0)
        test = self.train.dropna(axis=0)
        return train, test

    def feature_data(self, features):
        # 필요없는 속성값 제거 후 데이터 나누기
        train_target = self.train['Survived'].copy()
        train_input = self.train.drop(features + ['Survived'], axis=1)
        test_input = self.test.drop(features, axis=1)
        return train_input, train_target, test_input

    def slice_data(self, feature):
        id = self.train[self.train['Age'].isnull()]['PassengerId'].copy()
        train_target = self.train['Age'].copy().dropna().astype({'Age':'int'})
        train_input = self.train.drop(feature, axis=1).dropna().drop('Age', axis=1)
        test_input = self.train[self.train['Age'].isnull()].drop(feature + ['Age'], axis=1)
        return id, train_input, train_target, test_input


class ModelFactory:
    # 학습 모델 클래스
    def __init__(self, train_input, train_target):
        # train_input : 훈련 입력 데이터
        # train_target : 훈련 정답 데이터
        self.train_input = train_input
        self.train_target = train_target

    def hist_gradient(self):
        # 히스토그램 그래디언트 부스팅 학습 모델
        hgb = HistGradientBoostingClassifier()
        hgb.fit(self.train_input, self.train_target)
        return hgb

    def random_forest(self):
        # 랜덤 포레스트 학습 모델
        rf = RandomForestClassifier()
        rf.fit(self.train_input, self.train_target)
        return rf

    def random_search(self, params):
        # 랜덤 서치 학습 모델
        rs = RandomizedSearchCV(DecisionTreeClassifier(), params)
        rs.fit(self.train_input, self.train_target)
        return rs

    def extra_tree(self):
        # 엑스트라 트리 학습 모델
        et = ExtraTreesClassifier()
        et.fit(self.train_input, self.train_target)
        return et

    def gradient_boosting(self):
        # 그래디언트 부스팅 모델
        gb = GradientBoostingClassifier()
        gb.fit(self.train_input, self.train_target)
        return gb

    def light_gbm(self):
        lgbm = LGBMClassifier()
        lgbm.fit(self.train_input, self.train_target)
        return lgbm


def cross_validation(model, train_input, train_target):
    # 교차 검증 함수
    # model : 데이터 학습 모델
    # train_input : 학습 입력 데이터
    # train_target : 학습 정답 데이터
    return cross_validate(model, train_input, train_target,
                          return_train_score=True,
                          cv=StratifiedKFold(shuffle=True),
                          n_jobs=-1)


def feature_importance(model, train_input, train_target):
    # 속성값 중요도 검사 함수
    # model : 데이터 학습 모델
    # train_input : 학습 입력 데이터
    # train_target : 학습 정답 데이터
    return permutation_importance(model, train_input, train_target,
                                  n_repeats=1, n_jobs=-1)


def predict_data(model, test_input):
    # 예측값 반환 함수
    # model : 학습된 모델
    # test_input : 테스트 입력 데이터
    return model.predict(test_input)


def generate_submission(submission_path ,test_target, save_path):
    # 모델의 test 예측값 파일 생성 함수
    submission = pd.read_csv(submission_path)
    submission['Survived'] = test_target
    submission.to_csv(save_path, index=False)


if __name__ == "__main__":
    train_csv = ""
    test_csv = ""
    submission_path = ""
    save_path = ""
    if len(sys.argv) == 5:
        train_csv = sys.argv[1]
        test_csv = sys.argv[2]
        submission_path = sys.argv[3]
        save_path = sys.argv[4]
    else:
        print("usage : $ python analysis.py <train_csv> <train_csv> <train_csv> <train_csv>")
        exit(0)

    # 1. 데이터 읽어오기
    train, test = read_csv(train_csv, test_csv)

    # 2. 데이터 전처리 및 분석
    # 2-1) 오브젝트를 인트형으로 바꾸는 작업
    pre = PreProcessor(train, test)
    train, test = pre.object_to_int('Sex', 'Embarked')
    pre.train, pre.test = train, test

    # 2-2) 결측치 채우기
    # 2-2-1) 평균값으로 채움
    # train, test = pre.fill_value()
    # pre.train, pre.test = train, test
    # 2-2-2) 제거해서 해보자
    train, test = pre.del_nul()
    pre.train, pre.test = train, test

    # 2-3) 그래프 확인
    analyze = DataAnlysis(train, test)  # 데이터 그래프 클래스
    # analyze.correlation()   # 상관관계 그래프

    # 2-4) 필요없는 속성 제거
    features = [
        'PassengerId',
        'Name',
        'Ticket',

        # 'SibSp',
        # 'Parch',
        # 'Sex',
        # 'Pclass',
        # 'Age',
        'Fare',
        'Embarked'
    ]
    train_input, train_target, test_input = pre.feature_data(features)
    pd.set_option('display.max_columns', None)
    print(train_input.head(0))


    # 3. 모델 훈련
    lm = ModelFactory(train_input, train_target)
    model = lm.hist_gradient()
    # model = lm.light_gbm()


    # 4. 교차 검증 및 속성 중요도 파악악
    valid = cross_validation(model, train_input, train_target)  # 교차 검증
    print(np.mean(valid['train_score']), np.mean(valid['test_score']))
    imt = feature_importance(model, train_input, train_target)  # 속성 중요도 검사
    print(imt['importances_mean'])

    # 5. 예측값 만들기
    test_target = predict_data(model, test_input)


    # 6. 파일로 저장
    generate_submission(submission_path, test_target, save_path)



# 2-2-2) 결측치를 예측해보자. - 좀 어려울 거 같은데 흠...
    # Age 예측 부터 하자.
    # ID는 사용해야 하니 넣어두고
    # AGE와 결측치, 필요없는 값들을 제외하자.
    # features = [
    #     'PassengerId',
    #     'Name',
    #     'Ticket',
    #     'Cabin',
    #     'SibSp',
    #     'Parch',
    #     'Sex',
    #     # 'Pclass',
    #     # 'Age',
    #     # 'Fare',
    #     'Embarked'
    # ]
    # id, train_input, train_target, test_input = pre.slice_data(features)
    # # train_input.info()
    # # print(train_input.isnull().sum())
    # # print(train_target.isnull().sum())
    # pd.set_option('display.max_columns', None)
    # train_input.head(0)
    #
    # # 이제 예측을 해보자
    # # 모델이 안된다 왜안될까?
    # lm = ModelFactory(train_input, train_target)
    # model = lm.hist_gradient()
    # valid = cross_validation(model, train_input, train_target)  # 교차 검증
    # print(np.mean(valid['train_score']), np.mean(valid['test_score']))
    # imt = feature_importance(model, train_input, train_target)  # 속성 중요도 검사
    # print(imt['importances_mean'])