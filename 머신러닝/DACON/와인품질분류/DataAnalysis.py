import pandas as pd
import numpy as np
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_validate, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# train 데이터 - 훈련 데이터 5497개
# 결측치 없음
# quality(품질)을 제외한 나머지 특성을 가지고 훈련(입력)
# quality가 정답이 됨.(타깃)

# test 데이터 - 테스트 데이터 1000개
# 결측치 없음
# train의 콜롬 중 quality(품질)이 사라졌음.
# 사라진 quality에 대한 분류를 하면됨.
# 3~9까지 있음.

# # 속성간 상관관계
# plt.figure(figsize=(12,12))
# sns.heatmap(data=train.corr()[['quality']], annot=True)
# plt.show()

# # 변수별 분포
# plt.figure(figsize=(12, 12))
# for i in range(1, 13):
#     plt.subplot(3, 4, i)
#     # train의 i번째열의 전체 행
#     # 히스토그램, 빈도수 체크하는 것
#     sns.distplot(train.iloc[:, i])
# plt.tight_layout()
# plt.show()

# 품질 히스토 그램
# plt.figure(figsize=(12, 12))
# sns.displot(train.iloc[:,1])
# plt.show()

# # train에서 각 변수와 quality 변수 사이 분포 확인
# fig = plt.figure(figsize=(12, 12))
# for i in range(2, 13):
#     plt.subplot(3, 4, i-1)
#     # sns.barplot(x='quality', y=train.columns[i], data=train)
#     sns.scatterplot(x='quality', y=train.columns[i], data=train)
#     sns.boxplot(x='quality', y=train.columns[i], data=train)
# plt.tight_layout()
# plt.show()


# 현재 type이 오브젝트로 되어있으므로 Red = 1, white = 0로 변경
enc = LabelEncoder()
enc.fit(train['type'])
train['type'] = enc.transform(train['type'])
test['type'] = enc.transform(test['type'])
# train['type'] = train['type'].map({'white':0, 'red':1}).astype(int)
# test['type'] = test['type'].map({'white':0, 'red':1}).astype(int)

# 입력
# axis=1로 열방향 삭제
# features = ['index',
#     'fixed acidity',
#     'volatile acidity',
#     'citric acid',
#     'residual sugar',
#     'chlorides',
#     'free sulfur dioxide',
#     'total sulfur dioxide',
#     'density',
#     'pH',
#     'sulphates',
#     'alcohol',
#     'type']
features = [
    'index',
    'fixed acidity',
    # 'volatile acidity',
    'citric acid',
    'residual sugar',
    # 'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    # 'density',
    'pH',
    'sulphates',
    # 'alcohol',
    'type'
    ]
ss = StandardScaler()
train_target = train['quality'].copy()
train_input = train.drop(features + ['quality'], axis=1)
# train_input = pd.concat([train['alcohol'], train['density']], axis=1)
# 정답

# 테스트
test_input = test.drop(features, axis=1)
# test_input = pd.concat([test['alcohol'], test['density']], axis=1)
ss.fit(train_input)
train_scale = ss.transform(train_input)
test_scale = ss.transform(test_input)

pd.set_option('display.max_columns', None)
print(train_input.head(1))

# 학습
# 히스토그램 기반 그레디언트 부스팅
hgb = HistGradientBoostingClassifier(
    loss='categorical_crossentropy',
    learning_rate=0.1,
    max_iter=150,
    max_leaf_nodes=8,
    max_depth=90,
    early_stopping=True)
hgb.fit(train_scale, train_target)
scores = cross_validate(hgb, train_scale, train_target, return_train_score=True, cv=StratifiedKFold(shuffle=True), n_jobs=-1)
result = permutation_importance(hgb, train_scale, train_target, n_repeats=1, n_jobs=-1)
print(result.importances_mean)

# 엑스트라 트리
# et = ExtraTreesClassifier(n_jobs=-1)
# et.fit(train_input, train_target)
# scores = cross_validate(et, train_input, train_target, return_train_score=True, cv=StratifiedKFold(shuffle=True), n_jobs=-1)

## 그레디언트 부스팅
# gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1)
# gb.fit(train_input, train_target)
# scores = cross_validate(gb, train_input, train_target, return_train_score=True, cv=StratifiedKFold(shuffle=True), n_jobs=-1)

# # 랜덤 서치
# params = {'min_impurity_decrease': uniform(0.0001, 0.001),
#           'max_depth': randint(100, 1000),
#           'min_samples_split': randint(2, 20),
#           'min_samples_leaf': randint(1, 20)}
# gs = RandomizedSearchCV(DecisionTreeClassifier(), params, n_iter=200, n_jobs=-1)
# gs.fit(train_input, train_target)
# scores = cross_validate(gs, train_input, train_target, return_train_score=True, cv=StratifiedKFold(shuffle=True), n_jobs=-1)

# 랜덤 포레스트
# rf = RandomForestClassifier(max_depth=8, n_estimators=36, n_jobs=-1)
# scores = cross_validate(rf, train_input, train_target, return_train_score=True, cv=StratifiedKFold(shuffle=True), n_jobs=-1)
# rf.fit(train_input, train_target)
# print(rf.feature_importances_)


# lr = LogisticRegression(C=20, max_iter=5000)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

#
# submission = pd.read_csv('sample_submission.csv')
# submission['quality'] = test_target
# submission.to_csv('baseline1.csv', index=False)