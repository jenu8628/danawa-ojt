import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# 데이터 불러오기
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# 세트 나누기
# 훈련 세트와 테스트 세트 분배
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

# 랜덤 포레스트
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
# return_train_score : True - 검증 점수 + 훈련 세트에 대한 점수도 같이 반환
# RandomForestClassifier는 기본적으로 100개의 결정 트리를 사용
# oob_score=True : 각 결정 트리의 OOB 점수를 평균하여 출력
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 0.9973541965122431 0.8905151032797809

# 앞의 랜덤 포레스트 모델을
# 훈련 세트에 훈련한 후 특성 중요도 출력
rf.fit(train_input, train_target)
print(rf.feature_importances_)
# [0.23167441 0.50039841 0.26792718]

print(rf.oob_score_)
# 0.8934000384837406


# 엑스트라 트리
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 0.9974503966084433 0.8887848893166506

# 훈련 세트에 훈련한 후 특성 중요도 출력
et.fit(train_input, train_target)
print(et.feature_importances_)


# 그레이디언트 부스팅
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 0.8881086892152563 0.8720430147331015

# 트리의 개수를 늘리고 학습률을 증가시켜 보자!
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 0.9464595437171814 0.8780082549788999

gb.fit(train_input, train_target)
print(gb.feature_importances_)
# [0.15872278 0.68011572 0.16116151]


# 히스토그램 기반 그레이디언트 부스팅
# from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingClassifier

hgb =HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target, return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 0.9321723946453317 0.8801241948619236

hgb.fit(train_input, train_target)
# permutation_importance : 특성을 하나씩 랜덤하게 섞어서 모델의 성능이 변화하는지
# 관찰하여 어떤 특성이 중요한지를 계산
# 훈련 세트 뿐 아니라 테스트 세트에도 적용가능
# 사이킷런에서 제공하는 추정기 모델에 모두 사용 가능
# n_repeats : 랜덤하게 섞을 횟수 지정(default = 5)
result = permutation_importance(hgb, train_input, train_target,
                                n_repeats=10, random_state=42, n_jobs=-1)
print(result.importances_mean)
# [0.08876275 0.23438522 0.08027708]

result = permutation_importance(hgb, test_input, test_target,
                                n_repeats=10, random_state=42, n_jobs=-1)
print(result.importances_mean)
# [0.05969231 0.20238462 0.049     ]

print(hgb.score(test_input, test_target))
# 0.8723076923076923