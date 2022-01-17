import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# 데이터 불러오기
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# 세트 나누기
# 훈련 세트와 테스트 세트 분배
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
# 훈련 세트를 다시 훈련 세트와 검증 세트로 분배
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
# 즉 sub : 훈련 세트, val : 검증 세트, test : 테스트 세트

dt = DecisionTreeClassifier(random_state=42)
# dt.fit(sub_input, sub_target)
# print(dt.score(sub_input, sub_target))
# print(dt.score(val_input, val_target))
# 0.9971133028626413
# 0.864423076923077

# 교차 검증
from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target)
print(scores)
#{'fit_time': array([0.00700116, 0.00801253, 0.00597191, 0.00602841, 0.00600123]),
# 'score_time': array([0.00099945, 0.        , 0.00100088, 0.        , 0.        ]),
# 'test_score': array([0.86923077, 0.84615385, 0.87680462, 0.84889317, 0.83541867])}

# test_score : 검증 폴드의 점수, 여기선 5개로 나누었으니 5개의 평균이 된다.
print(np.mean(scores['test_score']))
# 0.855300214703487

# 기본적으로 회귀 모델일 경우 KFold 분할기를 사용하고
# 분류 모델일 경우 타깃 클래스를 골고루 나누기 위해 StratifiedKFold를 사용
from sklearn.model_selection import StratifiedKFold
# splitter(분할기) 지정
scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print(np.mean(scores['test_score']))
# 0.855300214703487

# 만약 훈련 세트를 섞은 후 10-폴드 교차 검증을 수행하려면
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))
# 0.8574181117533719

#
# # 그리드 서치
# from sklearn.model_selection import GridSearchCV
# # 기본 매개변수를 사용한 결정 트리 모델에서 min_impurity_decrease 매개변수의 최적값을 찾자.
# # 0.0001부터 0.0005까지 0.0001씩 증가하는 5개의 값을 시도하겠음
# params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
# # GridSearchCV 클래스에 탐색 대상 모델과 params 변수를 전달하여 그리드서치 객체를 생성
# # 그리드 서치 객체는 결정트리 모델 min_impurity_decrease 값을 바꿔가며 총 5번 실행
# # GridSearchCV의 cv 매개변수 기본값은 5
# # 따라서 min_impurity_decrease 값마다 5-폴드 교차 검증을 수행하므로
# # 결국 총 5 * 5 = 25개의 모델을 훈련!
# gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
# gs.fit(train_input, train_target)
# # 25개의 모델 중에서 검증 점수가 가장 높은 모델의 매개변수 조합으로
# # 전체 훈련 세트에서 자동으로 다시 모델을 훈련한다.
# dt = gs.best_estimator_
#
# # 검증 점수가 가장 높은 모델
# print(dt.score(train_input, train_target))
# # 0.9615162593804117
#
# # 검증 점수가 가장 높은 모델의 매개변수
# print(gs.best_params_)
# # {'min_impurity_decrease': 0.0001}
#
# # 각 매개변수에서 수행한 교차 검증의 평균 점수
# print(gs.cv_results_['mean_test_score'])
# # [0.86819297 0.86453617 0.86492226 0.86780891 0.86761605]
#
# # 가장 큰 값의 인덱스를 추출하여 확인해보기
# best_index = np.argmax(gs.cv_results_['mean_test_score'])
# print(gs.cv_results_['params'][best_index])
# # {'min_impurity_decrease': 0.0001}
#
# # min_impurity_decrease : 노드를 분할하기 위한 불순도 감소 최소량
# # max_depth : 트리의 깊이를 제한
# # min_samples_split: 노드를 나누기 위한 최소 샘플 수
# params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
#           'max_depth': range(5, 20, 1),
#           'min_samples_split': range(2, 100, 10)}
# # 교차 검증 횟수 : 9 * 15 * 10 = 1350개
# # 기본 5-폴드 교차 검증을 수행하므로 만들어지는 모델의 수는 6,750개
# gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
# gs.fit(train_input, train_target)
# print(gs.best_params_)
# # {'max_depth': 14, 'min_impurity_decrease': 0.0004, 'min_samples_split': 12}
# print(np.max(gs.cv_results_['mean_test_score']))
# # 0.8683865773302731


# 랜덤 서치
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20, 50),
          'min_samples_split': randint(2, 25),
          'min_samples_leaf': randint(1, 25)}
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)
# 검증 점수가 가장 높은 모델의 매개변수
print(gs.best_params_)
# {'max_depth': 39, 'min_impurity_decrease': 0.00034102546602601173, 'min_samples_leaf': 7, 'min_samples_split': 13}

# 교차 검증의 평균점수 중 최고값
print(np.max(gs.cv_results_['mean_test_score']))
# 0.8695428296438884

# 검증 점수가 가장 높은 모델
dt = gs.best_estimator_
print(dt.score(test_input, test_target))
# 0.86