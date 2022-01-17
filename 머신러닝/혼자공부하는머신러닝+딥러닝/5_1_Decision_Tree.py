from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd


wine = pd.read_csv('https://bit.ly/wine_csv_data')

# 데이터 분석
# 각 열의 데이터 타입과 누락된 데이터가 있는지 확인
wine.info()
# 열에 대한 간략한 통계를 출력
wine.describe()

# 데이터 분류
# 넘파일 배열로 교체, 데이터(입력)와 타겟(정답)으로 분류
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
# 훈련 세트와 테스트 세트로 분류
# 훈련 세트 : 5197개, 테스트 세트 : 1300개
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

# 데이터 전처리
# StandardScaler 클래스를 사용한 훈련 세트 전처리
# 표준 점수로 변환됨
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 로지스틱 회귀 모델 훈련
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
# 0.7808350971714451
# 0.7776923076923077
# 과소적합이라 볼 수 있다.
print(lr.coef_, lr.intercept_)
# [[ 0.51270274  1.6733911  -0.68767781]] [1.81777902]


dt = DecisionTreeClassifier()
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

# plt.figure(figsize=(10, 7))
# plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
# plt.show()

# max_depth로 가지치기
# 표준화 전처리가 필요없다.
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

# 특성 중요도
print(dt.feature_importances_)