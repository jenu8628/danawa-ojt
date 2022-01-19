# 참고 사항

### [DACON](https://dacon.io/competitions/open/235610/overview/description)에서 진행한 와인 품질 분류 학습

### 느낀점

사이킷런(sklearn), pandas, numpy라이브러리를 이용하였습니다.

처음엔 랜덤 서치, 랜덤 포레스트, 엑스트라 트리, 그레디언트 부스팅, 히스토그램 기반 그레디언트 부스팅 등 다양한 머신러닝 지도학습 알고리즘에 대입해 보았습니다.

교차 검증을 통해 측정하였는데 모든 알고리즘에 과대적합 측정치가 나와서 다른 방법을 시도해 보았습니다.

1. 랜덤 포레스트에서 하이퍼 파라미터를 변경해 가면서 과대적합을 줄이며 예측치를 높일 수 있도록 하였습니다.

2. 히스토그램 기반 그래디언트 부스팅의 하이퍼 파라미터를 변경해 가며 과대적합을 줄이는 노력을 했습니다.

3. 예측치를 원하는 quality 피쳐와 상관계수가 높은 피쳐들을 가지고 데이터 분석을 진행하는 과정을 거쳤습니다.

생각보다 예측치가 높게 나오지 않아 과대적합을 줄이고 예측점수를 높이는 방법을 생각해 보고 노력을 해야 할 것 같습니다.



# 데이터 분석

## train 데이터

- quality 품질

- fixed acidity 산도

- volatile acidity 휘발성산

- citric acid 시트르산

- residual sugar 잔당 : 발효 후 와인 속에 남아있는 당분

- chlorides 염화물

- free sulfur dioxide 독립 이산화황

- total sulfur dioxide 총 이산화황

- density 밀도

- pH 수소이온농도

- sulphates 황산염

- alcohol 도수

- type 종류

### 데이터의 종류와 특징

- 5497개 훈련 데이터

- 결측치 없음

- 입력 : quality를 제외한 나머지 특성

- 타깃 : quality

- 13개의 column - index제외 12개

### 속성간 상관관계

- index는 제외

- quality를 기준으로 보자

- alchol: 양의 상관관계

- density : 음의 상관관계

- alchol은 quality가 증가할 때 유사한 경향으로 증가

- density는 qulity가 감소할 때 유사한 경향으로 감소

```python
# 속성간 상관관계
plt.figure(figsize=(12,12))
sns.heatmap(data = train.corr(), annot=True)
plt.show()
```

![2022-01-18-10-32-27-image.png](C:\Users\admin\Desktop\study\머신러닝\DACON\와인품질분류\assets\2022-01-18-10-32-27-image.png)

![2022-01-18-16-30-45-image.png](C:\Users\admin\Desktop\study\머신러닝\DACON\와인품질분류\assets\2022-01-18-16-30-45-image.png)

### 각 변수별 분포

```python
plt.figure(figsize=(12, 12))
for i in range(1, 13):
    plt.subplot(3, 4, i)
    # train의 i번째열의 전체 행
    sns.distplot(train.iloc[:, i])
plt.tight_layout()
plt.show()
```

![2022-01-18-10-25-08-image.png](C:\Users\admin\Desktop\study\머신러닝\DACON\와인품질분류\assets\2022-01-18-10-25-08-image.png)

### train에서 각 변수와 quality 변수 사이 분포 확인

```python
fig = plt.figure(figsize=(12, 12))
for i in range(2, 13):
    plt.subplot(3, 4, i-1)
    sns.barplot(x='quality', y=train.columns[i], data=train)
plt.tight_layout()
plt.show()
```

![2022-01-18-10-50-50-image.png](C:\Users\admin\Desktop\study\머신러닝\DACON\와인품질분류\assets\2022-01-18-10-50-50-image.png)

### 품질의 분포도

```python
plt.figure(figsize=(12, 12))
sns.displot(train.iloc[:,1])
plt.show()
```

![2022-01-18-15-23-41-image.png](C:\Users\admin\Desktop\study\머신러닝\DACON\와인품질분류\assets\2022-01-18-15-23-41-image.png)
