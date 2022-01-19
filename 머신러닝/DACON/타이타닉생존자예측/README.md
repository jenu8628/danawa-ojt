# 참고 사항

### [DACON 타이타닉 생존자 구하기](https://dacon.io/competitions/open/235539/overview/description)에 대한 데이터 분석 및 예측입니다.



### 사용 모듈

```python
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```



### 본문 코드

```python
if __name__ == "__main__":
    train = ""
    test = ""
    if len(sys.argv) == 3:
        train = sys.argv[1]
        test = sys.argv[2]
    else:
        print("usage : $ python analysis.py <train_csv> <train_csv> <train_csv> <train_csv>")
        exit(0)

    train = pd.read_csv(train)
    test = pd.read_csv(test)
    train.info()

    analyze = DataAnlysis(train, test)
    # analyze.correlation()   # 상광관계 그래프
    
```







## 데이터 분석

### train 데이터

- PassengerID : 탑승객 고유 아이디
- Survival : 탑승객 생존 유무 (0: 사망, 1: 생존)
- Pclass : 등실의 등급
- Name : 이름
- Sex : 성별
- Age : 나이
- Sibsp : 함께 탐승한 형제자매, 아내, 남편의 수
- Parch : 함께 탐승한 부모, 자식의 수
- Ticket :티켓 번호
- Fare : 티켓의 요금
- Cabin : 객실번호
- Embarked : 배에 탑승한 항구 이름 ( C = Cherbourn, Q = Queenstown, S = Southampton)



### 데이터 불러오기

```python
def read_csv(train_csv, test_csv):
    return pd.read_csv(train_csv), pd.read_csv(test_csv)
```



### 데이터 분석

- 891개의 훈련 데이터
- 결측치 존재(Age, cabin, Embarked)
  1. cabin : 정보가 너무 적으므로 제외
  2. 평균으로 채운다.
     1. 70점대 나옴.
  3. 빈 곳의 해당 컬럼 값을 예측해서 채운다.
     1. Age와 Embarked는 상관관계 그래프 상에서도 큰 상관관계를 갖지 않는 수치를 보인다.
     2. 그렇다면 Age를 먼저 학습 할 때 ->Embarked를 제외
     3. Embarked를 모델학습할 때 Age를 제외
     4. 마지막으로 survived를 예측
- 필요없는 컬럼 정리
  - PassengerId
  - Ticket
  - Cabin
  - Name
- 문자열 -> 인트로 전환
  - Sex
  - Embarked

![image-20220119142807495](README.assets/image-20220119142807495.png)



### 그래프 클래스

```python
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
```



- 상관관계 그래프

![image-20220119150644719](README.assets/image-20220119150644719.png)

![image-20220119150355912](README.assets/image-20220119150355912.png)



### 데이터 전처리

