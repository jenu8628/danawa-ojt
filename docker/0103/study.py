import pandas as pd
import numpy as np

#단순 회귀분석(Simple Linear Regression), 어떤변수(독립변수)가 다른 변수(종속변수)에 영향을 준다면 두 변수 사이에 선형 관계가 있다고 할 수 있다.
# 단순 회귀분석을 활용하면, 새로운 독립변수 X값이 주어졌을 때 거기에 대응되는 종속 변수 Y 값을 예측 가능하다.


df = pd.read_csv('./auto-mpg.csv', header=None) #파일 받아서 읽어오는 것, 경로설정 다시할 것

df.columns = ['mpg','cylinders','displacement','horsepower','weight',
            'acceleration','model year','origin','name']

print(df.head()) # 데이터 프레임 상위 5개를 보여줌, 인자에 숫자를 적어서 보여지는 갯수를 조절할 수 있다.
print(df.info()) # 데이터 프레임의 정보를 출령한다. 행과 열의 크기, 컬럼명, 컬럼을 구성하는 값의 자료형 등을 출력해줌

print(df.describe()) # 컬럼별로 데이터의 개수(count), 데이터의 평균값(mean), 표준편차(std), 4분위수(25%, 50%, 75%), 최댓값(max)들의 정보를 알 수 있다.
print(df['horsepower'].unique()) # horsepower 열의 고유값 확인

# df['컬럼명'] or df.컬럼명 특정 열을 선택
# 행의 범위를 선택 : [0:3]의 슬라이싱 방법 0~2의 행을 가져옴 and ['인덱스명':'인덱스명']의 인덱스명을 직접 넣어서 범위 선택 이때는 처음과 끝의 행이 모두 포함된 결과
df['horsepower'].replace('?', np.nan, inplace=True) # ? -> np.nan으로 변경, inplace=True : 데이터프레임에 변경된 설정으로 덮어쓰겠다는 의미
# dropna : 결측치가 하나라도 존재하는 행을 버림
# axis = 0(행 default), 1(열) or 'index'/'columns'
# how='any' row또는 / 'all' 
df.dropna(subset=['horsepower'], axis=0, inplace=True)
