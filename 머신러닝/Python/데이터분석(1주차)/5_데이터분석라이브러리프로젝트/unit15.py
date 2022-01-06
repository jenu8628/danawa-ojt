import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 15-1 위키피디아 데이터 엑셀로 저장하기
# 헤더 : 열 이름, 인덱스 : 나라이름
df = pd.read_html('https://en.wikipedia.org/wiki/All-time_Olympic_Games_medal_table', header=0, index_col=0)
# iloc: 데이터의 순서에 따라 접근하는 슬라이싱 방식, 첫번째는 행, 두번째는 열
summer = df[1].iloc[:, :5]
summer.columns = ['경기수', '금', '은', '동', '계']

# sort_values() 함수 이용
# ascending - True: 오름차순, False: 내림차순
print(summer.sort_values('금', ascending=False))

# 엑셀 파일 저장
# summer.to_excel('하계올림픽메달.xlsx')

# 15-3 데이터 프레임 기초
# 행을 구분해주는 인덱스, 열을 구분해주는 컬럼
# 별도로 지정하지 않으면 인덱스는 리스트처럼 정수로 설정이 됨
# 한 번 설정된 인덱스는 변경되지 않는다.
index = pd.date_range('1/1/2000', periods=8)
print(index)

df = pd.DataFrame(np.random.rand(8, 3), index=index, columns=list('ABC'))
print(df)
# 열을 지정하면 series 데이터 구조(1차원 배열)로 표현됨
print(df['B'])
# 마스크 기능 가능
print(df['B'] > 0.4)
print(df[df['B'] > 0.4])

# 만약 행과 열을 뒤집으려면?? -> 전치행렬이라고 한다.
# 전치행렬의 약자 : T(transpose)
print(df[df['B'] > 0.4].T)

# 2차원 배열 형태의 데이터 프레임 연산
# 1. 행 방향 축을 기준으로 한 연산 (axis=0) default 위에서 아래 - 열 우선 계산
# 2. 열 방향 축을 기준으로 한 연산 (axis=1) 왼쪽에서 오른쪽 -> 행 우선 계산
# 어떠한 열과 다른 열이 행마다 연산 되어서 값이 나와야 한다면 or 열과 열끼리의 계산 axis = 0
# 1행을 쭉쭉 계산해서 새로운 열의 1행이 나온다면 or 행과 행끼리의 계산 axis = 1
# 헷갈리니 개념을 자주 보자
# A열의 값을 B열의 값으로 나눈 후 , 그 결과를 새로 만든 D열에 저장
index = pd.date_range('1/1/2020', periods=8)
df = pd.DataFrame(np.random.rand(8, 3), index=index, columns=list('ABC'))
df['D'] = df['A'] / df['B'] # A열의 값을 B열의 값으로 나눈 값을 D열에 저장
df['E'] = np.sum(df, axis=1)    # 행 우선 계산 값을 E열에 저장
df = df.sub(df['A'], axis=0)    # A열의 데이터를 기준으로 열 우선 계산
df = df.div(df['C'], axis=0)    # C열의 데이터를 기준으로 열 우선 계산산df.to_csv('test.csv')
# df.to_csv('test.csv')
df.head()

# 15-4 pandas로 인구 구조 분석하기
# 1. 데이터 읽어오기
df = pd.read_csv('../3_인구공공데이터/age.csv', encoding='cp949', index_col=0)
df = df.div(df['2019년02월_계_총인구수'], axis=0)    # 전체 데이터를 총인구수로 나눠서 비율로 변환
del df['2019년02월_계_총인구수'], df['2019년02월_계_연령구간인구수']
plt.rc('font', family="Malgun Gothic")
# 2 ~ 3 궁금한 지역 이름 입력받고 해당 지역의 인구 구조 저장하기
name = input('원하는 지역의 이름을 입력해주세요 : ')   # 2. 지역이름 입력
host = df.index.str.contains(name)  # 3. 해당 행을 찾아서 해당 지역의 인구 구조를 저장
df2 = df[host]

# # 4 ~ 5 궁금한 지역의 인구 구조와 가장 비슷한 인구 구조를 가진 지역 시각화하기
# # 궁금한 지역 인구 비율에서 선택 지역 인구 비율을 뺀다.
# x = df.sub(df2.iloc[0], axis=1)
# # x의 값을 제곱한다.
# y = np.power(x, 2)
# # 혹은 절댓값 이용
# # y = np.abs(x)
# # 한 행마다 모든 열의 값을 다 더한 값이 z
# z = y.sum(axis=1)
# # z를 정렬하여 5번째 까지의 인덱스(여기서는 동이름)를 뽑는다.
# i = z.sort_values().index[:5]
# # 구조가 가장 가까운 5번째 까지의 인덱스(동이름)을 데이터프레임에 넣어 전치하고
# # 그래프로 출력
# df.loc[i].T.plot()
# 위의 5과정을 한줄로 적으면
df.loc[np.power(df.sub(df2.iloc[0], axis=1), 2).sum(axis=1).sort_values().index[:5]].T.plot()
plt.show()