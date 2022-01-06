import csv
import matplotlib.pyplot as plt
import numpy as np

f = open('../3_인구공공데이터/age.csv')
data = csv.reader(f)
next(data)
data = list(data)

name = input('인구 구조가 알고 싶은 지역의 이름(읍면동 단위)을 입력해 주세요 : ')

home = []
for row in data:
    if name in row[0]:
        # for i in row[3:]:
        #     home.append(int(i))
        # 반복문 대신 numpy 이용
        # dtype은 리스트를 numpy배열로 저장할 때 데이터 타입을 정하는 옵션
        # 해당 나이의 인구 비율을 저장
        home = np.array(row[3:], dtype=int) / int(row[2])

mn = 1  # 최솟값을 저장할 변수 생성 및 초기화
result_name = ''    # 최솟값을 갖는 지역의 이름을 저장할 변수 생성 및 초기화
result = 0  # 최솟값을 갖는 지역의 연령대별 인구 비율을 저장할 배열 생성 및 초기화
for row in data:
    away = np.array(row[3:], dtype=int) / int(row[2])
    s = np.sum(abs(home - away))
    if s < mn and name not in row[0]:
        mn = s
        result_name = row[0]
        result = away
plt.rc('font', family="Malgun Gothic")
plt.title(name + ' 지역과 가장 비슷한 인구 구조를 가진 지역')
plt.plot(home, label=name)
plt.plot(result, label=result_name)
plt.legend()
plt.show()