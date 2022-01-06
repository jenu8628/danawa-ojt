import csv
import matplotlib.pyplot as plt
f = open('../1기온공공데이터/seoul.csv')
data = csv.reader(f)
next(data)

import random

# 6-1_1 hist() 사용하기
plt.hist([1, 1, 2, 3, 4, 5, 6, 6, 7, 8, 10])
plt.show()

# 6-1_2 주사위 시물레이션
dice = []
for _ in range(1000000):
    dice.append(random.randint(1, 6))
plt.hist(dice, bins=6)  # bins 옵션은 가로축의 구간 개수를 설정하는 속성
plt.show()

# 6-2 기온 데이터를 히스토그램으로 표현하기
# 6-2_1 1907년 ~2018년 서울의 최고 기온 데이터를 표현
result = []
for row in data:
    if row[-1] != '':
        result.append(float(row[-1]))
plt.hist(result, bins=100, color='r')
plt.show()

# 6-2_2 8월 달의 최고기온 표현
aug = []
for row in data:
    if row[0].split('-')[1] == '08':
        if row[-1] != '':
            aug.append(float(row[-1]))
plt.hist(aug, bins=100, color='r')
plt.show()

# 6-2_3 1월과 8월의 최고기온 데이터를 히스토그램으로 시각화하기
aug = []
jan = []
for row in data:
    month = row[0].split('-')[1]
    if row[-1] != '':
        # 8월의 최고기온 담기
        if month == '08':
            aug.append(float(row[-1]))
        # 1월의 최고기온 담기
        if month == '01':
            jan.append(float(row[-1]))
# 8월 최고기온 히스토그램 표현
plt.hist(aug, bins=100, color='r', label='Aug')
# 1월 최고기온 히스토그램 표현
plt.hist(jan, bins=100, color='b', label='Jan')
plt.legend()
plt.show()

# 6-3 상자 그림
# 6-3_1 상자 그림 그려보기
result = []
for _ in range(13):
    result.append(random.randint(1, 1000))
print(sorted(result))
plt.boxplot(result)
plt.show()

# 6-3_2 서울 최고기온 데이터 상자 그림으로 그려보기
result = []
for row in data:
    if row[-1] != '':
        result.append(float(row[-1]))
plt.boxplot(result)
plt.show()

# 6-3_3 1월과 8월의 상자 그림을 함께 그려보기
aug = []
jan = []
for row in data:
    month = row[0].split('-')[1]
    if row[-1] != '':
        if month == '08':
            aug.append(float(row[-1]))
        if month == '01':
            jan.append(float(row[-1]))
plt.boxplot([aug, jan])
plt.show()

# 6-3_4 월별로 해보기
month = [[] for _ in range(12)]
for row in data:
    mon = int(row[0].split('-')[1]) - 1
    if row[-1] != '':
        month[mon].append(float(row[-1]))
plt.boxplot(month)
plt.show()

# 6-3_5 8월 일별로 해보기
days = [[] for _ in range(31)]
for row in data:
    day = int(row[0].split('-')[2]) - 1
    if row[-1] != '':
        if row[0].split('-')[1] == '08':
            days[day].append(float(row[-1]))
# 그래프 스타일 지정
plt.style.use('ggplot')
# 그래프 크기 수정
plt.figure(figsize=(10,5), dpi=100)
# 이상치(아웃라이어) 값 생략
plt.boxplot(days, showfliers=False)
plt.show()