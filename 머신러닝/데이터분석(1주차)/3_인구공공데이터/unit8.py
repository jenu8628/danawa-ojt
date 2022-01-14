import csv
import matplotlib.pyplot as plt



# 막대그래프 그리기
f = open('age.csv')
data = csv.reader(f)
next(data)
result = []
name = input('인구 구조가 알고 싶은 지역의 이름(읍면동 단위)을 입력해주세요 : ')
for row in data:
    if name in row[0]:
        for i in row[3:]:
            result.append(int(i))
label = range(101)
plt.bar(result)
# plt.barh(range(101), result)
plt.show()

# 항아리 모양 그래프 그리기
# f = open('gender.csv')
# data = csv.reader(f)
# next(data)
# result = []
# name = input('찾고 싶은 지역의 이름을 알려주세요 : ')
# man = []
# woman = []
# for row in data:
#     if name in row[0]:
#         for i in range(101):
#             man.append(-int(row[i+3]))
#             woman.append(int(row[-(i+1)]))
# plt.rc('font', family='Malgun Gothic')
# plt.rcParams['axes.unicode_minus'] = False
# plt.title(name + ' 지역의 남녀 성별 인구 분포')
# woman.reverse()
# plt.barh(range(101), man, label='남성')
# plt.barh(range(101), woman, label='여성')
# plt.legend()
# plt.show()