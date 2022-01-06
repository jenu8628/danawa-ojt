import csv
import matplotlib.pyplot as plt

f = open('gender.csv')
data = csv.reader(f)
next(data)

# 10-1 꺾은선 그래프로 표현하기
# m, f = [], []
# name = input('궁금한 동네를 입렵해 주세요 : ')
# for row in data:
#     if name in row[0]:
#         for i in range(3, 104):
#             m.append(int(row[i]))
#             f.append(int(row[i+103]))
# plt.plot(m, label="Male")
# plt.plot(f, label="Femail")
# plt.legend()
# plt.show()

# 10-2 막대그래프로 표현하기
# result = []
# name = input('궁금한 동네를 입렵해 주세요 : ')
# for row in data:
#     if name in row[0]:
#         for i in range(3, 104):
#             result.append(int(row[i].replace(',', '')) - int(row[i+103].replace(',', '')))
#         break
# plt.bar(range(101), result)
# plt.show()

# 10-3 산점도
# plt.scatter([1, 2, 3, 4], [10, 30, 20, 40])
# plt.show()

# 10-4 버블 차트
# import random
# x, y, size = [], [], []
# for i in range(100):
#     x.append(random.randint(50, 100))
#     y.append(random.randint(50, 100))
#     size.append(random.randint(10, 100))
# # s : 원의 크기, c : 색깔, cmap : 컬러바 색상 종류, alpha: 투명도(opacity)
# plt.scatter(x, y, s=size, c=size, cmap='jet', alpha=0.7)
# # 컬러바
# plt.colorbar()
# plt.show()

# 10-5
import math
name = input('궁금한 동네를 입력하세요 : ')
m, f, size = [], [], []
for row in data:
    if name in row[0]:
        for i in range(3, 104):
            m.append(int(row[i].replace(',', '')))
            f.append(int(row[i+103].replace(',', '')))
            size.append(math.sqrt(int(row[i].replace(',', '')) + int(row[i+103].replace(',', ''))))
        break
plt.style.use('ggplot')
plt.rc('font', family="Malgun Gothic")
plt.figure(figsize=(10, 5), dpi=100)
plt.title(name+' 지역의 성별 인구 그래프')
plt.scatter(m, f, s=size, c=range(101), alpha=0.5, cmap='jet')
plt.colorbar()
plt.plot(range(max(m)), range(max(m)), 'g')
plt.xlabel('남성 인구수')
plt.ylabel('여성 인구수')
plt.show()
