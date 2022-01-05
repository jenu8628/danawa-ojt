import csv
import matplotlib.pyplot as plt

f = open('subwayfee.csv')
data = csv.reader(f)
next(data)
# 11-1 유임 승차 비율이 가장 높은 역
# mx, rate = 0, 0
# max_station = ""
# for row in data:
#     # 1: 호선, 3:역이름 4: 유임승차, 5: 유임하차, 6: 무임승차, 7: 무임하차
#     for i in range(4, 8):
#         row[i] = int(row[i].replace(',', ''))
#     if row[6] != 0 and (row[4] + row[6]) > 100000:
#         rate = row[4] / (row[4] + row[6])
#         if rate > mx:
#             mx = rate
#             mx_station = row[3] + ' ' + row[1]
# print(mx_station, round(mx*100, 2))

# 11-2 유무임 승하차 인원이 가장 많은 역
# mx = [0] * 4
# mx_station = [''] * 4
# label = ['유임승차', '유임하차', '무임승차', '무임하차']
# for row in data:
#     for i in range(4, 8):
#         if int(row[i].replace(',', '')) > mx[i-4]:
#             mx[i-4] = int(row[i].replace(',', ''))
#             mx_station[i-4] = row[3] + ' ' + row[1]
# for i in range(4):
#     print(label[i]+' : ' + mx_station[i], mx[i])

# 11-3 모든 역의 유무임 승하차 비율은 어떻게 될까
label = ['유임승차', '유임하차', '무임승차', '무임하차']
c = ['#14CCC0', '#389993', '#FF1C6A', '#CC14AF']
plt.rc('font', family='Malgun Gothic')
for row in data:
    for i in range(4, 8):
        row[i] = int(row[i].replace(',', ''))
    plt.title(row[3] + ' ' + row[1])
    plt.pie(row[4:8], labels=label, colors=c, autopct="%1.f%%")
    plt.axis('equal')
    # savefig() : 파일 저장
    plt.savefig(row[3] + ' ' + row[1]+'.png')
    plt.show()