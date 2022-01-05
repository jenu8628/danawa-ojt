import csv
import matplotlib.pyplot as plt

f = open('subwaytime.csv')
data = csv.reader(f)
next(data)
next(data)

# 12-1 출근 시간대 사람들이 가장 많이 타고 내리는 역은 어디일까
mx, mx_station = 0, ''
my, my_station = 0, ''
for row in data:
    for i in range(4, len(row)):
        row[i] = int(row[i].replace(',', ''))
    # 10, 12, 14 : 7시, 8시, 9시 승차 인원
    # 11, 13, 15 : 7시, 8시, 9시 하차 인원
    in_people = row[10:15:2]
    out_people = row[11:16:2]
    if sum(in_people) > mx:
        mx = sum(in_people)
        mx_station = row[3] + '(' + row[1] + ')'
    if sum(out_people) > my:
        my = sum(out_people)
        my_station = row[3] + '(' + row[1] + ')'
print(mx_station, mx)
print(my_station, my)

# 12-2 밤 11시에 사람들이 가장 많이 타는 역은 어디일까
mx, mx_station = 0, ''
t = int(input('몇 시의 승차인원이 가장 많은 역이 궁금하세요? : '))
for row in data:
    for i in range(4, len(row)):
        row[i] = int(row[i].replace(',', ''))
    idx = 4 + (t-4) * 2
    if row[idx] > mx:
        mx = row[idx]
        mx_station = row[3] + '(' + row[1] + ')'
print(mx_station, mx)

# 12-4 시간대 별로 승차 인원이 가장 많은 곳은?
mx, mx_station = [0] * 24, [''] * 24
for row in data:
    for j in range(24):
        idx = j * 2 + 4
        a = int(row[idx].replace(',', ''))
        if mx[j] < a:
            mx[j] = a
            mx_station[j] = row[3] + '(' + str(j+4) + ')'
for i in range(24):
    print(mx_station[i], mx[i])
plt.rc('font', family='Malgun Gothic')
plt.bar(range(24), mx, color='r')
plt.xticks(range(24), mx_station, rotation=90)
plt.show()


# 12-4 시간대별로 하차 인원이 가장 많은 역을 찾는 코드
mx, mx_station = [0] * 24, [''] * 24
for row in data:
    for j in range(24):
        idx = j * 2 + 5
        a = int(row[idx].replace(',', ''))
        if mx[j] < a:
            mx[j] = a
            mx_station[j] = row[3] + '(' + str(j+4) + ')'
for i in range(24):
    print(mx_station[i], mx[i])
plt.rc('font', family='Malgun Gothic')
plt.bar(range(24), mx, color='b')
plt.xticks(range(24), mx_station, rotation=90)
plt.show()

# 12-5 모든 지하철역에서 시간대별 승하차 인원을 모두 더하면
s_in, s_out = [0] * 24, [0] * 24
for row in data:
    for i in range(24):
        s_in[i] += int(row[(i*2) + 4].replace(',', ''))
        s_out[i] += int(row[(i*2) + 5].replace(',', ''))
plt.rc('font', family="Malgun Gothic")
plt.title("지하철 시간대 별 승하차 인원 추이")
plt.plot(s_in, 'r', label='승차')
plt.plot(s_out, 'b', label='하차')
plt.xticks(range(24), range(4, 28))
plt.legend()
plt.show()
