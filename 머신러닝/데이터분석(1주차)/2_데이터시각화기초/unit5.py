import csv
import matplotlib.pyplot as plt

# 5-2
f = open('../1기온공공데이터/seoul.csv')
data = csv.reader(f)
next(data)
result = []
for row in data:
    if row[-1] != '':
        result.append(float(row[-1]))
plt.plot(result, 'r')
plt.show()

# 5-3_1
f = open('../1기온공공데이터/seoul.csv')
data = csv.reader(f)
next(data)
result = []
for row in data:
    if row[-1] != '':
        if row[0].split('-')[1] == '08' and row[0].split('-')[2] == '07':
            result.append(float(row[-1]))
plt.plot(result, 'hotpink')
plt.show()

# 5-3_2
f = open('../1기온공공데이터/seoul.csv')
data = csv.reader(f)
next(data)
low = []    # 최저 기온 값을 저장할 리스트
high = []   # 최고 기온 값을 저장할 리스트
for row in data:
    if row[-1] != '' and row[-2] != '': # 최고 기온 값과 최저 기온 값이 존재한다면
        if 1983 <= int(row[0].split('-')[0]):   # 1983년 이후라면
            if row[0].split('-')[1] =='08' and row[0].split('-')[2] =='07': # 8월 7일이라면
                high.append(float(row[-1])) # 최고 기온 값을 high에 저장
                low.append(float(row[-2]))  # 최저 기온 값을 low에 저장
plt.rc('font', family='Malgun Gothic')  # 맑은 고딕을 기본 글꼴로 설정
plt.rcParams['axes.unicode_minus'] = False  #마이너스 기호 깨짐 방지
plt.title('내 생일의 기온 변화 그래프')    # 제목 설정
# high 리스트에 저장된 값을 hotpink 색으로 그리고 레이블을 표시
plt.plot(high, 'hotpink', label='high')
# low 리스트에 저장된 값을 skyblue 색으로 그리고 레이블을 표시
plt.plot(low, 'skyblue', label='low')
plt.legend()   # 범례 표시
plt.show()