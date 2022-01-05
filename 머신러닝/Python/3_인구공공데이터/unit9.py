import matplotlib.pyplot as plt
import csv

# 9-1 혈액형 비율 표현하기
# size = [2441, 2312, 1031, 1233]
# label =['A형', 'B형', 'AB형', 'O형']
# plt.rc('font', family="Malgun Gothic")
# color = ['darkmagenta', 'deeppink', 'hotpink', 'pink']
# # 동그란 원
# plt.axis('equal')
# # autopct는 auto percent를 의미, 소수점 아래 둘째 자리에서 반올림한 값을 표시 : %.f%%
# # explode는 돌출효과 표시가능!
# plt.pie(size, labels=label, autopct='%.1f%%', colors=color, explode=(0, 0, 0.1, 0))
# plt.legend()
# plt.show()

# 9-2 제주도의 성별 인구 비율 표현하기
f = open('gender.csv')
data = csv.reader(f)
next(data)
size = []
name = input('찾고 싶은 지역의 이름을 알려주세요 : ')
m = 0
f = 0
for row in data:
    if name in row[0]:
        for i in range(101):
            m += int(row[i+3].replace(',', ''))
            f += int(row[i+106].replace(',', ''))
        break
size.append(m)
size.append(f)
plt.rc('font', family="Malgun Gothic")
color = ['crimson', "darkcyan"]
plt.axis('equal')
plt.pie(size, labels=['남', '여'], autopct='%.1f%%', colors=color, startangle=90)
plt.title(name + ' 지역의 남녀 성별 비율')
plt.show()