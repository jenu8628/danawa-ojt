import csv
import matplotlib.pyplot as plt

# 5-2
# f = open('../기온공공데이터/seoul.csv')
# data = csv.reader(f)
# next(data)
# result = []
# for row in data:
#     if row[-1] != '':
#         result.append(float(row[-1]))
# plt.plot(result, 'r')
# plt.show()

# 5-3
f = open('../기온공공데이터/seoul.csv')
data = csv.reader(f)
next(data)
result = []
for row in data:
    if row[-1] != '':
        if row[0].split('-')[1] == '08' and row[0].split('-')[2] == '07':
            result.append(float(row[-1]))
plt.plot(result, 'hotpink')
plt.show()