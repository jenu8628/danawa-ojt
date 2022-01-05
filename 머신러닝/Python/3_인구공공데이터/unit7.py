import csv
import matplotlib.pyplot as plt

f = open('age.csv')
data = csv.reader(f)
next(data)
result = []
for row in data:
    if '신도림' in row[0]:
        for i in row[3:]:
            result.append(int(i))
print(result)
plt.style.use('ggplot')
plt.plot(result)
plt.show()