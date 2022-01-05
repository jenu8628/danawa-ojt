import csv

f = open('seoul.csv')
data = csv.reader(f)
header = next(data)
max_temp, max_date = -999, ''
for row in data:
    if row[-1] == '':
        row[-1] = -999
    row[-1] = float(row[-1])
    if row[-1] > max_temp:
        max_date = row[0]
        max_temp = row[-1]
f.close()
print(max_date, max_temp)