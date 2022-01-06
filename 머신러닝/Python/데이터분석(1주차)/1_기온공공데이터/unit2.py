import csv

# unit 2_1
def unit1():
    f = open('seoul.csv', 'r', encoding='cp949')
    data = csv.reader(f, delimiter=',')
    print(data)
    f.close()
    return

# unit 2_2
def unit2():
    f = open('seoul.csv', 'r', encoding='cp949')
    data = csv.reader(f)
    for row in data:
        print(row)
    f.close()
    return

def unit3():
    f = open('seoul.csv', 'r', encoding='cp949')
    data = csv.reader(f)
    header = next(data)
    print(header)
    f.close()
    return


# unit1()
# unit2()
unit3()