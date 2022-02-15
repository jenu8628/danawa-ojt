import pymysql
print('start')
conn = pymysql.connect(host='127.0.0.1',port=10000, user="lhw", password="1234")
cur = conn.cursor()

cur.execute("SHOW DATABASES")
print(cur)

cursor.close()
conn.close()
print('end')