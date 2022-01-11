import random
from time import sleep
gamenum = input('로또 게임 횟수를 입력하세요: ')
for i in range(int(gamenum)):
    ret = random.sample(range(1, 46), 6)
    ret.sort()
    print('로또번호[%d]: ' %(i+1), end='')
    print(ret)
    sleep(1)