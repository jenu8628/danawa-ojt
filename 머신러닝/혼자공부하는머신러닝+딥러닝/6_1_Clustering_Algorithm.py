import numpy as np
import matplotlib.pyplot as plt


# 데이터 준비
fruits = np.load('fruits_300.npy')
print(fruits.shape)     # (300, 100, 100)
print(fruits[0, 0, :])
# plt.imshow(fruits[0], cmap='gray')
# plt.show()

# 픽셀값 분석
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)
print(apple.shape)      # (100, 10000)
