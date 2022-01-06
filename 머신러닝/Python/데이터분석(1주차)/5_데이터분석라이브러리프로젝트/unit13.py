import numpy as np
import matplotlib.pyplot as plt

# 13-1 numpy 라이브러리 시작하기
# sqrt() : 제곱근
print(np.sqrt(2))   # 1.4142135623730951

# pi : 파이
print(np.pi)    # 3.141592653589793

# sin(), cos(), tan()와 같은 삼각함수 제공
print(np.sin(0), np.cos(0), np.tan(0))  # 0.0 1.0 0.0

# random 서브 라이브러리
# rand()
# ndarray에서 nd 는 N-Dimensional, 즉 N차원이라는 의미
# 기존 random 라이브러리의 randint() 함수의 실수 버전
a = np.random.rand(5)
print(a)    # [0.62681935 0.25138933 0.56977162 0.22690505 0.1167883 ]
print(type(a))  # <class 'numpy.ndarray'>

# choice()
print(np.random.choice(6, 10))  # [3 3 2 0 0 4 3 5 2 1]
# 한 번 뽑은 숫자를 다시 뽑지 못하게 하고 싶다면 replace 속성을 False로 설정
print(np.random.choice(10, 6, replace=False))   # [7 5 8 2 1 3]
# p 속성 : 확률을 지정할 수 있다. 인자의 합은 반드시 1이어야 한다.
print(np.random.choice(6, 10, p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.1]))    # [2 1 1 4 0 3 5 5 1 1]


# 13-2 numpy 라이브러리를 활용해 그래프 그리기
dice = np.random.choice(6, 1000000, p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.1])
plt.hist(dice, bins=6)
plt.show()

# np.random.rand(n) : 0~1사이에 있는 n개의 실수를 만듬
# np.random.randint(a, b, n) a이상 b미만인 정수 n개를 만듬
x = np.random.randint(10, 100, 200)
y = np.random.randint(10, 100, 200)
size = np.random.rand(200) * 100

plt.scatter(x, y, s=size, c=x, cmap='jet', alpha=0.7)
plt.colorbar()
plt.show()

# 13-3 numpy array
# ,가 없다.
a = np.array([1, 2, 3, 4])
print(a)    # [1 2 3 4]
# 인덱싱과 슬라이싱이 가능하다.
print(a[1], a[-1])  # 2 4
print(a[1:])    # [2 3 4]
# 한 가지 데이터타입만 저장가능하다.
b = np.array([1, 2, '3', 4])
print(b)    # ['1' '2' '3' '4']
# 배열 생성을 위해 꼭 리스특사 필요한 것은 아니다

# 0 또는 1로 배열을 채울 때 유용한 함수
# 0으로 이루어진 크기가 10인 배열 생성
print(np.zeros(10)) # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 1로 이루어진 크가기 10인 배열 생성
print(np.ones(10))  # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# 3행 X 3열의 단위 배열 생성
# 단위 배열(행렬) : 주대각선상의 요소가 1이며, 그 외 나머지 요소가 0인 배열(행렬)
print(np.eye(3))
#[[1. 0. 0.]
# [0. 1. 0.]
# [0. 0. 1.]]

# arange() 함수
print(np.arange(3)) # [0 1 2]
print(np.arange(3, 7))  # [3 4 5 6]
print(np.arange(3, 7, 2))   # [3 5]

# linspace() 함수
print(np.arange(1, 2, 0.1)) # [1.  1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9]
print(np.linspace(1, 2, 11))    # [1.  1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2. ]
# arange는 간격을 지정하고, linspace는 구간을 지정하는 차이

# 13-4 numpy array의 다양한 활용
# 배열에 어떤 연산이나 함수를 적용하면 배열의 모든 값이 한꺼번에 계산된다.
a = np.zeros(10) + 5
print(a)    # [5. 5. 5. 5. 5. 5. 5. 5. 5. 5.]
b = np.linspace(1, 2, 11)
print(np.sqrt(b))
# [1.         1.04880885 1.09544512 1.14017543 1.18321596 1.22474487
#  1.26491106 1.30384048 1.34164079 1.37840488 1.41421356]
# -파이 ~ 파이
c = np.arange(-np.pi, np.pi, np.pi/10)
# 사인함수 x: -파이 ~ 파이
plt.plot(c, np.sin(c), label="sin")
plt.plot(c, np.cos(c), label="Cosine")
plt.plot(c+np.pi/2, np.sin(c), label='x-translation')
plt.legend()
plt.show()

# mask 기능
a = np.arange(-5, 5)
print(a)    # [-5 -4 -3 -2 -1  0  1  2  3  4]
print(a < 0)    # [ True  True  True  True  True False False False False False]
print(a[a < 0]) # [-5 -4 -3 -2 -1]
mask1 = abs(a) > 3
print(a[mask1]) # [-5 -4  4]
mask2 = abs(a) % 2 == 0
# 둘 중 하나의 조건이라도 참일 경우(합집합)
print(a[mask1 + mask2]) # [-5 -4 -2  0  2  4]
# 두 가지 조건 모두 참일 경우(교집합)
print(a[mask1 * mask2]) # [-4  4]

# 버블차트 그리기
x = np.random.randint(-100, 100, 1000) # -100이상 100 미만에서 1000개의 랜덤값 추출
y = np.random.randint(-100, 100, 1000) # -100이상 100 미만에서 1000개의 랜덤값 추출
mask1 = abs(x) > 50
mask2 = abs(y) > 50
x = x[mask1 + mask2]
y = y[mask1 + mask2]
size = np.random.rand(len(x)) * 100
plt.scatter(x, y, s=size, c=x, cmap='jet', alpha=0.7)
plt.colorbar()
plt.show()