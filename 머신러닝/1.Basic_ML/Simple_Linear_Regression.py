import tensorflow as tf
import numpy as np


# Linear : 선형, regression : 회귀
# Hypothesis : 가설
# Cost function : 비용 함수

# H(x) Wx + b

x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]

# constant : 상수 선언
# Variable : 변수


w = tf.Variable(2.9)
b = tf.Variable(0.5)

hypothesis = w * x_data + b

# cost(W, b) = ((H(x**i) - (y**2))**2)의 합 * 1/m
# 가설에서 실제값을 뺀 것의 제곱들의 평균을 나타냄
# hypothesis - y_data : 에러 값
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# tf.reduce_mean()에 대해서 알아보자.
# reduce : 줄어든다의 의미로 차원을 하나 줄인다라고 생각
# 1차원
v = [1., 2., 3., 4.]
# 0차원
tf.reduce_mean(v)   # 2.5

# tf.square() : 넘겨받은 값을 제곱함
tf.square(3)    # 9


# Gradient(경사) descent(하강) : 경사 하강 알고리즘
# Cost가 최소가 되도록 하는 미니마이즈 알고리즘 중 가장유명함
# 경사를 하강하면서 minimize cost(W,b)의 W, b를 찾는 알고리즘이다.
learning_rate = 0.01
for i in range(100):
    # with ... as 파일 객체:
    # 파일이나 함수를 열고 해당 구문이 끝나면 자동으로 닫는다.
    # close같은 것을 빼먹는 경우를 위해 만들어졌음
    # GradientTape()은 보통 with 구문과 같이 쓰임
    # with 블럭 안에 있는 변수들의 변화, 정보를 tape에 기록한다.
    # tape의 gradient 메서드를 호출해서 이후에 경사도값 즉, 미분값(기울기)을 구함
    with tf.GradientTape() as tape:
        hypothesis = w * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    # gradient는 변수들에 대한 개별 미분값(기울기)을 구해서 tuple로 반환한다.
    # w_grad : cost함수에서 w의 미분값(기울기), b_grad : cost함수에서 b의 미분값(기울기)
    w_grad, b_grad = tape.gradient(cost, [w, b])
    # w, b의 값을 업데이트
    # assign_sub는 뺀값을 다시 그 값에 할당해줌
    # ex) A.assign_sub(B)
    # A = A-B, A -= B
    # learning_rate는 w_grad(gradient)값을 얼마나 반영할 지 정한다.
    # 작은 값이 쓰임
    # 기울기를 구했을 때 그 기울기를 얼마만큼 반영할 것인가이다.
    w.assign_sub(learning_rate * w_grad)
    b.assign_sub(learning_rate * b_grad)
    if i % 10 == 0:
        print('{:5}|{:10.4f}|{:10.4}|{:10.6f}'.format(i, w.numpy(), b.numpy(), cost))

# 우리가 만든 가설이 얼마나 맞는지 새로운 데이터로 해보자.
# 거의 근사한 값이 나온다는 것을 알 수 있다.
print(w*5 + b)  # tf.Tensor(5.0066934, shape=(), dtype=float32)
print(w*2.5 + b)    # tf.Tensor(2.4946523, shape=(), dtype=float32)

tf.random.set_seed(0)

x = [1., 2., 3., 4.]
y = [1., 3., 5., 7.]
w = tf.Variable(tf.random.normal([1], -100., 100.))
for step in range(300):
    hypothesis = w*x
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    alpha = 0.01
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(w, x) - y, x))
    descent= w - tf.multiply(alpha, gradient)
    w.assign(descent)

    if step % 10 == 0:
        print('{:5} | {:10.4f} | {:10.6f}'.format(step, cost.numpy(), w.numpy()[0]))



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

# 케라스를 이용한 표현
x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [11, 22, 33, 44, 53, 66, 77, 87, 95]

model = Sequential()

model.add(Dense(1, input_dim=1, activation='linear'))

sgd = optimizers.SGD(lr=0.01)

model.compile(optimizer=sgd, loss='mse', metrics=['mse'])

model.fit(x, y, epochs=300)
print(model.predict([8.5]))
