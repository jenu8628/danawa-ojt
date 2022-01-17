import tensorflow as tf
import numpy as np

# 데이터
# x1 = [73., 93., 89., 96., 73.]
# x2 = [80., 88., 91., 98., 66.]
# x3 = [75., 93., 90., 100., 70.]
# y = [152., 185., 180., 196., 142.]

# # 가중치
# w1 = tf.Variable(tf.random.normal([1]))
# w2 = tf.Variable(tf.random.normal([1]))
# w3 = tf.Variable(tf.random.normal([1]))
# b = tf.Variable(tf.random.normal([1]))
#
#
# learning_rate = 0.000001
#
# for i in range(1000+1):
#     with tf.GradientTape() as tape:
#         hypothesis = (w1 * x1) + (w2 * x2) + (w3 * x3) + b
#         cost = tf.reduce_mean(tf.square(hypothesis - y))
#     w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1, w2, w3, b])
#
#     w1.assign_sub(learning_rate * w1_grad)
#     w2.assign_sub(learning_rate * w2_grad)
#     w3.assign_sub(learning_rate * w3_grad)
#     b.assign_sub(learning_rate * b_grad)
#
#     if i % 50 == 0:
#         print("{:5} | {:12.4f}".format(i, cost.numpy()))

# 행렬을 사용한 구현
data = np.array([[73., 80., 75., 152.],
                 [93., 88., 93., 185.],
                 [89., 91., 90., 180.],
                 [96., 98., 100., 196.],
                 [73., 66., 70., 142.]], dtype=np.float32)
# 마지막행을 제외한 나머지
x = data[:, :-1]
# 마지막행
y = data[:, [-1]]

w = tf.Variable(tf.random.normal([3, 1]))
b = tf.Variable(tf.random.normal([1]))
def predict(x):
    return tf.matmul(x, w) + b


learning_rate = 0.00001
n_epochs = 2000
for i in range(n_epochs+1):
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean(tf.square(predict(x) - y))
    w_grad, b_grad = tape.gradient(cost, [w, b])
    w.assign_sub(learning_rate * w_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 50 == 0:
        print("{:5} | {:12.4f}".format(i, cost.numpy()))

