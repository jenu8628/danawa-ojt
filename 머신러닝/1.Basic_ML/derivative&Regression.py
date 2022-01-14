# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from tensorflow.keras import optimizers
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import urllib.request
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# 1. tape_gradient() : 미분 기능을 수행
# 예시 : 2*(w)**2 + 5를 w에 대해 미분
# <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>
# def f(w):
#     y = w ** 2
#     z = 2 * y + 5
#     return z





# 자돟 미분을 이용한 선형 회귀 구현
# 학습될 가중치 변수 선언
# w = tf.Variable(4.0)
# b = tf.Variable(1.0)
#
# # 가설 함수 정의
# def hypothesis(x):
#     return w * x + b
#
# # 현재 가설 w와 b는 각각 4, 1이므로 임이의 입력값을 넣는다면 결과는 다음과 같다
# x_test=[3.5, 5, 5.5, 6]
# # [15. 21. 23. 25.]
# print(hypothesis(x_test).numpy)
#
# # 다음과 같이 평균 제곱 오차를 손실 함수로서 정의한다
# def mes_loss(y_pred, y):
#     # 두 개의 차이값을 제곱한 평균 값
#     # reduce_mean : 평균 값을 구해줌
#     # squre : 제곱 값을 구해줌
#     return tf.reduce_mean(tf.square(y_pred-y))
#
# #여기서 사용할 데이터는 x와 y가 약 10배의 차이를 가지는 데이터이다.
# # 공부하는 시간
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# # 각 곻부하는 시간에 맵핑되는 성적
# y = [11, 22, 33, 44, 53, 66, 77, 87, 95]
# # 옵티마이저는 경사 하강법을 사용하되, 학습률(learning rate)는 0.01을 사용한다.
# # optimizers.SGD = gradient_descent를 가르킴
# optimizer = tf.optimizers.SGD(0.01)
#
# # 약 300번에 걸쳐서 경사하강법을 수행
# for i in range(301):
#     with tf.GradientTape() as tape:
#         # 현재 파라미터에 기반한 입력 x에 대한 예측값
#         y_pred = hypothesis(x)
#         # 평균 제곱 오차를 계산
#         cost = mes_loss(y_pred, y)
#     # 손실 함수에 대한 파라미터의 미분값 계산
#     gradients = tape.gradient(cost, [w, b])
#     # 파라미터 업데이트
#     # zip(현재 역할) : 리스트 두개를 받으면 해당 리스트의 인덱스 끼리 따로 모은다.
#     # 행렬을 뒤집는다라는 의미
#     # 즉,zip([5, 10], [15, 1]) = (5, 15), (10, 1)이 된다.
#     optimizer.apply_gradients(zip(gradients, [w, b]))
#     if i % 10 == 0:
#         print("epoch : {:3} | w의 값 : {:5.4f} | b의 값 : {:5.4} | cost : {:5.6f}".format(i, w.numpy(), b.numpy(), cost))




if __name__ == '__main__':

    # @tf.function
    # def hypothesis(x):
    #     return w * x + b
    #
    # @tf.function
    # def mse_loss(y_pred, y):
    #     return tf.reduce_mean(tf.square(y_pred - y))
    #
    # w = tf.Variable(2.)
    #
    # with tf.GradientTape() as tape:
    #     z = f(w)
    #
    # # [<tf.Tensor: shape=(), dtype=float32, numpy=8.0>]
    # gradients = tape.gradient(z, [w])
    # print(gradients)
    # w = tf.Variable(4.0)
    # b = tf.Variable(1.0)
    #
    # x_test = [3.5, 5, 5.5, 6]
    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # y = [11, 22, 33, 44, 53, 66, 77, 87, 95]
    #
    # optimizer = tf.optimizers.SGD(0.01)
    #
    # for i in range(31):
    #
    #     with tf.GradientTape() as tape:
    #         y_pred = hypothesis(x)
    #         cost = mse_loss(y_pred, y)
    #
    #     gradients = tape.gradient(cost, [w, b])
    #     optimizer.apply_gradients(zip(gradients, [w, b]))
    #
    #     # if i % 10 == 0:
    #     print("epoch : {:3} | w의 값 : {:5.4f} | b의 값 : {:5.4} | cost : {:5.6f}".format(i, w.numpy(), b.numpy(), cost))


    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # y = [11, 22, 33, 44, 53, 66, 77, 87, 95]
    #
    # model = Sequential()
    #
    # model.add(Dense(1, input_dim=1, activation='linear'))
    #
    # sgd = optimizers.SGD(lr=0.01)
    #
    # model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
    #
    # model.fit(x, y, epochs=300)
    # print(model.predict([8.5]))


    # def sigmoid(x):
    #     return 1/(1+np.exp(-x))
    # x = np.arange(-5.0, 5.0, 0.1)
    # y1 = sigmoid(0.5*x)
    # y2 = sigmoid(x)
    # y3 = sigmoid(2*x)
    #
    # plt.plot(x, y1, 'r--',)
    # plt.plot(x, y2, 'g')
    # plt.plot(x, y3, 'b--')
    # plt.plot([0, 0], [1.0, 0.0], ':')
    # plt.show()

    # x = np.array([-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50])
    # y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    #
    # model = Sequential()
    # model.add(Dense(1, input_dim=1, activation="sigmoid"))
    #
    # sgd = optimizers.SGD(lr=0.02)
    # model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_crossentropy'])
    # model.fit(x, y, epochs=200)
    # # plt.plot(x, model.predict(x), 'b', x, y, 'k.')
    # # plt.show()
    # print(model.predict([1, 2, 3, 4, 4.5]))
    # print(model.predict([11, 21, 31, 41, 500]))

    # 다중 선형 회귀
    # X = np.array([[70, 85, 11], [71, 89, 18], [50, 80, 20], [99, 20, 10], [50, 10, 10]])
    # y = np.array([73, 82, 72, 57, 34])
    #
    # model = Sequential()
    # model.add(Dense(1, input_dim=3, activation='linear'))
    #
    # sgd = optimizers.SGD(lr=0.0001)
    # model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
    # model.fit(X, y, epochs=2000)
    # print(model.predict(X))
    #
    # X_test = np.array([[20, 99, 10], [40, 50, 20]])
    # print(model.predict(X_test))

    # 다중 로지스틱 회귀
    # X = np.array([[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0]])
    # y = np.array([0, 0, 0, 1, 1, 1])
    #
    # model = Sequential()
    # model.add(Dense(1, input_dim=2, activation='sigmoid'))
    # model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['binary_crossentropy'])
    # model.fit(X, y, epochs=2000)
    # print(model.predict(X))
    urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/06.%20Machine%20Learning/dataset/Iris.csv", filename="Iris.csv")
    data = pd.read_csv('Iris.csv', encoding='latin1')
    # print('샘플의 개수 :', len(data))
    # print(data[:5])
    # print("품종 종류 :", data['Species'].unique(), sep="\n")

    sns.set(style="ticks", color_codes=True)
    # g = sns.pairplot(data, hue="Species", palette="husl")
    # sns.barplot(data['Species'], data['SepalWidthCm'], ci=None)
    # data['Species'].value_counts().plot(kind='bar')
    # Iris-virginica는 0, Iris-setosa는 1, Iris-versicolor는 2가 됨.
    data['Species'] = data['Species'].replace(['Iris-virginica', 'Iris-setosa', 'Iris-versicolor'], [0, 1, 2])
    # data['Species'].value_counts().plot(kind='bar')
    # plt.show()

    data_X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
    data_y = data['Species'].values

    # print(data_X[:5])
    # print(data_y[:5])
    (X_train, X_test, y_train, y_test) = train_test_split(data_X, data_y, train_size=0.8, random_state=1)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print(y_train[:5])
    print(y_test[:5])
    X = pd.DataFrame(X_train).head()
    y = pd.DataFrame(y_train).head()
    print(X)
    print(y)
    model = Sequential()
    model.add(Dense(3, input_dim=4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=200, batch_size=1, validation_data=(X_test, y_test))

    # epochs = range(1, len(history.history['accuracy']) + 1)
    a = [[5.0, 4.3, 5.5, 3.8], [7.5, 5.8, 4.6, 6.9], [2.9, 1.6, 3.2, 1.3], [3.5, 2.4, 1.8, 4.0]]
    print(model.predict(a))
    # plt.plot(epochs, history.history['loss'])
    # plt.plot(epochs, history.history['val_loss'])
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    # print(model.evaluate(X, y))