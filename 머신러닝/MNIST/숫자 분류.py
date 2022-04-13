import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

def load_data():
    mnist = tf.keras.datasets.mnist
    return mnist.load_data()

def preprocessing(x_train, x_test, y_train, y_test):
    # 정규화(0~1사이)
    x_train, x_test = x_train/255.0, x_test/255.0

    # (60000, 28, 28) -> (60000, 28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    return x_train, x_test, to_categorical(y_train), to_categorical(y_test)

def deepmodel(x_train, y_train):
    model = Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    print(model.summary())
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    return model

def show_number(x_train, y_train):
    print("Y[0] : ", y_train[0])
    plt.imshow(x_train[0], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()

def CNN(x_train, y_train):
    model = Sequential()
    model.add(Conv2D(kernel_size=(3, 3), filters=64, input_shape=(28, 28, 1), padding='same', activation='relu'))
    model.add(Conv2D(kernel_size=(3, 3), filters=64, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    print(model.summary())

    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4, baseline=0.2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=100, epochs=5, validation_split=0.2, callbacks=[early_stop])
    return model


if __name__ == '__main__':
    # 1. Fashion MNIST 데이터셋 불러오기
    (x_train, y_train), (x_test, y_test) = load_data()
    # x_train : 0~255사이의 값을 갖는 28 * 28 크기의 NumPy 어레이 손글씨 이미지 데이터
    # y_train : 0~9까지의 정수 값을 갖는 어레이

    # 첫번째 데이터 확인
    show_number(x_train, y_train)

    # 2. 데이터셋 전처리
    x_train, x_test, y_train, y_test = preprocessing(x_train, x_test, y_train, y_test)

    # 3. 모델링
    model = CNN(x_train, y_train)

    # 4. 정확도 평가
    loss, accuracy = model.evaluate(x_test, y_test)
    print(loss, accuracy)

    data = x_train[0]
    print(data.shape)
    print(model.predict(data))