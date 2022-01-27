from sklearn.model_selection import train_test_split
from tensorflow import keras

# 데이터 불러오기
(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()
# 픽셀 값 0~1 사이로 변환, 28 * 28 크기의 2차원 배열을 784 크기의 1차원 배열로 펼침
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
# 훈련 세트와 검증세트 나누기
train_input, val_input, train_target, val_target = \
    train_test_split(train_input, train_target, test_size=0.2, random_state=42)

# 2개의 층
# dense1 = keras.layers.Dense(100, activation="sigmoid", input_shape=(784,))
# dense2 = keras.layers.Dense(10, activation="softmax")
#
#
# model = keras.Sequential([dense1, dense2])
# model.summary()
#
# # 클래스 생성자 안에서 바로 클래스의 객체를 생성
# model = keras.Sequential([
#     keras.layers.Dense(100, activation="sigmoid", input_shape=(784,), name='hidden'),
#     keras.layers.Dense(10, activation="softmax", name='output')
# ])
# model.summary()
#
# # add() 메서드를 이용한 추가(제일 많이 쓰임)
# model = keras.Sequential()
# model.add(keras.layers.Dense(100, activation="sigmoid", input_shape=(784,), name='hidden'))
# model.add(keras.layers.Dense(10, activation="softmax", name='output'))
# model.summary()

# 렐루함수
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics='accuracy')

sgd = keras.optimizers.SGD()
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics='accuracy')