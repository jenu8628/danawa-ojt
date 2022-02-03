from tensorflow import keras
from sklearn.model_selection import train_test_split

# 모델을 만드는 함수
def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu'))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

# 데이터 받아오기
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
# 데이터 정규화
train_scaled = train_input / 255.0
# 데이터 나누기
train_scaled, val_scaled, train_target, val_target = \
    train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
# 모델 만들기
model = model_fn()
# 모델 구조 확인
model.summary()
# 모델 컴파일
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
# fit() 객체 담기
history = model.fit(train_scaled, train_target, epochs=5, verbose=0)