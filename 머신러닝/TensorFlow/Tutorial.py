import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# tf.keras.datasets 모듈은 Neural Network의 훈련에 사용할 수 있는 여러 데이터셋을 포함
fashion_mnist = tf.keras.datasets.fashion_mnist


# load_data() 함수를 호출하면 네 개의 넘파이(NumPy) 배열이 반환됨
# train_images와 train_labels 배열은 모델 학습에 사용되는 훈련 세트
# test_images와 test_labels 배열은 모델 테스트에 사용되는 테스트 세트
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# train_images : 0~255 사이의 값을 갖는 28*28 크기의 numpy 어레이
# train_labels : 0~9까지의 정수 값을 갖는 어레이, 이 값은 이미지에 있는 옷의 클래스(class)를 나타내는 레이블
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape  #(60000, 28, 28)
# 각 레이블은 0과 9 사이의 정수
train_labels    # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
# 테스트 세트 10000개의 이미지가 있음
test_images.shape   #(10000, 28, 28)

# 데이터 전처리

plt.figure()    # 그래프 사이즈조정
plt.imshow(train_images[0]) # 이미지 맵, 원하는 사이즈의 픽셀을 원하는 색으로 채워서 만든 그림
plt.colorbar()  # 칼라바
plt.grid(False) # grid : 격자 False이므로 격자 없애기
plt.show()

# 픽셀 값의 범위를 0~1 사이로 조정
train_images = train_images / 255.0
test_images = test_images / 255.0

# 훈련 세트에서 처음 25개의 이미지와 그 아래 클래스 이름 출력
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# 모델 구성
# 신경망 모델을 순서대로 구성
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# 모델 컴파일
# compile() 메서드를 이용해서 모델을 훈련하는데 사용할 옵티마이저, 손실 함수, 지표를 설정
model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

# 모델 피드
# fit() 메서드에 훈련에 사용할 이미지 데이터와 레이블을 입력
# 에포크 (epochs)는 60000개의 전체 이미지를 몇 번 학습할지 설정
model.fit(train_images, train_labels, epochs=10)

# 정확도 평가
# evaluate() 메서드를 이용해서 손실(loss)과 정확도(accuracy)를 각각 얻을 수 있음
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# 예측
# predict() 메서드를 사용하면 모델이 각 이미지의 클래스를 예측하는 결과를 확인할 수 있음.
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
predictions[0]
np.argmax(predictions[0])
test_labels[0]


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# 예측확인
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# 신뢰도 점수가 높을 때도 잘못 예측할 수 있음
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()