import tensorflow as tf

def load_data():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    return fashion_mnist.load_data()

def preprocessing(train_images, test_images):
    return train_images/255.0, test_images/255.0

def categorical_model(train_images, train_labels):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5)
    return model


if __name__ == '__main__':
    # 1. Fashion MNIST 데이터셋 불러오기
    (train_images, train_labels), (test_images, test_labels) = load_data()
    print(train_images[0])
    print(train_images.shape)
    # train_images : 0~255사이의 값을 갖는 28 * 28 크기의 NumPy 어레이
    # train_labels : 0~9까지의 정수 값을 갖는 어레이
    # lavels_dict = {0 : T-shirt/top,
    # 1 : Trouser,
    # 2 : Pullover,
    # 3 : Dress,
    # 4 : Coat,
    # 5 : Sandal,
    # 6 : Shirt,
    # 7 : Sneaker,
    # 8 : Bag,
    # 9 : Ankel boot}

    # # 2. 데이터셋 전처리
    # train_images, test_images = preprocessing(train_images, test_images)
    #
    # # 3. 모델링
    # model = categorical_model(train_images, train_labels)
    #
    # # 4. 정확도 평가
    # loss, accuracy = model.evaluate(test_images, test_labels)
    # print(loss, accuracy)