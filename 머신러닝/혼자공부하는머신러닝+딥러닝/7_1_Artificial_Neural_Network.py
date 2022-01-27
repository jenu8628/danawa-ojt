import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import SGDClassifier

# 패션 MNIST
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
print(train_input.shape, train_target.shape)    # (60000, 28, 28) (60000,)
print(test_input.shape, test_target.shape)      # (10000, 28, 28) (10000,)

fig, axs = plt.subplots(1, 10, figsize=(10, 10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
# plt.show()

print([train_target[i] for i in range(10)])     # [9, 0, 0, 3, 0, 2, 7, 2, 5, 5]
# data
# 0 : 티셔츠, 1: 바지, 2: 스웨터, 3: 드레스, 4: 코트, 5: 샌달, 6: 셔츠, 7: 니커즈, 8: 가방, 9: 앵클부츠
print(np.unique(train_target, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8),
# array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000], dtype=int64))

# 로지스틱 회귀로 패션 아이템 분류하기
train_scaled = train_input/255.0
train_scaled = train_scaled.reshape(-1, 28*28)
print(train_scaled.shape)       # (60000, 784)

sc = SGDClassifier(loss='log', max_iter=5, random_state=42)
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print(np.mean(scores['test_score']))        # 0.8192833333333333

# 밀집층 만들기
# 순서대로 뉴런개수, 뉴런의 출력에 적용할 함수, 입력의 크기를 나타낸다.
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
# 위의 밀집층을 가진 신경망 모델 만들기
model =keras.Sequential(dense)

# 인공 신경망으로 모델만들기
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
# 케라스 모델은 훈련하기 전에 설정 단계가 있다.
# 이런 설정을 model 객체의 compile()메서드에서 수행한다.
# 꼭 지정해야 할 것은 손실 함수의 종류
# 그다음 훈련 과정에서 계산하고 싶은 측정값을 지정
# 'sparse_categorical_crossentropy' : sparse(희소) 다중분류 크로스엔트로피, loss : 손실 함수
# 원-핫 인코딩으로 바꾸지 않고 그냥 사용 할 수 있다.
# 타깃값을 원-핫 인코딩으로 준비했다면 loss='categorical_crossentropy'로 지정
# 케라스는 모델이 훈련할 때 에포크마다 손실 값을 출력해줌
# accuracy는 정확도도 함께 출력해달라는 의미
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)
# 모델 성능 : evaluate() 메서드

print(model.evaluate(val_scaled, val_target))