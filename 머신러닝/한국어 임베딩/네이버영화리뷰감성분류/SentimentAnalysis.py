import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ------------------네이버 영화 리뷰 감성 분류 ------------------
# 데이터 다운로드 링크 : https://github.com/e9t/nsmc/

# 데이터 로드 함수
def load_data(url, filename):
    urllib.request.urlretrieve(url, filename=filename)
    print('데이터 다운로드 완료!')

# 전체 샘플 중 max_len 이하인 샘플의 비율
def below_threshold_len(max_len, nested_list):
    count = 0
    for sentence in nested_list:
        if(len(sentence) <= max_len):
            count = count + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))



def sentiment_predict(new_sentence):
    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
    okt = Okt()
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
    new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
    score = float(loaded_model.predict(pad_new)) # 예측
    if(score > 0.5):
        print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))


if __name__ == "__main__":
    # 1. 해당 url로 부터 데이터 로드
    # data_url = {"https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt": "ratings_train.txt",
    #        "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt": "ratings_test.txt"}
    # for url, filename in data_url.items():
    #     load_data(url, filename)

    train_data = pd.read_table('ratings_train.txt') # 훈련용 리뷰 개수 : 150,000
    test_data = pd.read_table('ratings_test.txt')   # 테스트용 리뷰 개서 : 50,000

    # 2. 데이터 정제하기
    # 트레인 데이터의 중복 유무
    # print((train_data['document'].nunique(), train_data['label'].nunique()))    # (146182, 2)
    # document 열의 중복 제거
    train_data.drop_duplicates(subset=['document'], inplace=True)    # 146183개
    test_data.drop_duplicates(subset=['document'], inplace=True)    # 146183개

    # print(train_data.isnull().sum())    # document에 1개 null값 존재
    # null값 제거
    train_data = train_data.dropna(how='any').copy()
    test_data = test_data.dropna(how='any').copy()

    # 한글과 공백을 제외하고 모두 제거
    train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
    test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")

    # 공백만 있거나 빈값을 가진 행이 있다면 Null로 변경
    train_data['document'] = train_data['document'].str.replace('^ +', "")    # white space 데이터를 empty value로 변경
    test_data['document'] = test_data['document'].str.replace('^ +', "")  # 공백은 empty 값으로 변경

      # Null 값 제거
    # 문자없으면 => null
    train_data['document'].replace('', np.nan, inplace=True)
    test_data['document'].replace('', np.nan, inplace=True)  # 공백은 Null 값으로 변경

    train_data = train_data.dropna(how='any')
    test_data = test_data.dropna(how='any')
    #
    # # 3. 토큰화
    # # 불용어
    # stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
    # okt = Okt()
    # X_train = []
    # X_test = []
    # # train 데이터 토큰화
    # for sentence in tqdm(train_data['document']):
    #     # 토큰화
    #     tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    #     stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    #     X_train.append(stopwords_removed_sentence)
    # # test 데이터 토큰화
    # for sentence in tqdm(test_data['document']):
    #     tokenized_sentence = okt.morphs(sentence, stem=True)  # 토큰화
    #     stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]  # 불용어 제거
    #     X_test.append(stopwords_removed_sentence)
    #
    # with open('train.pickle', 'wb') as f:
    #     pickle.dump(X_train, f)
    # with open('test.pickle', 'wb') as f:
    #     pickle.dump(X_test, f)

    # 4. 정수 인코딩
    with open('train.pickle', 'rb') as f:
        X_train = pickle.load(f)
    with open('test.pickle', 'rb') as f:
        X_test = pickle.load(f)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    threshold = 3
    total_cnt = len(tokenizer.word_index)
    rare_cnt = 0    # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0   # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq += value

        if value < threshold:
            rare_cnt += 1
            rare_freq += value

    print('단어 집합(vocabulary)의 크기 :', total_cnt) # 43752
    print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s' % (threshold - 1, rare_cnt))    # 24337
    print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)   # 55.62488571950996
    print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100) # 1.8715872104872904

    # 전체 단어 개수 중 빈도수 2이하인 단어는 제거
    # 0번 패딩 토큰을 고려하여 + 1
    vocab_size = total_cnt - rare_cnt + 1
    print('단어 집합의 크기 :', vocab_size)     # 19416

    # 케라스 토크나이저의 인자로 넘겨주고 텍스트 시퀀스를 정수 시퀀스로 변환
    tokenizer = Tokenizer(vocab_size)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    print(X_train[:3])

    y_train = np.array(train_data['label'])
    y_test = np.array(test_data['label'])

    # 5. 빈 샘플(empty samples) 제거
    drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
    # numpy이용 빈 샘플 제거
    X_train = np.delete(X_train, drop_train, axis=0)
    y_train = np.delete(y_train, drop_train, axis=0)

    # 6. 패딩
    # print('리뷰의 최대 길이 :', max(len(review) for review in X_train))
    # print('리뷰의 평균 길이 :', sum(map(len, X_train)) / len(X_train))
    # plt.hist([len(review) for review in X_train], bins=50)
    # plt.xlabel('length of samples')
    # plt.ylabel('number of samples')
    # plt.show()

    max_len = 30
    below_threshold_len(max_len, X_train)
    # 샘플의 길이를 30으로 맞춘다.
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)
    print(X_train)
    print(X_test)
    # ---------- LSTM으로 네이버 영화 리뷰 감성 분류하기 ----------

    # 하이퍼 파라미터
    # 임베딩 벡터의 차원 : 100, 은닉 상태의 크기 : 128, 모델 : 다 대 일 구조의 LSTM
    # 활성화 함수 : 시그모이드 함수
    # Why? 해당 모델 -> 마지막 시점에서 두 개의 선택지 중 하나를 예측하는 이진 분류 문제를 수행하는 모델
    # 이진 분류 -> 출력 층에 로지스틱 회귀를 사용해야 함. OK?
    # 손실 함수 : 크로스 엔트로피 함수, 하이퍼파라미터 - 배치 크기 : 64, 에포크 : 15

    embedding_dim = 100
    hidden_units = 128

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(LSTM(hidden_units))
    model.add(Dense(1, activation='sigmoid'))
    # 검증데이터 손실(val_loss)이 증가하면, 과적합 징후므로 검증 데이터 손실이 4회 증가하면
    # 정해진 에포크가 도달하지 못하였더라도 학습을 조기 종료(Early Stopping)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    # ModelCheckpoint를 사용하여 검증 데이터의 정확도(val_acc)가 이전보다 좋아질 경우에만 모델을 저장
    mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    # 훈련 데이터의 20%를 검증 데이터로 분리해서 사용
    # 검증 데이터를 통해서 훈련이 적절히 되고 있는지 확인
    history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)

    loaded_model = load_model('best_model.h5')
    print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))