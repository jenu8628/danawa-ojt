import pickle
from gensim.models import Word2Vec, KeyedVectors
from konlpy.tag import Mecab

def combine_text(x, y, z):
    filelist = [x, y, z]
    with open('corpus_mecab.txt', 'wb') as out_file:
        for filename in filelist:
            with open(filename, 'r', encoding='utf-8') as file:
                out_file.write(file.read().encode('utf-8'))

def save_model(corpus_fname, model_fname):
    print('corpus 시작')
    corpus = [sent.strip().split(" ") for sent in open(corpus_fname, 'r', encoding="utf-8").readlines()]
    with open('corpus.pickle', 'wb') as f:
        pickle.dump(corpus, f)
    print(corpus[0])
    print('corpus 완료')
    # with open('corpus.pickle', 'rb', encoding='utf-8') as f:
    #     corpus = pickle.load(f)
    # vector_size : 임베딩의 차원 수, workers : cpu 스레드 개수
    # sg : Skip-gram 모델인지 여부를 나타내는 하이퍼 파라미터(1 이면 Skip-gram, 0이면 CBOW 모델)
    print('word2vec 시작')
    model = Word2Vec(corpus, vector_size=100, workers=4, sg=1)
    print('word2vec 완료')
    model.save(model_fname)
    print('모델 저장 완료')



class WordEmbeddingSimilarWord:
    def __init__(self, model_file, dim=100):
        # self.model = KeyedVectors.load(model_file)
        self.model = KeyedVectors.load_word2vec_format(model_file)
        self.dim = dim
        self.tokenizer = Mecab(dicpath='C:/mecab/mecab-ko-dic')
        self.dictionary = self.load_dictionary(self.model)

    def load_dictionary(self, model):
        dictionary = []
        for word in model.wv.index2word:
            dictionary.append(word)
        return dictionary

    def get_sentence_vector(self, sentence):
        tokens = self.tokenizer.nouns(sentence)
        token_vecs = []
        for token in tokens :
            if token in self.dictionary :
                token_vecs.append(token)
            return token_vecs

    def most_similar(self, sentence, topn=10):
        token_vecs = self.get_sentence_vector(sentence)
        return self.model.wv.most_similar(token_vecs, topn=topn)


if __name__ == '__main__':
    # # 텍스트 합치기
    # combine_text('wiki_ko_mecab.txt', 'ratings_mecab.txt', 'korquad_mecab.txt')

    corpus_fname = 'corpus_mecab.txt'
    model_fname = 'word2vec'

    # 모델 생성 및 저장
    save_model(corpus_fname, model_fname)


    # 1번째 방법
    loaded_model = KeyedVectors.load_word2vec_format(model_fname)  # 모델 로드
    print(loaded_model.wv.vectors.shape)
    print(loaded_model.wv.most_similar("최민식", topn=5))
    print(loaded_model.wv.most_similar("남대문", topn=5))
    print(loaded_model.wv.similarity("헐크", '아이언맨'))
    print(loaded_model.wv.most_similar(positive=['어벤져스', '아이언맨'], negative=['스파이더맨'], topn=1))

    # 2번째 방법
    wv = WordEmbeddingSimilarWord(model_fname, 100)
    print(wv.most_similar("아이언맨과 배트맨", topn=5))
    print(wv.most_similar("롯데월드", topn=5))
    print(wv.most_similar("임시정부와 김구", topn=5))
