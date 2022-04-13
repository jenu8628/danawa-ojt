from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from soyspacing.countbase import CountSpace
import math

#----------------------------------------------------------------------------------------------------------------------
# # soynlp 비지도 기반 형태소 분석기
corpus_fname = 'processed_ratings.txt'
# model_fname = 'soyword.model'
#
# # 예제 파일 문장 리스트로 저장
# sentences = [sent.strip() for sent in open(corpus_fname, 'r', encoding='UTF8').readlines()]
# # 객체 선언
# word_extractor = WordExtractor(min_frequency=100,
#                                min_cohesion_forward=0.05,
#                                min_right_branching_entropy=0.0)
# word_extractor.train(sentences)     # 모델 학습
# word_extractor.save(model_fname)    # 모델 저장
#
# scores = word_extractor.word_scores()
# scores = {key:(scores[key].cohesion_forward * math.exp(scores[key].right_branching_entropy)) for key in scores.keys()}
#
# tokenizer = LTokenizer(scores=scores)
# tokens = tokenizer.tokenize("애비는 종이었다")
# print(tokens)

#----------------------------------------------------------------------------------------------------------------------
# soynlp 띄어쓰기 교정
model = CountSpace()
model.train(corpus_fname)
model.save_model('space-correct.model', json_format=False)
model.load_model('space-correct.model', json_format=False)
model.correct("어릴때보고 지금다시봐도 재밌어요")
print(model.correct("어릴때보고 지금다시봐도 재밌어요"))