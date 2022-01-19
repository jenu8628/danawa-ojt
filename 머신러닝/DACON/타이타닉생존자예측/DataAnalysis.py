import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnlysis:
    # 데이터 분석, 그래프 클래스
    plt.figure(figsize=(12, 12))

    def __init__(self, train, test):
        self.train = train
        self.test = test

    def correlation(self):
        # 상관관계 그래프 생성
        sns.heatmap(data=self.train.corr(), annot=True)
        plt.show()



if __name__ == "__main__":
    train = ""
    test = ""
    if len(sys.argv) == 3:
        train = sys.argv[1]
        test = sys.argv[2]
    else:
        print("usage : $ python analysis.py <train_csv> <train_csv> <train_csv> <train_csv>")
        exit(0)

    train = pd.read_csv(train)
    test = pd.read_csv(test)
    train.info()

    analyze = DataAnlysis(train, test)
    # analyze.correlation()   # 상광관계 그래프
