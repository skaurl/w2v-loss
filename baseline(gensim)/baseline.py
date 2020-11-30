import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from konlpy.tag import Mecab
from gensim.models import Word2Vec
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    with tf.device('/device:GPU:0'):
        mecab = Mecab()

        with open("/gdrive/My Drive/MacBook/ratings.txt", 'r') as f:
            data = pd.read_csv(f, sep = '\t')

        corpus = []

        for i in tqdm(range(len(data))):
            data.iloc[i, 1] = ' '.join(re.sub(r'[^0-9a-zA-Z가-힣]', ' ', str(data.iloc[i, 1]).strip()).split())
            corpus.append(mecab.morphs(data.iloc[i, 1]))

        model = Word2Vec(sentences=corpus, size=100, window=2, sg=0)

        x = []
        y = np.array(data['label'])

        max_len = 20 # avg = 16.278745

        for i in range(len(corpus)):
            tmp = corpus[i]
            for j in range(len(tmp)):
                try:
                    tmp[j] = model.wv.get_vector(tmp[j]).tolist()
                except:
                    tmp[j] = [0]*100
            if len(tmp) >= max_len:
                tmp = tmp[-max_len:]
            else:
                for j in range(max_len-len(tmp)):
                    tmp.insert(0,[0]*100)
            x.append(tmp)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

        x_train = np.array(x_train).reshape((len(x_train), max_len*100))
        x_test = np.array(x_test).reshape((len(x_test), max_len*100))

        print('train_shape : {} / {}'.format(x_train.shape, y_train.shape))
        print('test_shape : {} / {}'.format(x_test.shape, y_test.shape))

        model = KNeighborsClassifier(n_neighbors = 3)

        model.fit(x_train, y_train)

        y_test = list(y_test)
        y_pred = list(model.predict(x_test))

        print('accuracy_score = ', accuracy_score(y_test, y_pred))

        print(classification_report(y_test, y_pred))
        print(pd.crosstab(pd.Series(y_test), pd.Series(y_pred), rownames=['True'], colnames=['Predicted']))

if __name__ == '__main__':
    main()