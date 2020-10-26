import sys
sys.path.append('..')
import pickle
import pandas as pd
import numpy as np
import re
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

pkl_file = 'cbow_params.pkl'  # or 'skipgram_params.pkl'
txt_file = 'ratings.txt'

mecab = Mecab()

with open(sys.path[1] + '/word2vec/' + pkl_file, 'rb') as f:
    w2v = pickle.load(f)

with open(sys.path[1] + '/dataset/' + txt_file, 'r') as f:
    data = pd.read_csv(f, sep = '\t')

x = []
y = np.array(data['label'])

max_len = 20 #avg = 16.278745

for i in range(len(data)):
    data.iloc[i,1] = ' '.join(re.sub(r'[^0-9a-zA-Z가-힣]', ' ', str(data.iloc[i,1]).strip()).split())
    tmp = mecab.morphs(data.iloc[i,1])
    for j in range(len(tmp)):
        tmp[j] = w2v['word_vecs'][w2v['word_to_id'][tmp[j]]]
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