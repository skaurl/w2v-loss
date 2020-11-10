import sys
import os
sys.path.append('..')
try:
    import urllib.request
except ImportError:
    raise ImportError('Use Python3!')
import pickle
import pandas as pd
import numpy as np
import re
from konlpy.tag import Mecab
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams

url_base = 'https://raw.githubusercontent.com/e9t/nsmc/master/'

dataset_dir = os.path.dirname(os.path.abspath(__file__))

mecab = Mecab()

def _download():
    file_path = dataset_dir + '/' + "ratings.txt"
    if os.path.exists(file_path):
        return
    print('Downloading ...')
    try:
        urllib.request.urlretrieve(url_base + "ratings.txt", file_path)
    except urllib.error.URLError:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(url_base + "ratings.txt", file_path)
    print('Done')

def main():
    _download()
    file_path = dataset_dir + '/' + "ratings.txt"

    data = pd.read_csv(file_path, sep='\t', engine='python')[:100]

    for i in range(len(data)):
        data.iloc[i, 1] = ' '.join(re.sub(r'[^0-9a-zA-Z가-힣]', ' ', str(data.iloc[i, 1]).strip()).split())
        data.iloc[i, 1] = " ".join(mecab.morphs(data.iloc[i, 1]))

    df = data["document"].apply(lambda x: x.split())
    df = df.to_list()

    drop_train = [index for index, sentence in enumerate(df) if len(sentence) <= 1]
    df = np.delete(df, drop_train, axis=0)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df)

    word2idx = tokenizer.word_index
    idx2word = {v: k for k, v in word2idx.items()}

    with open(dataset_dir + '/' + "word2idx.pickle", 'wb') as f:
        pickle.dump(word2idx, f)

    with open(dataset_dir + '/' + "idx2word.pickle", 'wb') as f:
        pickle.dump(idx2word, f)

    df = tokenizer.texts_to_sequences(df)

    vocab_size = len(word2idx) + 1

    skip_grams = [skipgrams(sample, vocabulary_size=vocab_size, window_size=2) for sample in df]

    inout = []

    for i in range(len(skip_grams)):
        inout.extend(skip_grams[i][0])

    input = []
    output = []

    for i in range(len(inout)):
        inin = [0] * (vocab_size - 1)
        inin[inout[i][0] - 1] = 1
        input.append(inin)

        outout = [0] * (vocab_size - 1)
        outout[inout[i][1] - 1] = 1
        output.append(outout)

    input = np.array(input)
    output = np.array(output)

    np.save(dataset_dir + '/' + "input", input)
    np.save(dataset_dir + '/' + "output", input)

    return df, word2idx, idx2word, input, output

if __name__ == '__main__':
    main()