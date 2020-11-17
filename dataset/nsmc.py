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
from tqdm import tqdm
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

    data = pd.read_csv(file_path, sep='\t', engine='python')[:1000]

    for i in tqdm(range(len(data))):
        data.iloc[i, 1] = ' '.join(re.sub(r'[^가-힣]', ' ', str(data.iloc[i, 1]).strip()).split())
        data.iloc[i, 1] = " ".join(mecab.morphs(data.iloc[i, 1]))

    df = data["document"].apply(lambda x: x.split())
    df = df.to_list()

    stopwords = pd.read_csv(dataset_dir + '/' + "stopwords.csv", encoding="utf-8")
    stopwords = list(stopwords["stopwords"])

    for i in tqdm(range(len(df))):
        for j in range(len(stopwords)):
            while stopwords[j] in df[i]:
                df[i].remove(stopwords[j])

    drop_train = [index for index, sentence in enumerate(df) if len(sentence) <= 1]
    df = np.delete(df, drop_train, axis=0)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df)

    word2idx = tokenizer.word_index
    print(len(word2idx))
    with open(dataset_dir + '/' + "word2idx.pickle", 'wb') as f:
        pickle.dump(word2idx, f)

    encoded = tokenizer.texts_to_sequences(df)
    with open(dataset_dir + '/' + "encoded.pickle", 'wb') as f:
        pickle.dump(encoded, f)

    df = tokenizer.texts_to_sequences(df)
    with open(dataset_dir + '/' + "df.pickle", 'wb') as f:
        pickle.dump(df, f)

    with open(dataset_dir + '/' + "encoded.pickle", 'rb') as f:
        encoded = pickle.load(f)

    skip_grams = [skipgrams(sample, vocabulary_size=2957, window_size=2) for sample in encoded]
    with open(dataset_dir + '/' + "skip_grams.pickle", 'wb') as f:
        pickle.dump(skip_grams, f)

if __name__ == '__main__':
    main()