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

url_base = 'https://raw.githubusercontent.com/e9t/nsmc/master/'
key_file = {
    'total':'ratings.txt'
}
save_file = {
    'total':'ratings.npy'
}
vocab_file = 'nsmc.vocab.pkl'
dataset_dir = os.path.dirname(os.path.abspath(__file__))

mecab = Mecab()

def _download(file_name):
    file_path = dataset_dir + '/' + key_file[file_name]
    if os.path.exists(file_path):
        return
    print('Downloading ' + file_name + ' ... ')
    try:
        urllib.request.urlretrieve(url_base + key_file[file_name], file_path)
    except urllib.error.URLError:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(url_base + key_file[file_name], file_path)
    print('Done')

def load_vocab():
    vocab_path = dataset_dir + '/' + vocab_file
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            word_to_id, id_to_word = pickle.load(f)
        return word_to_id, id_to_word
    word_to_id = {}
    id_to_word = {}
    data_type = 'total'
    file_name = key_file[data_type]
    file_path = dataset_dir + '/' + file_name
    _download(data_type)
    df = ''
    for i in pd.read_csv(file_path, sep = '\t', engine='python')['document']:
        text = ' '.join(re.sub(r'[^0-9a-zA-Z가-힣]', ' ', str(i).strip()).split())
        text = ' '.join(mecab.morphs(text))
        df = df + '\n' + str(text)
    words = df.strip().split()
    for i, word in enumerate(words):
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word
    with open(vocab_path, 'wb') as f:
        pickle.dump((word_to_id, id_to_word), f)
    return word_to_id, id_to_word

def load_data(data_type='train'):
    save_path = dataset_dir + '/' + save_file[data_type]
    word_to_id, id_to_word = load_vocab()
    if os.path.exists(save_path):
        corpus = np.load(save_path)
        return corpus, word_to_id, id_to_word
    file_name = data_type
    file_path = dataset_dir + '/' + key_file[data_type]
    _download(file_name)
    df = ''
    for i in pd.read_csv(file_path, sep='\t', engine='python')['document']:
        text = ' '.join(re.sub(r'[^0-9a-zA-Z가-힣]', ' ', str(i).strip()).split())
        text = ' '.join(mecab.morphs(text))
        df = df + '\n' + str(text)
    words = df.strip().split()
    corpus = np.array([word_to_id[w] for w in words])
    np.save(save_path, corpus)
    return corpus, word_to_id, id_to_word

if __name__ == '__main__':
    load_data('total')