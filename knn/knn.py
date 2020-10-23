import sys
sys.path.append('..')
import pickle

pkl_file = 'cbow_params.pkl'  # or 'skipgram_params.pkl'

with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

print(data)