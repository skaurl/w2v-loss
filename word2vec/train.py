import sys
sys.path.append('..')
from common import config
from common.np import *
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from word2vec.cbow import CBOW
from word2vec.skip_gram import SkipGram
from common.util import create_contexts_target, to_cpu, to_gpu
from dataset import nsmc

window_size = 2**1
hidden_size = 100
batch_size = 2**7
max_epoch = 2**2

corpus, word_to_id, id_to_word = nsmc.load_data('total')

vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)
if config.GPU:
    contexts, target = to_gpu(contexts), to_gpu(target)

model = CBOW(vocab_size, hidden_size, window_size, corpus)
# model = SkipGram(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vecs
if config.GPU:
    word_vecs = to_cpu(word_vecs)
params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'cbow_params.pkl'  # or 'skipgram_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)