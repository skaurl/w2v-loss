import os
import argparse
import easydict
import joblib
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TerminateOnNaN, EarlyStopping
import archs
from scheduler import *

'''arch_names = archs.__dict__.keys()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='w2v_arcface',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: w2v)')
    parser.add_argument('--num-features', default=100, type=int,
                        help='dimention of embedded features')
    parser.add_argument('--scheduler', default='CosineAnnealing',
                        choices=['CosineAnnealing', 'None'],
                        help='scheduler: ' +
                            ' | '.join(['CosineAnnealing', 'None']) +
                            ' (default: CosineAnnealing)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=1024, type=int,
                        metavar='N', help='mini-batch size (default: 1024)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--min-lr', default=1e-4, type=float,
                        help='minimum learning rate')
    parser.add_argument('--momentum', default=0.5, type=float)
    args = parser.parse_args()
    return args'''

args = easydict.EasyDict({
    "name":None,
    "arch":"w2v_arcface",
    "num_features":100,
    "scheduler":"CosineAnnealing",
    "epochs":50,
    "batch_size":1024,
    "optimizer":"Adam",
    "lr":1e-2,
    "min_lr":1e-4,
    "momentum":0.5
})

def main():
    #args = parse_args()
    args.name = '%s_%dd' %(args.arch, args.num_features)
    os.makedirs('models/%s' %args.name, exist_ok=True)
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')
    joblib.dump(args, 'models/%s/args.pkl' %args.name)
    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    with open("/gdrive/My Drive/MacBook/skip_grams.pickle", 'rb') as f:
        skip_grams = pickle.load(f)

    first_elem = []
    second_elem = []
    labels = []

    for _, elem in enumerate(skip_grams):
        first_elem.extend(list(zip(*elem[0]))[0])
        second_elem.extend(list(zip(*elem[0]))[1])
        labels.extend(elem[1])

    first_elem = np.array(first_elem, dtype='int32')
    second_elem = np.array(second_elem, dtype='int32')
    labels = tf.keras.utils.to_categorical(labels, 2)

    X = [first_elem, second_elem]
    X_face = [first_elem, second_elem, labels]
    y = labels

    if args.optimizer == 'SGD':
        optimizer = SGD(lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = Adam(lr=args.lr)

    model = archs.__dict__[args.arch](args)
    model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
    model.summary()
    #early_stopping = EarlyStopping(monitor='loss', patience=3)
    callbacks = [
        ModelCheckpoint(os.path.join('models', args.name, 'model.hdf5'),
            verbose=1, save_best_only=True),
        CSVLogger(os.path.join('models', args.name, 'log.csv')),
        TerminateOnNaN()]
    if args.scheduler == 'CosineAnnealing':
        callbacks.append(CosineAnnealingScheduler(T_max=args.epochs, eta_max=args.lr, eta_min=args.min_lr, verbose=1))
    if 'face' in args.arch:
        # callbacks.append(LambdaCallback(on_batch_end=lambda batch, logs: print('W has nan value!!') if np.sum(np.isnan(model.layers[-4].get_weights()[0])) > 0 else 0))
        model.fit(X_face, y,
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1)
    else:
        model.fit(X, y,
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1)

    with open("/gdrive/My Drive/MacBook/" + args.arch + "_vector.pickle", 'wb') as f:
        pickle.dump(model.get_weights(), f)

if __name__ == '__main__':
    main()
