import os
import argparse
import joblib
import numpy as np
import pickle
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, TerminateOnNaN
import archs
from scheduler import *

arch_names = archs.__dict__.keys()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='w2v_sphereface',
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
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--min-lr', default=1e-3, type=float,
                        help='minimum learning rate')
    parser.add_argument('--momentum', default=0.5, type=float)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
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

    X = np.load(os.path.dirname(os.path.abspath(__file__))[:-8] + "dataset/" + "input.npy")
    X = X.reshape(X.shape[0],X.shape[1],1,1)
    y = np.load(os.path.dirname(os.path.abspath(__file__))[:-8] + "dataset/" + "output.npy")

    print(X.shape)

    if args.optimizer == 'SGD':
        optimizer = SGD(lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = Adam(lr=args.lr)
    model = archs.__dict__[args.arch](args)
    model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
    model.summary()
    callbacks = [
        ModelCheckpoint(os.path.join('models', args.name, 'model.hdf5'),
            verbose=1, save_best_only=True),
        CSVLogger(os.path.join('models', args.name, 'log.csv')),
        TerminateOnNaN()]
    if args.scheduler == 'CosineAnnealing':
        callbacks.append(CosineAnnealingScheduler(T_max=args.epochs, eta_max=args.lr, eta_min=args.min_lr, verbose=1))
    if 'face' in args.arch:
        # callbacks.append(LambdaCallback(on_batch_end=lambda batch, logs: print('W has nan value!!') if np.sum(np.isnan(model.layers[-4].get_weights()[0])) > 0 else 0))
        model.fit([X, y], y,
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

    with open(os.path.dirname(os.path.abspath(__file__)) + '/' + args.arch + "_vector.pickle", 'wb') as f:
        pickle.dump(model.get_weights(), f)

if __name__ == '__main__':
    main()