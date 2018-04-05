'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras_callback import TuneCallback  # Consider moving this into Tune???


def train_mnist(args, cfg, reporter):
    vars(args).update(cfg)
    batch_size = 128
    num_classes = 10
    epochs = 12

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(args.kernel1, args.kernel1),
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (args.kernel2, args.kernel2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(args.poolsize, args.poolsize)))
    model.add(Dropout(args.dropout1))
    model.add(Flatten())
    model.add(Dense(args.hidden, activation='relu'))
    model.add(Dropout(args.dropout2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(
                      lr=args.lr, momentum=args.momentum),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[TuneCallback(reporter)])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Keras MNIST Example')
    parser.add_argument('--steps', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--kernel1', type=int, default=3,
                        help='Size of first kernel (default: 3)')
    parser.add_argument('--kernel2', type=int, default=3,
                        help='Size of second kernel (default: 3)')
    parser.add_argument('--poolsize', type=int, default=2,
                        help='Size of Pooling (default: 2)')
    parser.add_argument('--dropout1', type=float, default=0.25,
                        help='Size of first kernel (default: 0.25)')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Size of Hidden Layer (default: 128)')
    parser.add_argument('--dropout2', type=float, default=0.5,
                        help='Size of first kernel (default: 0.5)')

    args = parser.parse_args()
    # import ipdb; ipdb.set_trace()
    train_mnist(args)



if __name__ == '__main__':
    import ray
    from ray import tune

    ray.init()
    tune.register_trainable("train_mnist", train_mnist)
    tune.run_experiments({"exp": {
        "run": "train_mnist",
        "config": {
            "ksize1": tune.grid_search([2, 3, 4]),
            "ksize2": tune.grid_search([2, 3, 4]),
            "poolsize": tune.grid_search([2, 3]),
        }
        }})
