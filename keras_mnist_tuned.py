'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras_callback import TuneCallback


def train_mnist(args, cfg, reporter):
    K.set_session(
        K.tf.Session(
            config=K.tf.ConfigProto(
                intra_op_parallelism_threads=args.threads,
                inter_op_parallelism_threads=args.threads)))
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
    model.add(
        Conv2D(
            32,
            kernel_size=(args.kernel1, args.kernel1),
            activation='relu',
            input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (args.kernel2, args.kernel2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(args.poolsize, args.poolsize)))
    model.add(Dropout(args.dropout1))
    model.add(Flatten())
    model.add(Dense(args.hidden, activation='relu'))
    model.add(Dropout(args.dropout2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.SGD(lr=args.lr, momentum=args.momentum),
        metrics=['accuracy'])

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(x_test, y_test),
        callbacks=[TuneCallback(reporter)])


if __name__ == '__main__':
    from helper import create_parser
    parser = create_parser()
    args = parser.parse_args()
    mnist.load_data()  # we do this because it's not threadsafe

    import ray
    from ray import tune
    from ray.tune.async_hyperband import AsyncHyperBandScheduler

    ray.init()
    #ray.init(redis_address="localhost:6379")
    sched = AsyncHyperBandScheduler(
        time_attr="timesteps_total",
        reward_attr="mean_accuracy",
        max_t=400,
        grace_period=20)
    tune.register_trainable("train_mnist",
                            lambda cfg, rprtr: train_mnist(args, cfg, rprtr))
    tune.run_experiments(
        {
            "exp": {
                "stop": {
                    "mean_accuracy": 0.99,
                    "timesteps_total": 300
                },
                "run": "train_mnist",
                "repeat": 100,
                "config": {
                    "lr": lambda spec: np.random.uniform(0.001, 0.1),
                    "momentum": lambda spec: np.random.uniform(0.1, 0.9),
                    "hidden": lambda spec: np.random.randint(32, 512),
                    "dropout1": lambda spec: np.random.uniform(0.2, 0.8),
                }
            }
        },
        verbose=0,
        scheduler=sched)
