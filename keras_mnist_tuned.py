import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


def train_mnist(cfg, reporter):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(cfg["ksize1"], cfg["ksize1"]),
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (cfg["ksize2"], cfg["ksize1"]), activation='relu'))
    model.add(MaxPooling2D(pool_size=(cfg["poolsize"], cfg["poolsize"])))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    for i in range(20):
        model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=0, steps_per_epoch=5)
        score = model.evaluate(x_test, y_test, verbose=0)
        reporter(timesteps_total=i, mean_accuracy=score)


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
