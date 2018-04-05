import keras


class TuneCallback(keras.callbacks.Callback):
    def __init__(self, reporter, logs={}):
        self.reporter = reporter
        self.iteration = 0

    def on_train_end(self, epoch, logs={}):
        self.reporter(timesteps_total=batch, done=1, mean_accuracy=logs["acc"])

    def on_batch_end(self, batch, logs={}):
        self.iteration += 1
        self.reporter(timesteps_total=self.iteration, mean_accuracy=logs["acc"])
