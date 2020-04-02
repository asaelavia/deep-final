from src.moco import EmulateMultiGPUBatchNorm

from src.moco import moco_training_step

import tensorflow as tf

# from tensorflow.keras.layers import Dense, Flatten, Conv2D
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
Conv2D = tf.keras.layers.Conv2D
Model = tf.keras.Model

from src.moco import MoCoQueue, update_model_via_ema

print(Model)
EMBEDDING_DIM = 64
queue = MoCoQueue(EMBEDDING_DIM, 256)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 127.5 - 1, x_test / 127.5 - 1
x_train, x_test = [tf.cast(i, 'float32') for i in (x_train, x_test)]

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(2)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(256, activation='relu')
        self.d2 = Dense(EMBEDDING_DIM, activation=None)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


# Create an instance of the model
model = MyModel()
model_ema = MyModel()

# Initialise the models and make the EMA model 90% similar to the main model
model(x_train[:1])
model_ema(x_train[:1])
update_model_via_ema(model, model_ema, 0.1)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

EPOCHS = 10

for epoch in range(EPOCHS):
    for x, y in train_ds:
        x_aug = x + 0.1 * tf.random.normal(x.shape, dtype='float32')
        moco_training_step(x, x_aug, queue, model, model_ema, optimizer)
print(train_accuracy)
print(test_accuracy)
