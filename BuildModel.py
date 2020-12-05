import tensorflow as tf
import numpy, GetData
import matplotlib.pyplot as plt
import pandas as pd


dat, labels = GetData.test1_all()
dat, labels = GetData.every_contiguous_set(5, dat, labels)

labels = [[n[0] * 319.2715, n[1] * 139.36778] for n in labels]  # Convert to degrees

# Normalize to -1 - 1 range
dat = [[[m[0] / 133.078125, m[1] / 26.625] for m in n] for n in dat]
labels = [[n[0] / 92, n[1] / 81] for n in labels]

dat = numpy.asarray(dat)
labels = numpy.asarray(labels)

modelName = "LSTM_newdata_len5_contiguous_gen5"

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, input_shape=(5, 2), activation="tanh"),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(2, activation="tanh")
],
                            name=modelName)  # TODO: Make sure you actually build your model before trying to train it!
model.summary()

model.compile(
    optimizer=tf.optimizers.Adam(0.001),
    loss=tf.keras.losses.mean_squared_error
)

history = model.fit(dat, labels, epochs=60, verbose=0, validation_split=0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.grid(True)

plt.show()

model.save(modelName)
