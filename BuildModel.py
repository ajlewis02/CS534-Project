import tensorflow as tf
import numpy, GetData
import matplotlib.pyplot as plt
import pandas as pd

dat, labels = GetData.every_contiguous_set(2)

dat = numpy.asarray(dat)
labels = numpy.asarray(labels)

print(dat[0])

modelName = "LSTM_len2_contiguous_gen3"

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, input_shape=(2, 2)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(2)
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
