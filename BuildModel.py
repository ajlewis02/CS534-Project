import tensorflow as tf
import numpy, GetData, matplotlib

dat, labels = GetData.old_data_raw()

dat = numpy.array(dat)
labels = numpy.array(labels)

modelName = "SET THIS"

model = tf.keras.Sequential(name=modelName)  # TODO: Make sure you actually build your model before trying to train it!

model.compile(
    optimizer=tf.optimizers.Adam(0.001),
    loss=tf.keras.losses.mean_squared_error
)

model.fit(dat, labels, epochs=20, verbose=0, validation_split=0.2)

model.save(modelName)
