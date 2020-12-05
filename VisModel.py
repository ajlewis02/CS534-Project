import tensorflow as tf
import matplotlib.pyplot as plt
import GetData

dat, labels = GetData.test1_all()
dat_sets, labels_sets = GetData.every_contiguous_set(2, dat, labels)

model_name = "LSTM_newdata_len2_contiguous_gen1"

labels_sets = [[m * 100 for m in n] for n in labels_sets]

model = tf.keras.models.load_model(model_name)
model.summary()
guess = model.predict(dat_sets)

#guess = [[m * 100 for m in n] for n in guess]

plt.plot([n[0] for n in dat][:500], c="r")
plt.plot([n[1] for n in dat][:500], c="y")

plt.plot([n[0] for n in labels_sets][:500], c="g")
plt.plot([n[1] for n in labels_sets][:500], c="b")

plt.plot([n[0] for n in guess][:500], c="lime")
plt.plot([n[1] for n in guess][:500], c="c")

print(model.evaluate(dat_sets, labels_sets))

plt.legend(["Bend", "Stretch", "Angle 1", "Angle 2", "Predicted Angle 1", "Predicted Angle 2"])
plt.title("Model " + model_name + " performance")
plt.show()
