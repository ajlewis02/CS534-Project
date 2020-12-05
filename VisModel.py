import tensorflow as tf
import matplotlib.pyplot as plt
import GetData

dat, labels = GetData.test1_all()
dat_sets, labels_sets = GetData.every_contiguous_set(5, dat, labels)

model_name = "LSTM_newdata_len5_contiguous_gen5"

labels_sets = [[n[0] * 319.2715, n[1] * 139.36778] for n in labels_sets]

# Normalize to -1 - 1 range
dat_sets = [[[m[0] / 133.078125, m[1] / 26.625] for m in n] for n in dat_sets]
labels_sets = [[n[0] / 92, n[1] / 81] for n in labels_sets]

model = tf.keras.models.load_model(model_name)
model.summary()
guess = model.predict(dat_sets)
print(model.evaluate(dat_sets, labels_sets))

# Reverse normalization for graph
labels_sets = [[n[0] * 92, n[1] * 81] for n in labels_sets]
guess = [[n[0] * 92, n[1] * 81] for n in guess]

plt.plot([n[0] for n in dat][:500], c="r")
plt.plot([n[1] for n in dat][:500], c="y")

plt.plot([n[0] for n in labels_sets][:500], c="g")
plt.plot([n[1] for n in labels_sets][:500], c="b")

plt.plot([n[0] for n in guess][:500], c="lime")
plt.plot([n[1] for n in guess][:500], c="c")


plt.legend(["Bend", "Stretch", "Angle 1", "Angle 2", "Predicted Angle 1", "Predicted Angle 2"])
plt.title("Model " + model_name + " performance")
plt.show()
