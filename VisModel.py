import tensorflow as tf
import matplotlib.pyplot as plt
import GetData

dat, labels = GetData.test1_all()
dat_sets, labels_sets = GetData.every_contiguous_set(20, dat, labels)

model_name = "LSTM_newdata_len20_contiguous_gen14"

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

# Pad for alignment
labels_sets = [[0, 0] for n in range(20)] + labels_sets

plt.plot([n[0] for n in dat][:500], c="r")
plt.plot([n[1] for n in dat][:500], c="y")

plt.plot([n[0] for n in labels_sets][:500], c="g")
plt.plot([n[1] for n in labels_sets][:500], c="b")

plt.plot([n[0] for n in guess][:500], c="lime")
plt.plot([n[1] for n in guess][:500], c="c")

n_within_thresh = [0, 0]
m_err = [0, 0]
total_err = [0, 0]
n_total = 0
thresh = 2

for n in range(len(guess)):
    n_total += 1
    if abs(guess[n][0] - labels_sets[n][0]) < thresh:
        n_within_thresh[0] += 1
    if abs(guess[n][1] - labels_sets[n][1]) < thresh:
        n_within_thresh[1] += 1

    total_err[0] += abs(guess[n][0] - labels_sets[n][0])
    total_err[1] += abs(guess[n][1] - labels_sets[n][1])

    if abs(guess[n][0] - labels_sets[n][0]) > m_err[0]:
        m_err[0] = abs(guess[n][0] - labels_sets[n][0])

    if abs(guess[n][1] - labels_sets[n][1]) > m_err[1]:
        m_err[1] = abs(guess[n][1] - labels_sets[n][1])

print()
print(n_within_thresh[0]/n_total * 100)
print(n_within_thresh[1]/n_total * 100)
print()
print(total_err[0]/n_total)
print(total_err[1]/n_total)
print()
print(m_err[0])
print(m_err[1])
print()
print(n_total)

plt.legend(["Bend", "Stretch", "Angle 1", "Angle 2", "Predicted Angle 1", "Predicted Angle 2"])
plt.title("Model " + model_name + " performance")
plt.show()
