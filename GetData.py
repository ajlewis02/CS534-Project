import csv


def old_data_raw():
    '''Returns two lists:
    dat, a 2d array which contains a [bend, stretch] pair for every timestep in the original dataset
    labels, a 2d array which contains a [angle1, angle2] pair for every timestep in the original dataset'''
    dat = []
    labels = []
    with open("old_data.csv", "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row in reader:
            dat.append([float(row[0]), float(row[1])])
            labels.append([float(row[2]), float(row[5])])

    return dat, labels


def every_contiguous_set(length):
    """
    Returns all contiguous sets of size length in the original dataset
    :param length: length of the sets of data
    :return: ([[[bend1, stretch1], [bend2, stretch2], ...], [[bend1, stretch1], ...], ...], [[avg_angle1, avg_angle2], ...])
    """
    dat, labels = old_data_raw()
    dat_sets = []
    label_sets = []
    for i in range(len(dat)-(length-1)):
        dat_sets.append([dat[j] for j in range(i, i + length)])
        label_sets.append([avg([labels[j][0] for j in range(i, i+length)]), avg([labels[j][1] for j in range(i, i+length)])])

    return dat_sets, label_sets


def avg(n):
    return sum(n)/len(n)
