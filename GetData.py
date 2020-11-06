import csv


def old_data_raw():
    '''Returns two lists:
    dat, a 2d array which contains a [bend, stretch] pair for every timestep in the original dataset
    labels, a 2d array which contains a [angle1, angle2] pair for every timestep in the original dataset'''
    dat = []
    labels = []
    with open("old_data.csv", "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  #Skip the header
        for row in reader:
            dat.append([float(row[0]), float(row[1])])
            labels.append([float(row[2]), float(row[5])])

    return dat, labels
