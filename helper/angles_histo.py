import csv
import cv2

lines = []
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	measurement = float(line[3])
	measurements.append(measurement)

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# the histogram of the data
n, bins, patches = plt.hist(measurements, 31, facecolor='green', alpha=0.75)

# add a 'best fit' line

plt.grid(True)

plt.show()
