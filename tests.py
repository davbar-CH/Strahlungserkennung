import matplotlib.pyplot as plt

import skimage as ski

import numpy
import numpy as np
from scipy import ndimage
from PIL import Image
from skimage import filters
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2

image = cv2.imread(f"D:\Matura Data komplett\Testpack\Testpack\DSC_0081.JPG")

punkte = np.array([
    [715, 1450],
    [3035, 188],
    [4666, 2228],
    [2379, 3989]
], dtype=np.int32)

punkte = punkte.reshape((-1, 1, 2))

# Maske erzeugen
maske = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.fillPoly(maske, [punkte], 255)

# Bild maskieren (alles außerhalb schwarz)
masked = cv2.bitwise_and(image, image, mask=maske)

# Bounding Box berechnen
x, y, w, h = cv2.boundingRect(punkte)

# Nur den Bereich innerhalb der Bounding Box ausschneiden
cropped = masked[y:y+h, x:x+w]

imageArray = numpy.array(cropped)

aGrayScaleArray = numpy.sum(imageArray,axis=2).astype(numpy.int64)

fig, ax = ski.filters.try_all_threshold(aGrayScaleArray, figsize=(10, 8), verbose=False)

plt.show()