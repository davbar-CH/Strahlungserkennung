#! /usr/bin/env python
"""
automated thresholding using otsu:
https://scipy-lectures.org/packages/scikit-image/auto_examples/plot_threshold.html
structure element for connectivity:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html
connected component filter:
https://docs.scipy.org/doc/scipy-1.2.3/reference/generated/scipy.ndimage.label.html
"""
import numpy
import numpy as np
from scipy import ndimage
from PIL import Image
from skimage import filters
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2

image = cv2.imread(r"D:\Matura Data komplett\Testpack\Testpack\DSC_0082.JPG")
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


# --- get image data as numpy array (if color tiff may be a 4D RGBA array)
imageArray = numpy.array(cropped)

# --- collapse array (sum all color channels to make grayscale)
aGrayScaleArray = numpy.sum(imageArray,axis=2).astype(numpy.int64)
# --- find threshold to separate objects from background
threshold = 280
# ---- threshold array
aBinaryArray = aGrayScaleArray>threshold
# ---- run connectivity filter using 2D-cross structure element
aStructure = ndimage.generate_binary_structure(2, 1)
aVal = ndimage.label(aBinaryArray, aStructure, output=None)
# ---- get object count
objectCount = len(set(aVal[0].flatten())) - 1

colored_labels = colors.ListedColormap(plt.cm.tab20.colors[:objectCount + 1])


# Visualisierung
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

Anfangspunkte = []
Endpunkte = []
# ---- get object dimensions
# aVal[0] is a labeled image where each object has a separate label (incl. background)
objectLabels = numpy.unique(aVal[0])
print("object labels: ",objectLabels)
for xx in objectLabels[1:]:
    aObj = aVal[0] == xx
    aP_all = numpy.argwhere(aObj)
    aP0 = numpy.min(aP_all,axis=0)
    aP1 = numpy.max(aP_all,axis=0)
    aLength = numpy.linalg.norm(aP0-aP1)
    if (100 < aLength < 250) and (300 < numpy.count_nonzero(aObj) < 4500):
        print("object ", xx, " has ", numpy.count_nonzero(aObj), " pixels and is ", aLength, " pixels long.")
        Anfangspunkte.append(aP0)
        Endpunkte.append(aP1)
        axes[3].plot((aP0[1], aP1[1]), (aP0[0], aP1[0]), color='red', linewidth=2)

    else:
        pass

for i, punkt in enumerate(Anfangspunkte):
    for j, vergleichspunkt in enumerate(Anfangspunkte + Endpunkte):
        # Berechne die Differenz zwischen Punkten
        differenz = numpy.linalg.norm(punkt - vergleichspunkt)
        # Überprüfe, ob die Differenz höchstens 2 beträgt
        if differenz <= 1:
            print(f"Punkt {punkt} (Index {i}) ähnelt Punkt {vergleichspunkt} (Index {j}) mit einer Differenz von {differenz:.2f}.")


# Originalbild
axes[0].imshow(cropped)
axes[0].set_title("Originalbild")
axes[0].axis('off')

# Graustufenbild
axes[1].imshow(aGrayScaleArray, cmap='gray')
axes[1].set_title("Graustufenbild")
axes[1].axis('off')

# Binärbild
axes[2].imshow(aBinaryArray, cmap='gray')
axes[2].set_title("Binärbild (nach Otsu-Threshold)")
axes[2].axis('off')

# Gelabelte Objekte
axes[3].set_facecolor('#20423c')
axes[3].imshow(aGrayScaleArray, cmap='gray')#hier statt alle gelabelte objetke nur die, die auch zeile 77 erfüllen

axes[3].set_title(f"Gelabelte Objekte (Anzahl: {objectCount})")
axes[3].axis('off')


plt.tight_layout()
plt.show()
