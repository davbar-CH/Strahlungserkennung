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

image = cv2.imread(r"C:\temp\Testpack\DSC_0082.JPG")

# Cut off borders (adjust border sizes as needed)
border_top = 50     # pixels to cut from top
border_bottom = 50  # pixels to cut from bottom  
border_left = 50    # pixels to cut from left
border_right = 50   # pixels to cut from right

# Apply border cropping
height, width = image.shape[:2]
image = image[border_top:height-border_bottom, border_left:width-border_right]

# Resize image to make it smaller (adjust scale_factor as needed)
scale_factor = 0.5  # Resize to 50% of original size
new_width = int(image.shape[1] * scale_factor)
new_height = int(image.shape[0] * scale_factor)
image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

# Scale the polygon points accordingly and adjust for border cropping
punkte = np.array([
    [715, 1450],
    [3035, 188],
    [4666, 2228],
    [2379, 3989]
], dtype=np.int32)

# Adjust points for border cropping first, then scale
punkte[:, 0] -= border_left   # Adjust x coordinates
punkte[:, 1] -= border_top    # Adjust y coordinates
punkte = (punkte * scale_factor).astype(np.int32)

# Debug: Print image properties
print("[DEBUG] image dtype:", image.dtype)
print("[DEBUG] image shape:", image.shape)
print("[DEBUG] image min:", image.min())
print("[DEBUG] image max:", image.max())

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

print("[DEBUG] cropped image dtype:", imageArray.dtype)
print("[DEBUG] cropped image shape:", imageArray.shape)
print("[DEBUG] cropped image min:", imageArray.min())
print("[DEBUG] cropped image max:", imageArray.max())

# --- collapse array (sum all color channels to make grayscale)
aGrayScaleArray = numpy.sum(imageArray,axis=2).astype(numpy.int64)
# Debug: Print grayscale stats
print("[DEBUG] grayscale array dtype:", aGrayScaleArray.dtype)
print("[DEBUG] grayscale array shape:", aGrayScaleArray.shape)
print("[DEBUG] grayscale min:", aGrayScaleArray.min())
print("[DEBUG] grayscale max:", aGrayScaleArray.max())
# --- find threshold to separate objects from background
threshold = 260
# Debug: Print threshold value
print("[DEBUG] threshold value:", threshold)
# ---- threshold array
aBinaryArray = aGrayScaleArray>threshold
# Debug: Print how many pixels are above threshold
print("[DEBUG] Number of pixels above threshold:", np.sum(aBinaryArray))
# ---- run connectivity filter using 2D-cross structure element
aStructure = ndimage.generate_binary_structure(2, 1)
aVal = ndimage.label(aBinaryArray, aStructure, output=None)
# Debug: Print unique labels found
print("[DEBUG] Unique labels found:", np.unique(aVal[0]))
# ---- get object count
objectCount = len(set(aVal[0].flatten())) - 1

print("[DEBUG] Object count:", objectCount)

colored_labels = colors.ListedColormap(numpy.random.rand(objectCount + 1, 3))
bounds = numpy.arange(objectCount + 2) - 0.5
norm = colors.BoundaryNorm(bounds, colored_labels.N)

# Visualisierung
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

# ---- get object dimensions
# aVal[0] is a labeled image where each object has a separate label (incl. background)
objectLabels = numpy.unique(aVal[0])
print("object labels: ",objectLabels)
fig.patch.set_facecolor('lightgrey')
# Optimized object property extraction
object_slices = ndimage.find_objects(aVal[0])
cnt = 0
for label in objectLabels[1:]:
    slc = object_slices[label - 1]  # label indices start at 1
    if slc is None:
        continue
    mask = (aVal[0][slc] == label)
    pixel_count = np.count_nonzero(mask)
    if pixel_count == 0:
        continue
    coords = np.argwhere(mask)
    # Offset coordinates by slice start
    coords += np.array([slc[0].start, slc[1].start])
    min_pt = coords.min(axis=0)
    max_pt = coords.max(axis=0)
    length = np.linalg.norm(max_pt - min_pt)
    new_colors = np.vstack([colored_labels.colors, [[0.0, 0.0, 0.0]]])  # Add black
    new_cmap = colors.ListedColormap(new_colors)
    new_norm = colors.BoundaryNorm(np.arange(objectCount + 3) - 0.5, new_cmap.N)
    if 100 < length < 120 and pixel_count < 2500:
        print(f"object {label} has {pixel_count} pixels and is {length:.2f} pixels long.")
        axes[3].imshow(aVal[0], cmap=new_cmap, norm=new_norm)
        axes[3].set_title(f"Gelabelte Objekte (Anzahl: {cnt+1})")
        axes[3].axis('off')
        axes[3].set_facecolor('black')  # Set axes background to black
        cnt += 1





# Erzeuge eine farbige Darstellung des gelabelten Bildes
# Ignoriere Label 0 (Hintergrund)


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



plt.tight_layout()
plt.show()
