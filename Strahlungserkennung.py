import glob
import os
import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt
from matplotlib import cm
import imutils

class StrahlPunkt:
    def __init__(self, xKoordinate, yKoordinate):
        self.xKoordinate = xKoordinate
        self.yKoordinate = yKoordinate

# Bilderliste und Pfad zum Ordner
bilder = []
folder_path = r"C:\temp\Testpack"
#r"C:\Dokumente 2\Matura\data"

# Bild und Pfad verbinden, zur Liste hinzufügen
for i in glob.glob(os.path.join(folder_path, "DSC_*.JPG")):
    bilder.append(i)

image = cv2.imread(bilder[1],cv2.IMREAD_GRAYSCALE)
rotiert = imutils.rotate(image, 51)

#Bilder nacheinander anzeigen lassen


#[26 26 26] Wert für Pixel 1599 und 2399

# Rhombus-Ecken
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

# Classic straight-line Hough transform
# Set a precision of 0.5 degree.
tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
h, theta, d = hough_line(cropped, theta=tested_angles)

# Generating figure 1
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(cropped, cmap=cm.gray)
ax[0].set_title('Input image')
ax[0].set_axis_off()

"""angle_step = 0.5 * np.diff(theta).mean()
d_step = 0.5 * np.diff(d).mean()
bounds = [
    np.rad2deg(theta[0] - angle_step),
    np.rad2deg(theta[-1] + angle_step),
    d[-1] + d_step,
    d[0] - d_step,
]
ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')"""

for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    x0_int = int(x0)
    y0_int = int(y0)
    x_strahl = x0_int
    #ax[1].axline((x0, y0), slope=-np.tan(angle)**-1)#hier cotan, macht mehr Sinn für mich (weil x*cos(angle) + y*sin(angle) = d)
    q = y0_int -(np.tan(angle + np.pi / 2)*x0_int) #eigentlich cotan, aber so ausgedrückt um 1/0 zu vermeiden
    q_int = int(q)
    bild_wert_liste = cropped[x0_int, y0_int]
    print(bild_wert_liste)

    if bild_wert_liste > 200:
        while bild_wert_liste >= 200:   #200 als Schwellenwert für weiss
            y_strahl = (np.tan(angle + np.pi / 2) * x_strahl) + q_int   #eigentlich cotan, aber so ausgedrückt um 1/0 zu vermeiden
            y_strahl_int = int(y_strahl)
            bild_wert_liste = cropped[x_strahl, y_strahl_int]
            x_strahl = x_strahl + 1
            y_ende = y_strahl
        ax[1].plot((x_strahl, x0_int), (y_ende, y0_int))
    else:
        pass


"""for x_strahl in range(-1, 3, 2):
            bild_wert = cropped[neuervektor[0] + x_strahl, neuervektor[1] - 1]
            if np.all(bild_wert >= 45):
                neuervektor = Fusspunkt + k * (x_strahl, y) #eigentlich (x0 +x) - x0), ((y0+y) - y0) aber vereinfacht

                k = k + 1"""

ax[1].imshow(cropped, cmap=cm.gray)
ax[1].set_ylim((cropped.shape[0], 0))
ax[1].set_axis_off()
ax[1].set_title('Detected lines')

"""width, height = cropped.shape
#print(height, width)
for y in range (0, height):
    for x in range (0, width):
        bild_wert = cropped[x, y]
        #print(bild_wert)
        if np.all(bild_wert >= 45):#45 als Schwellenwert für weiss
            strcord.append(x, y)
            Punkt = StrahlPunkt(x, y)
for i in strcord:
    if math.sqrt((strcord[0] - Fusspunkt[0])**2 + (strcord[1] - Fusspunkt[1])**2) < 2:"""

plt.tight_layout()
plt.show()
