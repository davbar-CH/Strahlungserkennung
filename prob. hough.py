import cv2
from shapely import Point
from skimage.feature import canny
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from skimage.transform import probabilistic_hough_line
from itertools import chain
from shapely.geometry import LineString
from shapely.ops import unary_union

# hier den jeweiligen Bildpfad einfügen oder ansonsten config file
folder_path = r"C:\Dokumente 2\Matura Data\Matura Data komplett\902D7200\902D7200"


def bilder_einfügen():
    try:
        bilder = []
        for i in glob.glob(os.path.join(folder_path, "DSC_0084.JPG")):
            bilder.append(i)
        print(bilder)
        print(f"Es wurden {len(bilder)} Bilder gefunden!")
        return bilder

    except Exception as e:
        print(f"Fehler beim Einfügen des Bildes aus der Datei:{e}")


bilder = bilder_einfügen()


def bilder_einlesen():
    try:
        image = bilder[0]
        image = cv2.imread(image)
        print(f"Gerade am Bild {0} dran")
        print("geht1")
        return image

    except Exception as e:
        print(f"Fehler beim Einlesen des Bildes:{e}")


def croppen():
    try:
        image_eingelesen = bilder_einlesen()

        # Skalierung des Bildes, damit das Programm schneller ist
        skalierung = 0.5
        new_width = int(image_eingelesen.shape[1] * skalierung)
        new_height = int(image_eingelesen.shape[0] * skalierung)
        image = cv2.resize(image_eingelesen, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Punkte für Ecken des Bildes, bilden Viereck, damit der Rand der Nebelkammer weg ist.
        # Punkte müssen ebenfalls skaliert werden
        punkte = np.array([
            [970, 1550],  # oben links
            [3000, 440],  # oben rechts
            [4400, 2187],  # unten rechts
            [2400, 3760]  # unten links
        ], dtype=np.int32)

        punkte_skal = (punkte * skalierung).astype(np.int32)

        punkte_reshape = punkte_skal.reshape((-1, 1, 2))

        # Maske erzeugen, damit die Punkte das Viereck bilden,
        # damit man das eigentliche Bild vom Hintergrund wegschneiden kann
        maske = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(maske, [punkte_reshape], 255)
        masked = cv2.bitwise_and(image, image, mask=maske)
        x, y, w, h = cv2.boundingRect(punkte_reshape)
        cropped = masked[y:y + h, x:x + w]

        # --- get image data as numpy array (if color tiff may be a 4D RGBA array)
        grau = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        print("geht2")

        return cropped, grau, punkte

    except Exception as e:
        print(f"Fehler beim Croppen:{e}")


def plot_ecken(Startpunkte, Endpunkte):
    start_dic = {Startpunkte[i][0]: Startpunkte[i][1] for i in range(0, len(Startpunkte))}
    end_dic = {Endpunkte[i][0]: Endpunkte[i][1] for i in range(0, len(Endpunkte))}

    x_min = min(chain(start_dic.keys(), end_dic.keys()))  # westlichster Punkt
    y_x_min = start_dic.get(x_min, end_dic.get(x_min))

    x_max = max(chain(start_dic.keys(), end_dic.keys()))  # östlichster Punkt
    y_x_max = start_dic.get(x_max, end_dic.get(x_max))

    y_min = min(chain(start_dic.values(), end_dic.values()))  # nördlichster Punkt
    x_y_min = None
    for key in chain(start_dic, end_dic):
        if start_dic.get(key) == y_min or end_dic.get(key) == y_min:
            x_y_min = key
            break

    y_max = max(chain(start_dic.values(), end_dic.values()))  # südlichster Punkt
    x_y_max = None
    for key in chain(start_dic, end_dic):
        if start_dic.get(key) == y_max or end_dic.get(key) == y_max:
            x_y_max = key
            break

    punkte = np.array([
        [x_y_min, y_min],
        [x_max, y_x_max],
        [x_y_max, y_max],
        [x_min, y_x_min]
    ], dtype=np.int32)

    return punkte


def punkt_auf_gerade(punkte_plot, Startpunkte, Endpunkte):
    geraden_liste = []
    längen_liste = []

    for punkt1, punkt2 in zip(punkte_plot, punkte_plot[1:]):
        m = (punkt1[1] - punkt2[1]) / (punkt1[0] - punkt2[0])
        q = punkt1[1] - (m * punkt1[0])
        ax[2].plot((punkt1[0], punkt2[0]), (punkt1[1], punkt2[1]), color="blue")
        geraden_liste.append((m, q))

    m1 = (punkte_plot[3][1] - punkte_plot[0][1]) / (punkte_plot[3][0] - punkte_plot[0][0])
    q1 = punkte_plot[3][1] - (m1 * punkte_plot[3][0])
    ax[2].plot((punkte_plot[3][0], punkte_plot[0][0]), (punkte_plot[3][1], punkte_plot[0][1]), color="blue")
    geraden_liste.append((m1, q1))

    fragment_linien = []
    for p0, p1 in zip(Startpunkte, Endpunkte):
        zu_nah = False
        for m, q in geraden_liste:
            y_p0 = (m * p0[0]) + q
            y_p1 = (m * p1[0]) + q

            # print((y_p0, p0[1]), (y_p1, p1[1]))
            if abs(p0[1] - y_p0) < 20 and abs(p1[1] - y_p1) < 20:
                zu_nah = True
                break
        if not zu_nah:
            linie = LineString([p0, p1])
            fragment_linien.append(linie)

    buffer_linien = [linie.buffer(0.5) for linie in fragment_linien]
    zusammen_polygone = unary_union(buffer_linien)
    zusammen_linien = zusammen_polygone.boundary

    boundary_punkte = []
    for linie in zusammen_linien.geoms:
        boundary_punkte.append(list(linie.coords))
        x, y = linie.xy
        ax[2].plot(x, y, color="red")

    for punkt in boundary_punkte:
        nördlichster_punkt = min(punkt, key=lambda p: p[1])
        südlichster_punkt = max(punkt, key=lambda p: p[1])
        snördlichster_punkt = Point(nördlichster_punkt)
        ssüdlichster_punkt = Point(südlichster_punkt)
        längen = snördlichster_punkt.distance(ssüdlichster_punkt)
        print(längen)
        längen_liste.append(längen)

    return längen_liste


# Line finding using the Probabilistic Hough Transform
cropped, grau, punkte = croppen()

edges = canny(grau, 4, 25, 40)
lines = probabilistic_hough_line(edges, threshold=5, line_length=30, line_gap=10)

# Generating figure 2
fig, axes = plt.subplots(1, 3, figsize=(15, 10), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(cropped, cmap="gray")
ax[0].set_title('Input image')

ax[1].imshow(edges, cmap="gray")
ax[1].set_title('Canny edges')

ax[2].imshow(edges * 0)
Startpunkte = []
Endpunkte = []
for line in lines:
    p0, p1 = line
    Startpunkte.append(p0)
    Endpunkte.append(p1)

punkte_plot = plot_ecken(Startpunkte, Endpunkte)
punkt_auf_gerade(punkte_plot, Startpunkte, Endpunkte)

ax[2].set_xlim((0, cropped.shape[1]))
ax[2].set_ylim((cropped.shape[0], 0))
ax[2].set_title('Probabilistic Hough')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()
