# ! /usr/bin/env python
"""
automated thresholding using otsu:
https://scipy-lectures.org/packages/scikit-image/auto_examples/plot_threshold.html
structure element for connectivity:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html
connected component filter:
https://docs.scipy.org/doc/scipy-1.2.3/reference/generated/scipy.ndimage.label.html
"""
import shapely
import numpy
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
import glob
import os
from pandas import DataFrame

# hier den jeweiligen Bildpfad einfügen oder ansonsten config file
folder_path = r"C:\Dokumente 2\Matura Data\Matura Data komplett\902D7200\Data Modus 2-2"


def bilder_einfügen():
    try:
        bilder = []
        for i in glob.glob(os.path.join(folder_path, "DSC_0367.JPG")):
            bilder.append(i)
        print(bilder)
        print(f"Es wurden {len(bilder)} Bilder gefunden!")
        return bilder

    except Exception as e:
        print(f"Fehler beim Einfügen des Bildes aus der Datei:{e}")


bilder = bilder_einfügen()
i = 0
Zähler_Liste = []
Winkel_Liste = []

while i < len(bilder):
    def bilder_einlesen():
        try:
            image = bilder[i]
            image = cv2.imread(image)
            print(f"Gerade am Bild {i} dran")
            print("geht1")
            return image

        except Exception as e:
            print(f"Fehler beim Einlesen des Bildes:{e}")


    # noinspection PyTypeChecker
    def croppen():
        try:
            image = bilder_einlesen()

            # Skalierung des Bildes, damit das Programm schneller ist
            skalierung = 0.5
            new_width = int(image.shape[1] * skalierung)
            new_height = int(image.shape[0] * skalierung)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Punkte für Ecken des Bildes, bilden Viereck, damit der Rand der Nebelkammer weg ist.
            # Punkte müssen ebenfalls skaliert werden
            punkte = np.array([
                [970, 1550],  # oben links
                [3000, 440],  # oben rechts
                [4400, 2187],  # unten rechts
                [2400, 3760]  # unten links
            ], dtype=np.int32)

            punkte = (punkte * skalierung).astype(np.int32)
            punkte = punkte.reshape((-1, 1, 2))

            # Maske erzeugen, damit die Punkte das Viereck bilden,
            # damit man das eigentliche Bild vom Hintergrund wegschneiden kann
            maske = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(maske, [punkte], 255)

            # Bild maskieren und alles andere wird schwarz
            masked = cv2.bitwise_and(image, image, mask=maske)

            # Viereck berechnen
            x, y, w, h = cv2.boundingRect(punkte)

            # Nur den Bereich innerhalb des Vierecks ausschneiden
            cropped = masked[y:y + h, x:x + w]

            # --- get image data as numpy array (if color tiff may be a 4D RGBA array)
            imageArray = numpy.array(cropped)
            print("geht2")
            return cropped, imageArray

        except Exception as e:
            print(f"Fehler beim Croppen:{e}")


    def labeling():
        try:
            temp, imageArray = croppen()
            # --- collapse array (sum all color channels to make grayscale)
            aGrayScaleArray = numpy.sum(imageArray, axis=2).astype(numpy.int64)
            # treshhold empirisch gefunden, alternativ otsu-methode, dauert aber
            threshold = 220  # 120 für Strahlungserkennung und 300 für Winkel
            # ---- threshold array
            aBinaryArray = aGrayScaleArray > threshold
            # ---- run connectivity filter using 2D-cross structure element
            aStructure = ndimage.generate_binary_structure(2,
                                                           2)  # hier 2,2 weil 2,1 würde Diagonale nicht erkennen
            aVal = ndimage.label(aBinaryArray, aStructure, output=None)

            print("geht3")
            return aVal, aGrayScaleArray, aBinaryArray

        except Exception as e:
            print(f"Fehler beim Labeln der Objekte:{e}")


    # für Winkel ein "ruhigeres" Bild nehmen
    def winkel(Anfangspunkte, Endpunkte):
        try:
            sichere_winkel = []
            mögliche_winkel = []
            def betrag(punkt1, punkt2):
                try:
                    betrag = np.sqrt((punkt1[0] - punkt2[0]) ** 2 + (punkt1[1] - punkt2[1]) ** 2)
                    return betrag

                except Exception as e:
                    print(f"Fehler beim Berechnen des Betrages:{e}")


            def winkel_berechnung(vektor1, vektor2):
                try:
                    betrag_vektor1 = betrag(vektor1[0], vektor1[1])
                    betrag_vektor2 = betrag(vektor2[0], vektor2[1])
                    winkel = np.arccos((np.dot(vektor1, vektor2)) / (betrag_vektor1 * betrag_vektor2))
                    return winkel

                except Exception as e:
                    print(f"Fehler bei der Winkelberechnung:{e}")


            def schnittpunkte(schneidende_vektoren, vektor_start_ende, vektor_vergs_verge):
                try:
                    anzahl_schnitte = schneidende_vektoren.count(True)
                    if anzahl_schnitte == 1:
                        winkel = winkel_berechnung(vektor_start_ende, vektor_vergs_verge)
                        sichere_winkel.append(winkel)

                    if anzahl_schnitte < 1:
                        for vektor_verg in schneidende_vektoren:
                            winkel = winkel_berechnung(vektor_start_ende, vektor_verg)
                            print(f"Winkel zwischen zwei Strahlen:{winkel}")

                    else:
                        print("Ist ein einzelner Strahl oder Buffer zu klein")

                except Exception as e:
                    print(f"Fehler bei der Schnittpunkt-Berechnung:{e}")

            schneidende_vektoren = []
            for start, ende in zip(Anfangspunkte, Endpunkte):
                vektor_start_ende = ((ende[0] - start[0]), (ende[1] - start[1]))
                schneidende_vektoren.append(vektor_start_ende)

                for verg_start, verg_ende in zip(Anfangspunkte[1:], Endpunkte[1:]):
                    vektor_vergs_verge = ((verg_ende[0] - verg_start[0]), (verg_ende[1] - verg_start[1]))
                    buffer_vektorSE = shapely.buffer(vektor_start_ende, 10)
                    buffer_vektorVsVe = shapely.buffer(vektor_vergs_verge, 10)

                    if shapely.intersects(buffer_vektorSE, buffer_vektorVsVe):
                        schneidende_vektoren.append(vektor_vergs_verge)

                schnittpunkte(schneidende_vektoren, vektor_start_ende, vektor_vergs_verge)

            return sichere_winkel, mögliche_winkel

        except Exception as e:
            print(f"Fehler bei der Winkelberechnung:{e}")


    def strahlungs_findung():
        try:
            aVal, aGrayScaleArray, aBinaryArray = labeling()
            cropped, temp = croppen()
            Anfangspunkte = []
            Endpunkte = []
            # ---- get object dimensions
            # aVal[0] is a labeled image where each object has a separate label (incl. background)
            objectLabels = numpy.unique(aVal[0])
            print("geht4")
            zähler = 0
            for xx in objectLabels[1:]:
                aObj = aVal[0] == xx
                aP_all = numpy.argwhere(aObj)

                if len(aP_all) > 170:  # PCA benötigt mind. 3 Punkte
                    pca = PCA(n_components=1)
                    pca.fit(aP_all)
                    richtung = pca.components_[0]
                    mittelpunkt = pca.mean_

                    start = mittelpunkt - 50 * richtung  # 50 für Visualisierung, 150 für Winkel Visualisierung
                    ende = mittelpunkt + 50 * richtung

                    Anfangspunkte.append(start)
                    Endpunkte.append(ende)

                    zähler = zähler + 1
                    print(zähler)
            winkel(Anfangspunkte, Endpunkte)
            visualisierung(cropped, aGrayScaleArray, aBinaryArray, Anfangspunkte, Endpunkte, zähler)
            return zähler

        except Exception as e:
            print(f"Fehler im Hauptprogramm:{e}")


    # eigentlich nur fürs Debuggen
    def visualisierung(cropped, aGrayScaleArray, aBinaryArray, Anfangspunkte, Endpunkte, zähler):

        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.ravel()

            # Originalbild
            axes[0].imshow(cropped)
            axes[0].set_title("Originalbild")
            axes[0].axis("off")

            # Graubild
            axes[1].imshow(aGrayScaleArray, cmap="gray")
            axes[1].set_title("Graustufenbild")
            axes[1].axis("off")

            # Binärbild
            axes[2].imshow(aBinaryArray, cmap="gray")
            axes[2].set_title("Binärbild (nach Threshold)")
            axes[2].axis("off")

            # Gelabelte Objekte
            for start, ende in zip(Anfangspunkte, Endpunkte):
                axes[3].set_facecolor("#20423c")
                axes[3].imshow(aGrayScaleArray,
                               cmap="gray")
                axes[3].set_title(f"Gelabelte Objekte (Anzahl: {zähler})")
                axes[3].axis("off")
                axes[3].plot([start[1], ende[1]], [start[0], ende[0]], "r-", linewidth=2)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Fehler bei der Visualisierung:{e}")


    zähler = strahlungs_findung()
    Zähler_Liste.append(zähler)
    i = i + 1

while len(Winkel_Liste) < len(Zähler_Liste):
    Winkel_Liste.append("0")

while len(Winkel_Liste) > len(Zähler_Liste):
    Zähler_Liste.append("0")


def final():
    dataframe = DataFrame({"Anzahl Strahlen": Zähler_Liste})
    dataframe.to_excel("Resultate3.xlsx", sheet_name="Anzahl", index=False)
    print("fertig")


#final()
