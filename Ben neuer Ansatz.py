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
from shapely import Point
from shapely.ops import nearest_points
from sklearn.decomposition import PCA
import glob
import os
from pandas import DataFrame

# hier den jeweiligen Bildpfad einfügen oder ansonsten config file
folder_path = r"C:\Dokumente 2\Matura Data\Matura Data komplett\902D7200\Data Modus 2-2"


def bilder_einfügen():
    try:
        bilder = []
        for i in glob.glob(os.path.join(folder_path, "DSC_0368.JPG")):
            bilder.append(i)
        print(bilder)
        print(f"Es wurden {len(bilder)} Bilder gefunden!")
        return bilder

    except Exception as e:
        print(f"Fehler beim Einfügen des Bildes aus der Datei:{e}")


bilder = bilder_einfügen()
i = 0
Zähler_Liste = []
sichere_Winkel_Liste = []
mögliche_Winkel_Liste = []
alle_Winkel_Liste = []

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
            imageArray = np.array(cropped)
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
    def winkel_funk(Anfangspunkte, Endpunkte):
        try:
            def winkel_berechnung(vektor1, vektor2):
                try:
                    betrag_vektor1 = np.linalg.vector_norm(vektor1)
                    betrag_vektor2 = np.linalg.vector_norm(vektor2)
                    winkel_rad = np.arccos((np.dot(vektor1, vektor2)) / (betrag_vektor1 * betrag_vektor2))
                    winkel_rad = np.clip(winkel_rad, -1.0, 1.0)
                    winkel_deg = np.rad2deg(winkel_rad)
                    return winkel_deg

                except Exception as e:
                    print(f"Fehler bei der Winkelberechnung:{e}")

            schnittpunkt_liste = []
            alle_winkel = []
            sichere_winkel = []
            mögliche_winkel = []

            for i in range(len(Anfangspunkte)):
                schneidende_vektoren = []
                anzahl_schnitte = 0
                linie1 = shapely.LineString([Anfangspunkte[i], Endpunkte[i]])
                vektor1 = np.array(Endpunkte[i]) - np.array(Anfangspunkte[i])

                for j in range(i + 1, len(Anfangspunkte)):
                    linie2 = shapely.LineString([Anfangspunkte[j], Endpunkte[j]])
                    vektor2 = np.array(Endpunkte[j]) - np.array(Anfangspunkte[j])

                    pt1, pt2 = nearest_points(linie1, linie2)

                    if pt1.distance(pt2) <= 50:
                        schnittpunkt_liste.append(Point((pt1.x + pt2.x) / 2, (pt1.y + pt2.y) / 2))
                        anzahl_schnitte = anzahl_schnitte + 1
                        schneidende_vektoren.append((vektor1, vektor2))

                if anzahl_schnitte == 1:
                    for vektor1, vektor2 in schneidende_vektoren:
                        winkel_deg = winkel_berechnung(vektor1, vektor2)
                        sichere_winkel.append(winkel_deg)
                        alle_winkel.append(winkel_deg)
                        print(f"Winkel zwischen zwei Strahlen:{winkel_deg}")

                if anzahl_schnitte > 1:
                    for vektor1, vektor2 in schneidende_vektoren:
                        winkel_deg = winkel_berechnung(vektor1, vektor2)
                        alle_winkel.append(winkel_deg)
                        print(f"Winkel zwischen zwei Strahlen:{winkel_deg}")
                        if winkel_deg > 10:
                            mögliche_winkel.append(winkel_deg)
                        else:
                            print("Wahrscheinlich ein falscher Winkel")

                if anzahl_schnitte < 1:
                    print("Wahrscheinlich ein einzelner Strahl")
            return sichere_winkel, mögliche_winkel, alle_winkel, schnittpunkt_liste

        except Exception as e:
            print(f"Fehler bei den Winkeln:{e}")


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

                    start = mittelpunkt - 50 * richtung
                    ende = mittelpunkt + 50 * richtung

                    Anfangspunkte.append(start)
                    Endpunkte.append(ende)

                    zähler = zähler + 1
                    print(zähler)

            sichere_winkel, mögliche_winkel, alle_winkel, schnittpunkt_liste = winkel_funk(
                Anfangspunkte, Endpunkte)
            visualisierung(cropped, aGrayScaleArray, aBinaryArray,
                           Anfangspunkte, Endpunkte, zähler,
                           alle_winkel, schnittpunkt_liste)
            return zähler, sichere_winkel, mögliche_winkel, alle_winkel

        except Exception as e:
            print(f"Fehler im Hauptprogramm:{e}")


    # eigentlich nur fürs Debuggen
    def visualisierung(cropped, aGrayScaleArray, aBinaryArray,
                       Anfangspunkte, Endpunkte, zähler,
                       alle_winkel, schnittpunkt_liste):

        try:
            fig, ax = plt.subplots(2, 2, figsize=(12, 10))

            # Originalbild
            ax[0, 0].imshow(cropped)
            ax[0, 0].set_title("Originalbild")
            ax[0, 0].axis("off")

            # Graubild
            ax[0, 1].imshow(aGrayScaleArray, cmap="gray")
            ax[0, 1].set_title("Graustufenbild")
            ax[0, 1].axis("off")

            # Binärbild
            ax[1, 0].imshow(aBinaryArray, cmap="gray")
            ax[1, 0].set_title("Binärbild (nach Threshold)")
            ax[1, 0].axis("off")

            # Gelabelte Objekte
            ax[1, 1].set_facecolor("#20423c")
            ax[1, 1].imshow(aGrayScaleArray, cmap="gray")
            ax[1, 1].set_title(f"Gelabelte Objekte (Anzahl: {zähler})")
            ax[1, 1].axis("off")

            for start, ende in zip(Anfangspunkte, Endpunkte):
                ax[1, 1].plot([start[1], ende[1]], [start[0], ende[0]], "r-", linewidth=2)

            for schnittpunkt, winkel in zip(schnittpunkt_liste, alle_winkel):
                ax[1, 1].plot(schnittpunkt.y, schnittpunkt.x, "o", markersize=5, color="green",
                              label=f"Winkel:{winkel:.2f}°")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Fehler bei der Visualisierung:{e}")


    zähler, sichere_winkel, mögliche_winkel, alle_winkel = strahlungs_findung()
    sichere_Winkel_Liste.extend(sichere_winkel)
    mögliche_Winkel_Liste.extend(mögliche_winkel)
    alle_Winkel_Liste.extend(alle_winkel)
    Zähler_Liste.append(zähler)
    i = i + 1

while len(mögliche_Winkel_Liste) < len(alle_Winkel_Liste) and len(sichere_Winkel_Liste) < len(Zähler_Liste):
    mögliche_Winkel_Liste.append("0")
    sichere_Winkel_Liste.append("0")


def final():
    dataframe1 = DataFrame({"Anzahl Strahlen": Zähler_Liste})
    dataframe1.to_excel("Resultate_zaehler.xlsx", sheet_name="Anzahl", index=False)
    dataframe2 = DataFrame({"Anzahl Strahlen": Zähler_Liste})
    dataframe2.to_excel("Resultate_winkel.xlsx", sheet_name="Winkel", index=False)
    print("fertig")


final()
