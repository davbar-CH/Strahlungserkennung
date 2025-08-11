import sys
from logging import exception
from types import NoneType

import cv2
from shapely import Point, MultiLineString
from skimage.feature import canny
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from skimage.transform import probabilistic_hough_line
from itertools import chain
from shapely.geometry import LineString
from shapely.ops import unary_union
from pandas import DataFrame

# hier den jeweiligen Bildpfad einfügen oder ansonsten config file
folder_path = r"C:\schulisches\Matura\Daten\matura data\length3"


# diese funktioniert für mich, in pycharm. Je nach IDE muss diese Funktion geändert werden
def debug_enabled():
    try:
        if sys.gettrace() is not None:
            return True
    except AttributeError:
        pass

    try:
        if sys.monitoring.get_tool(sys.monitoring.DEBUGGER_ID) is not None:
            return True
    except AttributeError:
        pass

    return False


def bilder_einfügen():
    try:
        bilder = []
        for i in glob.glob(os.path.join(folder_path, "DSC_5069.JPG")):
            bilder.append(i)
        print(bilder)
        print(f"Es wurden {len(bilder)} Bilder gefunden!")
        return bilder

    except Exception as e:
        print(f"Fehler beim Einfügen des Bildes aus der Datei:{e}")


bilder = bilder_einfügen()
i = 0
längen_liste_gesamt = []
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
                [1313, 63],  # oben links
                [2148, 646],  # oben rechts
                [1414, 1472],  # unten rechts
                [552, 738]  # unten links
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
        y_von_x_min = start_dic.get(x_min, end_dic.get(x_min))

        x_max = max(chain(start_dic.keys(), end_dic.keys()))  # östlichster Punkt
        y_von_x_max = start_dic.get(x_max, end_dic.get(x_max))

        y_min = min(chain(start_dic.values(), end_dic.values()))  # nördlichster Punkt
        x_von_y_min = None
        for key in chain(start_dic, end_dic):
            if start_dic.get(key) == y_min or end_dic.get(key) == y_min:
                x_von_y_min = key
                break

        y_max = max(chain(start_dic.values(), end_dic.values()))  # südlichster Punkt
        x_von_y_max = None
        for key in chain(start_dic, end_dic):
            if start_dic.get(key) == y_max or end_dic.get(key) == y_max:
                x_von_y_max = key
                break

        punkte = np.array([
            [x_von_y_min, y_min],
            [x_max, y_von_x_max],
            [x_von_y_max, y_max],
            [x_min, y_von_x_min]
        ], dtype=np.int32)

        return punkte


    def punkt_auf_gerade(punkte_plot, Startpunkte, Endpunkte, ax, toleranz=20):
        geraden_liste = []
        längen_liste = []

        # erstellt Geraden zur Filterung des Randes
        # verbindet jeweils die Punkte im Uhrzeigersinn, vom nördlisten Punkt aus
        for punkt1, punkt2 in zip(punkte_plot, punkte_plot[1:]):
            m = (punkt1[1] - punkt2[1]) / (punkt1[0] - punkt2[0])
            q = punkt1[1] - (m * punkt1[0])
            geraden_liste.append((m, q))
            if ax is not None:
                ax[2].plot((punkt1[0], punkt2[0]), (punkt1[1], punkt2[1]), color="blue")

        # verbindet den westlichsten Punkt mit dem Nördlisten
        m1 = (punkte_plot[3][1] - punkte_plot[0][1]) / (punkte_plot[3][0] - punkte_plot[0][0])
        q1 = punkte_plot[3][1] - (m1 * punkte_plot[3][0])
        geraden_liste.append((m1, q1))
        if ax is not None:
            ax[2].plot((punkte_plot[3][0], punkte_plot[0][0]), (punkte_plot[3][1], punkte_plot[0][1]), color="blue")

        # falls ein Punkt sich zu nah an einer der berechneten Begrenzungslinien befindet,
        # dann wird dieser herausgefiltert. Alle anderen Punkte werden übernommen
        fragment_linien = []
        for p0, p1 in zip(Startpunkte, Endpunkte):
            zu_nah = False
            for m, q in geraden_liste:
                y_p0 = (m * p0[0]) + q
                y_p1 = (m * p1[0]) + q

                if abs(p0[1] - y_p0) < toleranz and abs(p1[1] - y_p1) < toleranz:
                    zu_nah = True
                    break
            if not zu_nah:
                linie = LineString([p0, p1])
                fragment_linien.append(linie)

        if not fragment_linien:
            längen_liste.append("0")
            return längen_liste
        try:
            buffer_linien = [linie.buffer(0.5) for linie in fragment_linien]
            zusammen_polygone = unary_union(buffer_linien)
            zusammen_linien = zusammen_polygone.boundary
        except Exception:
            längen_liste.append("0")
            return längen_liste

        if isinstance(zusammen_linien, MultiLineString):
            linien_iterierbar = zusammen_linien.geoms
        elif isinstance(zusammen_linien, LineString):
            linien_iterierbar = [zusammen_linien]
        else:
            linien_iterierbar = []

        boundary_punkte = []
        for linie in linien_iterierbar:
            boundary_punkte.append(list(linie.coords))
            x, y = linie.xy
            if ax is not None:
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


    def hauptprogramm():
        # Strahlungserkennung mit PPHT
        cropped, grau, punkte = croppen()

        edges = canny(grau, 4, 25, 40)
        lines = probabilistic_hough_line(edges, threshold=5, line_length=20, line_gap=10)

        Startpunkte = []
        Endpunkte = []
        for line in lines:
            p0, p1 = line
            Startpunkte.append(p0)
            Endpunkte.append(p1)

        if debug_enabled() is True:
            fig, axes = plt.subplots(1, 3, figsize=(15, 10), sharex=True, sharey=True)

            punkte_plot = plot_ecken(Startpunkte, Endpunkte)
            punkt_auf_gerade(punkte_plot, Startpunkte, Endpunkte, axes)

            axes[0].imshow(cropped, cmap="gray")
            axes[0].set_title('Input image')

            axes[1].imshow(edges, cmap="gray")
            axes[1].set_title('Canny edges')

            axes[2].imshow(edges * 0)

            axes[2].set_xlim((0, cropped.shape[1]))
            axes[2].set_ylim((cropped.shape[0], 0))
            axes[2].set_title('Probabilistic Hough')

            for a in axes:
                a.set_axis_off()

            plt.tight_layout()
            plt.show()
        else:
            punkte_plot = plot_ecken(Startpunkte, Endpunkte)
            längen_liste = punkt_auf_gerade(punkte_plot, Startpunkte, Endpunkte, ax=None)

            return längen_liste

    try:
        längen_liste = hauptprogramm()
        längen_liste_gesamt.extend(längen_liste)
        i = i + 1
    except TypeError:
        print("Gerade im Debugmodus")
        i = i + 1

#dataframe1 = DataFrame({"Längen": längen_liste_gesamt})
#dataframe1.to_excel("Resultate_laengen.xlsx", sheet_name="Laengen", index=False)
