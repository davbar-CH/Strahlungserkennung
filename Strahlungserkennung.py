import glob
import os
import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt
from matplotlib import cm


# Bilder holen
def bilder_holen(folder_path):
    try:
        bilder = []
        for i in glob.glob(os.path.join(folder_path, "DSC_*.JPG")):
            bilder.append(i)

        return bilder

    except Exception as e:
        print(f"Fehler beim Bilderpfad, richtiger Pfad?:{e}")

def bilder_lesen(bilder):
    try:
        image = cv2.imread(bilder[1], cv2.IMREAD_GRAYSCALE)

        return image
    except Exception as e:
        print(f"Fehler bei der Einlesung des Bildes:{e}")

# or =oben-rechts, ol=oben-links, ur=unten-rechts, ul=unten-links
def bilder_croppen(image, OR, OL, UR, UL):
    try:
        punkte = np.array([
            [OR],
            [OL],
            [UR],
            [UL]
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
        cropped = masked[y:y + h, x:x + w]

        return cropped
    except Exception as e:
        print(f"Fehler bei der Erzeugung einer Maske:{e}")

def hough_transformation(cropped):
    try:
        # Classic straight-line Hough transform
        # Set a precision of 0.5 degree.
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h, theta, d = hough_line(cropped, theta=tested_angles)

        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        return x0, y0

    except Exception as e:
        print(f"Fehler bei der Hough-Transformation:{e}")

def darstellung_bedinungen(nrows, ncols):
    try:
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 6))
        ax = axes.ravel()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Fehler beim der Bilderdarstellung: {e}")

def canny_bilder(cropped):
    try:
        kanten = cv2.Canny(cropped, 100, 200, 3)
        return kanten
    except Exception as e:
        print(f"Fehler bei der Canny Kanten Detektion:{e}")

def Gerade_links(angle, x0, y0):
    try:
        bilder = bilder_holen(r"C:\Dokumente 2\Matura Data\Matura Data komplett\902D7200\Tests")
        bild_gelesen = bilder_lesen(bilder)
        cropped = bilder_croppen(bild_gelesen, (715, 1450), (3035, 188), (4666, 2228), (2379, 3989))
        kanten = canny_bilder(cropped)

        x0_int = int(x0)
        y0_int = int(y0)
        _, width = kanten.shape[:2]
        q = y0_int - (np.tan(angle + np.pi / 2) * x0_int)  # eigentlich cotan, aber so ausgedrückt um 1/0 zu vermeiden
        q_int = int(q)
        bild_wert_liste = kanten[x0_int, y0_int]

        for x_strahl in range(x0_int, width, -1):  # vom startpunkt aus (gefundene maxima) rückwärts
            if bild_wert_liste >= 200:
                # ax[1].axline((x0, y0), slope=-np.tan(angle)**-1)#hier cotan, macht mehr Sinn für mich
                # (weil x*cos(angle) + y*sin(angle) = d)
                y_strahl = (np.tan(angle + np.pi / 2) * x_strahl) + q_int
                y_strahl_int = int(y_strahl)
                bild_wert_liste = kanten[x_strahl, y_strahl_int]
                print(bild_wert_liste)
        anfangspunkt = (x_strahl, y_strahl_int)
        return anfangspunkt
    except Exception as e:
        print(f"Fehler bei der Konstruktion der Gerade:{e}")

def Gerade_rechts(angle, x0, y0):
    try:
        bilder = bilder_holen(r"C:\Dokumente 2\Matura Data\Matura Data komplett\902D7200\Tests")
        bild_gelesen = bilder_lesen(bilder)
        cropped = bilder_croppen(bild_gelesen, (715, 1450), (3035, 188), (4666, 2228), (2379, 3989))
        kanten = canny_bilder(cropped)

        x0_int = int(x0)
        y0_int = int(y0)
        _, width = kanten.shape[:2]
        q = y0_int - (np.tan(angle + np.pi / 2) * x0_int)  # eigentlich cotan, aber so ausgedrückt um 1/0 zu vermeiden
        q_int = int(q)
        bild_wert_liste = kanten[x0_int, y0_int]

        for x_strahl in range(x0_int, width): # vom startpunkt aus (gefundene maxima) vorwärts
            if bild_wert_liste >= 200:
                # ax[1].axline((x0, y0), slope=-np.tan(angle)**-1)#hier cotan, macht mehr Sinn für mich (weil x*cos(angle) + y*sin(angle) = d
                y_strahl = (np.tan(angle + np.pi / 2) * x_strahl) + q_int
                y_strahl_int = int(y_strahl)
                bild_wert_liste = kanten[x_strahl, y_strahl_int]
                print(bild_wert_liste)


        linien_laenge = np.sqrt(((x_strahl - x0_int)**2) + ((y_strahl_int - y0_int)**2))
        endpunkt = (x_strahl, y_strahl_int)

        if linien_laenge < 90: # 90 und 120 noch anpassen, da um die linienlänge rauszufiltern
            anfangspunkt = Gerade_links(angle, x0, y0)
        elif linien_laenge > 120:
            pass # bin mir nicht sicher ob "pass" ich will das es aufhört und die funktion nochmal durchgeht


        linien_laenge_final = np.sqrt((endpunkt[0] - anfangspunkt[0]) + (endpunkt[1] - anfangspunkt[1]))


        # Warte auf Frau Fritschi
        return linien_laenge_final
    except Exception as e:
        print(f"Fehler bei der Konstruktion der Gerade:{e}")

def Winkel(linien_laenge):
    try:
        a
    except Exception as e:
        print(f"Fehler bei der Winkelberechnung:{e}")

def Statistik(linien_laenge_final, linien_zahl, linien_winkel):
    print(linien_laenge_final)
    print(linien_zahl)
    print(linien_winkel)

def darstellung_ergebnisse(cropped, ax):
    try:
        ax[0].imshow(cropped, cmap=cm.gray)
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        ax[1].imshow(cropped, cmap=cm.gray)
        ax[1].set_ylim((cropped.shape[0], 0))
        ax[1].set_axis_off()
        ax[1].set_title('Detected lines')
    except Exception as e:
        print(f"Fehler beim Visualisieren: {e}")



"""for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    x0_int = int(x0)
    y0_int = int(y0)
    x_strahl = x0_int
    # ax[1].axline((x0, y0), slope=-np.tan(angle)**-1)#hier cotan, macht mehr Sinn für mich (weil x*cos(angle) + y*sin(angle) = d)
    q = y0_int - (np.tan(angle + np.pi / 2) * x0_int)  # eigentlich cotan, aber so ausgedrückt um 1/0 zu vermeiden
    q_int = int(q)
    bild_wert_liste = cropped[x0_int, y0_int]
    print(bild_wert_liste)

    if bild_wert_liste > 200:
        x_strahl = x0_int  # Initialize x_strahl
        y_ende = y0_int  # Initialize y_ende with default value

        while bild_wert_liste >= 200:  # 200 als Schwellenwert für weiss
            y_strahl = (np.tan(
                angle + np.pi / 2) * x_strahl) + q_int  # eigentlich cotan, aber so ausgedrückt um 1/0 zu vermeiden
            y_strahl_int = int(y_strahl)

            # Check bounds before accessing array
            if (x_strahl >= 0 and x_strahl < cropped.shape[0] and
                    y_strahl_int >= 0 and y_strahl_int < cropped.shape[1]):
                bild_wert_liste = cropped[x_strahl, y_strahl_int]
                x_strahl = x_strahl + 1
                y_ende = y_strahl
            else:
                break  # Exit loop if we go out of bounds

        ax[1].plot((x_strahl, x0_int), (y_ende, y0_int))
    else:
        pass"""





if __name__ == "__main__":
    main()
