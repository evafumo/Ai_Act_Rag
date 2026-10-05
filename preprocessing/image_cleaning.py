"""
Preprocessing delle immagini scannerizzate: deskew, denoising,
binarizzazione adattiva. Fa gran parte del lavoro prima ancora
che l'OCR entri in gioco.

FIX rispetto alla versione precedente: `clean_pipeline` binarizzava per
default (`binarize=True`), ma PaddleOCR (come quasi tutti i motori OCR
moderni basati su reti neurali) è addestrato su immagini in scala di grigi
o a colori, non su bianco/nero puro. Binarizzare "a monte" spesso cancella
dettagli fini (grazie ad antialiasing, sfumature attorno ai caratteri) di
cui il modello ha bisogno, peggiorando il riconoscimento invece di
migliorarlo. Il default ora è `binarize=False`; abilitalo solo se stai
usando un motore OCR classico basato su template matching (es. Tesseract
in modalità legacy) che ne trae effettivamente beneficio.
"""

from __future__ import annotations

import cv2
import numpy as np


def deskew(img: np.ndarray) -> np.ndarray:
    """
    Corregge l'inclinazione della pagina scannerizzata.

    INPUT:
        img (np.ndarray): immagine BGR o grayscale

    OUTPUT:
        np.ndarray: immagine raddrizzata
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 20:  # troppo poco contenuto per stimare l'angolo
        return img

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)


def denoise(img: np.ndarray) -> np.ndarray:
    """
    Riduce il rumore tipico degli scan (grana, sale e pepe).

    INPUT:
        img (np.ndarray): immagine BGR

    OUTPUT:
        np.ndarray: immagine ripulita
    """
    return cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10,
                                            templateWindowSize=7, searchWindowSize=21)


def adaptive_binarize(img: np.ndarray) -> np.ndarray:
    """
    Binarizzazione adattiva: gestisce meglio di Otsu l'illuminazione
    non uniforme, frequente negli scan da fotocopiatrice/telefono.

    Usala solo per motori OCR che ne beneficiano davvero (vedi nota in
    testa al file) — non per PaddleOCR/PPStructureV3.

    INPUT:
        img (np.ndarray): immagine BGR o grayscale

    OUTPUT:
        np.ndarray: immagine binaria (grayscale a 2 livelli)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )


def upscale_if_needed(img: np.ndarray, min_height: int = 1500) -> np.ndarray:
    """
    Ingrandisce l'immagine se la risoluzione è troppo bassa per un OCR affidabile.

    INPUT:
        img (np.ndarray)
        min_height (int): altezza minima desiderata in pixel

    OUTPUT:
        np.ndarray
    """
    h, w = img.shape[:2]
    if h >= min_height:
        return img
    scale = min_height / h
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def clean_pipeline(img: np.ndarray, binarize: bool = False) -> np.ndarray:
    """
    Applica in sequenza tutto il preprocessing.

    INPUT:
        img (np.ndarray): immagine grezza (BGR)
        binarize (bool): default False. PaddleOCR/PPStructureV3 gestisce
                          bene grayscale/colore; binarizzare a monte
                          tende a peggiorare il riconoscimento. Metti True
                          solo se usi un motore OCR classico che ne trae
                          beneficio.

    OUTPUT:
        np.ndarray
    """
    img = upscale_if_needed(img)
    img = denoise(img)
    img = deskew(img)
    if binarize:
        img = adaptive_binarize(img)
    return img
