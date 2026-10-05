"""
Rasterizza pagine PDF in immagini (necessario per PDF scannerizzati,
dove il testo non è selezionabile e va passato interamente a OCR).
Usa PyMuPDF (fitz) invece di pdf2image perché non richiede poppler
installato a livello di sistema ed è più leggero in RAM.
"""

from __future__ import annotations

import fitz  # PyMuPDF
import numpy as np
from pathlib import Path


def pdf_to_images(pdf_path: str, dpi: int = 250) -> list[np.ndarray]:
    """
    Converte ogni pagina di un PDF in un'immagine (array BGR, compatibile OpenCV).

    INPUT:
        pdf_path (str): percorso al PDF
        dpi (int): risoluzione di rasterizzazione (200-300 è un buon compromesso
                    tra qualità OCR e uso di RAM)

    OUTPUT:
        list[np.ndarray]: una immagine per pagina, in ordine
    """
    images = []
    zoom = dpi / 72  # fitz lavora in punti (72 dpi di base)
    matrix = fitz.Matrix(zoom, zoom)

    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img_bgr = img[:, :, ::-1] if pix.n == 3 else img  # RGB -> BGR per OpenCV
        images.append(np.ascontiguousarray(img_bgr))
    doc.close()

    return images


def load_single_image(image_path: str) -> np.ndarray:
    """
    Carica una singola immagine (jpg/png) come array BGR.

    INPUT:
        image_path (str): percorso all'immagine

    OUTPUT:
        np.ndarray
    """
    import cv2
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Impossibile leggere l'immagine: {image_path}")
    return img
