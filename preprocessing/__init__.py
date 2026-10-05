# Package: preprocessing
# Contiene le funzioni per rasterizzazione PDF/immagini, pulizia
# (deskew/denoise/binarizzazione) ed estrazione OCR+tabelle.

from .pdf_to_images import pdf_to_images, load_single_image
from .image_cleaning import clean_pipeline, deskew, denoise, adaptive_binarize, upscale_if_needed
from .ocr_table_engine import get_ocr_engine, get_structure_engine, extract_page_content, html_table_to_markdown

__all__ = [
    "pdf_to_images",
    "load_single_image",
    "clean_pipeline",
    "deskew",
    "denoise",
    "adaptive_binarize",
    "upscale_if_needed",
    "get_ocr_engine",
    "get_structure_engine",
    "extract_page_content",
    "html_table_to_markdown",
]
