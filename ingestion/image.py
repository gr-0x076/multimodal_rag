import os
import shutil
from pathlib import Path
from PIL import Image
import pytesseract
from knowledge.schema import Evidence


def _configure_tesseract():
    """Attempt to locate tesseract executable if not already in system PATH."""
    if shutil.which("tesseract"):
        return
    
    # Common Windows installation locations
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")
    ]
    for path in common_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return


def extract_image(image_path: str) -> Evidence:
    """
    Extracts text/OCR information from an image file and returns an Evidence object.
    
    Args:
        image_path: Path to the target image file.
        
    Returns:
        An Evidence object representing the image and its extracted content.
    """
    path_obj = Path(image_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image file not found at: {image_path}")
        
    _configure_tesseract()
    
    image = Image.open(path_obj)
    width, height = image.size
    
    ocr_text = ""
    try:
        ocr_text = pytesseract.image_to_string(image).strip()
    except Exception as e:
        # Fallback gracefully if tesseract binary is not installed in the OS
        ocr_text = f"[Image: {path_obj.name} (dimensions: {width}x{height}, format: {image.format}) - OCR unavailable: {str(e)}]"
        
    if not ocr_text:
        ocr_text = f"[Image: {path_obj.name} (dimensions: {width}x{height}) - No readable text detected]"
        
    evidence = Evidence(
        id=f"{path_obj.stem}_image",
        content=ocr_text,
        modality="image",
        source=path_obj.name,
        confidence=0.85,
        metadata={
            "width": width,
            "height": height,
            "format": image.format or path_obj.suffix.lstrip(".").upper()
        }
    )
    return evidence
