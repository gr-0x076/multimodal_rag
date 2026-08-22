import pymupdf as fitz
from pathlib import Path
from typing import List
from knowledge.schema import Evidence


def extract_pdf(pdf_path: str) -> List[Evidence]:
    """
    Extracts text from each page of a PDF file and returns a list of Evidence objects.
    
    Args:
        pdf_path: Path to the target PDF file.
        
    Returns:
        List of Evidence objects, one per non-empty page.
    """
    path_obj = Path(pdf_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    document = fitz.open(path_obj)
    evidence_list: List[Evidence] = []
    
    for page_number, page in enumerate(document, start=1):
        text = page.get_text().strip()
        if not text:
            continue
            
        item = Evidence(
            id=f"{path_obj.stem}_page_{page_number}",
            content=text,
            modality="pdf",
            source=path_obj.name,
            page=page_number,
            confidence=0.95
        )
        evidence_list.append(item)
        
    document.close()
    return evidence_list
