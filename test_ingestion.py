import json
from pathlib import Path
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

from ingestion.pdf import extract_pdf
from ingestion.image import extract_image
from knowledge.schema import Evidence


def create_sample_assets():
    """Generates sample test PDF and test image in data/ directory."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 1. Create sample PDF: data/architecture.pdf
    pdf_path = data_dir / "architecture.pdf"
    doc = fitz.open()
    
    # Page 1
    page1 = doc.new_page()
    page1.insert_text((50, 72), "ContextMesh Architecture Overview", fontsize=18)
    page1.insert_text((50, 110), "System Overview:\nThis system ingests multimodal documents including PDFs, images, and audio/video.", fontsize=11)
    
    # Page 2
    page2 = doc.new_page()
    page2.insert_text((50, 72), "Caching Layer Specification", fontsize=18)
    page2.insert_text((50, 110), "Redis is an in-memory caching system used to reduce database load and improve response latency.", fontsize=11)
    
    doc.save(str(pdf_path))
    doc.close()
    print(f"Created sample PDF at: {pdf_path}")
    
    # 2. Create sample Image: data/diagram.png
    img_path = data_dir / "diagram.png"
    img = Image.new("RGB", (600, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw simple boxes and text
    draw.rectangle([30, 70, 170, 130], outline=(0, 0, 0), width=2)
    draw.text((45, 90), "Application", fill=(0, 0, 0))
    
    draw.line([170, 100, 240, 100], fill=(0, 0, 0), width=2)
    draw.polygon([(240, 95), (250, 100), (240, 105)], fill=(0, 0, 0))
    
    draw.rectangle([250, 70, 390, 130], outline=(0, 0, 0), width=2)
    draw.text((275, 90), "Redis Cache", fill=(0, 0, 0))
    
    draw.line([390, 100, 460, 100], fill=(0, 0, 0), width=2)
    draw.polygon([(460, 95), (470, 100), (460, 105)], fill=(0, 0, 0))
    
    draw.rectangle([470, 70, 580, 130], outline=(0, 0, 0), width=2)
    draw.text((495, 90), "Database", fill=(0, 0, 0))
    
    img.save(str(img_path))
    print(f"Created sample Diagram at: {img_path}")


def run_tests():
    print("\n" + "=" * 50)
    print("Testing Multimodal Ingestion Pipeline (Person 2)")
    print("=" * 50)
    
    create_sample_assets()
    
    # Test PDF Extraction
    print("\n--- Testing PDF Extraction (ingestion/pdf.py) ---")
    pdf_results = extract_pdf("data/architecture.pdf")
    print(f"Extracted {len(pdf_results)} evidence item(s) from PDF:")
    for item in pdf_results:
        print(json.dumps(item.to_dict(), indent=2))
        assert isinstance(item, Evidence), "Item must be instance of Evidence"
        assert item.modality == "pdf"
        assert item.page is not None
        
    # Test Image Extraction
    print("\n--- Testing Image Extraction (ingestion/image.py) ---")
    image_result = extract_image("data/diagram.png")
    print("Extracted evidence item from Image:")
    print(json.dumps(image_result.to_dict(), indent=2))
    assert isinstance(image_result, Evidence), "Item must be instance of Evidence"
    assert image_result.modality == "image"
    
    print("\n" + "=" * 50)
    print(" All tests passed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    run_tests()
