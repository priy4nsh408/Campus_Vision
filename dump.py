# FILE: inspect_ocr_doctr.py

import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

fp = r"C:\Users\Priyansh\Desktop\El 5\rag-tutorial-v2\data\map.jpeg"

# -------------------------------
# Load Doctr OCR Model
# -------------------------------
print("Loading Doctr OCR model (this may take a few seconds)...")
model = ocr_predictor(pretrained=True, assume_straight_pages=True)

# -------------------------------
# Load Image
# -------------------------------
image = DocumentFile.from_images(fp)   # automatically loads image to tensor

# -------------------------------
# Run OCR
# -------------------------------
print("Running OCR...")
result = model(image)

# -------------------------------
# Extract Text Lines
# -------------------------------
full_text = []
for page in result.pages:
    for block in page.blocks:
        for line in block.lines:
            line_text = " ".join([word.value for word in line.words])
            full_text.append(line_text)

print("\nOCR lines:")
for line in full_text:
    print(line)

print("\n========== Full OCR text ==========\n")
print("\n".join(full_text))

print("\n" + "="*50)
print(f"Total OCR lines: {len(full_text)}")
total_chars = sum(len(x) for x in full_text)
print(f"Total characters: {total_chars}")
