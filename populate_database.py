"""
Populate Chroma DB from images (using Doctr OCR) and CSV files.

Doctr is a modern, production-ready OCR library built on PyTorch/TensorFlow with:
- Excellent accuracy on documents and complex layouts
- Fast inference
- Python 3.13 compatible
- No complicated dependencies

Install:
    pip install python-doctr[torch]
    # Or for TensorFlow: pip install python-doctr[tf]
    pip install pillow opencv-python pandas sentence-transformers langchain-text-splitters langchain_community chromadb

Note: Doctr will download models on first run (~50MB)
"""

import argparse
import os
import shutil
import json
import warnings
from typing import List, Callable, Any

from PIL import Image
import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Try to import Doctr
DOCTR_AVAILABLE = False
ocr_predictor = None

try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor as build_predictor
    
    print("✅ Doctr package imported successfully!")
    
    # Initialize the OCR predictor
    print("🔧 Initializing Doctr OCR (downloading models on first run, ~50MB)...")
    ocr_predictor = build_predictor(pretrained=True)
    print("✅ Doctr OCR engine ready!")
    DOCTR_AVAILABLE = True
    
except ImportError as e:
    print(f"⚠️ Doctr not available: {e}")
    print("OCR for images will be skipped. CSV files will still be processed.")
    
except Exception as e:
    print(f"⚠️ Failed to initialize Doctr: {e}")
    print("OCR for images will be skipped. CSV files will still be processed.")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
DATA_PATH = "data"


class EmbeddingAdapter:
    def __init__(self, emb: Any):
        self._orig = emb

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if hasattr(self._orig, "embed_documents") and callable(getattr(self._orig, "embed_documents")):
            return self._orig.embed_documents(texts)
        if hasattr(self._orig, "embed_texts") and callable(getattr(self._orig, "embed_texts")):
            return self._orig.embed_texts(texts)
        if callable(self._orig):
            return self._orig(texts)
        raise TypeError("Provided embedding object is not callable and has no embed_documents/embed_texts method.")

    def embed_query(self, text: str) -> List[float]:
        if hasattr(self._orig, "embed_query") and callable(getattr(self._orig, "embed_query")):
            return self._orig.embed_query(text)
        if hasattr(self._orig, "embed_texts") and callable(getattr(self._orig, "embed_texts")):
            out = self._orig.embed_texts([text])
            return out[0]
        if callable(self._orig):
            out = self._orig([text])
            return out[0]
        raise TypeError("Provided embedding object is not callable and has no embed_query method.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--single", type=str, default=None, help="Optional: path to a single file to process instead of scanning DATA_PATH")
    args = parser.parse_args()
    
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    source = args.single if args.single else DATA_PATH

    documents = load_documents_path_or_dir(source)
    if not documents:
        print("⚠️  No documents found in data folder.")
        return

    chunks = split_documents(documents)
    add_to_chroma(chunks)


def _doctr_ocr_image(image_path: str) -> tuple[str, list]:
    """
    Run Doctr OCR on the image and return:
      - full_text: combined OCR text (joined by newlines)
      - items: list of dicts {text, conf, left, top, width, height} from OCR results
    """
    # Load document
    doc = DocumentFile.from_images(image_path)
    
    # Perform OCR
    result = ocr_predictor(doc)
    
    items = []
    texts = []
    
    # Get image dimensions
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Parse results
    # Doctr structure: result.pages[0].blocks[i].lines[j].words[k]
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text_parts = []
                for word in line.words:
                    text = word.value
                    conf = word.confidence * 100  # Convert to percentage
                    
                    # Get bounding box (normalized coordinates)
                    # geometry is ((x_min, y_min), (x_max, y_max))
                    bbox = word.geometry
                    x_min, y_min = bbox[0]
                    x_max, y_max = bbox[1]
                    
                    # Convert to pixel coordinates
                    left = int(x_min * img_width)
                    top = int(y_min * img_height)
                    right = int(x_max * img_width)
                    bottom = int(y_max * img_height)
                    width = right - left
                    height = bottom - top
                    
                    item = {
                        "text": text.strip(),
                        "conf": int(conf),
                        "left": left,
                        "top": top,
                        "width": width,
                        "height": height,
                    }
                    items.append(item)
                    line_text_parts.append(text.strip())
                
                # Join words in a line with space
                if line_text_parts:
                    texts.append(" ".join(line_text_parts))
    
    full_text = "\n".join(texts)
    return full_text, items


def _process_csv(csv_path: str) -> List[Document]:
    """
    Process a CSV file and return a list of Documents.
    Each row becomes a document with all column data concatenated.
    Handles multiple encodings automatically.
    """
    documents = []
    
    # Try different encodings in order
    encodings = ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']
    df = None
    used_encoding = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            used_encoding = encoding
            print(f"✅ Successfully read CSV with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            # If it's not an encoding error, we should stop trying
            if df is None:
                print(f"❌ Error processing CSV {csv_path}: {e}")
                import traceback
                traceback.print_exc()
            return documents
    
    if df is None:
        print(f"❌ Could not read CSV {csv_path} with any supported encoding")
        print(f"   Tried: {', '.join(encodings)}")
        return documents
    
    try:
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        print(f"📊 CSV has {len(df)} rows and {len(df.columns)} columns")
        print(f"📊 Columns: {', '.join(df.columns.tolist())}")
        
        # Process each row
        for idx, row in df.iterrows():
            # Create text representation of the row
            # Format: "Column1: value1\nColumn2: value2\n..."
            row_text_parts = []
            for col in df.columns:
                value = row[col]
                # Skip NaN values
                if pd.notna(value):
                    row_text_parts.append(f"{col}: {value}")
            
            row_text = "\n".join(row_text_parts)
            
            # Create metadata
            metadata = {
                "source": csv_path,
                "filename": os.path.basename(csv_path),
                "row_index": int(idx),
                "data_type": "csv",
                "num_columns": len(df.columns),
                "encoding": used_encoding,
            }
            
            # Add first column value as identifier if available
            if len(df.columns) > 0:
                first_col = df.columns[0]
                if pd.notna(row[first_col]):
                    metadata["identifier"] = str(row[first_col])
            
            documents.append(Document(page_content=row_text, metadata=metadata))
        
        print(f"✅ Loaded CSV: {csv_path} ({len(documents)} rows)")
        
    except Exception as e:
        print(f"❌ Error processing CSV {csv_path}: {e}")
        import traceback
        traceback.print_exc()
    
    return documents


def load_documents_path_or_dir(path: str) -> List[Document]:
    """
    Load images (with OCR) and CSV files from a directory or a single file path.
    Returns a list of Documents.
    """
    image_ext = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    csv_ext = {".csv"}
    documents: List[Document] = []

    if os.path.isfile(path):
        file_paths = [path]
    else:
        file_paths = []
        for root, _, files in os.walk(path):
            for fname in sorted(files):
                fp = os.path.join(root, fname)
                file_paths.append(fp)

    for fp in file_paths:
        _, ext = os.path.splitext(fp)
        ext_lower = ext.lower()
        
        # Process CSV files
        if ext_lower in csv_ext:
            print(f"📄 Processing CSV: {fp}")
            csv_docs = _process_csv(fp)
            documents.extend(csv_docs)
        
        # Process image files with OCR
        elif ext_lower in image_ext:
            if not DOCTR_AVAILABLE or ocr_predictor is None:
                print(f"⚠️  Skipping image {fp} (Doctr not available)")
                continue
                
            try:
                print(f"📄 Processing Image: {fp}")
                # Run Doctr OCR
                text, items = _doctr_ocr_image(fp)

                # Basic postprocessing: normalize whitespace
                text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

                # Convert ocr_items to JSON string to avoid ChromaDB metadata errors
                metadata = {
                    "source": fp,
                    "page": 1,
                    "filename": os.path.basename(fp),
                    "ocr_items_json": json.dumps(items),
                    "num_boxes": len(items),
                    "ocr_engine": "Doctr",
                    "data_type": "image",
                }

                documents.append(Document(page_content=text, metadata=metadata))
                print(f"✅ Loaded OCR (Doctr): {fp} (chars: {len(text)}, boxes: {len(items)})")
            except Exception as e:
                print(f"❌ Error processing image {fp}: {e}")
                import traceback
                traceback.print_exc()

    return documents


def split_documents(documents: list[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def _build_local_sentence_transformer_embedding_fn(model_name: str = "all-MiniLM-L6-v2") -> Callable[[List[str]], List[List[float]]]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("Failed to import sentence-transformers. Install it with: pip install sentence-transformers") from e

    model = SentenceTransformer(model_name)

    def embed_texts(texts: List[str]) -> List[List[float]]:
        arr = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [x.tolist() for x in arr]

    return embed_texts


def add_to_chroma(chunks: List[Document]):
    embedding_source = None
    try:
        print("Trying to load embedding function from get_embedding_function()...")
        embedding_source = get_embedding_function()
    except Exception as e:
        print("⚠️  get_embedding_function() failed. Falling back to local sentence-transformers embeddings.")
        print(f"Reason: {e}")
        try:
            embedding_source = _build_local_sentence_transformer_embedding_fn()
        except Exception as e2:
            print("❌ Failed to create local embedding function as well.")
            raise e2

    embedding_obj = EmbeddingAdapter(embedding_source)

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_obj)

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = {}
    try:
        existing_items = db.get(include=[])
    except Exception:
        existing_items = {}

    existing_ids = set(existing_items.get("ids", [])) if isinstance(existing_items, dict) else set()
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        try:
            db.add_documents(new_chunks, ids=new_chunk_ids)
        except Exception:
            texts = [c.page_content for c in new_chunks]
            metadatas = [c.metadata for c in new_chunks]
            db.add_texts(texts=texts, metadatas=metadatas, ids=new_chunk_ids)
        db.persist()
        print("✅ Successfully added documents to database")
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        
        # For CSV files, use row_index; for images use page
        if chunk.metadata.get("data_type") == "csv":
            page = chunk.metadata.get("row_index", 0)
        else:
            page = chunk.metadata.get("page", 1)
            
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()