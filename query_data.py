# FILE: query_data.py
"""
Query the Chroma DB and answer using Ollama (updated for images + CSV support).

Notes:
 - Uses local fallback embeddings (sentence-transformers) if get_embedding_function() fails.
 - Uses langchain_chroma & langchain_ollama to avoid LangChain deprecation warnings.
 - Handles both image OCR data and CSV data in retrieval and display.
 - Add your question interactively or pass as positional argument.
"""

# -----------------------
# Silence noisy warnings
# -----------------------
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

import re
import os
# recommended environment tweak for tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"

# -----------------------
# Imports (updated)
# -----------------------
import argparse
import sys
import json
from typing import List, Tuple, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Use the newer packages to avoid LangChain deprecation warnings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama as Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


# -----------------------
# Embedding adapter + fallback
# -----------------------
class EmbeddingAdapter:
    """
    Wraps either:
      - an object that already implements embed_documents/embed_query, or
      - a plain callable f(texts: List[str]) -> List[List[float]]
    and exposes embed_documents(texts) and embed_query(text).
    This adapter satisfies what the Chroma wrapper expects.
    """
    def __init__(self, emb: Any):
        self._orig = emb

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if hasattr(self._orig, "embed_documents") and callable(getattr(self._orig, "embed_documents")):
            return self._orig.embed_documents(texts)
        if hasattr(self._orig, "embed_texts") and callable(getattr(self._orig, "embed_texts")):
            return self._orig.embed_texts(texts)
        if callable(self._orig):
            return self._orig(texts)
        raise TypeError("Embedding object must be callable or implement embed_documents/embed_texts")

    def embed_query(self, text: str) -> List[float]:
        if hasattr(self._orig, "embed_query") and callable(getattr(self._orig, "embed_query")):
            return self._orig.embed_query(text)
        if hasattr(self._orig, "embed_texts") and callable(getattr(self._orig, "embed_texts")):
            return self._orig.embed_texts([text])[0]
        if callable(self._orig):
            return self._orig([text])[0]
        raise TypeError("Embedding object must be callable or implement embed_query/embed_texts")


def _build_local_sentence_transformer_embedding_fn(model_name: str = "all-MiniLM-L6-v2"):
    """
    Build a fallback embedding callable using sentence-transformers:
      f(list[str]) -> list[list[float]]
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "Failed to import sentence-transformers. Install it with: pip install sentence-transformers"
        ) from e

    model = SentenceTransformer(model_name)

    def embed_texts(texts: List[str]) -> List[List[float]]:
        arr = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [x.tolist() for x in arr]

    return embed_texts


# -----------------------
# DB loader + retrieval
# -----------------------
def _load_db(debug: bool = False) -> Chroma:
    """
    Load Chroma DB, wrapping the embedding function/object in EmbeddingAdapter if needed.
    If get_embedding_function() raises, fall back to local sentence-transformers callable.
    """
    emb_source = None
    try:
        emb_source = get_embedding_function()
        if debug:
            print("[debug] get_embedding_function() succeeded; wrapping returned object.", file=sys.stderr)
    except Exception as e:
        # Fallback to local sentence-transformers
        if debug:
            print(f"[debug] get_embedding_function() raised an error: {e}", file=sys.stderr)
            print("[debug] Falling back to local sentence-transformers embeddings.", file=sys.stderr)
        emb_source = _build_local_sentence_transformer_embedding_fn()

    embedding_obj = EmbeddingAdapter(emb_source)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_obj)
    return db


def _retrieve_docs(db: Chroma, query_text: str, top_k: int) -> List[Tuple[Document, float]]:
    """
    Retrieve documents with relevance scores using similarity_search_with_score.
    Returns a list of (Document, score) tuples where score is the similarity distance.
    Lower scores = more relevant (better match).
    """
    try:
        # Use similarity_search_with_score to get actual relevance scores
        results = db.similarity_search_with_score(query_text, k=top_k)
        return results
    except Exception as e:
        try:
            # Fallback: use as_retriever without scores
            retriever = db.as_retriever(search_kwargs={"k": top_k})
            docs = retriever.get_relevant_documents(query_text)
            return [(d, None) for d in docs]
        except Exception:
            # final fallback: attempt to pull raw documents from DB (if possible)
            try:
                fetched = db.get(include=["documents", "metadatas"])
                docs = []
                for doc_text, meta in zip(fetched.get("documents", []), fetched.get("metadatas", []))[:top_k]:
                    docs.append((Document(page_content=doc_text, metadata=meta), None))
                return docs
            except Exception:
                return []


def _format_faculty_response(results: List[Tuple[Document, float]]) -> str:
    """
    Format faculty information from database results as a descriptive narrative
    instead of structured column-answer format.
    """
    faculty_narratives = []
    
    for doc, _score in results:
        content = (doc.page_content or "").strip()
        
        # Parse the structured format: Key: Value pairs separated by newlines
        faculty_details = {}
        lines = content.split('\n')
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                faculty_details[key] = value
        
        # Extract and format the description
        description = faculty_details.get('Description', '')
        
        if description:
            # Use the description directly as it's already well-formatted
            faculty_narratives.append(description)
        else:
            # Build a narrative from available fields
            name = faculty_details.get('Faculty', 'Faculty Member').strip()
            department = faculty_details.get('Department', 'the Department').strip()
            designation = faculty_details.get('Designation', 'Faculty').strip()
            floor = faculty_details.get('Floor', '').strip()
            email = faculty_details.get('Teacher Email Id', '').strip()
            
            # Only add if we have a valid name
            if name and name != 'Faculty Member':
                narrative = f"{name} is a {designation.lower()} in {department}."
                
                if floor:
                    narrative += f" They are located on the {floor}."
                
                if email:
                    narrative += f" Contact: {email}"
                
                faculty_narratives.append(narrative)
    
    # Combine all faculty information into a single descriptive response
    if faculty_narratives:
        return "\n\n".join(faculty_narratives)
    else:
        return "No faculty information found."


def _format_source_info(doc: Document, score: float = None) -> str:
    """
    Format source information based on document type (image OCR or CSV).
    Returns a human-readable string describing the source.
    """
    metadata = doc.metadata
    data_type = metadata.get("data_type", "unknown")
    filename = metadata.get("filename", "unknown")
    
    info_parts = [f"[File: {filename}]"]
    
    if data_type == "csv":
        # CSV source
        row_idx = metadata.get("row_index")
        identifier = metadata.get("identifier")
        if row_idx is not None:
            info_parts.append(f"(row {row_idx})")
        if identifier:
            info_parts.append(f"[ID: {identifier}]")
    elif data_type == "image":
        # Image OCR source
        ocr_engine = metadata.get("ocr_engine", "OCR")
        num_boxes = metadata.get("num_boxes", 0)
        info_parts.append(f"({ocr_engine}, {num_boxes} text boxes)")
    
    if score is not None:
        info_parts.append(f"(relevance: {score:.3f})")
    
    return " ".join(info_parts)


# -----------------------
# Core query function
# -----------------------
def query_rag(query_text: str, top_k: int = 5, debug: bool = False, show_context: bool = False, return_sources: bool = False, prioritize_database: bool = True, relevance_threshold: float = 0.5, suppress_output: bool = False):
    """
    Query Chroma, build a prompt with the retrieved context, and call Ollama.
    Returns the model's response text (string).
    
    Args:
        query_text: The question to ask
        top_k: Number of documents to retrieve
        debug: Print debug information
        show_context: Display the full retrieved context
        prioritize_database: If True, return database results directly before using Ollama (default: True)
        relevance_threshold: Minimum relevance score (0-1) to return database results. Lower = more lenient (default: 0.5)
        suppress_output: If True, suppress console output (default: False, used by web app)
    """
    db = _load_db(debug=debug)
    results = _retrieve_docs(db, query_text, top_k=top_k)

    # Debug: print retrieved docs
    if debug:
        print(f"\n[debug] Retrieved {len(results)} docs.", file=sys.stderr)
        for i, (doc, score) in enumerate(results):
            data_type = doc.metadata.get("data_type", "unknown")
            print(f"[debug] DOC {i} (type={data_type}, score={score}):", file=sys.stderr)
            print(f"  Content preview: {repr(doc.page_content[:200])}...", file=sys.stderr)
            print(f"  Metadata: {doc.metadata}", file=sys.stderr)

    # PRIORITY 1: Check relevance threshold FIRST (before building context)
    has_relevant_results = False
    
    # Check if this is a faculty query (for more lenient threshold)
    faculty_keywords = ["faculty", "professor", "teacher", "instructor", "staff", "dr.", "prof."]
    is_faculty_query = any(keyword in query_text.lower() for keyword in faculty_keywords)
    
    # Use lower threshold for faculty queries
    effective_threshold = 2.0 if is_faculty_query else relevance_threshold
    
    if prioritize_database and results:
        # Check if the best result meets the relevance threshold
        best_score = min([score for _, score in results if score is not None], default=None)
        
        if best_score is not None:
            if best_score < effective_threshold:  # Lower scores are better
                has_relevant_results = True
            if debug:
                print(f"[debug] Best relevance score: {best_score:.4f}, threshold: {effective_threshold}", file=sys.stderr)
        else:
            # If no scores available, assume they're relevant
            has_relevant_results = True
            if debug:
                print(f"[debug] No scores available, assuming relevant results", file=sys.stderr)
    
    # ONLY build context if results are relevant
    context_text = ""
    if has_relevant_results:
        context_parts = []
        for doc, _score in results:
            if hasattr(doc, "page_content") and doc.page_content:
                # Add only the content without source information
                context_parts.append(doc.page_content)
        
        context_text = "\n\n---\n\n".join(context_parts).strip()

    # Show context if requested
    if show_context and context_text:
        print("\n" + "="*60)
        print("RETRIEVED CONTEXT:")
        print("="*60)
        print(context_text)
        print("="*60 + "\n")

    # --- Greeting + Name Detection Logic (check this FIRST, before context check) ---
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
    lower_q = query_text.lower()

    # Detect greeting
    is_greeting = any(lower_q.startswith(g) or lower_q.endswith(g) for g in greetings)

    # Detect name (patterns: "I am X", "my name is X", "this is X", "my name's X")
    name_patterns = [
        r"\bi am ([A-Za-z]+)\b",
        r"\bmy name is ([A-Za-z]+)\b",
        r"\bmy name'?s ([A-Za-z]+)\b",
        r"\bthis is ([A-Za-z]+)\b",
    ]

    detected_name = None
    for pattern in name_patterns:
        match = re.search(pattern, lower_q, re.IGNORECASE)
        if match:
            detected_name = match.group(1).capitalize()
            break

    # If greeting with name
    if is_greeting and detected_name:
        response_text = f"Hello, {detected_name}! How can I help you today?"
        print("\n" + "="*60)
        print("RESPONSE:")
        print("="*60)
        print(response_text)
        print("="*60 + "\n")
        if return_sources:
            return response_text, []
        return response_text

    # If greeting alone
    if is_greeting:
        response_text = "Hello! How can I help you today?"
        print("\n" + "="*60)
        print("RESPONSE:")
        print("="*60)
        print(response_text)
        print("="*60 + "\n")
        if return_sources:
            return response_text, []
        return response_text

    # If only name was mentioned (without much else)
    if detected_name and len(query_text.split()) <= 6:
        response_text = f"Nice to meet you, {detected_name}! How can I assist you?"
        print("\n" + "="*60)
        print("RESPONSE:")
        print("="*60)
        print(response_text)
        print("="*60 + "\n")
        if return_sources:
            return response_text, []
        return response_text

    # --- Now handle context-based responses ---
    # PRIORITY 1: If we have relevant results, return them from database
    if prioritize_database and has_relevant_results and context_text:
        if is_faculty_query:
            # Format faculty results as descriptive text
            response_text = _format_faculty_response(results)
        else:
            # Format database results with sources
            response_text = "Based on our database:\n\n"
            
            for i, (doc, score) in enumerate(results, 1):
                try:
                    content = (doc.page_content or "").strip()
                    source_info = _format_source_info(doc, score)
                    response_text += f"{i}. {content}\n\n"
                except Exception:
                    response_text += f"{i}. {doc.page_content}\n\n"
        
        if not suppress_output:
            print("\n" + "="*60)
            print("RESPONSE (FROM DATABASE):")
            print("="*60)
            print(response_text)
            print("="*60 + "\n")
        
        if return_sources:
            return response_text, []
        return response_text
    
    # PRIORITY 2: Use Ollama LLM for general questions or when no relevant database results
    if not context_text:
        prompt = f"Answer the question concisely:\nQ: {query_text}\nA:"
        if has_relevant_results is False and prioritize_database and results:
            best_score = min([s for _, s in results if s is not None], default=None)
            print(f"Database results not relevant enough (best score: {best_score:.4f} >= {relevance_threshold}). Using Ollama LLM.")
        else:
            print("No database results or using LLM for general knowledge.")
    else:
        prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

    # Call Ollama and handle errors
    try:
        model = Ollama(model="mistral")
        
        # ChatOllama returns AIMessage
        response = model.invoke(prompt)

        # Extract content
        if hasattr(response, "content"):
            response_text = response.content
        else:
            response_text = str(response)

        response_text = response_text.strip()

    except Exception as e:
        print(f"[LLM call failed: {e}]", file=sys.stderr)
        if return_sources:
            return f"[LLM unavailable: {e}]\nRetrieved context:\n{context_text}", []
        return f"[LLM unavailable: {e}]\nRetrieved context:\n{context_text}"

    # Prepare structured sources list (info + full content)
    sources = []
    for i, (doc, score) in enumerate(results, 1):
        source_info = _format_source_info(doc, score)
        excerpt = ""
        try:
            excerpt = (doc.page_content or "").strip()
            # collapse newlines for a short preview
            excerpt = " ".join(excerpt.splitlines())
        except Exception:
            excerpt = ""
        sources.append({"info": source_info, "excerpt": excerpt})

    # Print response (CLI-friendly)
    print("\n" + "="*60)
    print("RESPONSE:")
    print("="*60)
    print(response_text)
    print("="*60)

    # Print detailed sources for CLI
    print("\nSOURCES:")
    print("-"*60)
    for i, s in enumerate(sources, 1):
        print(f"{i}. {s}")
    print("-"*60 + "\n")

    if return_sources:
        return response_text, sources

    return response_text


# -----------------------
# CLI entrypoint
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    # Make query_text optional; if not supplied, prompt interactively
    parser.add_argument("query_text", type=str, nargs="?", help="The query text. If omitted, you will be prompted in loop mode.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of docs to retrieve from Chroma")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    parser.add_argument("--show-context", action="store_true", help="Display the full retrieved context")
    args = parser.parse_args()

    # If a query is provided as argument, run once and exit
    if args.query_text:
        query_rag(args.query_text, top_k=args.top_k, debug=args.debug, show_context=args.show_context)
        return

    # Interactive loop mode
    print("\n" + "="*60)
    print("🤖 Student Assistant - Interactive Mode")
    print("="*60)
    print("Type your questions and press Enter.")
    print("Type 'exit', 'quit', or 'thank you' to stop.\n")
    
    exit_commands = {'exit', 'quit', 'thank you', 'thanks', 'bye', 'goodbye'}
    
    while True:
        try:
            query_text = input("💬 Your question: ").strip()
            
            # Check for empty input
            if not query_text:
                print("⚠️  Please enter a question.\n")
                continue
            
            # Check for exit commands (case-insensitive)
            if query_text.lower() in exit_commands:
                print("\n" + "="*60)
                print("👋 Thank you for using Student Assistant!")
                print("="*60 + "\n")
                break
            
            # Process the query
            query_rag(query_text, top_k=args.top_k, debug=args.debug, show_context=args.show_context)
            print()  # Add spacing between queries
            
        except KeyboardInterrupt:
            print("\n\n" + "="*60)
            print("👋 Session interrupted. Thank you!")
            print("="*60 + "\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            if args.debug:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()