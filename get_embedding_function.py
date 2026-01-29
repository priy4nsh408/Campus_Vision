from sentence_transformers import SentenceTransformer


def get_embedding_function():
    """
    Returns embedding function using sentence-transformers (local fallback).
    This is more reliable than Ollama embeddings and doesn't require external services.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def embed_texts(texts):
        """Embed multiple texts"""
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [x.tolist() for x in embeddings]
    
    return embed_texts
