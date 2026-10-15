import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# =====================================================
# Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

CHUNKS_FILE = (
    BASE_DIR
    / "../data"
    / "piu_chunks_improved.jsonl"
)

INDEX_FILE = (
    BASE_DIR
    / "../data"
    / "piu.index"
)

METADATA_FILE = (
    BASE_DIR
    / "../data"
    / "piu_chunks.pkl"
)


# =====================================================
# Config
# =====================================================

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

BATCH_SIZE = 32


# =====================================================
# Load chunks
# =====================================================

def load_chunks():

    chunks = []

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)

            except json.JSONDecodeError as e:

                print(
                    f"Skipping line {line_number}: {e}"
                )

                continue

            text = item.get(
                "text",
                "",
            ).strip()

            title = item.get(
                "title",
                "",
            ).strip()

            if not text:
                continue

            item["title"] = title
            item["text"] = text

            chunks.append(item)

    return chunks


# =====================================================
# Build text for embedding
# =====================================================

def get_embedding_text(chunk):

    title = chunk.get(
        "title",
        "",
    ).strip()

    text = chunk.get(
        "text",
        "",
    ).strip()

    if title:
        return (
            f"Title: {title}\n\n"
            f"{text}"
        )

    return text


# =====================================================
# Main
# =====================================================

def main():

    print("Loading chunks...")

    chunks = load_chunks()

    if not chunks:
        raise ValueError(
            "No valid chunks found."
        )

    print(
        f"Loaded {len(chunks)} chunks."
    )

    # IMPORTANT:
    # FAISS vector position must correspond
    # exactly to chunks[position].

    texts = [
        get_embedding_text(chunk)
        for chunk in chunks
    ]

    print("Loading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    print(
        "Embedding shape:",
        embeddings.shape,
    )

    # =================================================
    # FAISS cosine similarity
    # =================================================

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    # =================================================
    # Save
    # =================================================

    print("Saving FAISS index...")

    faiss.write_index(
        index,
        str(INDEX_FILE),
    )

    print("Saving metadata...")

    with open(
        METADATA_FILE,
        "wb",
    ) as f:

        pickle.dump(
            chunks,
            f,
        )

    # =================================================
    # Validation
    # =================================================

    if index.ntotal != len(chunks):

        raise ValueError(
            "Index and metadata count mismatch."
        )

    print()
    print("============================")
    print("INDEX BUILD COMPLETE")
    print("============================")

    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        f"Vectors: {index.ntotal}"
    )

    print(
        f"Dimension: {dimension}"
    )

    print(
        f"Index: {INDEX_FILE}"
    )

    print(
        f"Metadata: {METADATA_FILE}"
    )


if __name__ == "__main__":
    main()