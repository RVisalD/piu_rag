import pickle
import re
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


# =====================================================
# Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

INDEX_FILE = (
    BASE_DIR
    / "data"
    / "piu.index"
)

METADATA_FILE = (
    BASE_DIR
    / "data"
    / "piu_chunks.pkl"
)

QWEN_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "Qwen3-1.7B"
)


# =====================================================
# Config
# =====================================================

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

TOP_K = 5

# Start permissive.
# Tune this after testing your real questions.
MIN_SCORE = 0.30

# 1 means:
# previous chunk + matched chunk + next chunk
NEIGHBOR_WINDOW = 1


# =====================================================
# Load embedding model
# =====================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# =====================================================
# Load index
# =====================================================

print("Loading FAISS index...")

index = faiss.read_index(
    str(INDEX_FILE)
)


# =====================================================
# Load chunks
# =====================================================

print("Loading metadata...")

with open(
    METADATA_FILE,
    "rb",
) as f:

    chunks = pickle.load(f)


print(
    f"Chunks loaded: {len(chunks)}"
)

print(
    f"FAISS vectors: {index.ntotal}"
)


if index.ntotal != len(chunks):

    raise ValueError(
        "FAISS index and metadata do not match. "
        f"Index={index.ntotal}, "
        f"metadata={len(chunks)}"
    )


# =====================================================
# Build doc lookup
# =====================================================

chunks_by_doc = {}

for chunk in chunks:

    doc_id = chunk.get("doc_id")

    if doc_id is None:
        continue

    chunks_by_doc.setdefault(
        doc_id,
        []
    ).append(chunk)


for doc_id in chunks_by_doc:

    chunks_by_doc[doc_id].sort(
        key=lambda x: x.get(
            "chunk_id",
            0,
        )
    )


# =====================================================
# Load Qwen
# =====================================================

print("Loading Qwen...")

tokenizer = AutoTokenizer.from_pretrained(
    str(QWEN_MODEL_PATH)
)

model = AutoModelForCausalLM.from_pretrained(
    str(QWEN_MODEL_PATH),
    torch_dtype="auto",
    device_map="auto",
)

print("Qwen loaded.")


# =====================================================
# Retrieve
# =====================================================

def _tokens(text):
    """Simple lexical tokenizer for BM25-like retrieval."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


# Build a lightweight lexical index over the same fields users search for.
# This makes exact role/name queries ("head of computer science", "who is dean")
# complement semantic FAISS retrieval.
LEXICAL_DOCS = []
DOC_FREQ = Counter()

for chunk in chunks:
    searchable = " ".join([
        chunk.get("title", ""),
        chunk.get("embedding_text", ""),
        chunk.get("text", ""),
    ])
    toks = _tokens(searchable)
    counts = Counter(toks)
    LEXICAL_DOCS.append((toks, counts))
    for token in counts:
        DOC_FREQ[token] += 1

AVG_DOC_LEN = (
    sum(len(toks) for toks, _ in LEXICAL_DOCS) / max(len(LEXICAL_DOCS), 1)
)


def _bm25_scores(question):
    q_tokens = _tokens(question)
    if not q_tokens:
        return {}

    N = len(chunks)
    scores = {}
    k1 = 1.5
    b = 0.75

    for idx, (tokens, counts) in enumerate(LEXICAL_DOCS):
        dl = max(len(tokens), 1)
        score = 0.0
        for term in q_tokens:
            tf = counts.get(term, 0)
            if not tf:
                continue
            df = DOC_FREQ.get(term, 0)
            # Positive BM25 IDF.
            idf = np.log(1.0 + (N - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1 - b + b * dl / max(AVG_DOC_LEN, 1.0))
            score += idf * (tf * (k1 + 1)) / denom
        if score > 0:
            scores[idx] = float(score)
    return scores


def _role_boost(question, chunk):
    """Boost exact leadership-role matches without hard-coding names."""
    q = " ".join(_tokens(question))
    haystack = " ".join([
        chunk.get("title", ""),
        chunk.get("embedding_text", ""),
        chunk.get("text", ""),
    ]).lower()

    boost = 0.0
    role_words = ["head", "dean", "director", "rector", "provost", "coordinator", "manager"]
    for role in role_words:
        if role in q and role in haystack:
            boost += 0.08

    # Strongly reward department/faculty terms that occur in both query and record.
    q_terms = set(_tokens(question))
    h_terms = set(_tokens(haystack))
    important = q_terms & h_terms & {
        "computer", "science", "information", "technology", "technologies",
        "management", "systems", "engineering", "civil", "industrial",
        "architecture", "business", "administration", "economics", "finance",
        "banking", "trade", "logistics", "graduate", "faculty", "department"
    }
    boost += min(len(important) * 0.025, 0.15)
    return boost


def retrieve(question, top_k=TOP_K):
    query_vector = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    search_k = min(max(top_k * 10, 50), index.ntotal)
    scores, ids = index.search(query_vector, search_k)

    # Rank maps are used for Reciprocal Rank Fusion (RRF). This is robust
    # because semantic FAISS and BM25 scores are on different scales.
    vector_rank = {}
    vector_score = {}
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
        if idx == -1:
            continue
        vector_rank[int(idx)] = rank
        vector_score[int(idx)] = float(score)

    lexical_scores = _bm25_scores(question)
    lexical_order = sorted(lexical_scores, key=lexical_scores.get, reverse=True)
    lexical_rank = {idx: rank for rank, idx in enumerate(lexical_order, start=1)}

    candidates = set(vector_rank) | set(lexical_rank)
    fused = []
    RRF_K = 60.0

    for idx in candidates:
        v_rank = vector_rank.get(idx)
        l_rank = lexical_rank.get(idx)
        score = 0.0
        if v_rank is not None:
            score += 1.0 / (RRF_K + v_rank)
        if l_rank is not None:
            score += 1.0 / (RRF_K + l_rank)

        chunk = chunks[idx]
        score += _role_boost(question, chunk)
        fused.append((score, idx))

    fused.sort(reverse=True)

    results = []
    seen = set()
    for hybrid_score, idx in fused:
        chunk = chunks[idx]
        key = (chunk.get("doc_id"), chunk.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)

        # Keep a permissive semantic floor only when the candidate has no
        # lexical evidence. Exact lexical matches should not be discarded.
        sem = vector_score.get(idx, -1.0)
        lex = lexical_scores.get(idx, 0.0)
        if lex <= 0 and sem < MIN_SCORE:
            continue

        results.append({
            "score": float(hybrid_score),
            "semantic_score": float(sem),
            "lexical_score": float(lex),
            "doc_id": chunk.get("doc_id"),
            "chunk_id": chunk.get("chunk_id"),
            "title": chunk.get("title", ""),
            "text": chunk.get("text", ""),
        })
        if len(results) >= top_k:
            break

    return results


# =====================================================
# Neighbor expansion
# =====================================================

def expand_with_neighbors(
    result,
    window=NEIGHBOR_WINDOW,
):

    doc_id = result.get("doc_id")
    chunk_id = result.get("chunk_id")

    if doc_id is None:
        return result

    document_chunks = chunks_by_doc.get(
        doc_id,
        [],
    )

    if not document_chunks:
        return result

    position = None

    for i, chunk in enumerate(
        document_chunks
    ):

        if (
            chunk.get("chunk_id")
            == chunk_id
        ):

            position = i
            break

    if position is None:
        return result

    start = max(
        0,
        position - window,
    )

    end = min(
        len(document_chunks),
        position + window + 1,
    )

    selected = document_chunks[
        start:end
    ]

    expanded_result = result.copy()

    expanded_result["text"] = (
        "\n\n".join(
            chunk.get(
                "text",
                "",
            )
            for chunk in selected
            if chunk.get(
                "text",
                "",
            ).strip()
        )
    )

    expanded_result["context_chunk_ids"] = [
        chunk.get("chunk_id")
        for chunk in selected
    ]

    return expanded_result


# =====================================================
# Build context
# =====================================================

def build_context(results):

    sections = []

    for number, result in enumerate(
        results,
        start=1,
    ):

        section = f"""
SOURCE {number}

Title:
{result["title"]}

Information:
{result["text"]}
""".strip()

        sections.append(section)

    return "\n\n".join(
        sections
    )


# =====================================================
# Generate
# =====================================================

def answer_question(question):

    # 1. Search
    results = retrieve(
        question
    )

    if not results:

        return (
            "I could not verify that from the "
            "available Paragon International "
            "University information.",
            [],
        )

    # 2. Add nearby context
    expanded_results = [
        expand_with_neighbors(result)
        for result in results
    ]

    # 3. Build context
    context = build_context(
        expanded_results
    )

    system_prompt = """
You are an assistant for Paragon International University.

Your job is to answer questions using ONLY the
retrieved university information provided in the context.

Rules:

1. Do not invent facts.
2. Do not guess.
3. Use the title and information together.
4. If the information does not support the answer, say:
   "I could not verify that from the available Paragon International University information."
5. Answer the user's question directly.
6. Keep answers concise unless more explanation is needed.
7. For study plans, curricula, or course lists, include ALL semesters
  available in the context.
8. Do not omit semesters unless the user specifically asks for only
  certain semesters.
9. Do not mention embeddings, FAISS, vector databases,
   retrieval scores, chunks, or internal implementation.
10. If multiple retrieved sources conflict, do not choose
    one by guessing. State that the available information
   appears inconsistent.
""".strip()

    user_prompt = f"""
Retrieved Paragon International University information:

{context}

User question:

{question}

Answer using only the retrieved information above.
""".strip()

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=800,
            # Qwen3 warns that greedy decoding can cause endless repetition.
            # Use its recommended sampling settings for non-thinking mode.
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
        )

    generated = output[
        :,
        inputs.input_ids.shape[1]:
    ]

    answer = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0]

    return (
        answer.strip(),
        expanded_results,
    )


# =====================================================
# Interactive chat
# =====================================================

def main():

    print()
    print("============================")
    print("Paragon RAG Chatbot")
    print("============================")
    print("Type 'exit' or 'quit' to stop.")

    while True:

        question = input("\nYou: ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            break

        answer, results = answer_question(question)

        print("\nAssistant:")
        print(answer)

        # -----------------------------------------
        # Evaluation
        # -----------------------------------------

        satisfied = input(
            "\nWas this answer satisfactory? (y/n): "
        ).strip().lower()

        if satisfied == "y":
            print("✓ Answer marked as satisfactory.")

        elif satisfied == "n":

            reason = input(
                "What was wrong with the answer? "
            ).strip()

            print("\n--- DEBUG INFORMATION ---")

            for number, result in enumerate(results, start=1):

                print(
                    f"{number}. {result['title']}"
                    f" | chunk={result['chunk_id']}"
                    f" | semantic={result['semantic_score']:.3f}"
                    f" | lexical={result['lexical_score']:.3f}"
                )

                if "context_chunk_ids" in result:
                    print(
                        "   context chunks:",
                        result["context_chunk_ids"]
                    )

            print("-------------------------")
            print("Reason:", reason)

if __name__ == "__main__":
    main()
