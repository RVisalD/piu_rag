# PIU RAG Chatbot

A local command-line chatbot for Paragon International University information. It searches prepared document chunks using FAISS embeddings and lexical matching, then uses Qwen3-1.7B to generate an answer from the retrieved context.

## 1. Set up Python

Use Python 3.11 for these instructions. Run commands from the project directory (`piu_rag`). Internet access is needed to install dependencies and download models.

On macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy faiss-cpu sentence-transformers "transformers>=4.51.0,<5" accelerate huggingface_hub
```

On Windows PowerShell, create and activate the environment with the following commands, then run the same `python -m pip` commands above:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Qwen3 requires Transformers 4.51.0 or newer; these instructions keep the Transformers 4.x API used by the scripts. See the [official Qwen model card](https://huggingface.co/Qwen/Qwen3-1.7B). `accelerate` supports the script's `device_map="auto"` model loading.

## 2. Download the language model

With the virtual environment activated, run:

```bash
hf download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B
```

This downloads the model weights, tokenizer, and configuration to the exact directory expected by `chat.py`. Allow several gigabytes of disk space for the model and dependencies. The [Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli) describes the download command.

The second model, `sentence-transformers/all-MiniLM-L6-v2`, creates search embeddings. The scripts download it automatically on first use and reuse the Hugging Face cache afterward.

## 3. Prepare the data

Place your prepared university chunks at:

```text
data/piu_chunks_improved.jsonl
```

This file and the `models/` directory are ignored by Git, so a fresh clone needs the data supplied separately and the model downloaded again.

Use one JSON object per line. For example, this is a placeholder record to illustrate the format; replace the text with actual source content:

```json
{"doc_id":"example-document","chunk_id":0,"title":"Example title","text":"Replace with university source text."}
```

- `text`: nonempty string containing the source content.
- `doc_id`: document identifier, shared by chunks from the same document.
- `chunk_id`: integer indicating the chunk's order within its document. Each `(doc_id, chunk_id)` pair should be unique.
- `title`: optional string describing the content.
- `embedding_text`: optional string to embed instead of the combined title and text.

The index builder skips empty text and malformed JSON lines. Supply `doc_id` and `chunk_id` for correct chat deduplication and neighboring-chunk lookup. The current scripts consume prepared JSONL; they do not convert files in `data/raw/` into chunks.

## 4. Build the search index

```bash
python build_index.py
```

Wait for `INDEX BUILD COMPLETE`. This creates:

```text
data/piu.index        # FAISS vector index
data/piu_chunks.pkl   # Matching chunk metadata
```

Rebuild the index whenever you change the JSONL data or embedding model. Keep the generated index and metadata together so their records stay aligned.

## 5. Run the chatbot

```bash
python chat.py
```

After the models load, enter a question at the `You:` prompt, for example:

```text
You: What programs does Paragon International University offer?
```

Answers depend on the source information in your dataset. Each answer is followed by `Was this answer satisfactory? (y/n):`. Enter `y` to continue, or `n` to provide a reason and display retrieval debugging information. Feedback is printed in the terminal and is not saved to a file.

Type `exit` or `quit` at the `You:` prompt to stop. Each question is processed independently; the script does not include previous turns in the model prompt.

For later sessions on macOS or Linux:

```bash
source .venv/bin/activate
python chat.py
```

## Configuration

Edit the constants in the scripts to adjust behavior:

| Setting | Location | Default | Purpose |
| --- | --- | --- | --- |
| `EMBEDDING_MODEL` | Both scripts | `sentence-transformers/all-MiniLM-L6-v2` | Must match between indexing and chat. Rebuild after changing. |
| `BATCH_SIZE` | `build_index.py` | `32` | Number of chunks embedded per batch. |
| `QWEN_MODEL_PATH` | `chat.py` | `models/Qwen3-1.7B` | Local language model directory. |
| `TOP_K` | `chat.py` | `5` | Maximum retrieved matches. |
| `MIN_SCORE` | `chat.py` | `0.30` | Semantic threshold for candidates without lexical evidence. |
| `NEIGHBOR_WINDOW` | `chat.py` | `1` | Include one preceding and one following chunk from the same document. |

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `ModuleNotFoundError` | Activate `.venv` and rerun the dependency installation command. |
| Missing `piu_chunks_improved.jsonl` | Supply the prepared JSONL file under `data/`. |
| Missing `piu.index` or `piu_chunks.pkl` | Run `python build_index.py` before starting chat. |
| `No valid chunks found` | Check that JSONL records contain nonempty string values for `text`. |
| `KeyError: 'qwen3'` | Install the documented Transformers version range in the active environment. |
| Model cannot load from `models/Qwen3-1.7B` | Rerun the model download command and check that the directory contains configuration, tokenizer, and weight files. |
| Index and metadata count mismatch | Rebuild both files from the same JSONL dataset. |
| Slow generation or insufficient memory | Automatic device placement depends on available hardware. CPU generation can be slow. Close other memory-heavy applications; reduce `TOP_K` or `NEIGHBOR_WINDOW` to shorten context, or reduce `max_new_tokens` in `chat.py` to shorten responses. |

The setup commands are based on the current scripts; dependency installation and an end-to-end model run have not been verified as part of writing this README.
