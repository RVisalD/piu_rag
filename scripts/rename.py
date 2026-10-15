import json
import re
from pathlib import Path

FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "piu_chunks_improved.jsonl"
)


def title_to_doc_id(title):
    prefix = title.split("-", 1)[0].strip().lower()

    if not prefix:
        raise ValueError(f"Invalid title: {title!r}")

    return re.sub(r"\s+", "-", prefix)


def main():
    chunks = []

    # Validate everything before writing.
    for line_number, line in enumerate(
        FILE.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        chunk = json.loads(line)
        title = chunk.get("title")

        if not isinstance(title, str):
            raise ValueError(
                f"Missing or invalid title on line {line_number}"
            )

        chunk["doc_id"] = title_to_doc_id(title)
        chunks.append(chunk)

    temp_file = FILE.with_suffix(".tmp")
    temp_file.write_text(
        "".join(
            json.dumps(chunk, ensure_ascii=False) + "\n"
            for chunk in chunks
        ),
        encoding="utf-8",
    )
    temp_file.replace(FILE)

    print(f"Updated doc_id for {len(chunks)} chunks.")


if __name__ == "__main__":
    main()