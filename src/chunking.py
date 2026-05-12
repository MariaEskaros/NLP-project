from pathlib import Path
from typing import List, Dict


def chunk_text_by_words(
    text: str,
    chunk_size: int = 350,
    overlap: int = 70
) -> List[str]:

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def build_chunks_from_folder(
    normalized_dir: Path,
    chunk_size: int = 350,
    overlap: int = 70
) -> List[Dict]:

    all_chunks = []

    for file_path in sorted(normalized_dir.glob("*.txt")):

        text = file_path.read_text(encoding="utf-8")

        chunks = chunk_text_by_words(
            text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for i, chunk in enumerate(chunks):

            all_chunks.append({
                "chunk_id": f"{file_path.stem}_chunk_{i}",
                "episode": file_path.stem,
                "source_file": file_path.name,
                "text": chunk
            })

    return all_chunks