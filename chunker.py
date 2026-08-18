import re

def split_into_sentences(text):
    abbreviations = ["e.g.", "i.e.", "Dr.", "Mr.", "Mrs.", "Ms.", "U.S.", "vs.", "etc."]
    placeholder_map = {}
    protected_text = text
    for i, abbr in enumerate(abbreviations):
        placeholder = f"__ABBR{i}__"
        placeholder_map[placeholder] = abbr
        protected_text = protected_text.replace(abbr, placeholder)

    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z]|$)', protected_text)

    restored = []
    for s in sentences:
        for placeholder, abbr in placeholder_map.items():
            s = s.replace(placeholder, abbr)
        restored.append(s.strip())

    return [s for s in restored if s]


def split_long_sentence(sentence, size):
    words = sentence.split(" ")
    parts = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= size:
            current += (" " if current else "") + word
        else:
            if current:
                parts.append(current)
            current = word
    if current:
        parts.append(current)
    return parts


def chunk_text(text, size, min_size=100):
    """Chunks a single string of text. Returns list of chunk strings."""
    chunks = []
    buffer = ""

    def flush_buffer():
        nonlocal buffer
        if buffer:
            chunks.append(buffer)
            buffer = ""

    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]

    for paragraph in paragraphs:
        if len(paragraph) <= size:
            if len(buffer) + len(paragraph) + 2 <= size:
                buffer = (buffer + "\n\n" + paragraph) if buffer else paragraph
            else:
                flush_buffer()
                buffer = paragraph
            continue

        flush_buffer()
        sentences = split_into_sentences(paragraph)
        current_chunk = ""

        for sentence in sentences:
            if len(sentence) > size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                sub_parts = split_long_sentence(sentence, size)
                chunks.extend(sub_parts[:-1])
                current_chunk = sub_parts[-1] if sub_parts else ""
                continue

            if len(current_chunk) + len(sentence) + 1 <= size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            buffer = current_chunk

    flush_buffer()

    merged = []
    for chunk in chunks:
        if merged and len(merged[-1]) < min_size and len(merged[-1]) + len(chunk) + 2 <= size:
            merged[-1] = merged[-1] + "\n\n" + chunk
        else:
            merged.append(chunk)

    return merged


def chunker(pages, size, min_size=100):
    """
    pages: list of {"page_number": int, "text": str}
    Returns: list of {"text": str, "page_number": int}
    """
    all_chunks = []
    for page in pages:
        page_chunks = chunk_text(page["text"], size, min_size)
        for chunk_text_str in page_chunks:
            all_chunks.append({
                "text": chunk_text_str,
                "page_number": page["page_number"]
            })
    return all_chunks
