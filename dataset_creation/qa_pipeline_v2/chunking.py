import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Regex Patterns ──────────────────────────────────────────────────────────

PAGE_TAG_RE = re.compile(r"<page_number>(.*?)</page_number>", re.IGNORECASE)
PAGE_MARKER_RE = re.compile(r"__PAGE_(.*?)__")
TABLE_RE = re.compile(r"<table>.*?</table>", re.IGNORECASE | re.DOTALL)
HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s+(.+?)|\*\*(.+?)\*\*)\s*$", re.MULTILINE)
WATERMARK_RE = re.compile(r"<watermark>.*?</watermark>", re.IGNORECASE | re.DOTALL)
IMG_RE = re.compile(r"<img>.*?</img>", re.IGNORECASE | re.DOTALL)
SENTENCE_END_RE = re.compile(r'[.。!?"\']$')

# Financial text patterns for token estimation
NUMBER_RE = re.compile(r'\d[\d,.]*')
CURRENCY_RE = re.compile(r'[₹$€£]')


@dataclass
class Block:
    text: str
    page_numbers: List[int] = field(default_factory=list)
    section: str = "root"
    is_table: bool = False
    token_count: int = 0


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    sector: str
    company: str
    year: str
    section: str
    text: str
    page_numbers: List[int] = field(default_factory=list)
    has_table: bool = False
    token_count: int = 0
    oversized_table: bool = False
    is_truncated: bool = False
    images_removed: int = 0
    watermarks_removed: int = 0
    score: int = 0


# ── Public API ──────────────────────────────────────────────────────────────

def chunk_markdown_text(
    doc_id: str,
    sector: str,
    company: str,
    year: str,
    markdown_text: str,
    target_tokens: int = 800,
    max_tokens: int = 900,
    preserve_page_tags: bool = True,
    preserve_image_placeholders: bool = True,
) -> List[Chunk]:
    """
    Chunk markdown text with intelligent section tracking, page number preservation,
    and proper handling of tables and financial content.
    """
    markdown_text = _remove_hallucinations(markdown_text, doc_id)
    if not markdown_text.strip():
        logger.error(f"File {doc_id} skipped because the entire text was a hallucination.")
        return []

    blocks = _build_blocks(
        markdown_text,
        preserve_page_tags=preserve_page_tags,
        preserve_image_placeholders=preserve_image_placeholders,
    )
    return _group_blocks(
        doc_id=doc_id,
        sector=sector,
        company=company,
        year=year,
        blocks=blocks,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
    )


# ── Block Building ──────────────────────────────────────────────────────────

def _build_blocks(
    markdown_text: str,
    preserve_page_tags: bool = True,
    preserve_image_placeholders: bool = True,
) -> List[Block]:
    """Parse markdown into semantic blocks with metadata."""
    text = _normalize_text(markdown_text)
    
    images_removed = len(IMG_RE.findall(text))
    watermarks_removed = len(WATERMARK_RE.findall(text))
    
    text = WATERMARK_RE.sub("", text)
    
    if preserve_image_placeholders:
        text = IMG_RE.sub(lambda m: f"<!-- image: {m.group(0)} -->", text)
    else:
        text = IMG_RE.sub("", text)
    
    text = PAGE_TAG_RE.sub(lambda m: f"\n__PAGE_{m.group(1)}__\n", text)
    text = _normalize_table_whitespace(text)
    
    raw_blocks = _split_blocks_preserve_tables(text)
    blocks: List[Block] = []
    pending_pages: List[int] = []
    current_section = "root"
    document_title = None
    
    for raw in raw_blocks:
        pages, cleaned = _extract_pages(raw, preserve_tags=preserve_page_tags)
        cleaned = cleaned.strip()
        if not cleaned:
            pending_pages.extend(pages)
            continue
        
        heading, level = _extract_heading(cleaned)
        if heading:
            if level == 1:
                document_title = heading
                # H1 doesn't change current_section
            else:
                current_section = heading
        
        all_pages = pending_pages + pages
        pending_pages = []
        
        if not all_pages and blocks:
            all_pages = blocks[-1].page_numbers.copy()
        
        is_table = "<table" in cleaned.lower()
        token_count = _estimate_tokens(cleaned)
        
        blocks.append(Block(
            text=cleaned,
            page_numbers=sorted(set(all_pages)),
            section=current_section,
            is_table=is_table,
            token_count=token_count,
        ))
    
    return blocks


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip horizontal rules (3+ dashes on their own line)
    text = re.sub(r'^\s*---+\s*$', '', text, flags=re.MULTILINE)
    # Clean up resulting blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def _remove_hallucinations(text: str, doc_id: str = "Unknown", n_words: int = 150, threshold: int = 10) -> str:
    """
    Detects if a block of `n_words` repeats `threshold` or more times sequentially.
    Removes ONLY the hallucinated repeating blocks and keeps the rest of the text.
    Also saves the removed text to a debug log file.
    """
    matches = list(re.finditer(r'\S+', text))
    if len(matches) < n_words * threshold:
        return text
        
    from collections import Counter
    n_grams = []
    step = 10
    for i in range(0, len(matches) - n_words + 1, step):
        chunk = " ".join(m.group() for m in matches[i:i+n_words])
        n_grams.append((chunk, matches[i].start(), matches[i+n_words-1].end()))
        
    counts = Counter(ngram for ngram, _, _ in n_grams)
    if not counts:
        return text
        
    bad_ngrams = [ngram for ngram, count in counts.items() if count >= threshold]
    if not bad_ngrams:
        return text

    intervals_to_remove = []
    for target in bad_ngrams:
        indices = [i for i, (ngram, _, _) in enumerate(n_grams) if ngram == target]
        
        i = 0
        while i <= len(indices) - threshold:
            window = indices[i+threshold-1] - indices[i]
            if window <= (n_words * threshold * 2) // step:
                start_char = n_grams[indices[i]][1]
                j = i + threshold - 1
                while j + 1 < len(indices):
                    next_window = indices[j+1] - indices[j]
                    if next_window <= (n_words * 3) // step:
                        j += 1
                    else:
                        break
                end_char = n_grams[indices[j]][2]
                intervals_to_remove.append((start_char, end_char))
                i = j + 1
            else:
                i += 1
                
    if not intervals_to_remove:
        return text
        
    intervals_to_remove.sort()
    merged = []
    for start, end in intervals_to_remove:
        if not merged:
            merged.append([start, end])
        else:
            prev_start, prev_end = merged[-1]
            if start <= prev_end + 2000: # merge if close
                merged[-1][1] = max(prev_end, end)
            else:
                merged.append([start, end])
                
    result = []
    last_idx = 0
    
    debug_log_path = "hallucinations_debug.log"
    
    for start, end in merged:
        removed_text = text[start:end]
        result.append(text[last_idx:start])
        logger.warning(f"Removing hallucination block from char {start} to {end}.")
        result.append(f"\n\n[...HALLUCINATION REMOVED (Chars {start}-{end})...]\n\n")
        last_idx = end
        
        # Write to debug file
        try:
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write(f"=== DOC: {doc_id} | Removed Chars: {start} to {end} ===\n")
                f.write(removed_text[:1000] + "\n...[TRUNCATED IN LOG]...\n" if len(removed_text) > 1000 else removed_text + "\n")
                f.write("="*80 + "\n\n")
        except Exception as e:
            logger.error(f"Failed to write to hallucination debug log: {e}")
            
    result.append(text[last_idx:])
    
    return "".join(result)

def _normalize_table_whitespace(text: str) -> str:
    """Remove blank lines inside table tags to prevent splitting."""
    def _clean_table(match: re.Match) -> str:
        full = match.group(0)
        # Replace blank lines inside table with single newlines
        cleaned = re.sub(r'\n\s*\n', '\n', full)
        return cleaned
    
    return TABLE_RE.sub(_clean_table, text)


def _split_blocks_preserve_tables(text: str) -> List[str]:
    """Split text into blocks, keeping tables intact."""
    blocks: List[str] = []
    last = 0
    
    for match in TABLE_RE.finditer(text):
        before = text[last:match.start()]
        blocks.extend(_split_paragraphs(before))
        blocks.append(match.group(0))
        last = match.end()
    
    blocks.extend(_split_paragraphs(text[last:]))
    return [b for b in blocks if b.strip()]


def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs by blank lines."""
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


def _extract_pages(text: str, preserve_tags: bool = True) -> Tuple[List[int], str]:
    """
    Extract page numbers from markers.
    Returns (page_numbers, cleaned_text).
    """
    pages_raw = PAGE_MARKER_RE.findall(text)
    pages = []
    for pr in pages_raw:
        # Extract the first sequence of digits as the page number
        m = re.search(r'\d+', pr)
        if m:
            try:
                pages.append(int(m.group(0)))
            except ValueError:
                pass
    
    if preserve_tags:
        cleaned = PAGE_MARKER_RE.sub(
            lambda m: f'<page_number>{m.group(1)}</page_number>', text
        )
    else:
        cleaned = PAGE_MARKER_RE.sub("", text)
    
    return pages, cleaned


def _extract_heading(text: str) -> Tuple[str, int]:
    """
    Extract the first markdown heading from text.
    Returns (heading_text, level). Level 0 means no heading found.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('<page_number>') or line.startswith('<!--'):
            continue
        match = HEADING_RE.match(line)
        if match:
            if match.group(1):
                # It's a # heading
                full_match = match.group(0).lstrip()
                level = len(full_match) - len(full_match.lstrip('#'))
                heading = match.group(1).strip()
            else:
                # It's a ** heading
                level = 2  # Treat bold as H2/Section
                heading = match.group(2).strip()
            return heading, level
    return "", 0


# ── Token Estimation ─────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """
    Estimate token count for financial text.
    Numbers and currency symbols cost more tokens than regular words.
    """
    words = len(re.findall(r"\S+", text))
    if words == 0:
        return 0
    
    numbers = len(NUMBER_RE.findall(text))
    currency_symbols = len(CURRENCY_RE.findall(text))
    special_chars = len(re.findall(r'[%&/\\=+\-@#*]', text))
    
    # Base ratio for English text
    base_ratio = 1.3
    
    # Adjust for financial content density
    number_density = numbers / words
    special_density = (currency_symbols + special_chars) / words
    
    ratio = base_ratio + (number_density * 0.8) + (special_density * 0.5)
    ratio = min(ratio, 2.5)  # Cap at 2.5 to avoid overestimation
    
    return int(words * ratio)


# ── Chunk Grouping ───────────────────────────────────────────────────────────

def _group_blocks(
    doc_id: str,
    sector: str,
    company: str,
    year: str,
    blocks: List[Block],
    target_tokens: int,
    max_tokens: int,
) -> List[Chunk]:
    """Group blocks into chunks respecting token limits AND section boundaries."""
    chunks: List[Chunk] = []
    buffer: List[Block] = []
    buffer_tokens = 0
    buffer_pages: List[int] = []
    buffer_sections: List[str] = []
    buffer_has_table = False
    
    def flush(forced: bool = False, oversized_table: bool = False) -> None:
        nonlocal buffer, buffer_tokens, buffer_pages, buffer_sections, buffer_has_table
        if not buffer:
            return
        
        chunk_text = _assemble_chunk_text(buffer)
        # Use FIRST section (where chunk starts), not majority
        section = _resolve_start_section(buffer, chunks)
        page_numbers = sorted(set(buffer_pages))
        is_truncated = not _is_complete_sentence(chunk_text)
        
        chunk_id = f"{sector}/{company}/{year}:{len(chunks):04d}"
        
        chunks.append(Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            sector=sector,
            company=company,
            year=year,
            section=section,
            text=chunk_text,
            page_numbers=page_numbers,
            has_table=buffer_has_table,
            token_count=buffer_tokens,
            oversized_table=oversized_table,
            is_truncated=is_truncated,
        ))
        
        buffer = []
        buffer_tokens = 0
        buffer_pages = []
        buffer_sections = []
        buffer_has_table = False
    
    for i, block in enumerate(blocks):
        # Flush on section boundary if buffer is substantial
        if (buffer and 
            block.section != "root" and 
            buffer_sections and 
            block.section != buffer_sections[-1] and
            buffer_tokens >= target_tokens * 0.3):  # Don't flush for tiny buffers
            flush()
        
        # Handle oversized tables
        if not buffer and block.is_table and block.token_count > max_tokens:
            buffer = [block]
            buffer_tokens = block.token_count
            buffer_pages = list(block.page_numbers)
            buffer_sections = [block.section]
            buffer_has_table = True
            flush(oversized_table=True)
            continue
        
        # Determine if we should flush before adding this block
        should_flush = False
        
        if buffer:
            if block.is_table:
                if buffer_tokens + block.token_count > max_tokens:
                    should_flush = True
            elif buffer_tokens >= target_tokens:
                should_flush = True
            elif buffer_tokens + block.token_count > max_tokens:
                should_flush = True
        
        if should_flush:
            flush()
        
        buffer.append(block)
        buffer_tokens += block.token_count
        buffer_pages.extend(block.page_numbers)
        buffer_sections.append(block.section)
        buffer_has_table = buffer_has_table or block.is_table
    
    flush()
    return chunks

def _assemble_chunk_text(blocks: List[Block]) -> str:
    """Assemble blocks into clean chunk text."""
    parts = []
    for block in blocks:
        text = block.text.strip()
        # Ensure tables have surrounding whitespace
        if block.is_table:
            parts.append(text)
        else:
            parts.append(text)
    
    return "\n\n".join(parts).strip()


def _resolve_start_section(buffer: List[Block], previous_chunks: List[Chunk]) -> str:
    """
    Section is determined by where the chunk STARTS (first non-root section),
    NOT by majority content. This preserves semantic boundaries.
    """
    for block in buffer:
        if block.section and block.section != "root":
            return block.section
    
    if previous_chunks and previous_chunks[-1].is_truncated:
        return previous_chunks[-1].section
    
    return "root"


def _is_complete_sentence(text: str) -> bool:
    """Check if text ends with a sentence terminator."""
    stripped = text.rstrip()
    if not stripped:
        return False
    last_char = stripped[-1]
    return bool(SENTENCE_END_RE.match(last_char)) or last_char in ')}]'


# ── Utility Functions ────────────────────────────────────────────────────────

def validate_chunks(chunks: List[Chunk]) -> Dict[str, List[str]]:
    """
    Validate chunk quality and return issues found.
    """
    issues: Dict[str, List[str]] = {
        "warnings": [],
        "errors": [],
    }
    
    for i, chunk in enumerate(chunks):
        prefix = f"Chunk {chunk.chunk_id}"
        
        # Check for truncation
        if chunk.is_truncated:
            issues["warnings"].append(f"{prefix}: Ends with incomplete sentence")
        
        # Check for empty chunks
        if not chunk.text.strip():
            issues["errors"].append(f"{prefix}: Empty text")
        
        # Check table integrity
        if chunk.has_table:
            open_tables = chunk.text.lower().count("<table")
            close_tables = chunk.text.lower().count("</table>")
            if open_tables != close_tables:
                issues["errors"].append(
                    f"{prefix}: Mismatched table tags ({open_tables} open, {close_tables} close)"
                )
        
        # Check token bounds
        if chunk.token_count > 1000:
            issues["warnings"].append(
                f"{prefix}: Token count {chunk.token_count} exceeds recommended max"
            )
        
        # Check for section inheritance issues
        if chunk.section == "root" and not chunk.oversized_table:
            issues["warnings"].append(f"{prefix}: Section is 'root' (no heading found)")
    
    return issues


def merge_truncated_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """
    Post-process: merge truncated chunks with next chunk if possible.
    Use with caution - may violate max_tokens.
    """
    merged: List[Chunk] = []
    pending: Optional[Chunk] = None
    
    for chunk in chunks:
        if pending:
            # Try to merge
            combined_text = pending.text + "\n\n" + chunk.text
            combined_tokens = pending.token_count + chunk.token_count
            
            if combined_tokens <= 900:  # Within limit
                new_chunk = Chunk(
                    chunk_id=pending.chunk_id,
                    doc_id=pending.doc_id,
                    sector=pending.sector,
                    company=pending.company,
                    year=pending.year,
                    section=pending.section,
                    text=combined_text,
                    page_numbers=sorted(set(pending.page_numbers + chunk.page_numbers)),
                    has_table=pending.has_table or chunk.has_table,
                    token_count=combined_tokens,
                    oversized_table=pending.oversized_table or chunk.oversized_table,
                    is_truncated=chunk.is_truncated,
                )
                pending = new_chunk
                continue
            else:
                merged.append(pending)
                pending = None
        
        if chunk.is_truncated and not chunk.oversized_table:
            pending = chunk
        else:
            merged.append(chunk)
    
    if pending:
        merged.append(pending)
    
    return merged
