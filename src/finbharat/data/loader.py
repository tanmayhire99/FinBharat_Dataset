import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Table serialization helpers
# ---------------------------------------------------------------------------

def _html_table_to_markdown(html: str) -> str:
    """Convert a single HTML table to Markdown pipe format."""
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE)
    md_rows = []
    for i, row in enumerate(rows):
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.DOTALL | re.IGNORECASE)
        cells = [re.sub(r"<[^>]+>", " ", c).strip() for c in cells]
        cells = [re.sub(r"\s+", " ", c) for c in cells]
        md_rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            md_rows.append("|" + "|".join(["---"] * len(cells)) + "|")
    return "\n".join(md_rows)


def _html_table_to_linearized(html: str) -> str:
    """Convert a single HTML table to key-value linearized format."""
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE)
    if not rows:
        return ""
    header_cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", rows[0], re.DOTALL | re.IGNORECASE)
    headers = [re.sub(r"<[^>]+>", " ", c).strip() for c in header_cells]
    parts = []
    for row in rows[1:]:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.DOTALL | re.IGNORECASE)
        cells = [re.sub(r"<[^>]+>", " ", c).strip() for c in cells]
        if not any(c for c in cells):
            continue
        kvs = []
        for h, v in zip(headers, cells):
            if h and v:
                kvs.append(f"{h}: {v}")
        if kvs:
            parts.append(" | ".join(kvs))
    return "\n".join(parts)


def _serialize_tables(text: str, fmt: str = "markdown") -> str:
    """Replace all HTML tables in text with the requested format."""
    def replace_table(m: re.Match) -> str:
        html = m.group(0)
        if fmt == "markdown":
            return _html_table_to_markdown(html)
        elif fmt == "linearized":
            return _html_table_to_linearized(html)
        return html  # "html" passthrough

    return re.sub(r"<table[^>]*>.*?</table>", replace_table,
                  text, flags=re.DOTALL | re.IGNORECASE)


@dataclass
class VerificationAnchors:
    """Structured gold evidence available on hard and multihop QA pairs."""
    calculation_inputs: list[float]          # gold numeric operands
    cross_section_sources: list[str]         # hop chain — list of source section names
    alignment_status: str                    # "consistent" | "contradiction" | ""
    hop_count: int                           # len(cross_section_sources)
    is_red_flag: bool                        # True when alignment_status == "contradiction"


def _parse_verification_anchors(raw: dict) -> Optional["VerificationAnchors"]:
    va = raw.get("verification_anchors")
    if not va or not isinstance(va, dict):
        return None
    sources = va.get("cross_section_sources", [])
    if isinstance(sources, str):
        sources = [sources] if sources else []
    alignment = va.get("alignment_status", "")
    inputs_raw = va.get("calculation_inputs", [])
    inputs: list[float] = []
    for x in (inputs_raw or []):
        try:
            inputs.append(float(x))
        except (TypeError, ValueError):
            pass
    return VerificationAnchors(
        calculation_inputs=inputs,
        cross_section_sources=sources,
        alignment_status=alignment,
        hop_count=len(sources),
        is_red_flag=(str(alignment).lower() == "contradiction"),
    )


BRSR_KEYWORDS = frozenset([
    "brsr", "business responsibility", "sustainability report",
    "esg", "principle 1", "principle 2", "principle 3", "principle 4",
    "principle 5", "principle 6", "principle 7", "principle 8", "principle 9",
    "essential indicators", "leadership indicators",
    "responsible business conduct", "section c",
])


def is_brsr_section(section: str) -> bool:
    sl = section.lower()
    return any(kw in sl for kw in BRSR_KEYWORDS)


@dataclass
class QARecord:
    question: str
    answer: str
    evidence: str
    question_type: str
    requires_calculation: bool
    sector: str
    company: str
    year: str
    section: str
    page_numbers: list[int]
    bundle_id: str
    difficulty: str
    global_id: Optional[str] = None
    verification_anchors: Optional[VerificationAnchors] = None
    is_brsr: bool = False


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    sector: str
    company: str
    year: str
    section: str
    page_numbers: list[int]
    has_table: bool
    token_count: int
    oversized_table: bool
    score: int
    text: str


@dataclass
class BundleRecord:
    bundle_id: str
    score: int
    chunk_count: int
    section: str


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_qa_record(raw: dict, idx: int = 0) -> QARecord:
    section = raw.get("section", "")
    return QARecord(
        question=raw["question"],
        answer=raw["answer"],
        evidence=raw["evidence"],
        question_type=raw.get("question_type", "Unknown"),
        requires_calculation=raw.get("requires_calculation", False),
        sector=raw.get("sector", ""),
        company=raw.get("company", ""),
        year=raw.get("year", ""),
        section=section,
        page_numbers=raw.get("page_numbers", []),
        bundle_id=raw.get("bundle_id", ""),
        difficulty=raw.get("difficulty", "easy"),
        global_id=raw.get("global_id", f"{raw.get('company','unk')}_{idx}"),
        verification_anchors=_parse_verification_anchors(raw),
        is_brsr=is_brsr_section(section),
    )


def parse_chunk_record(raw: dict) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=raw["chunk_id"],
        doc_id=raw["doc_id"],
        sector=raw.get("sector", ""),
        company=raw.get("company", ""),
        year=raw.get("year", ""),
        section=raw.get("section", ""),
        page_numbers=raw.get("page_numbers", []),
        has_table=raw.get("has_table", False),
        token_count=raw.get("token_count", 0),
        oversized_table=raw.get("oversized_table", False),
        score=raw.get("score", 0),
        text=raw["text"],
    )


def parse_bundle_record(raw: dict) -> BundleRecord:
    return BundleRecord(
        bundle_id=raw["bundle_id"],
        score=raw.get("score", 0),
        chunk_count=raw.get("chunk_count", 0),
        section=raw.get("section", ""),
    )


SAMPLE_COMPANIES = [
    {"sector": "Private_Sector_Bank", "company": "HDFC_Bank"},
    {"sector": "Auto_Components_&_Equipments", "company": "Bosch"},
    {"sector": "Airline", "company": "Interglobe_Aviat"},
]


class FinBharatDataset:
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.e_m_root = self.data_root / "output_final_e_m"
        self.h_m_root = self.data_root / "output_final_h_m"
        self._chunk_cache: dict[str, list[ChunkRecord]] = {}

    def _company_dir(self, sector: str, company: str, source: str = "e_m") -> Path:
        root = self.e_m_root if source == "e_m" else self.h_m_root
        return root / sector / company / "FY2025"

    def load_easy_qa(self, sector: str, company: str) -> list[QARecord]:
        path = self._company_dir(sector, company) / "easy" / "test_easy_qa.jsonl"
        raw = load_jsonl(path)
        return [parse_qa_record(r, idx=i) for i, r in enumerate(raw)]

    def load_medium_qa(self, sector: str, company: str) -> list[QARecord]:
        path = self._company_dir(sector, company) / "medium" / "test_medium_qa.jsonl"
        raw = load_jsonl(path)
        return [parse_qa_record(r, idx=i) for i, r in enumerate(raw)]

    def load_hard_qa(self, sector: str, company: str) -> list[QARecord]:
        path = self._company_dir(sector, company, source="h_m") / "hard" / "hard_qa_pairs.jsonl"
        raw = load_jsonl(path)
        return [parse_qa_record(r, idx=i) for i, r in enumerate(raw)]

    def load_multihop_qa(self, sector: str, company: str) -> list[QARecord]:
        path = self._company_dir(sector, company, source="h_m") / "multihop" / "multihop_qa_pairs.jsonl"
        raw = load_jsonl(path)
        return [parse_qa_record(r, idx=i) for i, r in enumerate(raw)]

    def load_chunks(self, sector: str, company: str, source: str = "e_m") -> list[ChunkRecord]:
        cache_key = f"{source}:{sector}:{company}"
        if cache_key not in self._chunk_cache:
            path = self._company_dir(sector, company, source) / "chunks.jsonl"
            raw = load_jsonl(path)
            self._chunk_cache[cache_key] = [parse_chunk_record(r) for r in raw]
        return self._chunk_cache[cache_key]

    def load_bundles(self, sector: str, company: str, source: str = "e_m") -> list[BundleRecord]:
        path = self._company_dir(sector, company, source) / "semantic_bundles_details.jsonl"
        raw = load_jsonl(path)
        return [parse_bundle_record(r) for r in raw]

    def get_bundle_text(self, sector: str, company: str, bundle_id: str, source: str = "e_m") -> str:
        chunks = self.load_chunks(sector, company, source)
        match = re.match(r"(.+):(\d+)_to_(.+):(\d+)", bundle_id)
        if not match:
            return ""
        start_id = f"{match.group(1)}:{match.group(2)}"
        end_id = f"{match.group(3)}:{match.group(4)}"
        collecting = False
        texts = []
        for chunk in chunks:
            if chunk.chunk_id == start_id:
                collecting = True
            if collecting:
                texts.append(chunk.text)
            if chunk.chunk_id == end_id:
                collecting = False
                break
        return "\n\n".join(texts)

    def load_sample(
        self,
        companies: list[dict] | None = None,
        difficulty: str = "easy",
        max_per_company: int | None = None,
    ) -> list[QARecord]:
        if companies is None:
            companies = SAMPLE_COMPANIES

        loader_map = {
            "easy": self.load_easy_qa,
            "medium": self.load_medium_qa,
            "hard": self.load_hard_qa,
            "multihop": self.load_multihop_qa,
        }
        loader = loader_map.get(difficulty)
        if loader is None:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        all_records: list[QARecord] = []
        for comp in companies:
            records = loader(comp["sector"], comp["company"])
            for i, r in enumerate(records):
                r.global_id = f"{comp['company']}_{difficulty}_{i}"
            if max_per_company is not None:
                records = records[:max_per_company]
            all_records.extend(records)
        return all_records

    def build_context_for_qa(self, qa: QARecord, table_format: str = "html") -> str:
        """Build context string for a QA record.

        Args:
            qa: the QA record
            table_format: one of "html" (default), "markdown", "linearized"
        """
        source = "e_m" if qa.difficulty in ("easy", "medium") else "h_m"
        text = self.get_bundle_text(qa.sector, qa.company, qa.bundle_id, source)
        if table_format != "html" and text:
            text = _serialize_tables(text, fmt=table_format)
        return text

    def load_brsr_subset(
        self,
        companies: list[dict] | None = None,
        difficulties: list[str] | None = None,
        max_per_company: int | None = None,
    ) -> list[QARecord]:
        """Return only QA pairs from BRSR/ESG sections across all requested tiers.

        Uses e_m companies for easy/medium and h_m companies for hard/multihop
        to avoid trying to load h_m data for companies only in e_m.
        """
        difficulties = difficulties or ["easy", "medium", "hard", "multihop"]
        all_records: list[QARecord] = []
        for diff in difficulties:
            source = "e_m" if diff in ("easy", "medium") else "h_m"
            comp_list = companies or self.list_available_companies(source)
            try:
                records = self.load_sample(companies=comp_list, difficulty=diff,
                                           max_per_company=max_per_company)
                all_records.extend(r for r in records if r.is_brsr)
            except FileNotFoundError:
                continue  # company exists in one source but not the other
        return all_records

    def list_available_companies(self, source: str = "e_m") -> list[dict]:
        root = self.e_m_root if source == "e_m" else self.h_m_root
        companies = []
        for sector_dir in sorted(root.iterdir()):
            if not sector_dir.is_dir():
                continue
            for company_dir in sorted(sector_dir.iterdir()):
                if not company_dir.is_dir():
                    continue
                companies.append({
                    "sector": sector_dir.name,
                    "company": company_dir.name,
                })
        return companies
