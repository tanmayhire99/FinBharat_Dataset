import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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
    return QARecord(
        question=raw["question"],
        answer=raw["answer"],
        evidence=raw["evidence"],
        question_type=raw.get("question_type", "Unknown"),
        requires_calculation=raw.get("requires_calculation", False),
        sector=raw.get("sector", ""),
        company=raw.get("company", ""),
        year=raw.get("year", ""),
        section=raw.get("section", ""),
        page_numbers=raw.get("page_numbers", []),
        bundle_id=raw.get("bundle_id", ""),
        difficulty=raw.get("difficulty", "easy"),
        global_id=raw.get("global_id", f"{raw.get('company','unk')}_{idx}"),
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

    def build_context_for_qa(self, qa: QARecord) -> str:
        source = "e_m" if qa.difficulty in ("easy", "medium") else "h_m"
        return self.get_bundle_text(qa.sector, qa.company, qa.bundle_id, source)

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
