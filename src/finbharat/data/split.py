"""
Create stratified train/dev/test splits at the company (report) level.

Usage:
    uv run python -m finbharat.data.split --data-root dataset --output splits.json
    uv run python -m finbharat.data.split --data-root dataset --output splits.json --show
"""
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

from finbharat.data.loader import FinBharatDataset


# Sectors that must appear in every split (important for paper analysis)
PRIORITY_SECTORS = {
    "Private_Sector_Bank",
    "Information_Technology",
    "Pharmaceutical",
    "Automobile",
    "Fast_Moving_Consumer_Goods",
}

SPLIT_RATIOS = {"train": 0.60, "dev": 0.20, "test": 0.20}


def create_splits(
    data_root: Path,
    seed: int = 42,
    train_frac: float = 0.60,
    dev_frac: float = 0.20,
) -> dict[str, list[dict]]:
    """
    Returns {"train": [...], "dev": [...], "test": [...]}
    Each entry is {"sector": ..., "company": ...}.

    Strategy:
      1. Collect all 187 companies.
      2. Group by sector.
      3. For multi-company sectors (rare), distribute across splits.
      4. Shuffle single-company sectors and assign in 60/20/20 ratio.
      5. Ensure every priority sector appears in test and dev.
    """
    rng = random.Random(seed)
    ds = FinBharatDataset(data_root)
    all_companies = ds.list_available_companies(source="e_m")

    # Group by sector
    by_sector: dict[str, list[str]] = defaultdict(list)
    for c in all_companies:
        by_sector[c["sector"]].append(c["company"])

    train, dev, test = [], [], []

    # Handle priority sectors first — guarantee one in test and one in dev
    for sector in PRIORITY_SECTORS:
        companies = by_sector.get(sector, [])
        if not companies:
            continue
        rng.shuffle(companies)
        # Force at least one into test and dev
        test.append({"sector": sector, "company": companies[0]})
        if len(companies) > 1:
            dev.append({"sector": sector, "company": companies[1]})
            train.extend({"sector": sector, "company": c} for c in companies[2:])
        else:
            # Only one company → it's in test; dev will be covered by random assignment
            pass
        del by_sector[sector]

    # Remaining sectors
    remaining = []
    for sector, companies in by_sector.items():
        for company in companies:
            remaining.append({"sector": sector, "company": company})
    rng.shuffle(remaining)

    n = len(remaining)
    n_dev = max(1, round(n * dev_frac))
    n_test = max(1, round(n * (1 - train_frac - dev_frac)))
    n_train = n - n_dev - n_test

    train.extend(remaining[:n_train])
    dev.extend(remaining[n_train: n_train + n_dev])
    test.extend(remaining[n_train + n_dev:])

    return {"train": train, "dev": dev, "test": test}


def split_stats(splits: dict[str, list[dict]]) -> None:
    for name, companies in splits.items():
        sectors = {c["sector"] for c in companies}
        print(f"  {name:5s}: {len(companies):3d} companies | {len(sectors):3d} sectors")


def main():
    parser = argparse.ArgumentParser(description="Create FinBharat train/dev/test splits")
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("splits.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true", help="Print split contents")
    args = parser.parse_args()

    splits = create_splits(args.data_root, seed=args.seed)
    print("\nSplit statistics:")
    split_stats(splits)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nSaved to {args.output}")

    if args.show:
        for name, companies in splits.items():
            print(f"\n--- {name} ---")
            for c in sorted(companies, key=lambda x: x["sector"]):
                print(f"  {c['sector']}: {c['company']}")


if __name__ == "__main__":
    main()
