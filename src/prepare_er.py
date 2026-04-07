"""
prepare_er.py — TUẦN 4: Chuẩn bị cho Entity Resolution

Công việc:
  1. Load mentions.parquet (output của extract_entities.py)
  2. Normalize và đếm entity unique
  3. Tính co-occurrence matrix trong cùng document
  4. Lọc cặp có weight >= COOCCURRENCE_MIN_WEIGHT
  5. Tạo gold set template (CSV trống để annotate tay)
  6. In báo cáo thống kê NER

Output:
  - output/cooccurrence.csv        — cặp entity co-occur
  - output/gold_set_template.csv   — file template cho annotation tay

Chạy: python src/prepare_er.py
"""
import sys
import itertools
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    MENTIONS_PARQUET, COOCCURRENCE_CSV, GOLD_SET_CSV, OUTPUT_DIR,
    COOCCURRENCE_MIN_WEIGHT, GOLD_SET_SIZE
)
from src.utils import normalize_text, normalize_entity_name, logger


# ── Blocking helpers ─────────────────────────────────────────────────────────

def get_block_key(name: str) -> str:
    """
    Tạo block key cho ER blocking:
    - 3 ký tự đầu của tên normalized
    Dùng để giảm số cặp so sánh trong ER (tuần 5).
    """
    norm = normalize_entity_name(name)
    return norm[:3] if len(norm) >= 3 else norm


# ── Load & preprocess mentions ────────────────────────────────────────────────

def load_mentions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}.\n"
            "Hãy chạy trước: python src/extract_entities.py"
        )
    df = pd.read_parquet(path)
    logger.info("Đọc mentions: %d dòng từ %s", len(df), path)
    return df


def normalize_mentions(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột normalized và block_key."""
    df = df.copy()
    df["entity_norm"] = df["entity_text"].apply(normalize_entity_name)

    # Bỏ mention quá ngắn sau normalize
    df = df[df["entity_norm"].str.len() >= 2].copy()

    df["block_key"] = df["entity_norm"].apply(get_block_key)
    logger.info("  → %d mention hợp lệ sau normalize", len(df))
    return df


# ── Thống kê NER ─────────────────────────────────────────────────────────────

def print_ner_report(df: pd.DataFrame) -> None:
    """Báo cáo thống kê mention."""
    print("\n" + "=" * 65)
    print("BÁO CÁO NER — TUẦN 4")
    print("=" * 65)
    print(f"Tổng mention (sau normalize): {len(df):,}")
    print(f"Số doc: {df['doc_id'].nunique():,}")
    print(f"Entity unique (normalized): {df['entity_norm'].nunique():,}")
    print()

    print("─ Phân bố theo entity type ─")
    print(df["entity_type"].value_counts().to_string())
    print()

    print("─ Top 50 entity phổ biến nhất ─")
    top = (
        df.groupby(["entity_norm", "entity_type"])
        .size()
        .reset_index(name="mention_count")
        .sort_values("mention_count", ascending=False)
        .head(50)
    )
    print(top.to_string(index=False))
    print()

    print("─ Phân bố số mention / doc ─")
    per_doc = df.groupby("doc_id").size()
    print(f"  Mean: {per_doc.mean():.1f}")
    print(f"  Median: {per_doc.median():.0f}")
    print(f"  Max: {per_doc.max()}")
    print(f"  Doc không có mention: {df['doc_id'].nunique()} / {df['doc_id'].max() + 1}")

    print()
    print("─ Entity unique theo block_key (top 10 block lớn nhất) ─")
    block_sizes = df.groupby("block_key")["entity_norm"].nunique().sort_values(ascending=False)
    print(block_sizes.head(10).to_string())
    print("=" * 65)


# ── Co-occurrence ─────────────────────────────────────────────────────────────

def compute_cooccurrence(
    df: pd.DataFrame,
    min_weight: int = COOCCURRENCE_MIN_WEIGHT,
) -> pd.DataFrame:
    """
    Với mỗi document, tạo tất cả cặp entity (i, j) xuất hiện cùng nhau.
    Đếm số doc mà cặp đó cùng xuất hiện (co-occurrence weight).

    Trả về DataFrame: entity_a, entity_type_a, entity_b, entity_type_b, weight
    """
    logger.info("Tính co-occurrence (threshold weight >= %d)...", min_weight)

    # Group by doc: lấy distinct entity trong mỗi doc
    doc_entities = (
        df.groupby("doc_id")[["entity_norm", "entity_type"]]
        .apply(lambda g: list(g.drop_duplicates().itertuples(index=False, name=None)))
        .reset_index(name="entities")
    )

    cooc = defaultdict(int)

    for _, row in doc_entities.iterrows():
        entities = row["entities"]  # list of (entity_norm, entity_type)
        if len(entities) < 2:
            continue
        # Tạo tất cả cặp (không phân biệt thứ tự)
        for (ea, ta), (eb, tb) in itertools.combinations(entities, 2):
            # Chuẩn hóa thứ tự để tránh trùng (a,b) và (b,a)
            if ea > eb:
                ea, ta, eb, tb = eb, tb, ea, ta
            cooc[((ea, ta), (eb, tb))] += 1

    logger.info("  → %d cặp co-occur (trước khi lọc threshold)", len(cooc))

    # Chuyển thành DataFrame và lọc
    records = []
    for (ea_ta, eb_tb), weight in cooc.items():
        if weight >= min_weight:
            (ea, ta), (eb, tb) = ea_ta, eb_tb
            records.append({
                "entity_a":      ea,
                "type_a":        ta,
                "entity_b":      eb,
                "type_b":        tb,
                "weight":        weight,
            })

    cooc_df = pd.DataFrame(records).sort_values("weight", ascending=False)
    logger.info("  → %d cặp sau khi lọc weight >= %d", len(cooc_df), min_weight)

    return cooc_df


def print_cooccurrence_stats(cooc_df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("THỐNG KÊ CO-OCCURRENCE")
    print("=" * 65)
    print(f"Tổng cặp co-occur: {len(cooc_df):,}")
    if len(cooc_df) == 0:
        print("(Không có cặp nào — hãy giảm COOCCURRENCE_MIN_WEIGHT trong config.py)")
        return

    print(f"Weight trung bình: {cooc_df['weight'].mean():.1f}")
    print(f"Weight max: {cooc_df['weight'].max()}")
    print()

    print("─ Phân bố loại cặp entity ─")
    cooc_df["pair_type"] = cooc_df["type_a"] + " × " + cooc_df["type_b"]
    print(cooc_df["pair_type"].value_counts().to_string())
    print()

    print("─ Top 20 cặp co-occur mạnh nhất ─")
    print(
        cooc_df.head(20)[["entity_a", "type_a", "entity_b", "type_b", "weight"]]
        .to_string(index=False)
    )
    print("=" * 65)


# ── Gold set template ─────────────────────────────────────────────────────────

def create_gold_set_template(
    df: pd.DataFrame,
    n: int = GOLD_SET_SIZE,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample ngẫu nhiên n mention để làm gold set annotation thủ công.

    Output CSV có cột:
      mention_id, doc_id, entity_text, entity_norm, entity_type,
      source, specialty, manual_canonical (để con người điền)

    Hướng dẫn annotation:
      - manual_canonical: điền canonical form của entity
        Ví dụ: "htn" → "hypertension"
               "heart attack" → "myocardial infarction"
               "aspirin" → "aspirin" (nếu đã đúng)
      - Nếu không chắc: để trống hoặc ghi "SKIP"
    """
    # Stratified sample theo entity_type để đủ cả Drug và Disease
    samples = []
    for etype, group in df.groupby("entity_type"):
        n_sample = min(n // df["entity_type"].nunique(), len(group))
        samples.append(group.sample(n=n_sample, random_state=random_state))

    # Nếu vẫn còn thiếu, lấy thêm random
    current = pd.concat(samples)
    if len(current) < n:
        remaining = df[~df.index.isin(current.index)]
        extra = remaining.sample(
            n=min(n - len(current), len(remaining)),
            random_state=random_state
        )
        current = pd.concat([current, extra])

    gold = current.head(n).copy()
    gold = gold.reset_index(drop=True)
    gold["mention_id"] = gold.index
    gold["manual_canonical"] = ""   # cột trống cho người annotate

    # Chọn cột output
    cols = ["mention_id", "doc_id", "entity_text", "entity_norm",
            "entity_type", "source", "specialty", "manual_canonical"]
    cols = [c for c in cols if c in gold.columns]
    gold = gold[cols]

    logger.info("  → Gold set template: %d mention", len(gold))
    return gold


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== TUẦN 4: Chuẩn bị Entity Resolution ===")

    # 1. Load mentions
    df = load_mentions(MENTIONS_PARQUET)

    # 2. Normalize
    df = normalize_mentions(df)

    # 3. Báo cáo NER
    print_ner_report(df)

    # 4. Co-occurrence
    cooc_df = compute_cooccurrence(df, min_weight=COOCCURRENCE_MIN_WEIGHT)
    print_cooccurrence_stats(cooc_df)

    # Lưu co-occurrence
    cooc_df.to_csv(COOCCURRENCE_CSV, index=False)
    logger.info("Đã lưu co-occurrence: %s", COOCCURRENCE_CSV)

    # 5. Gold set template
    logger.info("Tạo gold set template (%d mention)...", GOLD_SET_SIZE)
    gold_df = create_gold_set_template(df, n=GOLD_SET_SIZE)
    gold_df.to_csv(GOLD_SET_CSV, index=False)
    logger.info("Đã lưu gold set template: %s", GOLD_SET_CSV)

    print("\n" + "=" * 65)
    print("VIỆC CẦN LÀM SAU KHI CHẠY SCRIPT NÀY:")
    print("=" * 65)
    print(f"1. Mở file: {GOLD_SET_CSV}")
    print("   Điền cột 'manual_canonical' cho từng mention.")
    print("   Ví dụ: 'htn' → 'hypertension', 'heart attack' → 'myocardial infarction'")
    print("   Mục tiêu: annotate ~150 cặp, dùng để đánh giá ER ở tuần 5.")
    print()
    print(f"2. Review file co-occurrence: {COOCCURRENCE_CSV}")
    print("   Đặc biệt chú ý top 50 cặp có weight cao nhất.")
    print("   Các cặp Drug×Disease co-occur mạnh = ứng viên hidden relationship.")
    print()
    print("3. Bước tiếp theo (Tuần 5): python src/entity_resolution.py")
    print("=" * 65)

    logger.info("=== Chuẩn bị ER hoàn tất ===")


if __name__ == "__main__":
    main()
