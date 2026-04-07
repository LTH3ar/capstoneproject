"""
extract_entities.py — TUẦN 3: NER trên Bộ 3 (Medical Transcriptions)

Công việc:
  1. Đọc mtsamples.csv
  2. Chạy scispaCy (en_ner_bc5cdr_md) trên cột 'transcription'
  3. Lưu kết quả entity mention ra output/mentions.parquet
  4. Thống kê sơ bộ số lượng mention theo type

Chuẩn bị trước khi chạy:
  pip install scispacy
  pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

Chạy: python src/extract_entities.py
      python src/extract_entities.py --limit 500   # chạy thử 500 doc đầu
"""
import sys
import argparse
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    BO3_CSV, MENTIONS_PARQUET, OUTPUT_DIR, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
)
from src.utils import normalize_text, safe_str, logger

# ── Nhãn entity từ en_ner_bc5cdr_md ─────────────────────────────────────────
# CHEMICAL → Drug/Chemical
# DISEASE  → Disease/Condition
ENTITY_TYPE_MAP = {
    "CHEMICAL": "Drug",
    "DISEASE":  "Disease",
}

# ── Từ điển bổ sung (dictionary-based NER) ────────────────────────────────────
# Để tăng recall cho abbreviations y khoa phổ biến
ABBREVIATION_MAP = {
    "htn":   "hypertension",
    "dm":    "diabetes mellitus",
    "dm2":   "type 2 diabetes mellitus",
    "dm1":   "type 1 diabetes mellitus",
    "cad":   "coronary artery disease",
    "copd":  "chronic obstructive pulmonary disease",
    "chf":   "congestive heart failure",
    "afib":  "atrial fibrillation",
    "af":    "atrial fibrillation",
    "mi":    "myocardial infarction",
    "cva":   "cerebrovascular accident",
    "dvt":   "deep vein thrombosis",
    "pe":    "pulmonary embolism",
    "uti":   "urinary tract infection",
    "gerd":  "gastroesophageal reflux disease",
    "ckd":   "chronic kidney disease",
    "esrd":  "end-stage renal disease",
    "ra":    "rheumatoid arthritis",
    "sle":   "systemic lupus erythematosus",
    "ms":    "multiple sclerosis",
    "tia":   "transient ischemic attack",
    "bph":   "benign prostatic hyperplasia",
}


def load_spacy_model():
    """Load scispaCy model, in hướng dẫn cài nếu chưa có."""
    try:
        import spacy
        nlp = spacy.load("en_ner_bc5cdr_md")
        logger.info("Đã load model: en_ner_bc5cdr_md")
        return nlp
    except OSError:
        logger.error(
            "Chưa cài model scispaCy.\n"
            "Chạy lệnh sau:\n"
            "  pip install scispacy\n"
            "  pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/"
            "releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
        )
        sys.exit(1)
    except ImportError:
        logger.error("Chưa cài spacy. Chạy: pip install scispacy spacy")
        sys.exit(1)


def load_transcriptions(csv_path: Path, limit: int = None) -> pd.DataFrame:
    """Đọc mtsamples.csv, lọc dòng có transcription hợp lệ."""
    logger.info("Đọc Bộ 3: %s", csv_path)
    df = pd.read_csv(csv_path, index_col=0)
    logger.info("  → %d dòng gốc", len(df))

    # Giữ các cột cần thiết
    df = df[["description", "medical_specialty", "sample_name",
             "transcription", "keywords"]].copy()

    # Bỏ dòng không có transcription
    df = df.dropna(subset=["transcription"])
    df = df[df["transcription"].str.strip().str.len() > 50]
    df = df.reset_index(drop=True)
    df["doc_id"] = df.index

    logger.info("  → %d doc có transcription hợp lệ", len(df))

    if limit:
        df = df.head(limit)
        logger.info("  → Chạy giới hạn %d doc đầu tiên (--limit)", limit)

    return df


def extract_abbreviations(text: str, doc_id: int) -> list[dict]:
    """
    Dictionary-based NER để bổ sung các abbreviation y khoa.
    Tìm từng word trong văn bản, map với ABBREVIATION_MAP.
    """
    mentions = []
    words = text.lower().split()
    seen = set()
    for word in words:
        word_clean = word.strip(".,;:()")
        if word_clean in ABBREVIATION_MAP and word_clean not in seen:
            mentions.append({
                "doc_id":        doc_id,
                "entity_text":   word_clean,
                "entity_type":   "Disease",   # hầu hết abbrev là tên bệnh
                "canonical_hint": ABBREVIATION_MAP[word_clean],
                "source":        "abbrev_dict",
            })
            seen.add(word_clean)
    return mentions


def run_ner(
    df: pd.DataFrame,
    nlp,
    batch_size: int = 50,
    n_process: int = 1,
) -> list[dict]:
    """
    Chạy scispaCy NER trên tất cả transcription.
    Trả về list các dict mention.

    n_process=1 mặc định để tương thích Windows (multiprocessing có thể lỗi).
    """
    from tqdm import tqdm

    texts = df["transcription"].tolist()
    doc_ids = df["doc_id"].tolist()
    specialties = df["medical_specialty"].fillna("").tolist()

    mentions = []
    logger.info("Chạy scispaCy NER trên %d document...", len(texts))

    # scispaCy pipe: batch để tăng tốc
    docs = nlp.pipe(texts, batch_size=batch_size, n_process=n_process)

    for doc, doc_id, specialty in tqdm(
        zip(docs, doc_ids, specialties),
        total=len(texts),
        desc="  NER"
    ):
        for ent in doc.ents:
            entity_type = ENTITY_TYPE_MAP.get(ent.label_, ent.label_)
            entity_text = normalize_text(ent.text)
            if len(entity_text) < 2 or len(entity_text) > 80:
                continue   # lọc noise
            mentions.append({
                "doc_id":        doc_id,
                "entity_text":   entity_text,
                "entity_type":   entity_type,
                "canonical_hint": "",   # sẽ điền sau khi ER
                "source":        "scispacy",
                "specialty":     normalize_text(specialty),
            })

        # Bổ sung abbreviation matching
        abbrev_mentions = extract_abbreviations(texts[doc_ids.index(doc_id)], doc_id)
        for m in abbrev_mentions:
            m["specialty"] = normalize_text(specialty)
        mentions.extend(abbrev_mentions)

    logger.info("  → Tổng %d entity mention (thô)", len(mentions))
    return mentions


def print_ner_stats(mentions_df: pd.DataFrame) -> None:
    """In thống kê NER cơ bản."""
    print("\n" + "=" * 60)
    print("THỐNG KÊ NER")
    print("=" * 60)
    print(f"Tổng số mention: {len(mentions_df):,}")
    print(f"Số doc có mention: {mentions_df['doc_id'].nunique():,}")
    print()

    print("Phân bố theo entity type:")
    print(mentions_df["entity_type"].value_counts().to_string())
    print()

    print("Phân bố theo nguồn NER:")
    print(mentions_df["source"].value_counts().to_string())
    print()

    print("Top 30 entity phổ biến nhất:")
    top = (
        mentions_df
        .groupby(["entity_text", "entity_type"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(30)
    )
    print(top.to_string(index=False))
    print()

    # Trung bình mention mỗi doc
    per_doc = mentions_df.groupby("doc_id").size()
    print(f"Trung bình mention / doc: {per_doc.mean():.1f}")
    print(f"Median mention / doc: {per_doc.median():.1f}")
    print(f"Max mention / doc: {per_doc.max()}")
    print("=" * 60)


# ── Specialty nodes (không cần NER, lấy trực tiếp từ metadata) ───────────────

def ingest_specialty_nodes(df: pd.DataFrame, driver) -> None:
    """Tạo node Specialty trong Neo4j từ cột medical_specialty."""
    from src.utils import neo4j_session, run_batch

    specialties = df["medical_specialty"].dropna().unique()
    rows = [{"name": normalize_text(s)} for s in specialties if s.strip()]

    Q = """
    UNWIND $rows AS row
    MERGE (sp:Specialty {name: row.name})
    """
    with neo4j_session(driver) as session:
        run_batch(session, Q, rows, 200)
    logger.info("  ✓ %d Specialty nodes đã ingest.", len(rows))


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="NER trên Medical Transcriptions")
    parser.add_argument("--limit", type=int, default=None,
                        help="Giới hạn số doc xử lý (để test nhanh)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Batch size cho scispaCy pipe (default: 50)")
    parser.add_argument("--skip-neo4j", action="store_true",
                        help="Bỏ qua ingest Specialty vào Neo4j")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("=== TUẦN 3: NER trên Bộ 3 (Medical Transcriptions) ===")

    if not BO3_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {BO3_CSV}")

    # Load model
    nlp = load_spacy_model()

    # Load data
    df = load_transcriptions(BO3_CSV, limit=args.limit)

    # Run NER
    mentions = run_ner(df, nlp, batch_size=args.batch_size, n_process=1)

    # Lưu ra parquet
    mentions_df = pd.DataFrame(mentions)
    mentions_df.to_parquet(MENTIONS_PARQUET, index=False)
    logger.info("Đã lưu mentions vào: %s (%d dòng)", MENTIONS_PARQUET, len(mentions_df))

    # Thống kê
    print_ner_stats(mentions_df)

    # Ingest Specialty nodes vào Neo4j (nếu không skip)
    if not args.skip_neo4j:
        try:
            from src.utils import get_neo4j_driver
            driver = get_neo4j_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            ingest_specialty_nodes(df, driver)
            driver.close()
        except Exception as e:
            logger.warning("Bỏ qua ingest Neo4j (--skip-neo4j để tắt cảnh báo): %s", e)

    logger.info("=== NER hoàn tất ===")
    logger.info("File output: %s", MENTIONS_PARQUET)
    logger.info("Bước tiếp theo: chạy python src/prepare_er.py")


if __name__ == "__main__":
    main()
