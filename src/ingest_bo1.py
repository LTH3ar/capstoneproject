"""
ingest_bo1.py — TUẦN 2: Ingest Bộ 1 (Drugs, Side Effects & Medical Conditions)

Nodes tạo ra:   Drug, Disease, SideEffect, DrugClass, BrandName
Edges tạo ra:   TREATS, CAUSES, BELONGS_TO, HAS_BRAND

Chạy: python src/ingest_bo1.py
Yêu cầu: Neo4j đang chạy, file .env đã cấu hình đúng.
"""
import sys
import logging
from pathlib import Path

import pandas as pd

# Thêm root vào path để import được config, utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    BO1_CSV, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, BATCH_SIZE
)
from src.utils import (
    normalize_text, parse_comma_list, parse_side_effects,
    get_neo4j_driver, neo4j_session, run_batch, create_constraints,
    safe_str, safe_float, logger
)

# ── Cypher queries ────────────────────────────────────────────────────────────

# 1. Merge Drug + Disease + edge TREATS
Q_DRUG_DISEASE = """
UNWIND $rows AS row
MERGE (drug:Drug {generic_name: row.generic_name})
  ON CREATE SET
    drug.drug_name  = row.drug_name,
    drug.rating     = row.rating,
    drug.rx_otc     = row.rx_otc,
    drug.aliases    = [row.drug_name],
    drug.source     = ['bo1']
  ON MATCH SET
    drug.rating     = CASE WHEN row.rating > 0 THEN row.rating ELSE drug.rating END

MERGE (dis:Disease {name: row.disease_name})
  ON CREATE SET
    dis.description = row.disease_description,
    dis.aliases     = [],
    dis.source      = ['bo1']

MERGE (drug)-[t:TREATS]->(dis)
  ON CREATE SET
    t.rating = row.rating,
    t.source = 'bo1'
"""

# 2. Merge BrandName + edge HAS_BRAND (1 row = 1 brand)
Q_BRAND = """
UNWIND $rows AS row
MATCH (drug:Drug {generic_name: row.generic_name})
MERGE (b:BrandName {name: row.brand_name})
MERGE (drug)-[:HAS_BRAND]->(b)
"""

# 3. Merge DrugClass + edge BELONGS_TO (1 row = 1 class)
Q_CLASS = """
UNWIND $rows AS row
MATCH (drug:Drug {generic_name: row.generic_name})
MERGE (c:DrugClass {name: row.class_name})
MERGE (drug)-[:BELONGS_TO]->(c)
"""

# 4. Merge SideEffect + edge CAUSES (1 row = 1 side effect)
Q_SIDE_EFFECT = """
UNWIND $rows AS row
MATCH (drug:Drug {generic_name: row.generic_name})
MERGE (se:SideEffect {name: row.se_name})
MERGE (drug)-[:CAUSES]->(se)
"""


# ── Preprocessing ─────────────────────────────────────────────────────────────

def load_and_preprocess(csv_path: Path) -> pd.DataFrame:
    logger.info("Đọc Bộ 1: %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info("  → %d dòng, %d cột", len(df), len(df.columns))

    # Normalize các cột text chính
    # Dùng generic_name; nếu rỗng thì fallback sang drug_name
    df["_gname"] = df.apply(
        lambda r: normalize_text(safe_str(r["generic_name"])) or normalize_text(safe_str(r["drug_name"])),
        axis=1
    )
    df["_dname"] = df["drug_name"].apply(lambda x: normalize_text(safe_str(x)))
    df["_disease"] = df["medical_condition"].apply(lambda x: normalize_text(safe_str(x)))
    df["_disease_desc"] = df["medical_condition_description"].apply(lambda x: safe_str(x)[:500])
    df["_rating"] = df["rating"].apply(lambda x: safe_float(x))
    df["_rx_otc"] = df["rx_otc"].apply(lambda x: normalize_text(safe_str(x)))

    # Bỏ dòng thiếu tên thuốc hoặc tên bệnh
    df = df[df["_gname"].str.len() > 0]
    df = df[df["_disease"].str.len() > 0]
    logger.info("  → %d dòng sau khi lọc dòng thiếu tên.", len(df))

    return df


# ── Build batch lists ─────────────────────────────────────────────────────────

def build_drug_disease_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "generic_name":       r["_gname"],
            "drug_name":          r["_dname"],
            "rating":             r["_rating"],
            "rx_otc":             r["_rx_otc"],
            "disease_name":       r["_disease"],
            "disease_description": r["_disease_desc"],
        })
    return rows


def build_brand_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in df.iterrows():
        brands = parse_comma_list(safe_str(r["brand_names"]))
        for brand in brands:
            rows.append({"generic_name": r["_gname"], "brand_name": brand})
    logger.info("  → %d cặp (Drug, BrandName)", len(rows))
    return rows


def build_class_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in df.iterrows():
        classes = parse_comma_list(safe_str(r["drug_classes"]))
        for cls in classes:
            rows.append({"generic_name": r["_gname"], "class_name": cls})
    logger.info("  → %d cặp (Drug, DrugClass)", len(rows))
    return rows


def build_side_effect_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in df.iterrows():
        effects = parse_side_effects(safe_str(r["side_effects"]))
        for se in effects:
            rows.append({"generic_name": r["_gname"], "se_name": se})
    logger.info("  → %d cặp (Drug, SideEffect)", len(rows))
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== INGEST BỘ 1: Drugs, Side Effects & Medical Conditions ===")

    # Kiểm tra file tồn tại
    if not BO1_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {BO1_CSV}")

    df = load_and_preprocess(BO1_CSV)

    # Kết nối Neo4j
    driver = get_neo4j_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    with neo4j_session(driver) as session:
        # Tạo constraints trước
        logger.info("Tạo constraints và index...")
        create_constraints(session)

        # 1. Drug + Disease + TREATS
        logger.info("Ingest Drug → Disease (TREATS)...")
        rows = build_drug_disease_rows(df)
        count = run_batch(session, Q_DRUG_DISEASE, rows, BATCH_SIZE)
        logger.info("  ✓ %d cặp Drug-Disease đã ingest.", count)

        # 2. BrandName + HAS_BRAND
        logger.info("Ingest BrandName (HAS_BRAND)...")
        rows = build_brand_rows(df)
        if rows:
            run_batch(session, Q_BRAND, rows, BATCH_SIZE)
        logger.info("  ✓ %d cặp Drug-BrandName đã ingest.", len(rows))

        # 3. DrugClass + BELONGS_TO
        logger.info("Ingest DrugClass (BELONGS_TO)...")
        rows = build_class_rows(df)
        if rows:
            run_batch(session, Q_CLASS, rows, BATCH_SIZE)
        logger.info("  ✓ %d cặp Drug-DrugClass đã ingest.", len(rows))

        # 4. SideEffect + CAUSES
        logger.info("Ingest SideEffect (CAUSES)...")
        rows = build_side_effect_rows(df)
        if rows:
            run_batch(session, Q_SIDE_EFFECT, rows, BATCH_SIZE)
        logger.info("  ✓ %d cặp Drug-SideEffect đã ingest.", len(rows))

        # Sanity check
        logger.info("--- Sanity check ---")
        result = session.run("""
            MATCH (n) RETURN labels(n)[0] AS type, count(*) AS cnt
            ORDER BY cnt DESC
        """)
        for rec in result:
            logger.info("  %s: %d nodes", rec["type"], rec["cnt"])

        result = session.run("""
            MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS cnt
            ORDER BY cnt DESC
        """)
        for rec in result:
            logger.info("  [%s]: %d edges", rec["rel"], rec["cnt"])

    driver.close()
    logger.info("=== Bộ 1 ingest hoàn tất ===")


if __name__ == "__main__":
    main()
