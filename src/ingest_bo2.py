"""
ingest_bo2.py — TUẦN 2: Ingest Bộ 2 (Disease Symptom Prediction Dataset)

Nodes tạo ra:   Disease (update thêm), Symptom
Edges tạo ra:   HAS_SYMPTOM (với thuộc tính severity)
                PRECAUTION (precaution text làm thuộc tính node Disease)

Chạy: python src/ingest_bo2.py
Lưu ý: Chạy sau ingest_bo1.py để Disease node từ Bộ 1 được update description.
"""
import sys
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    BO2_DATASET_CSV, BO2_SEVERITY_CSV, BO2_DESC_CSV, BO2_PRECAUTION_CSV,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, BATCH_SIZE
)
from src.utils import (
    normalize_text, normalize_symptom_name,
    get_neo4j_driver, neo4j_session, run_batch, create_constraints,
    safe_str, safe_float, logger
)

# ── Cypher queries ────────────────────────────────────────────────────────────

Q_DISEASE = """
UNWIND $rows AS row
MERGE (d:Disease {name: row.disease_name})
  ON CREATE SET
    d.description = row.description,
    d.precautions = row.precautions,
    d.aliases     = [],
    d.source      = ['bo2']
  ON MATCH SET
    d.description = CASE WHEN d.description IS NULL OR d.description = ''
                         THEN row.description ELSE d.description END,
    d.precautions = CASE WHEN d.precautions IS NULL
                         THEN row.precautions ELSE d.precautions END,
    d.source      = CASE WHEN NOT 'bo2' IN d.source
                         THEN d.source + ['bo2'] ELSE d.source END
"""

Q_SYMPTOM = """
UNWIND $rows AS row
MERGE (s:Symptom {name: row.symptom_name})
  ON CREATE SET
    s.default_severity = row.severity,
    s.aliases          = [row.raw_name],
    s.source           = ['bo2']
  ON MATCH SET
    s.default_severity = CASE WHEN row.severity > 0
                              THEN row.severity ELSE s.default_severity END
"""

Q_HAS_SYMPTOM = """
UNWIND $rows AS row
MATCH (d:Disease {name: row.disease_name})
MATCH (s:Symptom  {name: row.symptom_name})
MERGE (d)-[hs:HAS_SYMPTOM]->(s)
  ON CREATE SET hs.severity = row.severity, hs.source = 'bo2'
  ON MATCH SET  hs.severity = CASE WHEN row.severity > 0
                                   THEN row.severity ELSE hs.severity END
"""


# ── Load & preprocess ─────────────────────────────────────────────────────────

def load_severity(path: Path) -> dict[str, int]:
    """Trả về dict: symptom_name_normalized → severity_weight."""
    df = pd.read_csv(path)
    severity = {}
    for _, r in df.iterrows():
        raw = safe_str(r["Symptom"])
        norm = normalize_symptom_name(raw)
        if norm:
            severity[norm] = int(safe_float(r["weight"], 0))
    logger.info("  Đọc severity: %d triệu chứng có weight.", len(severity))
    return severity


def load_descriptions(path: Path) -> dict[str, str]:
    """Trả về dict: disease_name_normalized → description."""
    df = pd.read_csv(path)
    descs = {}
    for _, r in df.iterrows():
        name = normalize_text(safe_str(r["Disease"]))
        desc = safe_str(r["Description"])[:500]
        if name:
            descs[name] = desc
    return descs


def load_precautions(path: Path) -> dict[str, list[str]]:
    """Trả về dict: disease_name_normalized → list of precaution strings."""
    df = pd.read_csv(path)
    prec_map = {}
    for _, r in df.iterrows():
        name = normalize_text(safe_str(r["Disease"]))
        precs = [safe_str(r.get(f"Precaution_{i}", "")) for i in range(1, 5)]
        precs = [p for p in precs if p]
        if name:
            prec_map[name] = precs
    return prec_map


def load_disease_symptom_pairs(
    dataset_path: Path,
    severity_map: dict[str, int],
) -> tuple[list[dict], list[dict], set[str]]:
    """
    Đọc dataset.csv, melt symptom columns thành cặp (disease, symptom).
    Trả về: (disease_rows, pair_rows, all_symptom_names)
    """
    df = pd.read_csv(dataset_path)
    logger.info("  dataset.csv: %d dòng", len(df))

    symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]

    # Melt: từ wide → long
    melted = df.melt(
        id_vars=["Disease"],
        value_vars=symptom_cols,
        var_name="col",
        value_name="symptom_raw"
    ).dropna(subset=["symptom_raw"])

    melted["disease_norm"] = melted["Disease"].apply(lambda x: normalize_text(safe_str(x)))
    melted["symptom_norm"] = melted["symptom_raw"].apply(normalize_symptom_name)
    melted = melted[melted["symptom_norm"].str.len() > 0]

    all_symptoms = set(melted["symptom_norm"].unique())
    logger.info("  → %d cặp (disease, symptom), %d triệu chứng unique",
                len(melted), len(all_symptoms))

    # Disease rows (distinct)
    disease_names = melted["disease_norm"].unique()

    return melted, disease_names, all_symptoms


# ── Build batch lists ─────────────────────────────────────────────────────────

def build_disease_rows(
    disease_names,
    desc_map: dict[str, str],
    prec_map: dict[str, list[str]],
) -> list[dict]:
    rows = []
    for name in disease_names:
        if not name:
            continue
        rows.append({
            "disease_name": name,
            "description":  desc_map.get(name, ""),
            "precautions":  prec_map.get(name, []),
        })
    return rows


def build_symptom_rows(
    all_symptoms: set[str],
    severity_map: dict[str, int],
) -> list[dict]:
    rows = []
    for sym in all_symptoms:
        rows.append({
            "symptom_name": sym,
            "raw_name":     sym,   # alias ban đầu = chính nó
            "severity":     severity_map.get(sym, 0),
        })
    return rows


def build_pair_rows(melted: pd.DataFrame, severity_map: dict[str, int]) -> list[dict]:
    rows = []
    for _, r in melted.iterrows():
        rows.append({
            "disease_name":  r["disease_norm"],
            "symptom_name":  r["symptom_norm"],
            "severity":      severity_map.get(r["symptom_norm"], 0),
        })
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== INGEST BỘ 2: Disease Symptom Dataset ===")

    for path in [BO2_DATASET_CSV, BO2_SEVERITY_CSV, BO2_DESC_CSV, BO2_PRECAUTION_CSV]:
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

    # Load auxiliary data
    severity_map = load_severity(BO2_SEVERITY_CSV)
    desc_map     = load_descriptions(BO2_DESC_CSV)
    prec_map     = load_precautions(BO2_PRECAUTION_CSV)

    # Load main dataset
    melted, disease_names, all_symptoms = load_disease_symptom_pairs(
        BO2_DATASET_CSV, severity_map
    )

    # Kết nối Neo4j
    driver = get_neo4j_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    with neo4j_session(driver) as session:
        # Đảm bảo constraint đã có (idempotent)
        create_constraints(session)

        # 1. Merge Disease nodes (update nếu đã có từ Bộ 1)
        logger.info("Ingest Disease nodes...")
        disease_rows = build_disease_rows(disease_names, desc_map, prec_map)
        run_batch(session, Q_DISEASE, disease_rows, BATCH_SIZE)
        logger.info("  ✓ %d Disease nodes.", len(disease_rows))

        # 2. Merge Symptom nodes
        logger.info("Ingest Symptom nodes...")
        symptom_rows = build_symptom_rows(all_symptoms, severity_map)
        run_batch(session, Q_SYMPTOM, symptom_rows, BATCH_SIZE)
        logger.info("  ✓ %d Symptom nodes.", len(symptom_rows))

        # 3. HAS_SYMPTOM edges
        logger.info("Ingest HAS_SYMPTOM edges...")
        pair_rows = build_pair_rows(melted, severity_map)
        run_batch(session, Q_HAS_SYMPTOM, pair_rows, BATCH_SIZE)
        logger.info("  ✓ %d HAS_SYMPTOM edges.", len(pair_rows))

        # Sanity check
        logger.info("--- Sanity check ---")
        for label in ["Disease", "Symptom", "Drug", "SideEffect"]:
            result = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
            cnt = result.single()["cnt"]
            logger.info("  %s: %d nodes", label, cnt)

        result = session.run("MATCH ()-[r:HAS_SYMPTOM]->() RETURN count(r) AS cnt")
        logger.info("  HAS_SYMPTOM: %d edges", result.single()["cnt"])

    driver.close()
    logger.info("=== Bộ 2 ingest hoàn tất ===")


if __name__ == "__main__":
    main()
