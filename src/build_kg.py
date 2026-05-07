"""
build_kg.py — TUẦN 6: Merge Bộ 3 (NER + ER) vào KG, sinh CO_MENTIONED + TREATED_BY_SPECIALTY.

Pipeline:
  1. Load: mentions.parquet (Tuần 3), er_mapping.csv (Tuần 5)
  2. Build "real canonical map" — filter cosmetic-only mappings của ER (chỉ khác punct → revert)
  3. MERGE Drug/Disease nodes từ Bộ 3, update aliases[] + source[] cho node có sẵn
  4. Re-aggregate co-occurrence ở doc level theo canonical thật, sinh CO_MENTIONED
  5. Sinh TREATED_BY_SPECIALTY từ metadata Bộ 3

Yêu cầu: Neo4j chạy + APOC enabled (đã có trong devcontainer).
Chạy: python src/build_kg.py
"""
import sys
import string
import logging
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, BATCH_SIZE,
    COOCCURRENCE_MIN_WEIGHT, OUTPUT_DIR
)
from src.utils import (
    normalize_text, normalize_entity_name,
    get_neo4j_driver, neo4j_session, run_batch, logger,
)
# Import ABBREVIATION_MAP để guarantee mọi mention "htn", "dm"... đều map về
# canonical đầy đủ — không phụ thuộc cột canonical_hint (có thể rỗng nếu
# scispaCy nhận diện trước abbrev_dict).
from src.extract_entities import ABBREVIATION_MAP

# ── Đường dẫn input (Tuần 5 đã ghi vào output_week_5/) ──────────────────────
ROOT = Path(__file__).parent.parent
WEEK5_DIR        = ROOT / "output_week_5"
MENTIONS_PARQUET = WEEK5_DIR / "mentions.parquet"
ER_MAPPING_CSV   = WEEK5_DIR / "er_mapping.csv"

# Hằng số riêng cho bước này
MIN_MENTION_LEN          = 3       # Bỏ canonical quá ngắn
MAX_ALIASES_PER_NODE     = 25      # Cap aliases để tránh phình node
MIN_SPECIALTY_COUNT      = 2       # Tối thiểu n doc để tạo edge TREATED_BY_SPECIALTY

# Generic terms scispaCy nhận nhầm là Drug/Disease nhưng không phải entity y khoa
# có ý nghĩa cho KG (lifestyle factors, lab values, IV fluids, sinh hoá phẩm).
# Chúng làm nhiễu top CO_MENTIONED nếu giữ lại.
GENERIC_BLACKLIST = {
    # Lifestyle / substance
    "alcohol", "tobacco", "smoke", "smoking", "caffeine", "nicotine",
    "marijuana", "cannabis", "cocaine", "heroin",
    # Lab values & sinh hoá phẩm (không phải drug điều trị)
    "creatinine", "potassium", "sodium", "calcium", "magnesium", "phosphate",
    "chloride", "glucose", "hemoglobin", "albumin", "bilirubin", "urea",
    "cholesterol", "triglyceride", "lactate", "ammonia",
    # IV fluids / inert solvents
    "saline", "water", "dextrose", "lactate ringer",
    # Generic (quá rộng)
    "medication", "medications", "drug", "drugs", "fluid", "fluids",
    "oxygen", "air", "gas", "blood", "plasma", "serum", "urine",
    # Đo lường / đơn vị bị NER nhận nhầm
    "po", "pos", "qd", "qid", "tid", "bid", "iv",
}


# ── Cypher queries ────────────────────────────────────────────────────────────

# MERGE Drug — match trên generic_name; nếu trùng với Bo1 → update aliases + source
Q_DRUG_MERGE = """
UNWIND $rows AS row
MERGE (d:Drug {generic_name: row.name})
  ON CREATE SET
    d.aliases       = row.aliases,
    d.source        = ['bo3'],
    d.mention_count = row.mention_count
  ON MATCH SET
    d.aliases       = apoc.coll.toSet(coalesce(d.aliases, []) + row.aliases),
    d.source        = apoc.coll.toSet(coalesce(d.source, []) + ['bo3']),
    d.mention_count = row.mention_count
"""

Q_DISEASE_MERGE = """
UNWIND $rows AS row
MERGE (d:Disease {name: row.name})
  ON CREATE SET
    d.aliases       = row.aliases,
    d.source        = ['bo3'],
    d.mention_count = row.mention_count
  ON MATCH SET
    d.aliases       = apoc.coll.toSet(coalesce(d.aliases, []) + row.aliases),
    d.source        = apoc.coll.toSet(coalesce(d.source, []) + ['bo3']),
    d.mention_count = row.mention_count
"""

# CO_MENTIONED — dùng APOC để MATCH theo dynamic label
# Lưu ý: edge CO_MENTIONED là directed (do Cypher), nhưng semantic là undirected.
# Thứ tự (a, b) đã được normalize a < b trong Python để tránh trùng.
Q_COMENTIONED = """
UNWIND $rows AS row
CALL apoc.merge.node(
    [row.la],
    CASE row.la WHEN 'Drug'
         THEN {generic_name: row.a}
         ELSE {name: row.a} END,
    {}, {}
) YIELD node AS a
CALL apoc.merge.node(
    [row.lb],
    CASE row.lb WHEN 'Drug'
         THEN {generic_name: row.b}
         ELSE {name: row.b} END,
    {}, {}
) YIELD node AS b
MERGE (a)-[r:CO_MENTIONED]-(b)
  ON CREATE SET r.weight = row.weight, r.source = 'bo3'
  ON MATCH  SET r.weight = row.weight
"""

# Specialty + edge TREATED_BY_SPECIALTY
Q_SPECIALTY = """
UNWIND $rows AS row
MERGE (sp:Specialty {name: row.specialty})
WITH sp, row
MATCH (d:Disease {name: row.disease})
MERGE (d)-[r:TREATED_BY_SPECIALTY]->(sp)
  ON CREATE SET r.count = row.count, r.source = 'bo3'
  ON MATCH  SET r.count = row.count
"""


# ── Logic xử lý ER mapping ────────────────────────────────────────────────────

def _word_set(text: str) -> set[str]:
    """Tập từ sau khi bỏ punctuation (để so sánh cosmetic vs real merge)."""
    if not isinstance(text, str):
        return set()
    cleaned = text.lower().translate(str.maketrans("", "", string.punctuation))
    return set(cleaned.split())


def build_real_canonical_map(mapping_df: pd.DataFrame) -> dict[str, str]:
    """
    Filter cosmetic-only mappings.

    Quy tắc:
      - Nếu canonical_entity == 'SKIP' hoặc rỗng    → drop (không merge)
      - Nếu word_set(raw) == word_set(canonical)    → cosmetic only → dùng normalize_entity_name(raw)
      - Else (genuine semantic merge)               → dùng normalize_entity_name(canonical)
    """
    real = {}
    n_skip = n_cosmetic = n_real = 0

    for _, row in mapping_df.iterrows():
        raw = row["entity_raw"]
        canon = row.get("canonical_entity", "")
        if not isinstance(raw, str) or not raw.strip():
            continue
        if not isinstance(canon, str) or not canon.strip() or canon == "SKIP":
            n_skip += 1
            continue
        if _word_set(raw) == _word_set(canon):
            real[raw] = normalize_entity_name(raw)
            n_cosmetic += 1
        else:
            real[raw] = normalize_entity_name(canon)
            n_real += 1

    logger.info("ER mapping classified: %d skip, %d cosmetic, %d real merge",
                n_skip, n_cosmetic, n_real)
    return real


def to_canonical(
    text: str,
    canonical_hint: str,
    real_map: dict[str, str],
) -> str:
    """
    Lookup canonical cho 1 mention. Ưu tiên (cao → thấp):
      1. ABBREVIATION_MAP — bắt buộc với mọi mention key trong dict, kể cả
         khi mention đến từ scispaCy (không có canonical_hint). Đảm bảo
         "htn" luôn → "hypertension", không tạo node duplicate.
      2. canonical_hint từ mentions.parquet — guard cho trường hợp khác.
      3. real_map (er_mapping sau khi filter cosmetic) — kết quả Tuần 5.
      4. normalize_entity_name(text) — fallback.
    """
    text_lower = text.lower().strip()
    if text_lower in ABBREVIATION_MAP:
        return normalize_entity_name(ABBREVIATION_MAP[text_lower])
    if isinstance(canonical_hint, str) and canonical_hint.strip():
        return normalize_entity_name(canonical_hint)
    if text in real_map:
        return real_map[text]
    return normalize_entity_name(text)


# ── Group mentions theo canonical ─────────────────────────────────────────────

def group_mentions(mentions_df: pd.DataFrame, real_map: dict[str, str]):
    """
    Trả về:
      - node_data: dict[canonical] → {aliases:set, types:Counter, mentions:int}
      - doc_canon: dict[doc_id] → set[canonical]   (để build co-occurrence)
      - mention_specialty: dict[(canon, specialty)] → count   (để build TREATED_BY_SPECIALTY)
    """
    node_data = defaultdict(lambda: {"aliases": set(), "types": Counter(), "mentions": 0})
    doc_canon = defaultdict(set)
    mention_specialty = defaultdict(int)

    skipped_short = skipped_blacklist = 0
    for row in mentions_df.itertuples(index=False):
        canonical_hint = getattr(row, "canonical_hint", "")
        canon = to_canonical(row.entity_text, canonical_hint, real_map)

        if not canon or len(canon) < MIN_MENTION_LEN:
            skipped_short += 1
            continue
        # Filter cả raw text lẫn canonical against blacklist
        if (row.entity_text.lower() in GENERIC_BLACKLIST
                or canon.lower() in GENERIC_BLACKLIST):
            skipped_blacklist += 1
            continue

        node_data[canon]["aliases"].add(row.entity_text)
        node_data[canon]["types"][row.entity_type] += 1
        node_data[canon]["mentions"] += 1
        doc_canon[row.doc_id].add(canon)

        # Specialty (chỉ tính cho Disease, theo plan)
        if row.entity_type == "Disease" and isinstance(row.specialty, str) and row.specialty.strip():
            mention_specialty[(canon, row.specialty.strip())] += 1

    logger.info("Grouped: %d unique canonical (skipped %d short, %d blacklist)",
                len(node_data), skipped_short, skipped_blacklist)
    return node_data, doc_canon, mention_specialty


# ── Build node insert rows ────────────────────────────────────────────────────

def build_node_rows(node_data: dict) -> tuple[list[dict], list[dict], dict[str, str]]:
    """Trả về (drug_rows, disease_rows, canon_to_label)."""
    drug_rows, disease_rows = [], []
    canon_label = {}

    for canon, data in node_data.items():
        # Mode type: nhãn xuất hiện nhiều nhất trong cluster
        mode_type = data["types"].most_common(1)[0][0]
        canon_label[canon] = mode_type

        aliases = sorted(data["aliases"] - {canon})[:MAX_ALIASES_PER_NODE]
        record = {
            "name":          canon,
            "aliases":       aliases,
            "mention_count": data["mentions"],
        }
        if mode_type == "Drug":
            drug_rows.append(record)
        else:
            disease_rows.append(record)

    logger.info("Node rows: %d Drug, %d Disease", len(drug_rows), len(disease_rows))
    return drug_rows, disease_rows, canon_label


# ── Build CO_MENTIONED ────────────────────────────────────────────────────────

def build_cooccurrence_rows(
    doc_canon: dict[int, set[str]],
    canon_label: dict[str, str],
    min_weight: int = COOCCURRENCE_MIN_WEIGHT,
) -> list[dict]:
    """Re-aggregate co-occurrence trên canonical đã merge."""
    cooc = defaultdict(int)
    for canons in doc_canon.values():
        if len(canons) < 2:
            continue
        for a, b in combinations(sorted(canons), 2):
            cooc[(a, b)] += 1

    rows = []
    for (a, b), w in cooc.items():
        if w < min_weight:
            continue
        rows.append({
            "a": a, "la": canon_label.get(a, "Disease"),
            "b": b, "lb": canon_label.get(b, "Disease"),
            "weight": w,
        })

    logger.info("Co-occurrence: %d total pairs, %d after filter (weight >= %d)",
                len(cooc), len(rows), min_weight)
    return rows


# ── Build TREATED_BY_SPECIALTY ────────────────────────────────────────────────

def build_specialty_rows(mention_specialty: dict, min_count: int) -> list[dict]:
    rows = [
        {"disease": canon, "specialty": spec, "count": cnt}
        for (canon, spec), cnt in mention_specialty.items()
        if cnt >= min_count
    ]
    logger.info("Specialty edges: %d (filter count >= %d)", len(rows), min_count)
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== TUẦN 6: Build KG v1.0 (merge Bộ 3) ===")

    # Verify input
    if not MENTIONS_PARQUET.exists():
        raise FileNotFoundError(f"Không tìm thấy: {MENTIONS_PARQUET}")
    if not ER_MAPPING_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy: {ER_MAPPING_CSV}")

    # 1. Load
    logger.info("Đọc mentions.parquet...")
    mentions_df = pd.read_parquet(MENTIONS_PARQUET)
    logger.info("  → %d mentions, %d unique entity_text",
                len(mentions_df), mentions_df["entity_text"].nunique())

    logger.info("Đọc er_mapping.csv...")
    mapping_df = pd.read_csv(ER_MAPPING_CSV)
    logger.info("  → %d mapping rows", len(mapping_df))

    # 2. Filter cosmetic mappings
    real_map = build_real_canonical_map(mapping_df)

    # 3. Group mentions theo canonical
    logger.info("Group mentions theo canonical...")
    node_data, doc_canon, mention_specialty = group_mentions(mentions_df, real_map)

    # 4. Build node rows
    drug_rows, disease_rows, canon_label = build_node_rows(node_data)

    # 5. Build co-occurrence rows (re-aggregate trên canonical thật)
    cooc_rows = build_cooccurrence_rows(doc_canon, canon_label)

    # 6. Build specialty rows
    spec_rows = build_specialty_rows(mention_specialty, MIN_SPECIALTY_COUNT)

    # ── Write to Neo4j ────────────────────────────────────────────────────────
    driver = get_neo4j_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    with neo4j_session(driver) as session:
        logger.info("MERGE Drug nodes (Bộ 3)...")
        run_batch(session, Q_DRUG_MERGE, drug_rows, BATCH_SIZE)
        logger.info("  ✓ %d Drug rows merged.", len(drug_rows))

        logger.info("MERGE Disease nodes (Bộ 3)...")
        run_batch(session, Q_DISEASE_MERGE, disease_rows, BATCH_SIZE)
        logger.info("  ✓ %d Disease rows merged.", len(disease_rows))

        logger.info("MERGE CO_MENTIONED edges...")
        run_batch(session, Q_COMENTIONED, cooc_rows, BATCH_SIZE)
        logger.info("  ✓ %d CO_MENTIONED edges merged.", len(cooc_rows))

        logger.info("MERGE TREATED_BY_SPECIALTY edges...")
        run_batch(session, Q_SPECIALTY, spec_rows, BATCH_SIZE)
        logger.info("  ✓ %d TREATED_BY_SPECIALTY edges merged.", len(spec_rows))

        # ── Sanity check ──────────────────────────────────────────────────────
        logger.info("--- Sanity check KG v1.0 ---")
        for label in ["Disease", "Drug", "Symptom", "SideEffect",
                      "DrugClass", "BrandName", "Specialty"]:
            res = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt").single()
            logger.info("  %-12s: %6d", label, res["cnt"])

        logger.info("---")
        for rel in ["TREATS", "CAUSES", "BELONGS_TO", "HAS_BRAND",
                    "HAS_SYMPTOM", "CO_MENTIONED", "TREATED_BY_SPECIALTY"]:
            res = session.run(
                f"MATCH ()-[r:{rel}]->() RETURN count(r) AS cnt"
            ).single()
            logger.info("  [%-22s]: %6d", rel, res["cnt"])

        # Source distribution
        logger.info("--- Disease source distribution ---")
        res = session.run("""
            MATCH (d:Disease)
            RETURN
              CASE
                WHEN 'bo1' IN d.source AND 'bo2' IN d.source AND 'bo3' IN d.source THEN 'bo1+bo2+bo3'
                WHEN 'bo1' IN d.source AND 'bo2' IN d.source THEN 'bo1+bo2'
                WHEN 'bo1' IN d.source AND 'bo3' IN d.source THEN 'bo1+bo3'
                WHEN 'bo2' IN d.source AND 'bo3' IN d.source THEN 'bo2+bo3'
                WHEN 'bo1' IN d.source THEN 'bo1 only'
                WHEN 'bo2' IN d.source THEN 'bo2 only'
                WHEN 'bo3' IN d.source THEN 'bo3 only'
                ELSE 'unknown'
              END AS bucket,
              count(*) AS cnt
            ORDER BY cnt DESC
        """)
        for r in res:
            logger.info("  %-15s: %5d", r["bucket"], r["cnt"])

    driver.close()
    logger.info("=== KG v1.0 hoàn tất ===")
    logger.info("Backup recommend: docker exec ... tar volume hoặc neo4j-admin dump")
    logger.info("Bước tiếp: Tuần 7 — Graph Analytics (Louvain, PageRank, Dijkstra)")


if __name__ == "__main__":
    main()
