"""
utils.py — Các hàm tiện ích dùng chung cho toàn pipeline.

Quy tắc normalize TEXT (nhất quán cho cả 3 bộ dữ liệu):
  - lowercase
  - strip whitespace đầu/cuối
  - thay nhiều dấu cách liên tiếp bằng 1 dấu cách
  - bỏ các ký tự đặc biệt không cần thiết (giữ dấu gạch ngang)
"""
import re
import logging
from contextlib import contextmanager
from typing import Any

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Text normalization ────────────────────────────────────────────────────────

# Từ "noise" trong tên bệnh/triệu chứng (bỏ để giúp matching tốt hơn)
_MEDICAL_NOISE_WORDS = {
    "acute", "chronic", "severe", "mild", "moderate",
    "primary", "secondary", "unspecified", "nos",
}

def normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản cơ bản: lowercase, strip, collapse spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_entity_name(name: str) -> str:
    """
    Chuẩn hóa tên entity (disease/drug/symptom) mạnh hơn normalize_text:
    - Bỏ dấu câu (giữ dấu gạch ngang)
    - Bỏ noise words y khoa
    - Dùng cho blocking trong ER
    """
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)   # bỏ dấu câu, giữ gạch ngang
    name = re.sub(r"\s+", " ", name).strip()
    tokens = [t for t in name.split() if t not in _MEDICAL_NOISE_WORDS]
    return " ".join(tokens)


def normalize_symptom_name(name: str) -> str:
    """
    Chuyển snake_case sang dạng tự nhiên.
    Ví dụ: 'skin_rash' → 'skin rash', 'continuous_sneezing' → 'continuous sneezing'
    """
    if not isinstance(name, str):
        return ""
    return normalize_text(name.replace("_", " "))


def parse_comma_list(text: str, max_item_len: int = 80) -> list[str]:
    """
    Tách chuỗi ngăn cách bởi dấu phẩy thành list.
    Lọc bỏ item rỗng hoặc quá dài (thường là câu văn, không phải tên entity).
    """
    if not isinstance(text, str) or not text.strip():
        return []
    items = [normalize_text(x) for x in text.split(",")]
    return [x for x in items if x and len(x) <= max_item_len]


def parse_side_effects(text: str) -> list[str]:
    """
    Trích xuất tên tác dụng phụ từ chuỗi free-text trong Bộ 1.
    Chiến lược: split theo dấu phẩy, giữ item ngắn (< 60 ký tự).
    """
    return parse_comma_list(text, max_item_len=60)


# ── Neo4j helpers ─────────────────────────────────────────────────────────────

def get_neo4j_driver(uri: str, user: str, password: str):
    """Tạo và trả về Neo4j driver. Raise lỗi rõ ràng nếu kết nối thất bại."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        logger.info("Kết nối Neo4j thành công: %s", uri)
        return driver
    except ImportError:
        raise ImportError("Chưa cài thư viện neo4j. Chạy: pip install neo4j")
    except Exception as e:
        raise ConnectionError(
            f"Không thể kết nối Neo4j tại {uri}.\n"
            f"Kiểm tra: (1) Neo4j Desktop đang chạy, (2) mật khẩu đúng trong .env\n"
            f"Lỗi gốc: {e}"
        )


@contextmanager
def neo4j_session(driver):
    """Context manager để dùng session an toàn."""
    session = driver.session()
    try:
        yield session
    finally:
        session.close()


def run_batch(session, query: str, rows: list[dict[str, Any]], batch_size: int = 500) -> int:
    """
    Chạy Cypher query theo batch dùng UNWIND.
    Trả về tổng số record đã xử lý.
    """
    from tqdm import tqdm
    total = 0
    for i in tqdm(range(0, len(rows), batch_size), desc="  Inserting batches"):
        batch = rows[i: i + batch_size]
        session.run(query, rows=batch)
        total += len(batch)
    return total


def create_constraints(session) -> None:
    """Tạo toàn bộ constraint và index cho KG. Idempotent (IF NOT EXISTS)."""
    statements = [
        # Unique constraints
        "CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (n:Disease) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT drug_generic_name IF NOT EXISTS FOR (n:Drug) REQUIRE n.generic_name IS UNIQUE",
        "CREATE CONSTRAINT symptom_name IF NOT EXISTS FOR (n:Symptom) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT side_effect_name IF NOT EXISTS FOR (n:SideEffect) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT drug_class_name IF NOT EXISTS FOR (n:DrugClass) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT brand_name_name IF NOT EXISTS FOR (n:BrandName) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT specialty_name IF NOT EXISTS FOR (n:Specialty) REQUIRE n.name IS UNIQUE",
        # Index cho tìm kiếm nhanh
        "CREATE INDEX disease_aliases IF NOT EXISTS FOR (n:Disease) ON (n.aliases)",
        "CREATE INDEX drug_aliases IF NOT EXISTS FOR (n:Drug) ON (n.aliases)",
    ]
    for stmt in statements:
        session.run(stmt)
        logger.debug("Executed: %s", stmt[:60])
    logger.info("Tạo constraint và index hoàn tất.")


# ── DataFrame helpers ─────────────────────────────────────────────────────────

def safe_str(val) -> str:
    """Chuyển pandas value thành str, trả về '' nếu NaN/None."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def safe_float(val, default: float = 0.0) -> float:
    """Chuyển value thành float an toàn."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default
