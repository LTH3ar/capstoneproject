"""
config.py — Cấu hình tập trung cho toàn bộ pipeline.
Copy .env.example thành .env và điền mật khẩu Neo4j vào đó.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Thư mục gốc của project ──────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent

# ── Đường dẫn dữ liệu ────────────────────────────────────────────────────────
BO1_CSV = ROOT_DIR / "drugs-side-effects-and-medical-condition" / "drugs_side_effects_drugs_com.csv"

BO2_DIR = ROOT_DIR / "disease-symptom-description-dataset"
BO2_DATASET_CSV    = BO2_DIR / "dataset.csv"
BO2_SEVERITY_CSV   = BO2_DIR / "Symptom-severity.csv"
BO2_DESC_CSV       = BO2_DIR / "symptom_Description.csv"
BO2_PRECAUTION_CSV = BO2_DIR / "symptom_precaution.csv"

BO3_CSV = ROOT_DIR / "medical_transcriptions" / "mtsamples.csv"

# ── Thư mục output ───────────────────────────────────────────────────────────
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MENTIONS_PARQUET   = OUTPUT_DIR / "mentions.parquet"
COOCCURRENCE_CSV   = OUTPUT_DIR / "cooccurrence.csv"
GOLD_SET_CSV       = OUTPUT_DIR / "gold_set_template.csv"
ER_MAPPING_CSV     = OUTPUT_DIR / "er_mapping.csv"

# ── Kết nối Neo4j ─────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# ── Hằng số xử lý ────────────────────────────────────────────────────────────
BATCH_SIZE = 500          # Số dòng mỗi batch khi ingest vào Neo4j
COOCCURRENCE_MIN_WEIGHT = 3   # Threshold co-occurrence để giữ edge
GOLD_SET_SIZE = 150           # Số mention lấy mẫu tay cho gold set
ER_AUTO_MERGE_THRESHOLD  = 0.92   # Score >= này → auto merge
ER_REVIEW_THRESHOLD      = 0.80   # Score trong [0.80, 0.92] → review tay
