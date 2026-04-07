# Knowledge Graph Integration for Heterogeneous Medical Data

Đồ án cuối kỳ — tích hợp dữ liệu y khoa có cấu trúc và phi cấu trúc thành Knowledge Graph trong Neo4j, hỗ trợ chẩn đoán y khoa.

---

## Cấu trúc thư mục

```
capstoneproject/
├── data/
│   ├── drugs-side-effects-and-medical-condition/
│   │   └── drugs_side_effects_drugs_com.csv        # Bộ 1
│   ├── disease-symptom-description-dataset/
│   │   ├── dataset.csv                             # Bộ 2 (chính)
│   │   ├── Symptom-severity.csv
│   │   ├── symptom_Description.csv
│   │   └── symptom_precaution.csv
│   └── medical_transcriptions/
│       └── mtsamples.csv                           # Bộ 3
├── src/
│   ├── config.py              # Cấu hình đường dẫn, Neo4j, hằng số
│   ├── utils.py               # Hàm tiện ích (normalize, Neo4j helpers)
│   ├── ingest_bo1.py          # [Tuần 2] Ingest Drugs & Side Effects
│   ├── ingest_bo2.py          # [Tuần 2] Ingest Disease Symptom Dataset
│   ├── extract_entities.py    # [Tuần 3] NER trên Medical Transcriptions
│   └── prepare_er.py          # [Tuần 4] Co-occurrence + Gold Set
├── notebooks/
│   ├── eda_bo1.py             # [Tuần 1] EDA Bộ 1
│   ├── eda_bo2.py             # [Tuần 1] EDA Bộ 2
│   └── eda_bo3.py             # [Tuần 1] EDA Bộ 3
├── output/                    # File trung gian (parquet, CSV)
├── docs/                      # Tài liệu ontology, báo cáo
├── .env.example               # Template cấu hình Neo4j
├── requirements.txt
└── README.md
```

---

## Cài đặt môi trường

### 1. Python dependencies

```bash
pip install -r requirements.txt
```

### 2. scispaCy model (bắt buộc cho Tuần 3)

```bash
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

### 3. Cấu hình Neo4j

Sau khi cài Neo4j Desktop và tạo database:

```bash
cp .env.example .env
# Mở .env, điền mật khẩu Neo4j vào NEO4J_PASSWORD
```

**Bắt buộc bật 2 plugin trong Neo4j Desktop:**
- APOC (Advanced Procedures)
- Graph Data Science (GDS)

---

## Chạy từng bước

### Tuần 1 — EDA (không cần Neo4j)

```bash
# Mở bằng VS Code Jupyter extension hoặc Jupyter Lab
jupyter lab notebooks/eda_bo1.py
jupyter lab notebooks/eda_bo2.py
jupyter lab notebooks/eda_bo3.py
```

### Tuần 2 — Ingest dữ liệu structured

```bash
# Bắt buộc chạy ingest_bo1 trước để tạo constraints
python src/ingest_bo1.py
python src/ingest_bo2.py
```

Sau khi chạy, KG v0.1 có:
- ~900 Drug nodes
- ~950 Disease nodes
- ~130 Symptom nodes
- SideEffect, DrugClass, BrandName nodes
- Các edge: TREATS, CAUSES, BELONGS_TO, HAS_BRAND, HAS_SYMPTOM

### Tuần 3 — NER (cần scispaCy model)

```bash
# Chạy thử với 500 doc đầu (nhanh, ~2-3 phút)
python src/extract_entities.py --limit 500

# Chạy toàn bộ (~5.000 doc, ~15-30 phút)
python src/extract_entities.py

# Bỏ qua ingest Neo4j nếu chưa setup
python src/extract_entities.py --skip-neo4j
```

Output: `output/mentions.parquet`

### Tuần 4 — Chuẩn bị Entity Resolution

```bash
python src/prepare_er.py
```

Output:
- `output/cooccurrence.csv` — cặp entity co-occur trong cùng document
- `output/gold_set_template.csv` — **cần annotate tay** (xem hướng dẫn bên dưới)

---

## Annotation Gold Set (Tuần 4 → Tuần 5)

Sau khi chạy `prepare_er.py`, mở file `output/gold_set_template.csv`:

1. Mở bằng Excel hoặc Google Sheets
2. Điền cột `manual_canonical` cho mỗi mention
3. Ví dụ:

| entity_text | entity_type | manual_canonical |
|-------------|-------------|-----------------|
| htn | Disease | hypertension |
| heart attack | Disease | myocardial infarction |
| aspirin | Drug | aspirin |
| tylenol | Drug | acetaminophen |

4. Ghi chú: nếu không chắc, để trống hoặc ghi `SKIP`
5. Mục tiêu: annotate ~150 cặp trước tuần 5

---

## Schema Knowledge Graph

### Node types

| Label | Properties | Nguồn |
|-------|-----------|-------|
| `Drug` | generic_name, drug_name, rating, rx_otc, aliases[], source[] | Bộ 1, Bộ 3 (NER) |
| `Disease` | name, description, precautions[], aliases[], source[] | Bộ 1, Bộ 2, Bộ 3 (NER) |
| `Symptom` | name, default_severity, aliases[], source[] | Bộ 2, Bộ 3 (NER) |
| `SideEffect` | name | Bộ 1 |
| `DrugClass` | name | Bộ 1 |
| `BrandName` | name | Bộ 1 |
| `Specialty` | name | Bộ 3 |

### Relationship types

| Relationship | Properties | Nguồn |
|-------------|-----------|-------|
| `(Drug)-[:TREATS]->(Disease)` | rating, source | Bộ 1 |
| `(Drug)-[:CAUSES]->(SideEffect)` | — | Bộ 1 |
| `(Drug)-[:BELONGS_TO]->(DrugClass)` | — | Bộ 1 |
| `(Drug)-[:HAS_BRAND]->(BrandName)` | — | Bộ 1 |
| `(Disease)-[:HAS_SYMPTOM]->(Symptom)` | severity | Bộ 2 |
| `(Disease)-[:TREATED_BY_SPECIALTY]->(Specialty)` | count | Bộ 3 |
| `(*)-[:CO_MENTIONED {weight}]-(*)` | weight, source | Bộ 3 |

---

## Tiến độ hiện tại

### ✅ Hoàn thành (Tuần 1–4)

- [x] Cấu trúc project, requirements.txt
- [x] `src/config.py` — cấu hình tập trung
- [x] `src/utils.py` — normalize text, Neo4j helpers
- [x] `notebooks/eda_bo1.py` — EDA Bộ 1
- [x] `notebooks/eda_bo2.py` — EDA Bộ 2
- [x] `notebooks/eda_bo3.py` — EDA Bộ 3
- [x] `src/ingest_bo1.py` — ingest Drugs & Side Effects
- [x] `src/ingest_bo2.py` — ingest Disease Symptom
- [x] `src/extract_entities.py` — NER scispaCy + dictionary
- [x] `src/prepare_er.py` — co-occurrence + gold set template

### 🔲 Cần làm (Tuần 5–8)

#### Tuần 5 — Entity Resolution
- [ ] `src/entity_resolution.py`
  - Blocking theo 3 ký tự đầu + Metaphone
  - Matching: Jaro-Winkler (jellyfish) + Sentence Embedding cosine similarity
  - Combined score: `0.4 × jaro + 0.6 × cosine`
  - Auto-merge nếu score >= 0.92, review tay nếu 0.80–0.92
  - Đánh giá trên gold set: Precision, Recall, F1
  - Output: `output/er_mapping.csv`

#### Tuần 6 — Merge vào Neo4j + CO_MENTIONED
- [ ] `src/build_kg.py`
  - Dùng er_mapping.csv để link mention về canonical entity
  - Tạo/update node từ Bộ 3 (entity mới chưa có trong KG)
  - Tạo edge `CO_MENTIONED` từ cooccurrence.csv
  - Tạo edge `TREATED_BY_SPECIALTY` từ metadata Bộ 3
  - KG v1.0 hoàn chỉnh → backup bằng neo4j-admin dump

#### Tuần 7 — Graph Analytics
- [ ] `notebooks/graph_analytics.ipynb`
  - Shortest path (Dijkstra): Symptom → Disease → Drug
  - Community detection (Louvain)
  - Centrality: PageRank, Betweenness
  - Link Prediction (bonus): Adamic-Adar hoặc Node2Vec

#### Tuần 8 — Visualization & Báo cáo
- [ ] Neo4j Bloom: tạo 3–5 perspective cho demo
- [ ] `notebooks/visualization.ipynb`: xuất subgraph bằng pyvis
- [ ] Báo cáo cuối kỳ (PDF, ~20-30 trang)
- [ ] Slide thuyết trình (~15-20 slide)
- [ ] Video demo 3 phút (backup cho trình bày live)

---

## Cypher Query mẫu

```cypher
-- Thuốc điều trị một bệnh cụ thể
MATCH (d:Drug)-[t:TREATS]->(dis:Disease {name: "hypertension"})
RETURN d.generic_name, t.rating ORDER BY t.rating DESC LIMIT 10;

-- Triệu chứng của một bệnh (kèm mức độ nghiêm trọng)
MATCH (dis:Disease {name: "diabetes mellitus"})-[hs:HAS_SYMPTOM]->(s:Symptom)
RETURN s.name, hs.severity ORDER BY hs.severity DESC;

-- Thuốc cùng nhóm với aspirin
MATCH (d:Drug {generic_name: "aspirin"})-[:BELONGS_TO]->(c:DrugClass)<-[:BELONGS_TO]-(other:Drug)
RETURN other.generic_name, c.name;

-- Entity co-mentioned mạnh nhất với "hypertension"
MATCH (dis:Disease {name: "hypertension"})-[co:CO_MENTIONED]-(other)
RETURN labels(other)[0] AS type, other.name, co.weight
ORDER BY co.weight DESC LIMIT 20;

-- Tổng quan KG
MATCH (n) RETURN labels(n)[0] AS type, count(*) AS count ORDER BY count DESC;
MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS count ORDER BY count DESC;
```

---

## Lưu ý kỹ thuật

1. **Luôn dùng `MERGE` không dùng `CREATE`** — script idempotent, chạy lại không sinh trùng.
2. **Backup Neo4j trước mỗi bước merge lớn:**
   ```bash
   neo4j-admin database dump neo4j --to-path=./backups
   ```
3. **Thứ tự chạy script bắt buộc:**
   `ingest_bo1.py` → `ingest_bo2.py` → `extract_entities.py` → `prepare_er.py` → `entity_resolution.py` → `build_kg.py`
4. **Normalize text phải nhất quán** — dùng hàm từ `src/utils.py`, không tự normalize riêng.
5. **scispaCy trên Windows** — dùng `n_process=1` (multiprocessing có thể lỗi trên Windows).

---

*Phiên bản README: 1.0 | Cập nhật: 07/04/2026*
