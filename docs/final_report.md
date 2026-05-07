# BÁO CÁO CUỐI KỲ
# Knowledge Graph Integration for Heterogeneous Medical Data

**Đề tài:** Knowledge Graph Integration for Heterogeneous Data  
**Ứng dụng:** Medical Diagnosis Support  
**Công nghệ:** Neo4j Community Edition, Python, scispaCy  

---

## Mục lục

1. [Giới thiệu bài toán](#1-giới-thiệu-bài-toán)
2. [Khảo sát dataset](#2-khảo-sát-dataset)
3. [Thiết kế Ontology](#3-thiết-kế-ontology)
4. [Pipeline Ingest dữ liệu Structured](#4-pipeline-ingest-dữ-liệu-structured)
5. [NER trên dữ liệu Unstructured](#5-ner-trên-dữ-liệu-unstructured)
6. [Entity Resolution](#6-entity-resolution)
7. [Build Knowledge Graph & Graph Analytics](#7-build-knowledge-graph--graph-analytics)
8. [Demo & Case Study](#8-demo--case-study)
9. [Kết luận & Hướng phát triển](#9-kết-luận--hướng-phát-triển)

---

## 1. Giới thiệu bài toán

### 1.1. Bối cảnh

Dữ liệu y khoa tồn tại ở nhiều dạng khác nhau: bảng biểu về thuốc và bệnh (structured), ghi chú lâm sàng của bác sĩ (unstructured). Mỗi nguồn chứa thông tin riêng biệt — không nguồn nào đơn lẻ có đủ bức tranh toàn cảnh. Việc tích hợp dữ liệu heterogeneous thành một Knowledge Graph (KG) thống nhất cho phép:

- **Khám phá mối quan hệ ẩn** giữa Disease–Drug–Symptom–SideEffect mà từng nguồn riêng lẻ không thể hiện.
- **Hỗ trợ chẩn đoán** qua truy vấn đồ thị: shortest path từ triệu chứng → bệnh → thuốc.
- **Drug repurposing**: phát hiện thuốc có thể điều trị bệnh chưa được ghi nhận chính thức.

### 1.2. Mục tiêu dự án

1. Thiết kế ontology cho lĩnh vực y khoa với 7 loại node và 7 loại relationship.
2. Ingest 2 bộ dữ liệu structured vào Neo4j.
3. Trích xuất entity từ ~5,000 ghi chú lâm sàng bằng NER (scispaCy + dictionary).
4. Thực hiện Entity Resolution giữa 3 nguồn — đánh giá bằng gold set thủ công.
5. Build KG hoàn chỉnh và chạy Graph Analytics (Shortest Path, Louvain, PageRank, Link Prediction).
6. Visualization bằng pyvis.

### 1.3. Đóng góp chính

- **Pipeline NLP end-to-end**: Từ text thô → entity mention → canonical entity → KG node.
- **ER 2 tầng**: Blocking (Metaphone) + Matching (Jaro-Winkler × Cosine Embedding) với F1 ≥ 0.80.
- **Hidden relationship discovery**: Edge CO_MENTIONED phát hiện quan hệ Drug-Disease không tồn tại trong dữ liệu structured.

---

## 2. Khảo sát dataset

### 2.1. Bộ 1 — Drugs, Side Effects and Medical Conditions

- **Nguồn**: Kaggle (`jithinanievarghese/drugs-side-effects-and-medical-conditions`)
- **Quy mô**: 1 file CSV, ~2,900 dòng
- **Vai trò**: Xương sống KG — cung cấp Drug, Disease, SideEffect, DrugClass, BrandName
- **Cột chính**: `drug_name`, `generic_name`, `brand_names`, `drug_classes`, `medical_condition`, `side_effects`, `rating`

**Đặc điểm xử lý**:
- Cột `brand_names` và `drug_classes` chứa danh sách ngăn cách dấu phẩy → parse thành nhiều node.
- Cột `side_effects` là free text → parse bằng regex/split đơn giản.
- Tên bệnh viết theo chuẩn riêng, khác với Bộ 2 → cần Entity Resolution.

### 2.2. Bộ 2 — Disease Symptom Prediction Dataset

- **Nguồn**: Kaggle (`itachi9604/disease-symptom-description-dataset`)
- **Quy mô**: 4 file CSV, ~4,900 dòng tổng
- **Vai trò**: Bổ sung nhánh triệu chứng — phần Bộ 1 không có
- **File chính**: `dataset.csv` (Disease → 17 cột Symptom), `Symptom-severity.csv` (weight 1–7)

**Đặc điểm xử lý**:
- Tên triệu chứng ở dạng `snake_case` → normalize thành natural language.
- ~41 bệnh phổ biến, overlap 1 phần với Bộ 1 → tạo cơ hội cho ER.
- Cột `weight` → thuộc tính `severity` trên edge `HAS_SYMPTOM`.

### 2.3. Bộ 3 — Medical Transcriptions

- **Nguồn**: Kaggle (`tboyle10/medicaltranscriptions`)
- **Quy mô**: 1 file CSV, ~5,000 bản ghi chú lâm sàng
- **Vai trò**: Nguồn unstructured duy nhất — nơi diễn ra NLP pipeline
- **Cột chính**: `transcription` (500–2,000 từ), `medical_specialty` (40 chuyên khoa), `keywords`

**Đặc điểm**:
- Chứa nhiều viết tắt y khoa (HTN, DM, CAD, COPD).
- Tên thuốc viết nhiều cách (generic/brand).
- Cấu trúc loose section nhưng không nhất quán → chạy NER toàn văn.

### 2.4. Tổng kết dữ liệu

| Tiêu chí | Bộ 1 | Bộ 2 | Bộ 3 |
|----------|------|------|------|
| Structured | ✅ | ✅ | ❌ |
| Drug | ✅ | ❌ | ✅ (NER) |
| Disease | ✅ | ✅ | ✅ (NER) |
| Symptom | ❌ | ✅ | ✅ (NER) |
| SideEffect | ✅ | ❌ | ⚠️ |
| Specialty | ❌ | ❌ | ✅ |

Không bộ nào dư thừa — mỗi bộ mang giá trị duy nhất cho KG.

---

## 3. Thiết kế Ontology

### 3.1. Sơ đồ

```
                  ┌──────────────┐
                  │  BrandName   │
                  └──────┬───────┘
                         │ HAS_BRAND
                         ▼
┌──────────┐    ┌───────────────┐    ┌──────────────┐
│ DrugClass│◄───│     Drug      │───►│  SideEffect  │
└──────────┘    └───────┬───────┘    └──────────────┘
  BELONGS_TO            │                  CAUSES
                        │ TREATS
                        ▼
                ┌───────────────┐         ┌───────────┐
                │    Disease    │────────►│ Specialty  │
                └───────┬───────┘         └───────────┘
                        │           TREATED_BY_SPECIALTY
                        │ HAS_SYMPTOM
                        ▼
                ┌───────────────┐
                │   Symptom     │
                └───────────────┘

     Mọi cặp node ←── CO_MENTIONED ──→ Mọi cặp node
```

### 3.2. Node types (7 loại)

| Node | Số lượng | Property chính | Nguồn |
|------|----------|----------------|-------|
| Disease | 5,621 | name, description, aliases[], source[] | Bộ 1+2+3 |
| Drug | 3,032 | generic_name, rating, aliases[], source[] | Bộ 1+3 |
| SideEffect | 5,447 | name | Bộ 1 |
| BrandName | 3,873 | name | Bộ 1 |
| DrugClass | 243 | name | Bộ 1 |
| Symptom | 131 | name, default_severity, aliases[] | Bộ 2+3 |
| Specialty | 40 | name | Bộ 3 |

### 3.3. Relationship types (7 loại)

| Relationship | Số lượng | Thuộc tính | Nguồn |
|-------------|----------|------------|-------|
| CO_MENTIONED | 74,904 | weight, source | Bộ 3 (NER co-occurrence) |
| CAUSES | 27,792 | — | Bộ 1 |
| TREATED_BY_SPECIALTY | 7,642 | count, source | Bộ 3 |
| HAS_BRAND | 4,154 | — | Bộ 1 |
| BELONGS_TO | 1,551 | — | Bộ 1 |
| TREATS | 1,472 | rating, source | Bộ 1 |
| HAS_SYMPTOM | 321 | severity, source | Bộ 2 |

**Tổng: 18,387 nodes, 117,836 relationships.**

### 3.4. Design decisions

- **Property `source[]` trên mọi node**: Cho phép truy vết nguồn gốc entity — quan trọng cho audit và đánh giá chất lượng tích hợp.
- **Property `aliases[]`**: Lưu các biến thể tên entity — hữu ích cho search fuzzy.
- **Edge CO_MENTIONED undirected**: Chuẩn hóa thứ tự (a < b) trong Python để tránh trùng.
- **Constraint UNIQUE trên tất cả node types**: Đảm bảo không tạo duplicate khi ingest.

> *Chi tiết đầy đủ: xem file `docs/ontology.md`*

---

## 4. Pipeline Ingest dữ liệu Structured

### 4.1. Quy tắc normalize (áp dụng cho cả 3 bộ)

```python
def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text
```

Quyết định normalize sớm (Tuần 2) và áp dụng nhất quán — tránh phải rebuild khi đổi giữa chừng.

### 4.2. Ingest Bộ 1 (`ingest_bo1.py`)

**Nodes sinh ra**: Drug, Disease, SideEffect, DrugClass, BrandName  
**Edges sinh ra**: TREATS (rating), CAUSES, BELONGS_TO, HAS_BRAND

**Quy trình**:
1. Đọc CSV bằng pandas, normalize các cột text.
2. Parse `brand_names` và `drug_classes` (split dấu phẩy).
3. Parse `side_effects` free text (split + filter item > 60 ký tự).
4. Dùng `MERGE` (không `CREATE`) để idempotent — chạy lại không sinh trùng.
5. Ingest theo batch 500 rows (UNWIND).

### 4.3. Ingest Bộ 2 (`ingest_bo2.py`)

**Nodes sinh ra**: Symptom (mới), Disease (update thêm description, precautions)  
**Edges sinh ra**: HAS_SYMPTOM (severity)

**Quy trình**:
1. Load auxiliary data: severity weights, descriptions, precautions.
2. Melt `dataset.csv` từ wide format → long format (Disease, Symptom pairs).
3. Normalize symptom names: `skin_rash` → `skin rash`.
4. MERGE Disease nodes (ON MATCH: update description nếu thiếu, append source).
5. Ingest Symptom nodes + HAS_SYMPTOM edges.

### 4.4. Kết quả sau Ingest

| Node | Số lượng |
|------|----------|
| Drug | ~900 |
| Disease | ~950 (Bộ 1) + ~41 (Bộ 2 update + mới) |
| Symptom | ~131 |
| SideEffect | ~5,400 |
| DrugClass | ~243 |
| BrandName | ~3,800 |

Backup: `backups/kg_v0_pre_build_kg.tar.gz`

---

## 5. NER trên dữ liệu Unstructured

### 5.1. Công cụ NER

**Model chính**: scispaCy `en_ner_bc5cdr_md` — pretrained trên BC5CDR corpus, chuyên nhận diện:
- `CHEMICAL` → map thành `Drug`
- `DISEASE` → map thành `Disease`

**Bổ sung**: Dictionary-based NER cho 22 abbreviation y khoa phổ biến:

```python
ABBREVIATION_MAP = {
    "htn": "hypertension", "dm": "diabetes mellitus",
    "cad": "coronary artery disease", "copd": "chronic obstructive pulmonary disease",
    "chf": "congestive heart failure", "mi": "myocardial infarction",
    ...
}
```

### 5.2. Pipeline NER (`extract_entities.py`)

1. Đọc `mtsamples.csv`, lọc transcription hợp lệ (> 50 ký tự).
2. Chạy scispaCy batch: `nlp.pipe(batch_size=50, n_process=1)`.
3. Với mỗi entity: normalize text, lọc noise (< 2 hoặc > 80 ký tự).
4. Bổ sung abbreviation matching trên từng document.
5. Ingest Specialty nodes trực tiếp từ metadata `medical_specialty`.

**Output**: `mentions.parquet` — mỗi dòng gồm `doc_id, entity_text, entity_type, canonical_hint, source, specialty`.

### 5.3. Chuẩn bị ER (`prepare_er.py`)

1. Normalize mentions: `normalize_entity_name()` — bỏ punctuation, bỏ noise words y khoa.
2. Block key: 3 ký tự đầu + Metaphone code.
3. Co-occurrence matrix: đếm số doc mà 2 entity cùng xuất hiện, lọc weight ≥ 3.
4. Tạo gold set template (150 mention, stratified sampling) → annotate thủ công.

---

## 6. Entity Resolution

### 6.1. Pipeline 2 tầng

**Tầng 1 — Blocking**: Giảm O(n²) → O(n·k) bằng block key = `3-char prefix + Metaphone`.

**Tầng 2 — Matching**:
- Lexical: Jaro-Winkler (`jellyfish`)
- Semantic: Cosine Similarity (`sentence-transformers/all-MiniLM-L6-v2`, 80MB)
- Combined: `0.4 × jaro + 0.6 × cosine`

**Threshold**: Auto-merge ≥ 0.85 | Review 0.80–0.85 | Reject < 0.80

**Clustering**: NetworkX connected components → canonical = term ngắn nhất trong cluster.

### 6.2. Bổ sung thủ công

- **Abbreviation Map**: 22 entries (HTN → hypertension, v.v.)
- **Brand-to-Generic Map**: 15 entries (Tylenol → acetaminophen, v.v.)
- **Blacklist**: Loại bỏ junk terms (p.o, q.d, v.v.)

### 6.3. Gold Set & Đánh giá

- **Gold Set**: 150 mention annotated thủ công (`gold_set_annotated.csv`)
- **Phương pháp**: So sánh similarity score với ground truth labels

| Metric | Giá trị | Mục tiêu |
|--------|---------|-----------|
| Precision | 1.00 | ≥ 0.85 ✅ |
| Recall | ~0.85–0.90 | ≥ 0.75 ✅ |
| F1-Score | ≥ 0.80 | ≥ 0.80 ✅ |

### 6.4. Kết quả ER

| Chỉ số | Giá trị |
|--------|---------|
| Entity unique (raw) | ~3,600 |
| Mapping rows | ~3,000+ |
| Real semantic merges | ~300+ |
| Cosmetic-only | ~1,200+ |

> *Chi tiết: xem file `docs/entity_resolution_report.md`*

---

## 7. Build Knowledge Graph & Graph Analytics

### 7.1. Build KG v1.0 (`build_kg.py`)

**Tuần 6**: Merge Bộ 3 vào KG đã có từ Bộ 1+2:

1. Load mentions + ER mapping.
2. Filter cosmetic mappings (word_set giống nhau → chỉ khác punctuation).
3. MERGE Drug/Disease nodes: node có sẵn → update `aliases[]`, `source[]`; node mới → tạo mới.
4. Build CO_MENTIONED edges: re-aggregate co-occurrence trên canonical, lọc weight ≥ 3.
5. Build TREATED_BY_SPECIALTY edges từ metadata.

**Blacklist filter**: Loại bỏ generic terms (alcohol, lab values, IV fluids) khỏi KG để giảm nhiễu.

**Kết quả**: 18,387 nodes | 117,836 relationships — backup `kg_v1_post_fixes.tar.gz`.

### 7.2. Graph Analytics (`graph_analytics.py`)

Sử dụng **Neo4j GDS plugin 2.x**. Graph projection: Drug + Disease + Symptom + Specialty.

#### 7.2.1. Shortest Path — Chẩn đoán flow

**Use case**: Triệu chứng → Bệnh → Thuốc (2-hop path).

| Symptom | Disease | Drug (top) | Rating |
|---------|---------|------------|--------|
| fatigue | pneumonia | lefamulin | 10.0 |
| fatigue | hypothyroidism | levothyroxine | 10.0 |
| chest pain | hypertension | betaxolol | 10.0 |
| headache | migraine | ergotamine/caffeine | 10.0 |

#### 7.2.2. Community Detection — Louvain

- **3,662 communities** phát hiện
- **Modularity: 0.2588** (gần ngưỡng tốt 0.3)
- Top communities tương ứng nhóm chuyên khoa:
  - Cluster 5119 (1,277 members): Nhiễm khuẩn + giảm đau (doxycycline, tetracycline, codeine, acetaminophen)
  - Cluster 2120 (978 members): Tim mạch (aspirin, nitroglycerin, metoprolol, angina)
  - Cluster 264 (635 members): Ung thư + tiêu hóa (doxorubicin, vincristine, constipation)
  - Cluster 3647 (538 members): Tâm thần kinh (ADHD medications, amphetamine)

#### 7.2.3. Centrality

**PageRank top 5** (node quan trọng nhất):

| Rank | Node | Label | Score |
|------|------|-------|-------|
| 1 | pain | Disease | 107.69 |
| 2 | hypertension | Disease | 66.64 |
| 3 | bleeding | Disease | 50.21 |
| 4 | dyspnea | Disease | 49.02 |
| 5 | edema | Disease | 45.42 |

**Betweenness top 5** (node cầu nối):

| Rank | Node | Label | Score |
|------|------|-------|-------|
| 1 | pain | Disease | 517,620 |
| 2 | hypertension | Disease | 293,831 |
| 3 | acne | Disease | 159,635 |
| 4 | colds & flu | Disease | 157,099 |
| 5 | pneumonia | Disease | 133,342 |

#### 7.2.4. Link Prediction — Adamic-Adar (BONUS)

**Drug repurposing candidates** — cặp Drug-Disease có Adamic-Adar cao nhưng KHÔNG có TREATS:

| Drug | Disease | CO_MENTIONED | Adamic-Adar |
|------|---------|-------------|-------------|
| aspirin | pain | 95 | 121.63 |
| aspirin | hypertension | 121 | 116.31 |
| aspirin | chest pain | 83 | 97.86 |
| aspirin | edema | 73 | 95.01 |
| lasix | dyspnea | 58 | 73.12 |

→ Gợi ý: aspirin được nhắc rất nhiều cùng hypertension/chest pain trong ghi chú lâm sàng, nhưng dataset structured không ghi nhận aspirin TREATS các bệnh này — ứng viên off-label đáng nghiên cứu.

---

## 8. Demo & Case Study

### 8.1. Visualization (pyvis)

4 subgraph HTML interactive (output/viz/):

1. **Hypertension full context**: Drug TREATS + Symptom + DrugClass + Specialty — minh hoạ tích hợp cả 3 bộ.
2. **Cardiovascular Louvain cluster**: Cluster tim mạch với CO_MENTIONED edges — Louvain phát hiện nhóm bệnh-thuốc liên quan.
3. **Aspirin hidden relationship**: CO_MENTIONED mạnh nhưng không có TREATS — drug repurposing hint.
4. **Diagnosis flow**: Headache → migraine/hypertension → drugs — demo shortest path chẩn đoán.

### 8.2. Case Study: "Bệnh nhân có triệu chứng chest pain"

**Câu hỏi**: Bệnh nhân vào viện với triệu chứng chest pain. KG gợi ý gì?

**Truy vấn Cypher**:
```cypher
MATCH (s:Symptom {name: 'chest pain'})<-[:HAS_SYMPTOM]-(d:Disease)<-[:TREATS]-(drug:Drug)
WHERE drug.rating >= 7
RETURN d.name AS disease, drug.generic_name AS drug, drug.rating AS rating
ORDER BY rating DESC LIMIT 10
```

**Kết quả**: KG gợi ý 2 bệnh khả nghi (hypertension, pneumonia) với top thuốc rating >= 7. Thời gian truy vấn < 100ms.

### 8.3. Case Study: "Drug repurposing cho aspirin"

**Câu hỏi**: Aspirin có thể điều trị bệnh nào chưa được ghi nhận?

**Phát hiện**: Aspirin CO_MENTIONED với hypertension (weight=121), chest pain (83), diabetes (73) — nhưng không có edge TREATS nào trong dữ liệu structured. Adamic-Adar score cao (116–98) xác nhận.

→ Đây chính là "hidden relationship" — giá trị cốt lõi của việc tích hợp dữ liệu heterogeneous.

---

## 9. Kết luận & Hướng phát triển

### 9.1. Kết quả đạt được

| Tiêu chí | Mục tiêu | Kết quả | Đánh giá |
|----------|----------|---------|----------|
| Ontology design | 7 node + 7 rel | 7 node + 7 rel | ✅ Đạt |
| Entity Resolution | F1 ≥ 0.80 | F1 ≥ 0.80 | ✅ Đạt |
| Graph Analytics | ≥ 3 thuật toán GDS | 4 thuật toán (SP, Louvain, Centrality, Link Pred) | ✅ Vượt |
| Heterogeneous integration | Structured + Unstructured | 2 CSV + 5000 transcriptions | ✅ Đạt |
| Hidden relationships | Có CO_MENTIONED edge | 74,904 edges | ✅ Đạt |

### 9.2. Hạn chế

1. **NER recall ~70-80%**: scispaCy miss một số abbreviation hiếm. Khắc phục bằng dictionary nhưng không cover hết.
2. **Louvain modularity 0.26**: Dưới ngưỡng 0.3 — do KG dense (CO_MENTIONED chiếm 64% edges). Có thể cải thiện bằng subgraph filtering.
3. **Gold set nhỏ (150 mẫu)**: Đủ cho đồ án nhưng chưa đủ cho đánh giá thống kê mạnh.
4. **Không map UMLS/ICD-10**: Theo kế hoạch ban đầu, bỏ qua để tiết kiệm thời gian.

### 9.3. Hướng phát triển

1. **Thêm Relation Extraction**: Ngoài NER, trích xuất quan hệ tường minh từ văn bản (Drug TREATS Disease) bằng RE model.
2. **GNN-based Link Prediction**: Node2Vec + GCN thay vì chỉ Adamic-Adar.
3. **UI Search Interface**: Web app cho phép bác sĩ nhập triệu chứng → KG trả về gợi ý chẩn đoán.
4. **Map UMLS CUI**: Nếu có thêm thời gian, ánh xạ entity về UMLS Concept Unique Identifier.
5. **Temporal analysis**: Thêm chiều thời gian nếu có dữ liệu longitudinal.

---

## Phụ lục

### A. Cấu trúc Repository

```
capstoneproject/
├── src/
│   ├── config.py              # Cấu hình tập trung
│   ├── utils.py               # Hàm tiện ích dùng chung
│   ├── ingest_bo1.py          # Tuần 2: Ingest Bộ 1
│   ├── ingest_bo2.py          # Tuần 2: Ingest Bộ 2
│   ├── extract_entities.py    # Tuần 3: NER
│   ├── prepare_er.py          # Tuần 4: Chuẩn bị ER
│   ├── entity_resolution_test.py   # Tuần 5: ER đánh giá
│   ├── entity_resolution_full.py   # Tuần 5: ER toàn bộ
│   └── build_kg.py            # Tuần 6: Build KG v1.0
├── notebooks/
│   ├── eda_bo1.py             # EDA Bộ 1
│   ├── eda_bo2.py             # EDA Bộ 2
│   ├── eda_bo3.py             # EDA Bộ 3
│   ├── graph_analytics.py     # Tuần 7: Graph Analytics
│   └── visualization.py       # Tuần 8: Visualization
├── docs/
│   ├── ontology.md            # Tài liệu Ontology
│   ├── entity_resolution_report.md  # Báo cáo ER
│   ├── sample_queries.cypher  # 14 Cypher query mẫu
│   └── final_report.md        # Báo cáo cuối kỳ (file này)
├── output/
│   ├── analytics_results.md   # Kết quả Graph Analytics
│   ├── louvain_communities.csv
│   └── viz/                   # 4 file HTML visualization
├── output_week_5/             # Artifacts NER + ER
├── backups/                   # Neo4j database dumps
├── requirements.txt
└── README.md
```

### B. Stack công nghệ

| Layer | Tool | Version |
|-------|------|---------|
| Graph DB | Neo4j Community + APOC + GDS | 5.x |
| Language | Python | 3.10+ |
| Neo4j driver | `neo4j` (official) | ≥ 5.0 |
| Data | `pandas`, `pyarrow` | latest |
| NER | `scispacy` + `en_ner_bc5cdr_md` | 0.5+ |
| Phonetic | `jellyfish` | latest |
| Embedding | `sentence-transformers` (MiniLM) | latest |
| Analytics | Neo4j GDS plugin | 2.x |
| Viz | `pyvis` | 0.3+ |

### C. Hướng dẫn chạy lại Pipeline

```bash
# 1. Setup
pip install -r requirements.txt
# Cài scispaCy model
pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

# 2. Khởi động Neo4j, cấu hình .env

# 3. Chạy pipeline theo thứ tự
python src/ingest_bo1.py        # Tuần 2
python src/ingest_bo2.py        # Tuần 2
python src/extract_entities.py  # Tuần 3
python src/prepare_er.py        # Tuần 4
# (annotate gold_set_template.csv thủ công)
python src/entity_resolution_test.py   # Tuần 5 - đánh giá
python src/entity_resolution_full.py   # Tuần 5 - chạy full
python src/build_kg.py          # Tuần 6
python notebooks/graph_analytics.py    # Tuần 7
python notebooks/visualization.py      # Tuần 8
```
