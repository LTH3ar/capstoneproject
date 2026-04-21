# Knowledge Graph Integration for Heterogeneous Medical Data

Đồ án cuối kỳ — tích hợp dữ liệu y khoa có cấu trúc và phi cấu trúc thành Knowledge Graph trong Neo4j, hỗ trợ chẩn đoán y khoa.

---

## Cấu trúc thư mục

```text
capstoneproject/
├── .devcontainer/
│   ├── devcontainer.json         # Cấu hình môi trường VS Code
│   ├── docker-compose.yaml       # Cấu hình container Python & Neo4j
│   └── Dockerfile                # Build image Python & Jupyter
├── drugs-side-effects-and-medical-condition/ # Bộ 1 (Dữ liệu có cấu trúc)
├── disease-symptom-description-dataset/      # Bộ 2 (Dữ liệu có cấu trúc)
├── medical_transcriptions/                   # Bộ 3 (Dữ liệu phi cấu trúc)
├── src/
│   ├── config.py                 # Cấu hình đường dẫn, Neo4j, hằng số
│   ├── utils.py                  # Hàm tiện ích (normalize, Neo4j helpers)
│   ├── ingest_bo1.py             # [Tuần 2] Ingest Drugs & Side Effects
│   ├── ingest_bo2.py             # [Tuần 2] Ingest Disease Symptom Dataset
│   ├── extract_entities.py       # [Tuần 3] NER trên Medical Transcriptions
│   ├── prepare_er.py             # [Tuần 4] Co-occurrence + Gold Set
│   ├── entity_resolution_test.py # [Tuần 5] Đánh giá ER trên Gold Set (F1-score)
│   └── entity_resolution_full.py # [Tuần 5] Chạy ER full pipeline tạo Mapping
├── notebooks/
│   ├── eda_bo1.py                # [Tuần 1] EDA Bộ 1
│   ├── eda_bo2.py                # [Tuần 1] EDA Bộ 2
│   └── eda_bo3.py                # [Tuần 1] EDA Bộ 3
├── output_week_5/                # Thư mục chứa kết quả ER Tuần 5 (er_mapping.csv)
├── .env.example                  # Template cấu hình Neo4j
├── .gitignore                    # Cấu hình git
├── requirements.txt              # Danh sách thư viện Python
└── README.md
```

---

## Cài đặt môi trường local

### 1. Python dependencies

```bash
pip install -r requirements.txt
pip install jellyfish sentence-transformers scikit-learn networkx
```

### 2. scispaCy model (bắt buộc cho Tuần 3)

```bash
pip install scispacy
pip install [https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz](https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz)
```

### 3. Cấu hình Neo4j

Sau khi cài đặt Neo4j thông qua Docker/Devcontainer hoặc chạy cục bộ:

```bash
cp .env.example .env
# Mở .env, điền thông tin kết nối Neo4j (Ví dụ: bolt://neo4j:7687)
```

**Bắt buộc bật 2 plugin trong Neo4j:**
- APOC (Advanced Procedures)
- Graph Data Science (GDS)

---

## Cài đặt môi trường (Devcontainer - Khuyên dùng)

Project này đã được cấu hình sẵn **Docker Compose & VS Code Devcontainer**. Thay vì phải cài đặt thủ công Python, JupyterLab và Neo4j, bạn có thể khởi động toàn bộ hệ thống chỉ với 1 cú click chuột.

### 1. Yêu cầu hệ thống
- Cài đặt [Docker Desktop](https://www.docker.com/products/docker-desktop/) và đảm bảo Docker đang chạy.
- Cài đặt [Visual Studio Code](https://code.visualstudio.com/).
- Cài đặt extension **Dev Containers** của Microsoft trong VS Code.

### 2. Khởi động môi trường
1. Mở thư mục `capstoneproject` bằng VS Code.
2. Nhấn `Ctrl + Shift + P` (hoặc `Cmd + Shift + P` trên Mac), gõ và chọn:
   **`Dev Containers: Rebuild and Reopen in Container`**
3. *Lưu ý: Lần chạy đầu tiên sẽ mất khoảng 5-10 phút để Docker tải image Python, Neo4j, tự động cài đặt các plugin (APOC, GDS) và tải mô hình NLP scispaCy. Các lần sau sẽ khởi động ngay lập tức.*

### 3. Truy cập các dịch vụ (Đã tự động forward port)
Sau khi Devcontainer khởi động xong, các dịch vụ sẽ chạy ngầm:

- **JupyterLab:** `http://localhost:8888` (Không cần token/password)
- **Neo4j Browser:** `http://localhost:7474`
  - URL kết nối: `neo4j://localhost:7687` (hoặc `bolt://neo4j:7687` nếu kết nối từ code Python bên trong container).
  - Username mặc định: `neo4j`
  - Password mặc định: `capstone123` (được cấu hình trong `.devcontainer/docker-compose.yml`).

### 4. Cấu hình biến môi trường
File `.env` không còn bắt buộc do mọi thứ đã cấu hình cứng trong docker-compose. Tuy nhiên, hãy đảm bảo `src/config.py` của bạn trỏ đúng vào mạng nội bộ của Docker:
```python
NEO4J_URI = "bolt://neo4j:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "capstone123"

---

## Chạy từng bước

### Tuần 1 — EDA

```bash
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

### Tuần 3 — NER (cần scispaCy model)

```bash
# Chạy thử với 500 doc đầu (nhanh, ~2-3 phút)
python src/extract_entities.py --limit 500

# Chạy toàn bộ (~5.000 doc, ~3 phút)
python src/extract_entities.py
```
*Output: `output/mentions.parquet`*

### Tuần 4 — Chuẩn bị Entity Resolution

```bash
python src/prepare_er.py
```
*Output:*
- `output/cooccurrence.csv` — cặp entity co-occur trong cùng document.
- `output/gold_set_template.csv` — file template để gán nhãn thủ công (đã hoàn thiện thành `gold_set_annotated.csv`).

### Tuần 5 — Entity Resolution (Pipeline 2 Tầng)

```bash
# 1. Đánh giá thuật toán trên 150 mẫu (Jaro-Winkler + Cosine Similarity)
python src/entity_resolution_test.py

# 2. Chạy thuật toán gom cụm (Clustering) trên toàn bộ 88k mentions
python src/entity_resolution_full.py
```
*Output: Bảng tra cứu `er_mapping.csv` (lưu tại `output/` hoặc `output_week_5/`).*

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

### ✅ Hoàn thành (Tuần 1–5)

- [x] Cấu trúc project, requirements.txt, Devcontainer/Docker setup
- [x] `src/config.py` — cấu hình tập trung
- [x] `src/utils.py` — normalize text, Neo4j helpers
- [x] Tuần 1: EDA các bộ dữ liệu
- [x] Tuần 2: Ingest Drugs, Side Effects & Disease Symptom
- [x] Tuần 3: NER scispaCy + dictionary
- [x] Tuần 4: Prepare ER (co-occurrence + gold set annotated)
- [x] Tuần 5: Pipeline Entity Resolution 2 tầng
  - [x] Đánh giá trên gold set: Jaro-Winkler + Sentence Embedding (F1 >= 0.80)
  - [x] Lọc từ nhiễu y khoa (garbage filtering) & áp dụng Rule-based Dictionary (RxNorm/SNOMED)
  - [x] Chạy full pipeline tạo NetworkX graph clustering → `er_mapping.csv`

### 🔲 Cần làm (Tuần 6–8)

#### Tuần 6 — Merge vào Neo4j + CO_MENTIONED
- [ ] `src/build_kg.py`
  - Dùng `er_mapping.csv` để link mention về canonical entity
  - Tạo/update node từ Bộ 3 (entity mới chưa có trong KG)
  - Tạo edge `CO_MENTIONED` từ `cooccurrence.csv`
  - Tạo edge `TREATED_BY_SPECIALTY` từ metadata Bộ 3
  - KG v1.0 hoàn chỉnh → backup bằng `neo4j-admin dump`

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

*Cập nhật lần cuối: Tuần 5 - Hoàn thành Entity Resolution.*
