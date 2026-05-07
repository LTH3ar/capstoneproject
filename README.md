# Knowledge Graph Integration for Heterogeneous Medical Data

Đồ án cuối kỳ — Tích hợp dữ liệu y khoa **có cấu trúc** (CSV thuốc, bệnh, triệu chứng) và **phi cấu trúc** (ghi chú lâm sàng) thành **một Knowledge Graph thống nhất** trong Neo4j.

**KG hoàn chỉnh:** 18,387 nodes | 117,836 relationships | 7 node types | 7 relationship types

---

## 🚀 HƯỚNG DẪN DEMO NHANH

> Phần này hướng dẫn cách demo dự án cho giảng viên/hội đồng. Chỉ cần **Neo4j đang chạy** với KG đã build sẵn.

### Bước 0 — Kiểm tra Neo4j đang chạy

```bash
# Kiểm tra kết nối
python -c "from neo4j import GraphDatabase; d = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j','capstone123')); d.verify_connectivity(); print('OK'); d.close()"
```

Nếu dùng Devcontainer:
```bash
# Neo4j Browser: http://localhost:7474
# Login: neo4j / capstone123
```

---

### 📌 Demo 1 — Tổng quan KG (2 phút)

Mở **Neo4j Browser** (`http://localhost:7474`) và chạy:

```cypher
-- Đếm node theo type
MATCH (n) RETURN labels(n)[0] AS type, count(*) AS count ORDER BY count DESC;

-- Đếm relationship theo type
MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS count ORDER BY count DESC;
```

**Kịch bản**: *"KG của chúng tôi có 18,387 nodes gồm 7 loại (Disease, Drug, Symptom...) và 117,836 relationships. Dữ liệu tích hợp từ 3 nguồn: 2 bộ CSV structured + 5,000 ghi chú lâm sàng phi cấu trúc."*

Tiếp tục — show 6 bệnh overlap cả 3 bộ (minh chứng tích hợp):
```cypher
MATCH (d:Disease)
WHERE 'bo1' IN d.source AND 'bo2' IN d.source AND 'bo3' IN d.source
RETURN d.name AS disease, d.aliases, d.source;
```

---

### 📌 Demo 2 — Chẩn đoán Flow: Triệu chứng → Bệnh → Thuốc (3 phút)

**Kịch bản**: *"Bệnh nhân vào viện với triệu chứng chest pain. KG gợi ý chẩn đoán gì?"*

```cypher
-- Shortest path: Symptom → Disease → Drug
MATCH (s:Symptom {name: 'chest pain'})<-[:HAS_SYMPTOM]-(d:Disease)<-[:TREATS]-(drug:Drug)
WHERE drug.rating IS NOT NULL AND drug.rating > 0
RETURN d.name AS disease, drug.generic_name AS drug, drug.rating AS rating
ORDER BY rating DESC LIMIT 10;
```

*"KG trả về 2 bệnh khả nghi (hypertension, pneumonia) kèm thuốc sort theo rating, truy vấn < 100ms."*

Thử với triệu chứng khác:
```cypher
-- Headache
MATCH (s:Symptom {name: 'headache'})<-[:HAS_SYMPTOM]-(d:Disease)<-[:TREATS]-(drug:Drug)
WHERE drug.rating >= 7
RETURN d.name, drug.generic_name, drug.rating ORDER BY drug.rating DESC LIMIT 10;

-- Fatigue
MATCH (s:Symptom {name: 'fatigue'})<-[:HAS_SYMPTOM]-(d:Disease)<-[:TREATS]-(drug:Drug)
WHERE drug.rating >= 7
RETURN d.name, drug.generic_name, drug.rating ORDER BY drug.rating DESC LIMIT 10;
```

---

### 📌 Demo 3 — Hidden Relationship & Drug Repurposing (3 phút)

**Kịch bản**: *"Đây là giá trị cốt lõi — phát hiện mối quan hệ ẩn mà KHÔNG nguồn dữ liệu nào ghi nhận."*

```cypher
-- Aspirin: co-mentioned nhiều với diseases nhưng KHÔNG có edge TREATS
MATCH (drug:Drug {generic_name: 'aspirin'})-[co:CO_MENTIONED]-(dis:Disease)
WHERE NOT EXISTS { MATCH (drug)-[:TREATS]->(dis) }
  AND co.weight >= 30
RETURN dis.name AS disease, co.weight AS co_mention_weight
ORDER BY co.weight DESC LIMIT 10;
```

*"Aspirin được nhắc cùng hypertension 121 lần trong ghi chú lâm sàng, cùng chest pain 83 lần — nhưng không có edge TREATS trong dữ liệu structured. Đây là ứng viên off-label use đáng nghiên cứu."*

So sánh với edge TREATS chính thức:
```cypher
-- Aspirin thực sự TREATS bệnh nào (theo dataset structured)?
MATCH (drug:Drug {generic_name: 'aspirin'})-[t:TREATS]->(dis:Disease)
RETURN dis.name, t.rating ORDER BY t.rating DESC;
```

---

### 📌 Demo 4 — Thuốc thay thế cùng nhóm dược lý (1 phút)

**Kịch bản**: *"Bệnh nhân dị ứng metoprolol, tìm thuốc thay thế cùng class?"*

```cypher
MATCH (drug:Drug {generic_name: 'metoprolol'})-[:BELONGS_TO]->(c:DrugClass)
      <-[:BELONGS_TO]-(alt:Drug)
WHERE alt.generic_name <> 'metoprolol'
RETURN c.name AS drug_class, alt.generic_name AS alternative, alt.rating AS rating
ORDER BY rating DESC LIMIT 10;
```

---

### 📌 Demo 5 — Community Detection (2 phút)

**Kịch bản**: *"Louvain tìm được 3,662 cụm bệnh-thuốc tương ứng nhóm chuyên khoa."*

```cypher
-- Đọc kết quả từ file (đã chạy sẵn Tuần 7)
-- Hoặc chạy trực tiếp nếu GDS plugin còn active:

CALL gds.graph.project('demo', ['Drug','Disease','Symptom','Specialty'],
  {CO_MENTIONED: {orientation:'UNDIRECTED', properties:{weight:{defaultValue:1.0}}},
   TREATS: {orientation:'UNDIRECTED', properties:{weight:{defaultValue:1.0}}},
   HAS_SYMPTOM: {orientation:'UNDIRECTED', properties:{weight:{defaultValue:1.0}}}
  })
YIELD nodeCount, relationshipCount
RETURN nodeCount, relationshipCount;

CALL gds.louvain.stream('demo', {relationshipWeightProperty:'weight'})
YIELD nodeId, communityId
WITH communityId, collect(gds.util.asNode(nodeId)) AS members
WITH communityId, size(members) AS sz,
     [m IN members[..5] | coalesce(m.name, m.generic_name)] AS sample
ORDER BY sz DESC LIMIT 10
RETURN communityId, sz AS size, sample;

-- Dọn dẹp
CALL gds.graph.drop('demo');
```

---

### 📌 Demo 6 — Visualization (pyvis) (2 phút)

Mở các file HTML trong browser — **không cần Neo4j**:

| File | Nội dung | Cách mở |
|------|----------|---------|
| `output/viz/01_hypertension_context.html` | Hypertension: Drug + Symptom + DrugClass + Specialty | Double-click hoặc `start output\viz\01_hypertension_context.html` |
| `output/viz/02_cardio_cluster.html` | Cardiovascular Louvain cluster | Double-click |
| `output/viz/03_aspirin_hidden.html` | Aspirin off-label candidates | Double-click |
| `output/viz/04_diagnosis_flow.html` | Headache → Disease → Drug path | Double-click |

**Tips demo**: Kéo thả node, hover xem chi tiết, zoom in/out.

---

### 📌 Demo 7 — Entity Resolution (1 phút)

**Kịch bản**: *"Cùng một bệnh viết theo 3 cách khác nhau → ER merge thành 1 node."*

```cypher
-- Xem aliases của hypertension (đến từ nhiều nguồn)
MATCH (d:Disease {name: 'hypertension'})
RETURN d.name, d.aliases, d.source, d.mention_count;

-- So sánh: tên thuốc có nhiều aliases
MATCH (d:Drug)
WHERE size(d.aliases) > 3
RETURN d.generic_name, d.aliases, d.source
ORDER BY size(d.aliases) DESC LIMIT 10;
```

---

### 📌 Demo 8 — PageRank & Betweenness (1 phút)

Nếu đã có GDS projection, chạy nhanh:
```cypher
CALL gds.graph.project('demo', ['Drug','Disease','Symptom'],
  {CO_MENTIONED: {orientation:'UNDIRECTED', properties:{weight:{defaultValue:1.0}}},
   TREATS: {orientation:'UNDIRECTED', properties:{weight:{defaultValue:1.0}}}})
YIELD nodeCount;

-- Top 10 node quan trọng nhất
CALL gds.pageRank.stream('demo')
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS n, score
RETURN labels(n)[0] AS label, coalesce(n.name, n.generic_name) AS name, score
ORDER BY score DESC LIMIT 10;

CALL gds.graph.drop('demo');
```

Hoặc show file kết quả sẵn: `output/analytics_results.md`

---

## 🎤 KỊCH BẢN THUYẾT TRÌNH ĐỀ XUẤT (15 phút)

| Thời gian | Nội dung | Slide / Demo |
|-----------|----------|--------------|
| 0:00–2:00 | Giới thiệu bài toán + mở đầu bằng case study | Slide: "BN có chest pain, KG tìm ra 3 bệnh + 5 thuốc trong 200ms" |
| 2:00–4:00 | Khảo sát 3 bộ dữ liệu | Slide: bảng so sánh 3 bộ |
| 4:00–5:00 | Ontology design | Slide: sơ đồ 7 node + 7 rel |
| 5:00–7:00 | NER + Entity Resolution | Slide: pipeline diagram + ER metrics (F1 ≥ 0.80) |
| 7:00–9:00 | **Demo live — Neo4j Browser** | Demo 1 (tổng quan) + Demo 2 (chẩn đoán flow) |
| 9:00–11:00 | **Demo live — Hidden relationship** | Demo 3 (aspirin off-label) |
| 11:00–12:00 | Graph Analytics results | Slide: Louvain clusters + PageRank top 10 |
| 12:00–13:00 | **Demo live — Visualization** | Mở HTML pyvis |
| 13:00–14:00 | Kết luận + hạn chế | Slide |
| 14:00–15:00 | Q&A | — |

**Tip**: Mở sẵn Neo4j Browser + 4 tab HTML trước khi bắt đầu. Copy-paste query từ `docs/sample_queries.cypher`.

---

## 📁 Cấu trúc thư mục

```
capstoneproject/
├── src/
│   ├── config.py                 # Cấu hình tập trung
│   ├── utils.py                  # Normalize text, Neo4j helpers
│   ├── ingest_bo1.py             # [Tuần 2] Ingest Drugs & Side Effects
│   ├── ingest_bo2.py             # [Tuần 2] Ingest Disease Symptom
│   ├── extract_entities.py       # [Tuần 3] NER (scispaCy + dictionary)
│   ├── prepare_er.py             # [Tuần 4] Co-occurrence + Gold Set
│   ├── entity_resolution_test.py # [Tuần 5] ER evaluation (F1)
│   ├── entity_resolution_full.py # [Tuần 5] ER full pipeline
│   └── build_kg.py               # [Tuần 6] Build KG v1.0
├── notebooks/
│   ├── eda_bo1.py / eda_bo2.py / eda_bo3.py  # [Tuần 1] EDA
│   ├── graph_analytics.py        # [Tuần 7] GDS analytics
│   └── visualization.py          # [Tuần 8] pyvis HTML
├── docs/
│   ├── ontology.md               # Tài liệu Ontology (sơ đồ + stats)
│   ├── entity_resolution_report.md  # Báo cáo ER (pipeline + metrics)
│   ├── sample_queries.cypher     # 14 Cypher query mẫu
│   └── final_report.md           # Báo cáo cuối kỳ (9 chương)
├── output/
│   ├── analytics_results.md      # Kết quả Graph Analytics
│   ├── louvain_communities.csv   # Community mapping
│   └── viz/                      # 4 file HTML visualization
│       ├── 01_hypertension_context.html
│       ├── 02_cardio_cluster.html
│       ├── 03_aspirin_hidden.html
│       └── 04_diagnosis_flow.html
├── output_week_5/                # NER + ER artifacts
│   ├── mentions.parquet          # Entity mentions từ NER
│   ├── cooccurrence.csv          # Co-occurrence matrix
│   ├── gold_set_annotated.csv    # Gold set (150 mẫu annotated)
│   └── er_mapping.csv            # ER mapping table
├── backups/                      # Neo4j database dumps
│   ├── kg_v0_pre_build_kg.tar.gz
│   └── kg_v1_post_fixes.tar.gz
├── drugs-side-effects-and-medical-condition/  # Bộ 1
├── disease-symptom-description-dataset/       # Bộ 2
├── medical_transcriptions/                    # Bộ 3
├── .devcontainer/                # Docker + VS Code setup
├── .env                          # Neo4j credentials
├── requirements.txt
└── README.md
```

---

## 🔧 Cài đặt & Chạy lại từ đầu

### Cách 1 — Devcontainer (khuyên dùng)

1. Cài [Docker Desktop](https://www.docker.com/products/docker-desktop/) + [VS Code](https://code.visualstudio.com/) + extension **Dev Containers**
2. Mở thư mục project → `Ctrl+Shift+P` → `Dev Containers: Rebuild and Reopen in Container`
3. Truy cập: Neo4j Browser `http://localhost:7474` | JupyterLab `http://localhost:8888`

### Cách 2 — Local

```bash
# 1. Python dependencies
pip install -r requirements.txt
pip install jellyfish sentence-transformers scikit-learn networkx

# 2. scispaCy model
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

# 3. Cấu hình Neo4j
cp .env.example .env
# Sửa .env: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
# Bật plugin APOC + GDS trong Neo4j Desktop
```

### Chạy pipeline đầy đủ

```bash
python src/ingest_bo1.py             # Tuần 2: Ingest Bộ 1
python src/ingest_bo2.py             # Tuần 2: Ingest Bộ 2
python src/extract_entities.py       # Tuần 3: NER (~3 phút)
python src/prepare_er.py             # Tuần 4: Co-occurrence + Gold Set
# → Annotate output/gold_set_template.csv thủ công
python src/entity_resolution_test.py # Tuần 5: Đánh giá trên Gold Set
python src/entity_resolution_full.py # Tuần 5: ER full → er_mapping.csv
python src/build_kg.py               # Tuần 6: Build KG v1.0
python notebooks/graph_analytics.py  # Tuần 7: Graph Analytics
python notebooks/visualization.py    # Tuần 8: 4 HTML visualization
```

---

## 📊 Thống kê KG v1.0

| Node | Count | | Relationship | Count |
|------|-------|-|-------------|-------|
| Disease | 5,621 | | CO_MENTIONED | 74,904 |
| SideEffect | 5,447 | | CAUSES | 27,792 |
| BrandName | 3,873 | | TREATED_BY_SPECIALTY | 7,642 |
| Drug | 3,032 | | HAS_BRAND | 4,154 |
| DrugClass | 243 | | BELONGS_TO | 1,551 |
| Symptom | 131 | | TREATS | 1,472 |
| Specialty | 40 | | HAS_SYMPTOM | 321 |
| **Tổng** | **18,387** | | **Tổng** | **117,836** |

---

## ✅ Trạng thái hoàn thành

- [x] Tuần 1: EDA 3 bộ dữ liệu
- [x] Tuần 2: Ingest structured (Bộ 1 + 2)
- [x] Tuần 3: NER (scispaCy + dictionary)
- [x] Tuần 4: Chuẩn bị ER (co-occurrence + gold set)
- [x] Tuần 5: Entity Resolution 2 tầng (F1 ≥ 0.80)
- [x] Tuần 6: Build KG v1.0 (merge + CO_MENTIONED)
- [x] Tuần 7: Graph Analytics (Shortest Path, Louvain, PageRank, Link Prediction)
- [x] Tuần 8: Visualization (4 pyvis HTML) + Tài liệu (Ontology, ER report, Final report)

*Cập nhật: 06/05/2026 — Hoàn thành toàn bộ pipeline.*
