# Ontology Design — Medical Knowledge Graph

## 1. Tổng quan

Knowledge Graph y khoa tích hợp 3 nguồn dữ liệu khác nhau (2 structured + 1 unstructured) thành một đồ thị thống nhất, được thiết kế theo nguyên tắc:

- **Tối thiểu nhưng đủ**: 7 loại node chính, 7 loại relationship — không quá phức tạp để demo nhưng đủ giàu để khám phá mối quan hệ ẩn.
- **Truy vết nguồn gốc**: Mọi node/edge đều có thuộc tính `source` để biết entity đến từ bộ dữ liệu nào.
- **Khử trùng lặp**: Field `aliases[]` lưu các biến thể tên entity (abbreviation, brand name, tên chuẩn hóa khác nhau giữa các nguồn).

---

## 2. Sơ đồ Ontology

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

     Mọi node ←──── CO_MENTIONED ────→ Mọi node
        (edge phát hiện từ văn bản phi cấu trúc)
```

---

## 3. Chi tiết Node Types

### 3.1. Disease (5,621 nodes)

| Property | Type | Mô tả |
|----------|------|--------|
| `name` | string | Tên bệnh chuẩn hóa (lowercase, unique constraint) |
| `description` | string | Mô tả ngắn về bệnh (từ Bộ 1 hoặc Bộ 2) |
| `precautions` | list[string] | Các biện pháp phòng ngừa (từ Bộ 2) |
| `aliases` | list[string] | Các biến thể tên (abbreviation, tên từ NER) |
| `source` | list[string] | Nguồn gốc: `['bo1']`, `['bo2']`, `['bo3']`, hoặc kết hợp |
| `mention_count` | int | Số lần mention trong Bộ 3 (nếu có) |

### 3.2. Drug (3,032 nodes)

| Property | Type | Mô tả |
|----------|------|--------|
| `generic_name` | string | Tên hoạt chất gốc (unique constraint) |
| `drug_name` | string | Tên thuốc thương mại phổ biến |
| `rating` | float | Điểm đánh giá hiệu quả (0–10, từ Bộ 1) |
| `rx_otc` | string | Kê đơn (Rx) hay OTC |
| `aliases` | list[string] | Các biến thể tên |
| `source` | list[string] | Nguồn gốc |
| `mention_count` | int | Số lần mention trong Bộ 3 |

### 3.3. Symptom (131 nodes)

| Property | Type | Mô tả |
|----------|------|--------|
| `name` | string | Tên triệu chứng chuẩn hóa (unique constraint) |
| `default_severity` | int | Mức độ nghiêm trọng mặc định (1–7, từ Bộ 2) |
| `aliases` | list[string] | Các biến thể tên (snake_case gốc, NER text) |
| `source` | list[string] | Nguồn gốc |

### 3.4. SideEffect (5,447 nodes)

| Property | Type | Mô tả |
|----------|------|--------|
| `name` | string | Tên tác dụng phụ (unique constraint) |

### 3.5. DrugClass (243 nodes)

| Property | Type | Mô tả |
|----------|------|--------|
| `name` | string | Tên nhóm dược lý (unique constraint) |

### 3.6. BrandName (3,873 nodes)

| Property | Type | Mô tả |
|----------|------|--------|
| `name` | string | Tên thương hiệu thuốc (unique constraint) |

### 3.7. Specialty (40 nodes)

| Property | Type | Mô tả |
|----------|------|--------|
| `name` | string | Tên chuyên khoa y tế (unique constraint) |

---

## 4. Chi tiết Relationship Types

| Relationship | Chiều | Thuộc tính | Nguồn | Số lượng |
|--------------|-------|------------|-------|----------|
| `(Drug)-[:TREATS]->(Disease)` | Drug → Disease | `rating`, `source` | Bộ 1 | 1,472 |
| `(Drug)-[:CAUSES]->(SideEffect)` | Drug → SideEffect | — | Bộ 1 | 27,792 |
| `(Drug)-[:BELONGS_TO]->(DrugClass)` | Drug → DrugClass | — | Bộ 1 | 1,551 |
| `(Drug)-[:HAS_BRAND]->(BrandName)` | Drug → BrandName | — | Bộ 1 | 4,154 |
| `(Disease)-[:HAS_SYMPTOM]->(Symptom)` | Disease → Symptom | `severity`, `source` | Bộ 2 | 321 |
| `(*)-[:CO_MENTIONED]-(*)` | Any ↔ Any | `weight`, `source` | Bộ 3 (NER) | 74,904 |
| `(Disease)-[:TREATED_BY_SPECIALTY]->(Specialty)` | Disease → Specialty | `count`, `source` | Bộ 3 | 7,642 |

### 4.1. Giải thích relationship CO_MENTIONED

Edge `CO_MENTIONED` là "hidden relationship" cốt lõi của dự án:

- **Sinh ra từ đâu**: Bộ 3 (Medical Transcriptions) — mỗi khi 2 entity cùng xuất hiện trong 1 bản ghi chú lâm sàng, tăng weight +1.
- **Threshold**: Chỉ giữ cặp có `weight >= 3` (lọc nhiễu).
- **Ý nghĩa**: Phát hiện mối liên hệ ẩn không có trong dữ liệu structured. Ví dụ khi Drug X và Disease Y không có edge TREATS nhưng CO_MENTIONED weight cao → gợi ý off-label use.
- **Đặc tính**: Undirected, thứ tự (a, b) đã chuẩn hóa a < b để tránh trùng.

---

## 5. Constraints & Indexes

```cypher
-- Unique constraints (bắt buộc trước khi ingest)
CREATE CONSTRAINT disease_name IF NOT EXISTS
  FOR (n:Disease) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT drug_generic_name IF NOT EXISTS
  FOR (n:Drug) REQUIRE n.generic_name IS UNIQUE;
CREATE CONSTRAINT symptom_name IF NOT EXISTS
  FOR (n:Symptom) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT side_effect_name IF NOT EXISTS
  FOR (n:SideEffect) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT drug_class_name IF NOT EXISTS
  FOR (n:DrugClass) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT brand_name_name IF NOT EXISTS
  FOR (n:BrandName) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT specialty_name IF NOT EXISTS
  FOR (n:Specialty) REQUIRE n.name IS UNIQUE;

-- Indexes cho tìm kiếm nhanh trên aliases
CREATE INDEX disease_aliases IF NOT EXISTS
  FOR (n:Disease) ON (n.aliases);
CREATE INDEX drug_aliases IF NOT EXISTS
  FOR (n:Drug) ON (n.aliases);
```

---

## 6. Thống kê KG hoàn chỉnh

| Chỉ số | Giá trị |
|--------|---------|
| Tổng nodes | **18,387** |
| Tổng relationships | **117,836** |
| Node types | 7 |
| Relationship types | 7 |
| Nguồn dữ liệu | 3 (2 structured + 1 unstructured) |
| Disease nodes có ≥2 nguồn | ~300+ nodes overlap |

### Phân bố Disease theo nguồn

| Nguồn | Số lượng Disease |
|-------|-----------------|
| Chỉ Bộ 3 (NER) | ~4,990 |
| Chỉ Bộ 1 | ~16 |
| Chỉ Bộ 2 | ~20 |
| Bộ 1 + Bộ 3 | ~253 |
| Bộ 2 + Bộ 3 | ~15 |
| Bộ 1 + Bộ 2 + Bộ 3 | ~6 |
| Bộ 1 + Bộ 2 | ~316 |

→ Minh chứng rõ ràng rằng KG đã thành công tích hợp dữ liệu từ nhiều nguồn heterogeneous.
