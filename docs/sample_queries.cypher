-- ============================================================
-- BỘ CYPHER QUERY MẪU — Medical Knowledge Graph
-- ============================================================
-- 10+ query demo cho các use case chẩn đoán, khám phá, phân tích
-- ============================================================

-- ──────────────────────────────────────────────────────────────
-- QUERY 1: Tìm thuốc điều trị 1 bệnh cụ thể (sort theo rating)
-- Use case: Bác sĩ muốn biết thuốc tốt nhất cho hypertension
-- ──────────────────────────────────────────────────────────────
MATCH (drug:Drug)-[t:TREATS]->(d:Disease {name: 'hypertension'})
WHERE drug.rating IS NOT NULL
RETURN drug.generic_name AS drug, t.rating AS rating
ORDER BY rating DESC
LIMIT 10;

-- ──────────────────────────────────────────────────────────────
-- QUERY 2: Shortest path — Triệu chứng → Bệnh → Thuốc
-- Use case: Bệnh nhân có triệu chứng "chest pain", gợi ý chẩn đoán
-- ──────────────────────────────────────────────────────────────
MATCH path = (s:Symptom {name: 'chest pain'})<-[:HAS_SYMPTOM]-(d:Disease)<-[:TREATS]-(drug:Drug)
WHERE drug.rating IS NOT NULL AND drug.rating > 0
RETURN d.name AS disease, drug.generic_name AS drug, drug.rating AS rating
ORDER BY rating DESC
LIMIT 10;

-- ──────────────────────────────────────────────────────────────
-- QUERY 3: Tìm tất cả triệu chứng của 1 bệnh
-- Use case: Liệt kê triệu chứng để hỗ trợ chẩn đoán phân biệt
-- ──────────────────────────────────────────────────────────────
MATCH (d:Disease {name: 'pneumonia'})-[hs:HAS_SYMPTOM]->(s:Symptom)
RETURN s.name AS symptom, hs.severity AS severity
ORDER BY severity DESC;

-- ──────────────────────────────────────────────────────────────
-- QUERY 4: Tìm tác dụng phụ của 1 thuốc
-- Use case: Kiểm tra tác dụng phụ trước khi kê đơn
-- ──────────────────────────────────────────────────────────────
MATCH (drug:Drug {generic_name: 'aspirin'})-[:CAUSES]->(se:SideEffect)
RETURN se.name AS side_effect
LIMIT 20;

-- ──────────────────────────────────────────────────────────────
-- QUERY 5: Tìm thuốc cùng nhóm dược lý (alternative drugs)
-- Use case: Bệnh nhân dị ứng thuốc A, tìm thuốc thay thế cùng class
-- ──────────────────────────────────────────────────────────────
MATCH (drug:Drug {generic_name: 'metoprolol'})-[:BELONGS_TO]->(c:DrugClass)
       <-[:BELONGS_TO]-(alt:Drug)
WHERE alt.generic_name <> 'metoprolol'
RETURN c.name AS drug_class, alt.generic_name AS alternative, alt.rating AS rating
ORDER BY rating DESC
LIMIT 10;

-- ──────────────────────────────────────────────────────────────
-- QUERY 6: Hidden relationship — Drug co-mentioned với Disease
--          nhưng KHÔNG có edge TREATS (off-label hint)
-- Use case: Drug repurposing — khám phá ứng viên điều trị mới
-- ──────────────────────────────────────────────────────────────
MATCH (drug:Drug)-[co:CO_MENTIONED]-(dis:Disease)
WHERE NOT EXISTS { MATCH (drug)-[:TREATS]->(dis) }
  AND co.weight >= 30
RETURN drug.generic_name AS drug, dis.name AS disease,
       co.weight AS co_mention_weight
ORDER BY co.weight DESC
LIMIT 15;

-- ──────────────────────────────────────────────────────────────
-- QUERY 7: Thống kê node theo label
-- Use case: Kiểm tra sức khỏe dữ liệu KG
-- ──────────────────────────────────────────────────────────────
MATCH (n)
RETURN labels(n)[0] AS type, count(*) AS count
ORDER BY count DESC;

-- ──────────────────────────────────────────────────────────────
-- QUERY 8: Thống kê relationship theo type
-- ──────────────────────────────────────────────────────────────
MATCH ()-[r]->()
RETURN type(r) AS relationship, count(*) AS count
ORDER BY count DESC;

-- ──────────────────────────────────────────────────────────────
-- QUERY 9: Tìm Disease được điều trị bởi nhiều chuyên khoa nhất
-- Use case: Bệnh nào có tính liên chuyên khoa cao?
-- ──────────────────────────────────────────────────────────────
MATCH (d:Disease)-[r:TREATED_BY_SPECIALTY]->(sp:Specialty)
WITH d, count(sp) AS num_specialties, collect(sp.name) AS specialties
WHERE num_specialties >= 3
RETURN d.name AS disease, num_specialties, specialties
ORDER BY num_specialties DESC
LIMIT 15;

-- ──────────────────────────────────────────────────────────────
-- QUERY 10: Tìm Drug "cầu nối" — điều trị nhiều bệnh nhất
-- Use case: Thuốc đa năng, có tiềm năng repurposing
-- ──────────────────────────────────────────────────────────────
MATCH (drug:Drug)-[:TREATS]->(d:Disease)
WITH drug, count(d) AS num_diseases
WHERE num_diseases >= 5
RETURN drug.generic_name AS drug, num_diseases
ORDER BY num_diseases DESC
LIMIT 15;

-- ──────────────────────────────────────────────────────────────
-- QUERY 11: Tìm Disease overlap cả 3 bộ dữ liệu
-- Use case: Kiểm chứng Entity Resolution — bệnh nào được link thành công?
-- ──────────────────────────────────────────────────────────────
MATCH (d:Disease)
WHERE 'bo1' IN d.source AND 'bo2' IN d.source AND 'bo3' IN d.source
RETURN d.name AS disease, d.aliases AS aliases, d.source AS sources
ORDER BY d.name;

-- ──────────────────────────────────────────────────────────────
-- QUERY 12: Top co-mentioned Drug-Disease pairs (weighted)
-- Use case: Cặp thuốc-bệnh nào được nhắc cùng nhau nhiều nhất
--           trong ghi chú lâm sàng?
-- ──────────────────────────────────────────────────────────────
MATCH (drug:Drug)-[co:CO_MENTIONED]-(dis:Disease)
RETURN drug.generic_name AS drug, dis.name AS disease, co.weight AS weight
ORDER BY weight DESC
LIMIT 20;

-- ──────────────────────────────────────────────────────────────
-- QUERY 13: Tìm tất cả brand names của 1 thuốc
-- Use case: Bệnh nhân hỏi "Tylenol có giống Panadol không?"
-- ──────────────────────────────────────────────────────────────
MATCH (drug:Drug)-[:HAS_BRAND]->(b:BrandName)
WHERE drug.generic_name CONTAINS 'acetaminophen'
RETURN drug.generic_name AS generic, collect(b.name) AS brands;

-- ──────────────────────────────────────────────────────────────
-- QUERY 14: Shortest path tổng quát giữa 2 node bất kỳ
-- Use case: Khám phá mối liên hệ giữa 2 entity xa nhau
-- ──────────────────────────────────────────────────────────────
MATCH path = shortestPath(
  (a:Disease {name: 'diabetes'})-[*..5]-(b:Drug {generic_name: 'aspirin'})
)
RETURN [n IN nodes(path) | coalesce(n.name, n.generic_name)] AS path_nodes,
       [r IN relationships(path) | type(r)] AS path_rels,
       length(path) AS hops;
