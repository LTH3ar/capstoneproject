# %% [markdown]
# # Tuần 7 — Graph Analytics trên Medical Knowledge Graph v1.0
#
# **Yêu cầu của plan (Tuần 7):**
# 1. **Shortest Path** (bắt buộc) — Dijkstra: Symptom → Disease → Drug
# 2. **Community Detection** (bắt buộc) — Louvain
# 3. **Centrality** (bắt buộc) — PageRank + Betweenness
# 4. **Link Prediction** (bonus) — Adamic-Adar
#
# Dùng **Neo4j Graph Data Science (GDS) 2.x** plugin.
# KG đã chứa 18.387 nodes / 117.836 edges sau Tuần 6 (đã fix).
#
# Chạy:
#   - VS Code Jupyter: chạy từng cell (`# %%`).
#   - Terminal: `python notebooks/graph_analytics.py` — chạy end-to-end.
#
# Output: bảng kết quả + file `output/analytics_results.md` để nhúng báo cáo.

# %%
import sys
import io
from pathlib import Path
from collections import defaultdict

# UTF-8 stdout cho Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OUTPUT_DIR
from src.utils import get_neo4j_driver, neo4j_session, logger

GRAPH_NAME = "medicalKG"
RESULT_FILE = OUTPUT_DIR / "analytics_results.md"


# ── Connection helper ────────────────────────────────────────────────────────
def fetch(session, cypher: str, **params) -> pd.DataFrame:
    """Run cypher → DataFrame."""
    return pd.DataFrame([dict(r) for r in session.run(cypher, **params)])


driver = get_neo4j_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
session = driver.session()
logger.info("Connected. GDS version: %s",
            session.run("RETURN gds.version() AS v").single()["v"])

# Mở file output để ghi báo cáo Markdown
result_md = []
def md(line: str = ""):
    print(line)
    result_md.append(line)


# %% [markdown]
# ## 1. Project subgraph cho phân tích
#
# GDS yêu cầu in-memory graph projection. Project 4 loại node và 5 loại edge
# có ý nghĩa về cấu trúc liên kết clinical:
# - Nodes: `Drug`, `Disease`, `Symptom`, `Specialty`
# - Edges: `TREATS`, `HAS_SYMPTOM`, `CO_MENTIONED` (có weight),
#          `TREATED_BY_SPECIALTY`, `BELONGS_TO`
#
# Bỏ `SideEffect` và `BrandName` — không tham gia chẩn đoán flow,
# chỉ làm tăng noise và slowdown thuật toán.

# %%
# Drop nếu đã tồn tại từ lần chạy trước (idempotent)
session.run(f"""
    CALL gds.graph.exists('{GRAPH_NAME}') YIELD exists
    WITH exists
    WHERE exists
    CALL gds.graph.drop('{GRAPH_NAME}') YIELD graphName
    RETURN graphName
""")

# Mỗi relationship type có property `weight`:
#   - CO_MENTIONED: dùng weight gốc (số lần co-occur)
#   - các edges structural (TREATS/HAS_SYMPTOM/TREATED_BY_SPECIALTY): default 1.0
#     (cần thiết để Louvain weighted, PageRank weighted hoạt động trên graph chung)
result = session.run(f"""
CALL gds.graph.project(
    '{GRAPH_NAME}',
    ['Drug', 'Disease', 'Symptom', 'Specialty'],
    {{
        TREATS:               {{orientation: 'UNDIRECTED',
                                properties: {{weight: {{property: 'weight', defaultValue: 1.0}}}}}},
        HAS_SYMPTOM:          {{orientation: 'UNDIRECTED',
                                properties: {{weight: {{property: 'weight', defaultValue: 1.0}}}}}},
        TREATED_BY_SPECIALTY: {{orientation: 'UNDIRECTED',
                                properties: {{weight: {{property: 'weight', defaultValue: 1.0}}}}}},
        CO_MENTIONED:         {{orientation: 'UNDIRECTED',
                                properties: {{weight: {{property: 'weight', defaultValue: 1.0}}}}}}
    }}
) YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
""").single()

md("# Tuần 7 — Graph Analytics Results\n")
md(f"## Graph projection `{GRAPH_NAME}`")
md(f"- Nodes: **{result['nodeCount']:,}**")
md(f"- Relationships: **{result['relationshipCount']:,}** (UNDIRECTED, count gấp đôi vì duplicate hai chiều)")
md("")


# %% [markdown]
# ## 2. Shortest Path — Symptom → Disease → Drug (chẩn đoán flow)
#
# **Use case:** "Bệnh nhân có triệu chứng X. Đường ngắn nhất tới thuốc?"
#
# Pure Cypher path-finding (không cần GDS) — vì path 2 hop đơn giản,
# dùng `MATCH path = ...` đủ.

# %%
md("## 1. Shortest Path — Symptom → Disease → Drug\n")

# Demo 5 symptom điển hình
demo_symptoms = ["fatigue", "chest pain", "headache", "fever", "vomiting"]

for sym in demo_symptoms:
    df = fetch(session, """
        MATCH path = (s:Symptom {name: $sym})<-[:HAS_SYMPTOM]-(d:Disease)<-[:TREATS]-(drug:Drug)
        WHERE drug.rating IS NOT NULL AND drug.rating > 0
        RETURN d.name AS disease, drug.generic_name AS drug, drug.rating AS rating
        ORDER BY rating DESC LIMIT 5
    """, sym=sym)
    md(f"### Symptom `{sym}` → top 5 chẩn đoán + thuốc (sort theo rating)\n")
    if df.empty:
        md(f"_Không có path 2 hop nào từ `{sym}`._\n")
    else:
        md("| Disease | Drug | Rating |")
        md("|---------|------|--------|")
        for _, r in df.iterrows():
            md(f"| {r['disease']} | {r['drug']} | {r['rating']} |")
        md("")


# %% [markdown]
# ## 3. Community Detection — Louvain
#
# **Use case:** "Có những cụm bệnh – thuốc – triệu chứng nào trong KG?"
#
# Cụm lớn thường tương ứng với nhóm chuyên khoa (cardio, pulmo, neuro, ...).
# Dùng weight CO_MENTIONED làm trọng số — cụm chặt là cụm cùng được nhắc nhiều
# trong cùng transcription.

# %%
md("## 2. Community Detection — Louvain\n")

# Stream mode để lấy result trực tiếp về DataFrame
df = fetch(session, f"""
    CALL gds.louvain.stream('{GRAPH_NAME}', {{relationshipWeightProperty: 'weight'}})
    YIELD nodeId, communityId
    WITH communityId, gds.util.asNode(nodeId) AS n
    RETURN
      communityId,
      labels(n)[0] AS label,
      coalesce(n.name, n.generic_name) AS name
""")

# Tổng kết
total_communities = df["communityId"].nunique()
md(f"- **Tổng số community:** {total_communities:,}")
md(f"- **Modularity** _(chạy mode='stats' bên dưới)_")

stats = session.run(f"""
    CALL gds.louvain.stats('{GRAPH_NAME}', {{relationshipWeightProperty: 'weight'}})
    YIELD modularity, communityCount, communityDistribution
    RETURN modularity, communityCount,
           communityDistribution.mean AS sizeMean,
           communityDistribution.max AS sizeMax,
           communityDistribution.p90 AS sizeP90
""").single()
md(f"  - Modularity: **{stats['modularity']:.4f}** (>= 0.3 là tốt)")
md(f"  - Community size: mean={stats['sizeMean']:.1f}, p90={stats['sizeP90']}, max={stats['sizeMax']}")
md("")

# Top 10 cộng đồng lớn nhất + sample members
top_communities = (
    df.groupby("communityId")
    .size()
    .reset_index(name="size")
    .sort_values("size", ascending=False)
    .head(10)
)
md("### Top 10 community lớn nhất (sample members)\n")
md("| Cluster ID | Size | Sample members (Disease/Drug/Symptom) |")
md("|------------|------|----------------------------------------|")
for _, row in top_communities.iterrows():
    cid = row["communityId"]
    members = df[df["communityId"] == cid]
    sample = members.head(8)
    sample_str = ", ".join(
        f"{m['name']} ({m['label'][0]})"
        for _, m in sample.iterrows()
    )
    md(f"| {cid} | {row['size']} | {sample_str} |")
md("")

# Lưu full mapping community → file
df.to_csv(OUTPUT_DIR / "louvain_communities.csv", index=False)
md(f"_Full community mapping saved → `output/louvain_communities.csv`_\n")


# %% [markdown]
# ## 4. Centrality — PageRank + Betweenness
#
# **PageRank:** node "quan trọng" về mặt connectivity (nhiều liên kết với
# node khác cũng quan trọng).
# **Betweenness:** node "cầu nối" — nằm trên nhiều shortest path nhất.

# %%
md("## 3. Centrality\n")

# PageRank
df_pr = fetch(session, f"""
    CALL gds.pageRank.stream('{GRAPH_NAME}', {{relationshipWeightProperty: 'weight'}})
    YIELD nodeId, score
    WITH gds.util.asNode(nodeId) AS n, score
    RETURN labels(n)[0] AS label,
           coalesce(n.name, n.generic_name) AS name,
           score
    ORDER BY score DESC LIMIT 30
""")
md("### Top 20 PageRank — node trung tâm của KG\n")
md("| Rank | Label | Name | Score |")
md("|------|-------|------|-------|")
for i, r in df_pr.head(20).iterrows():
    md(f"| {i+1} | {r['label']} | {r['name']} | {r['score']:.4f} |")
md("")

# Betweenness — dùng sample 1500 source vì betweenness O(VE) đắt
df_bw = fetch(session, f"""
    CALL gds.betweenness.stream('{GRAPH_NAME}', {{samplingSize: 1500, samplingSeed: 42}})
    YIELD nodeId, score
    WITH gds.util.asNode(nodeId) AS n, score
    WHERE score > 0
    RETURN labels(n)[0] AS label,
           coalesce(n.name, n.generic_name) AS name,
           score
    ORDER BY score DESC LIMIT 30
""")
md("### Top 20 Betweenness — node cầu nối (sample 1500 sources)\n")
md("| Rank | Label | Name | Score |")
md("|------|-------|------|-------|")
for i, r in df_bw.head(20).iterrows():
    md(f"| {i+1} | {r['label']} | {r['name']} | {r['score']:.1f} |")
md("")


# %% [markdown]
# ## 5. Link Prediction — Adamic-Adar (BONUS)
#
# **Use case:** "Drug X có thể TREATS Disease Y nào chưa biết?"
#
# Adamic-Adar score cao giữa Drug-Disease chưa có cạnh TREATS
# = ứng viên cho off-label/drug-repurposing.

# %%
md("## 4. Link Prediction — Adamic-Adar (BONUS)\n")
md("**Drug-Disease pair có Adamic-Adar score cao nhưng KHÔNG có TREATS:**\n")

# GDS Adamic-Adar yêu cầu pair input. Lấy candidates từ CO_MENTIONED weight cao
df_lp = fetch(session, """
    MATCH (drug:Drug)-[co:CO_MENTIONED]-(dis:Disease)
    WHERE NOT EXISTS { MATCH (drug)-[:TREATS]->(dis) }
      AND co.weight >= 30
    WITH drug, dis, co.weight AS coWeight,
         gds.alpha.linkprediction.adamicAdar(drug, dis,
            {relationshipQuery: 'CO_MENTIONED'}) AS aa
    RETURN drug.generic_name AS drug, dis.name AS disease,
           coWeight, aa AS adamicAdar
    ORDER BY aa DESC LIMIT 15
""")
md("| Rank | Drug | Disease | CO_MENTIONED weight | Adamic-Adar |")
md("|------|------|---------|---------------------|-------------|")
for i, r in df_lp.iterrows():
    md(f"| {i+1} | {r['drug']} | {r['disease']} | {r['coWeight']} | {r['adamicAdar']:.3f} |")
md("")


# %% [markdown]
# ## 6. Cleanup — drop projection
#
# In-memory graph projection chiếm RAM, drop sau khi xong.

# %%
session.run(f"CALL gds.graph.drop('{GRAPH_NAME}') YIELD graphName")
logger.info("Dropped projection '%s'", GRAPH_NAME)

session.close()
driver.close()


# %% [markdown]
# ## 7. Lưu kết quả ra file Markdown

# %%
RESULT_FILE.write_text("\n".join(result_md), encoding="utf-8")
print(f"\n✓ Báo cáo đã lưu: {RESULT_FILE}")
print(f"✓ Community mapping: {OUTPUT_DIR / 'louvain_communities.csv'}")
print()
print("Bước tiếp (Tuần 8): Visualization (Bloom + pyvis), báo cáo PDF, slide.")
