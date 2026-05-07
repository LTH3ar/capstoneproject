# %% [markdown]
# # Tuần 8 — Visualization (pyvis)
#
# Xuất 4 subgraph HTML tiêu biểu để nhúng vào báo cáo và demo:
#
# 1. **Hypertension full context** — bệnh có overlap cả 3 bộ. Hiển thị
#    drug TREATS + symptom HAS_SYMPTOM + drug class + specialty.
# 2. **Cardiovascular Louvain cluster** — top community lớn từ Tuần 7.
# 3. **Hidden relationship** — aspirin × diseases có CO_MENTIONED nhưng
#    không có TREATS (off-label hint).
# 4. **Diagnosis path** — Symptom → Disease → Drug, demo flow chẩn đoán.
#
# Output: 4 file HTML trong `output/viz/` — mở trong browser hoặc nhúng iframe.

# %%
import sys
import io
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyvis.network import Network

from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OUTPUT_DIR
from src.utils import get_neo4j_driver, logger

VIZ_DIR = OUTPUT_DIR / "viz"
VIZ_DIR.mkdir(exist_ok=True)

# Color & shape theo label — nhất quán cho mọi subgraph
LABEL_STYLE = {
    "Drug":       {"color": "#3498db", "shape": "dot",       "size": 14},  # blue
    "Disease":    {"color": "#e74c3c", "shape": "dot",       "size": 18},  # red
    "Symptom":    {"color": "#2ecc71", "shape": "diamond",   "size": 14},  # green
    "SideEffect": {"color": "#f39c12", "shape": "triangle",  "size": 12},  # orange
    "DrugClass":  {"color": "#9b59b6", "shape": "square",    "size": 14},  # purple
    "BrandName":  {"color": "#1abc9c", "shape": "dot",       "size": 10},  # teal
    "Specialty":  {"color": "#34495e", "shape": "star",      "size": 16},  # dark
}
EDGE_STYLE = {
    "TREATS":               {"color": "#3498db", "title": "TREATS"},
    "HAS_SYMPTOM":          {"color": "#2ecc71", "title": "HAS_SYMPTOM"},
    "CAUSES":               {"color": "#f39c12", "title": "CAUSES"},
    "BELONGS_TO":           {"color": "#9b59b6", "title": "BELONGS_TO"},
    "HAS_BRAND":            {"color": "#1abc9c", "title": "HAS_BRAND"},
    "TREATED_BY_SPECIALTY": {"color": "#34495e", "title": "TREATED_BY_SPECIALTY"},
    "CO_MENTIONED":         {"color": "#bdc3c7", "title": "CO_MENTIONED"},
}


# ── Helper ────────────────────────────────────────────────────────────────────

def write_html_utf8(net: Network, path: Path) -> None:
    """pyvis.write_html() dùng cp1252 mặc định trên Windows → fail với Unicode.
    Dùng generate_html() rồi ghi UTF-8 thủ công."""
    html = net.generate_html(notebook=False)
    Path(path).write_text(html, encoding="utf-8")


def make_network(title: str) -> Network:
    """Tạo pyvis Network với cấu hình thống nhất."""
    net = Network(
        height="700px", width="100%", bgcolor="#ffffff", font_color="black",
        directed=False, notebook=False, cdn_resources="in_line",
    )
    net.barnes_hut(
        gravity=-12000, central_gravity=0.3,
        spring_length=120, spring_strength=0.04, damping=0.85,
    )
    net.set_options("""
    var options = {
      "interaction": { "hover": true, "tooltipDelay": 100 },
      "physics": { "stabilization": { "iterations": 200 } }
    }
    """)
    return net


def add_node(net: Network, node_id: str, label: str, name: str,
             extra_title: str = "") -> None:
    style = LABEL_STYLE.get(label, {"color": "#888", "shape": "dot", "size": 12})
    title = f"{label}: {name}"
    if extra_title:
        title += f"\n{extra_title}"
    net.add_node(
        node_id, label=name[:30], title=title,
        color=style["color"], shape=style["shape"], size=style["size"],
    )


def add_edge(net: Network, src: str, dst: str, rel_type: str,
             weight: float = 1.0, label: str = "") -> None:
    style = EDGE_STYLE.get(rel_type, {"color": "#cccccc", "title": rel_type})
    # value = thickness; weight làm dày cạnh CO_MENTIONED
    value = max(1, min(weight, 20)) if rel_type == "CO_MENTIONED" else 2
    net.add_edge(
        src, dst, color=style["color"], title=f"{rel_type}: {label or weight}",
        value=value, label=label if label else "",
    )


def graph_from_query(net: Network, session, cypher: str, **params) -> int:
    """
    Chạy Cypher query trả về paths/relationships, build pyvis network.
    Cypher phải return field `path` hoặc tuple (n, r, m).
    Returns: số node thêm.
    """
    seen_nodes = set()
    n_nodes_added = 0
    for rec in session.run(cypher, **params):
        # Path or (n, r, m) tuple — handle both
        if "path" in rec.keys():
            elements = rec["path"]
            for el in elements.nodes:
                nid = el.element_id
                if nid not in seen_nodes:
                    add_node(net, nid, list(el.labels)[0],
                             el.get("name") or el.get("generic_name") or "?")
                    seen_nodes.add(nid)
                    n_nodes_added += 1
            for r in elements.relationships:
                add_edge(
                    net, r.start_node.element_id, r.end_node.element_id,
                    r.type,
                    weight=r.get("weight", 1),
                    label=str(r.get("weight", "")) if r.type == "CO_MENTIONED" else "",
                )
        else:
            # Tuple of (n, r, m)
            n, r, m = rec["n"], rec["r"], rec["m"]
            for x in (n, m):
                nid = x.element_id
                if nid not in seen_nodes:
                    add_node(net, nid, list(x.labels)[0],
                             x.get("name") or x.get("generic_name") or "?")
                    seen_nodes.add(nid)
                    n_nodes_added += 1
            add_edge(
                net, n.element_id, m.element_id, r.type,
                weight=r.get("weight", 1),
                label=str(r.get("weight", "")) if r.type == "CO_MENTIONED" else "",
            )
    return n_nodes_added


# %%
driver = get_neo4j_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
session = driver.session()


# %% [markdown]
# ## Subgraph 1 — Hypertension full clinical context
#
# Bệnh `hypertension` overlap cả 3 bộ → hiển thị tốt heterogeneous integration:
# - Drug TREATS (top 15 theo rating)
# - Symptom HAS_SYMPTOM
# - DrugClass mà các Drug đó thuộc về
# - Specialty điều trị

# %%
net = make_network("Hypertension Clinical Context")
graph_from_query(net, session, """
    MATCH path = (drug:Drug)-[t:TREATS]->(d:Disease {name: 'hypertension'})
    WHERE drug.rating IS NOT NULL AND drug.rating >= 7
    RETURN path LIMIT 15
""")
graph_from_query(net, session, """
    MATCH path = (d:Disease {name: 'hypertension'})-[:HAS_SYMPTOM]->(:Symptom)
    RETURN path
""")
graph_from_query(net, session, """
    MATCH (drug:Drug)-[:TREATS]->(:Disease {name: 'hypertension'})
    WHERE drug.rating >= 7
    WITH drug LIMIT 10
    MATCH path = (drug)-[:BELONGS_TO]->(:DrugClass)
    RETURN path
""")
graph_from_query(net, session, """
    MATCH path = (:Disease {name: 'hypertension'})-[r:TREATED_BY_SPECIALTY]->(:Specialty)
    WHERE r.count >= 5
    RETURN path
""")

out_file = VIZ_DIR / "01_hypertension_context.html"
write_html_utf8(net, out_file)
print(f"[1/4] Hypertension context → {out_file}  ({len(net.nodes)} nodes, {len(net.edges)} edges)")


# %% [markdown]
# ## Subgraph 2 — Cardiovascular cluster (Louvain cluster 2120)
#
# Cluster lớn từ Tuần 7. Hiển thị các CO_MENTIONED edge mạnh trong cụm.

# %%
import pandas as pd
COMMUNITIES_CSV = OUTPUT_DIR / "louvain_communities.csv"

if COMMUNITIES_CSV.exists():
    df_comm = pd.read_csv(COMMUNITIES_CSV)
    # Top community theo size, lấy cluster có chứa "angina" hoặc "nitroglycerin" (cardiovascular)
    cardio_clusters = df_comm[df_comm["name"].str.contains(
        "angina|nitroglycerin|metoprolol", case=False, na=False
    )]["communityId"].unique()

    if len(cardio_clusters) > 0:
        target_cluster = cardio_clusters[0]
        members = df_comm[df_comm["communityId"] == target_cluster]
        # Take top 40 members để vẽ — quá nhiều sẽ rối
        member_names = members["name"].head(40).tolist()
        print(f"  Cardio cluster {target_cluster}: {len(members)} members, viz {len(member_names)}")

        net = make_network(f"Louvain Cluster {target_cluster} — Cardiovascular")
        graph_from_query(net, session, """
            MATCH (a)-[r:CO_MENTIONED]-(b)
            WHERE (a.name IN $names OR a.generic_name IN $names)
              AND (b.name IN $names OR b.generic_name IN $names)
              AND r.weight >= 8
            RETURN a AS n, r, b AS m
        """, names=member_names)
        out_file = VIZ_DIR / "02_cardio_cluster.html"
        write_html_utf8(net, out_file)
        print(f"[2/4] Cardio cluster → {out_file}  ({len(net.nodes)} nodes, {len(net.edges)} edges)")
    else:
        print("[2/4] Skipped — không tìm thấy cardio cluster trong louvain output")
else:
    print(f"[2/4] Skipped — không thấy {COMMUNITIES_CSV}. Chạy Tuần 7 trước.")


# %% [markdown]
# ## Subgraph 3 — Hidden relationship: aspirin off-label hints
#
# Aspirin có Adamic-Adar cao với hypertension/chest pain/edema/dyspnea
# nhưng không có TREATS edge. Demo "drug repurposing" của plan.

# %%
net = make_network("Hidden Relationship — Aspirin Off-label Candidates")
graph_from_query(net, session, """
    MATCH (drug:Drug {generic_name: 'aspirin'})-[r:CO_MENTIONED]-(dis:Disease)
    WHERE NOT EXISTS { MATCH (drug)-[:TREATS]->(dis) }
      AND r.weight >= 30
    RETURN drug AS n, r, dis AS m
    ORDER BY r.weight DESC LIMIT 15
""")
# Thêm edge TREATS hợp lệ của aspirin để so sánh (cùng đồ thị)
graph_from_query(net, session, """
    MATCH path = (drug:Drug {generic_name: 'aspirin'})-[:TREATS]->(:Disease)
    RETURN path LIMIT 10
""")
out_file = VIZ_DIR / "03_aspirin_hidden.html"
write_html_utf8(net, out_file)
print(f"[3/4] Aspirin hidden → {out_file}  ({len(net.nodes)} nodes, {len(net.edges)} edges)")


# %% [markdown]
# ## Subgraph 4 — Diagnosis path: headache → migraine → drugs
#
# Demo flow chẩn đoán y khoa: triệu chứng → bệnh suy ra → thuốc gợi ý.

# %%
net = make_network("Diagnosis Flow — headache")
graph_from_query(net, session, """
    MATCH path = (s:Symptom {name: 'headache'})<-[:HAS_SYMPTOM]-(d:Disease)
    RETURN path
""")
graph_from_query(net, session, """
    MATCH path = (s:Symptom {name: 'headache'})<-[:HAS_SYMPTOM]-(d:Disease)<-[t:TREATS]-(drug:Drug)
    WHERE drug.rating IS NOT NULL AND drug.rating >= 7
    RETURN path LIMIT 25
""")
out_file = VIZ_DIR / "04_diagnosis_flow.html"
write_html_utf8(net, out_file)
print(f"[4/4] Diagnosis flow → {out_file}  ({len(net.nodes)} nodes, {len(net.edges)} edges)")


# %%
session.close()
driver.close()

print()
print("=" * 60)
print(f"4 subgraph HTML đã xuất → {VIZ_DIR}")
print("=" * 60)
print("Cách dùng:")
print("  - Mở từng file HTML trong browser để xem interactive viz")
print("  - Nhúng vào báo cáo bằng <iframe src='viz/01_hypertension_context.html'>")
print("  - Hoặc screenshot từng cảnh để chèn slide")
