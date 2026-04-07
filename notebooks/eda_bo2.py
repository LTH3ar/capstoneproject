# %% [markdown]
# # EDA — Bộ 2: Disease Symptom Prediction Dataset
# **Tuần 1** — Khám phá 4 file CSV trong `disease-symptom-description-dataset/`

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import BO2_DATASET_CSV, BO2_SEVERITY_CSV, BO2_DESC_CSV, BO2_PRECAUTION_CSV
from src.utils import normalize_text, normalize_symptom_name, safe_str

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_colwidth", 80)

# %% [markdown]
# ## 1. dataset.csv — Bảng chính Disease → Symptoms

# %%
df_main = pd.read_csv(BO2_DATASET_CSV)
print(f"Shape: {df_main.shape}")
print(f"Cột: {df_main.columns.tolist()}")
print(f"\nSố bệnh unique: {df_main['Disease'].nunique()}")
df_main.head(10)

# %%
# Symptom columns
symptom_cols = [c for c in df_main.columns if c.startswith("Symptom_")]
print(f"Số cột symptom: {len(symptom_cols)}")

# Đếm số triệu chứng mỗi bệnh (bỏ NaN)
df_main["n_symptoms"] = df_main[symptom_cols].notna().sum(axis=1)
print("\nSố triệu chứng / bệnh:")
print(df_main["n_symptoms"].describe())

# %%
fig, ax = plt.subplots(figsize=(8, 4))
df_main["n_symptoms"].value_counts().sort_index().plot.bar(ax=ax, color="steelblue", edgecolor="white")
ax.set_xlabel("Số triệu chứng")
ax.set_ylabel("Số bệnh")
ax.set_title("Phân bố số triệu chứng / bệnh (Bộ 2)")
plt.tight_layout()
plt.savefig("../output/eda_bo2_symptoms_per_disease.png", dpi=100)
plt.show()

# %%
# Melt để xem triệu chứng phổ biến
melted = df_main.melt(
    id_vars=["Disease"],
    value_vars=symptom_cols,
    value_name="symptom_raw"
).dropna(subset=["symptom_raw"])

melted["symptom_norm"] = melted["symptom_raw"].apply(normalize_symptom_name)
melted = melted[melted["symptom_norm"].str.len() > 0]

print(f"\nTổng cặp (Disease, Symptom): {len(melted)}")
print(f"Symptom unique: {melted['symptom_norm'].nunique()}")

# Top 30 triệu chứng phổ biến nhất
top_symptoms = melted["symptom_norm"].value_counts().head(30)
fig, ax = plt.subplots(figsize=(10, 6))
top_symptoms.plot.barh(ax=ax, color="salmon")
ax.set_xlabel("Số bệnh có triệu chứng này")
ax.set_title("Top 30 triệu chứng phổ biến nhất (Bộ 2)")
plt.tight_layout()
plt.savefig("../output/eda_bo2_top_symptoms.png", dpi=100)
plt.show()

# %% [markdown]
# ## 2. Symptom-severity.csv — Mức độ nghiêm trọng

# %%
df_sev = pd.read_csv(BO2_SEVERITY_CSV)
print(f"Shape: {df_sev.shape}")
print(df_sev.head(10))

# %%
print("\nPhân bố weight:")
print(df_sev["weight"].value_counts().sort_index())

fig, ax = plt.subplots(figsize=(7, 4))
df_sev["weight"].value_counts().sort_index().plot.bar(
    ax=ax, color="mediumseagreen", edgecolor="white"
)
ax.set_xlabel("Severity Weight (1–7)")
ax.set_ylabel("Số triệu chứng")
ax.set_title("Phân bố mức độ nghiêm trọng triệu chứng")
plt.tight_layout()
plt.savefig("../output/eda_bo2_severity_dist.png", dpi=100)
plt.show()

# %%
# Top 10 triệu chứng nặng nhất (weight cao)
print("Top 10 triệu chứng nghiêm trọng nhất:")
print(df_sev.nlargest(10, "weight")[["Symptom", "weight"]].to_string(index=False))

# %% [markdown]
# ## 3. symptom_Description.csv — Mô tả bệnh

# %%
df_desc = pd.read_csv(BO2_DESC_CSV)
print(f"Shape: {df_desc.shape}")
print("Sample mô tả:")
for _, r in df_desc.head(5).iterrows():
    print(f"\n  {r['Disease']}: {r['Description'][:120]}...")

# %% [markdown]
# ## 4. symptom_precaution.csv — Cách phòng ngừa

# %%
df_prec = pd.read_csv(BO2_PRECAUTION_CSV)
print(f"Shape: {df_prec.shape}")
print(df_prec.head(10))

# %%
# Số precaution mỗi bệnh
prec_cols = [c for c in df_prec.columns if c.startswith("Precaution_")]
df_prec["n_precautions"] = df_prec[prec_cols].notna().sum(axis=1)
print("Số precaution / bệnh:")
print(df_prec["n_precautions"].value_counts().sort_index())

# %% [markdown]
# ## 5. Overlap với Bộ 1 — Entity Resolution opportunities

# %%
from src.config import BO1_CSV

df_bo1 = pd.read_csv(BO1_CSV, low_memory=False)
bo1_diseases = set(df_bo1["medical_condition"].dropna().apply(normalize_text))
bo2_diseases = set(df_main["Disease"].dropna().apply(normalize_text))

exact_overlap = bo1_diseases & bo2_diseases
fuzzy_candidates = []

# Check tên gần giống nhau (chứa cùng từ đầu)
for d2 in bo2_diseases:
    for d1 in bo1_diseases:
        if d2 != d1 and (d2 in d1 or d1 in d2):
            fuzzy_candidates.append((d2, d1))

print(f"Bộ 1 có {len(bo1_diseases)} bệnh unique")
print(f"Bộ 2 có {len(bo2_diseases)} bệnh unique")
print(f"Overlap chính xác: {len(exact_overlap)} bệnh")
print(f"Fuzzy match candidates: {len(fuzzy_candidates)} cặp")
print("\nMột số cặp cần Entity Resolution:")
for a, b in fuzzy_candidates[:15]:
    print(f"  '{a}' ←→ '{b}'")

# %% [markdown]
# ## 6. Tóm tắt EDA Bộ 2
#
# | Metric | Giá trị |
# |--------|---------|
# | Số bệnh | 41 |
# | Cặp (Disease, Symptom) | ~4.900 |
# | Symptom unique | ~130 |
# | Overlap chính xác với Bộ 1 | ~X |
#
# **Lưu ý cho ingest (Tuần 2):**
# - Tên triệu chứng viết dạng snake_case → cần normalize
# - Symptom severity (1-7) → thuộc tính edge HAS_SYMPTOM
# - ~41 bệnh overlap một phần với Bộ 1 → cần ER

print("EDA Bộ 2 hoàn tất.")
