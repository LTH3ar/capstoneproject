# %% [markdown]
# # EDA — Bộ 1: Drugs, Side Effects & Medical Conditions
# **Tuần 1** — Khám phá dữ liệu để hiểu cấu trúc trước khi ingest.
#
# File CSV: `drugs-side-effects-and-medical-condition/drugs_side_effects_drugs_com.csv`
#
# Chạy file này trong VS Code (Jupyter extension) hoặc convert bằng jupytext.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import BO1_CSV
from src.utils import normalize_text, parse_comma_list, parse_side_effects, safe_str

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_colwidth", 80)

print(f"Đọc file: {BO1_CSV}")
df = pd.read_csv(BO1_CSV, low_memory=False)
print(f"Shape: {df.shape}")

# %% [markdown]
# ## 1. Tổng quan cấu trúc

# %%
print("Tên cột:")
print(df.columns.tolist())
print()
print("Dtype:")
print(df.dtypes)

# %%
# Missing values
missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(1)
missing_report = pd.DataFrame({"missing": missing, "pct": missing_pct})
print("Missing values:\n", missing_report[missing_report["missing"] > 0])

# %%
# 20 dòng đầu — cái nhìn tổng quát
df[["drug_name", "generic_name", "medical_condition", "rating",
    "drug_classes", "brand_names"]].head(20)

# %% [markdown]
# ## 2. Thống kê Drug

# %%
n_drugs = df["drug_name"].nunique()
n_generic = df["generic_name"].nunique()
print(f"Số drug_name unique: {n_drugs}")
print(f"Số generic_name unique: {n_generic}")

# %%
# Phân bố rating
fig, ax = plt.subplots(figsize=(8, 4))
df["rating"].dropna().plot.hist(bins=20, ax=ax, color="steelblue", edgecolor="white")
ax.set_xlabel("Rating (out of 10)")
ax.set_ylabel("Số thuốc")
ax.set_title("Phân bố Rating của thuốc (Bộ 1)")
plt.tight_layout()
plt.savefig("../output/eda_bo1_rating_dist.png", dpi=100)
plt.show()
print(df["rating"].describe())

# %%
# Top 20 drug theo số bệnh điều trị
top_drug_disease = (
    df.groupby("generic_name")["medical_condition"]
    .nunique()
    .sort_values(ascending=False)
    .head(20)
)
fig, ax = plt.subplots(figsize=(10, 5))
top_drug_disease.plot.barh(ax=ax, color="steelblue")
ax.set_xlabel("Số bệnh khác nhau được điều trị")
ax.set_title("Top 20 Drug điều trị nhiều bệnh nhất")
plt.tight_layout()
plt.savefig("../output/eda_bo1_top_drugs.png", dpi=100)
plt.show()

# %% [markdown]
# ## 3. Thống kê Disease

# %%
n_diseases = df["medical_condition"].nunique()
print(f"Số bệnh unique: {n_diseases}")

# Top 20 bệnh được nhiều thuốc điều trị nhất
top_diseases = (
    df.groupby("medical_condition")["generic_name"]
    .nunique()
    .sort_values(ascending=False)
    .head(20)
)
fig, ax = plt.subplots(figsize=(10, 5))
top_diseases.plot.barh(ax=ax, color="salmon")
ax.set_xlabel("Số thuốc điều trị")
ax.set_title("Top 20 bệnh được nhiều thuốc điều trị nhất")
plt.tight_layout()
plt.savefig("../output/eda_bo1_top_diseases.png", dpi=100)
plt.show()

# %% [markdown]
# ## 4. Brand Names & Drug Classes

# %%
# Phân bố số brand name mỗi thuốc
df["_n_brands"] = df["brand_names"].apply(
    lambda x: len(parse_comma_list(safe_str(x)))
)
print("Số brand name / thuốc:")
print(df["_n_brands"].describe())

# %%
# Top drug classes
all_classes = []
for val in df["drug_classes"].dropna():
    all_classes.extend(parse_comma_list(safe_str(val)))

class_series = pd.Series(all_classes).value_counts().head(20)
fig, ax = plt.subplots(figsize=(10, 5))
class_series.plot.barh(ax=ax, color="mediumseagreen")
ax.set_xlabel("Số thuốc thuộc nhóm")
ax.set_title("Top 20 Drug Classes phổ biến nhất")
plt.tight_layout()
plt.savefig("../output/eda_bo1_drug_classes.png", dpi=100)
plt.show()

# %% [markdown]
# ## 5. Side Effects (text analysis)

# %%
# Độ dài chuỗi side_effects
df["_se_len"] = df["side_effects"].apply(lambda x: len(safe_str(x)))
print("Độ dài chuỗi side_effects (ký tự):")
print(df["_se_len"].describe())

# %%
# Số side effect rút ra được sau parse
df["_n_se"] = df["side_effects"].apply(lambda x: len(parse_side_effects(safe_str(x))))
print("Số side effect sau parse / thuốc:")
print(df["_n_se"].describe())

fig, ax = plt.subplots(figsize=(8, 4))
df["_n_se"].clip(upper=40).plot.hist(bins=30, ax=ax, color="orchid", edgecolor="white")
ax.set_xlabel("Số side effect")
ax.set_ylabel("Số thuốc")
ax.set_title("Phân bố số Side Effect / thuốc (Bộ 1)")
plt.tight_layout()
plt.savefig("../output/eda_bo1_side_effects_dist.png", dpi=100)
plt.show()

# %%
# Top 30 side effects phổ biến nhất
all_se = []
for val in df["side_effects"].dropna():
    all_se.extend(parse_side_effects(safe_str(val)))

se_series = pd.Series(all_se).value_counts().head(30)
print("Top 30 side effects:")
print(se_series.to_string())

# %% [markdown]
# ## 6. Rx/OTC và Pregnancy Category

# %%
print("Phân bố rx_otc:")
print(df["rx_otc"].value_counts())
print()
print("Phân bố pregnancy_category:")
print(df["pregnancy_category"].value_counts())

# %% [markdown]
# ## 7. Sample dữ liệu — đọc tay

# %%
# 10 dòng ngẫu nhiên để đọc tay
sample = df.sample(10, random_state=42)
for _, r in sample.iterrows():
    print(f"Drug: {r['generic_name']} | Disease: {r['medical_condition']} | Rating: {r['rating']}")
    print(f"  Side effects (first 120 chars): {str(r['side_effects'])[:120]}")
    print(f"  Brand names: {str(r['brand_names'])[:80]}")
    print()

# %% [markdown]
# ## 8. Tóm tắt EDA Bộ 1
#
# | Metric | Giá trị |
# |--------|---------|
# | Tổng số dòng | ~2.900 |
# | Drug unique | ~900 |
# | Disease unique | ~950 |
# | Drug classes | ~200+ |
# | Side effect unique (parsed) | ~300+ |
#
# **Lưu ý cho ingest (Tuần 2):**
# - Cột `generic_name` có missing ~X% → fallback sang `drug_name`
# - Side effects là free text → cần parse bằng regex/split
# - Brand names và drug_classes cách nhau bởi dấu phẩy
# - Tên bệnh trong `medical_condition` sẽ overlap một phần với Bộ 2 → cơ hội ER

print("EDA Bộ 1 hoàn tất. Xem các file PNG trong thư mục output/")
