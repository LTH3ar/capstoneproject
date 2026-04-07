# %% [markdown]
# # EDA — Bộ 3: Medical Transcriptions
# **Tuần 1** — Khám phá `medical_transcriptions/mtsamples.csv`
#
# Đây là nguồn dữ liệu phi cấu trúc chính của dự án.
# Mục tiêu: hiểu phân bố chuyên khoa, độ dài text, và chất lượng dữ liệu.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from src.config import BO3_CSV
from src.utils import normalize_text, safe_str

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_colwidth", 100)

df = pd.read_csv(BO3_CSV, index_col=0)
print(f"Shape: {df.shape}")
print(f"Cột: {df.columns.tolist()}")

# %% [markdown]
# ## 1. Missing values & tổng quan

# %%
missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(1)
print("Missing values:")
print(pd.DataFrame({"missing": missing, "pct": missing_pct}))

# %%
# Bản ghi hợp lệ (có transcription)
valid = df.dropna(subset=["transcription"])
valid = valid[valid["transcription"].str.strip().str.len() > 50]
print(f"\nTổng dòng: {len(df)}")
print(f"Dòng có transcription hợp lệ: {len(valid)}")
print(f"Dòng bị loại: {len(df) - len(valid)}")

# %% [markdown]
# ## 2. Phân bố chuyên khoa (medical_specialty)

# %%
specialty_counts = df["medical_specialty"].value_counts()
print(f"\nSố chuyên khoa unique: {specialty_counts.shape[0]}")
print("\nTop 20 chuyên khoa:")
print(specialty_counts.head(20))

# %%
fig, ax = plt.subplots(figsize=(12, 6))
specialty_counts.head(20).plot.barh(ax=ax, color="steelblue")
ax.set_xlabel("Số bản ghi")
ax.set_title("Top 20 Medical Specialty (Bộ 3)")
plt.tight_layout()
plt.savefig("../output/eda_bo3_specialty_dist.png", dpi=100)
plt.show()

# %% [markdown]
# ## 3. Phân bố độ dài văn bản

# %%
df["_n_chars"] = df["transcription"].apply(lambda x: len(safe_str(x)))
df["_n_words"] = df["transcription"].apply(
    lambda x: len(safe_str(x).split()) if pd.notna(x) else 0
)
df["_n_sentences"] = df["transcription"].apply(
    lambda x: len(re.split(r"[.!?]+", safe_str(x))) if pd.notna(x) else 0
)

print("Độ dài transcription (ký tự):")
print(df["_n_chars"].describe())
print("\nSố từ / bản ghi:")
print(df["_n_words"].describe())

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df["_n_words"][df["_n_words"] > 0].clip(upper=3000).plot.hist(
    bins=50, ax=axes[0], color="steelblue", edgecolor="white"
)
axes[0].set_xlabel("Số từ")
axes[0].set_title("Phân bố độ dài (số từ)")

df["_n_words"][df["_n_words"] > 0].clip(upper=3000).plot.box(
    ax=axes[1], color="steelblue"
)
axes[1].set_title("Boxplot độ dài (số từ)")

plt.tight_layout()
plt.savefig("../output/eda_bo3_text_length.png", dpi=100)
plt.show()

# %% [markdown]
# ## 4. Phân tích Keywords

# %%
all_keywords = []
for kw_str in df["keywords"].dropna():
    kws = [normalize_text(k) for k in str(kw_str).split(",")]
    all_keywords.extend([k for k in kws if k])

kw_series = pd.Series(all_keywords).value_counts().head(40)
print(f"Tổng keywords: {len(all_keywords)}")
print(f"Unique keywords: {pd.Series(all_keywords).nunique()}")
print("\nTop 40 keywords:")
print(kw_series.to_string())

# %%
fig, ax = plt.subplots(figsize=(10, 6))
kw_series.head(25).plot.barh(ax=ax, color="salmon")
ax.set_xlabel("Tần suất")
ax.set_title("Top 25 Keywords phổ biến nhất (Bộ 3)")
plt.tight_layout()
plt.savefig("../output/eda_bo3_top_keywords.png", dpi=100)
plt.show()

# %% [markdown]
# ## 5. Sample transcription — đọc tay

# %%
# Đọc thủ công 5 bản ghi từ các chuyên khoa khác nhau
for specialty in ["Cardiology", "Neurology", "Orthopedic", "Gastroenterology", "Psychiatry"]:
    subset = df[df["medical_specialty"] == specialty]
    if len(subset) == 0:
        continue
    row = subset.sample(1, random_state=42).iloc[0]
    print(f"\n{'='*60}")
    print(f"SPECIALTY: {specialty}")
    print(f"Description: {row['description']}")
    print(f"Keywords: {str(row['keywords'])[:100]}")
    print(f"Transcription (first 400 chars):\n  {str(row['transcription'])[:400]}")

# %% [markdown]
# ## 6. Tìm viết tắt y khoa phổ biến

# %%
# Đếm tần suất các cụm viết hoa ngắn (2-5 ký tự) trong transcription — gợi ý abbreviation
abbrev_pattern = re.compile(r"\b[A-Z]{2,5}\b")
all_abbrevs = []

for text in df["transcription"].dropna().sample(500, random_state=42):
    abbrevs = abbrev_pattern.findall(str(text))
    all_abbrevs.extend(abbrevs)

abbrev_counts = pd.Series(all_abbrevs).value_counts().head(30)
print("Top 30 viết tắt tìm thấy trong transcription:")
print(abbrev_counts.to_string())

# %% [markdown]
# ## 7. Phân bố theo chuyên khoa × độ dài

# %%
top_specialties = specialty_counts.head(10).index
df_top = df[df["medical_specialty"].isin(top_specialties)].copy()

fig, ax = plt.subplots(figsize=(12, 5))
specialty_order = df_top.groupby("medical_specialty")["_n_words"].median().sort_values(ascending=False).index
df_top.boxplot(column="_n_words", by="medical_specialty", ax=ax,
               order=specialty_order.tolist())
ax.set_xlabel("Specialty")
ax.set_ylabel("Số từ")
ax.set_title("Phân bố độ dài transcription theo chuyên khoa")
plt.suptitle("")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("../output/eda_bo3_length_by_specialty.png", dpi=100)
plt.show()

# %% [markdown]
# ## 8. Tóm tắt EDA Bộ 3
#
# | Metric | Giá trị |
# |--------|---------|
# | Tổng bản ghi | ~5.000 |
# | Bản ghi có transcription | ~4.900+ |
# | Số chuyên khoa | ~40 |
# | Độ dài trung bình | ~500-1.500 từ |
#
# **Lưu ý cho NER (Tuần 3):**
# - Viết tắt phổ biến: HTN, DM, CAD, COPD... cần dictionary-based NER bổ sung
# - Chuyên khoa Surgery chiếm phần lớn → nhiều tên bệnh liên quan đến phẫu thuật
# - Độ dài không đều → một số doc rất ngắn, nên filter < 50 từ trước khi NER
# - Keywords là weak supervision tốt để đánh giá recall của NER

print("EDA Bộ 3 hoàn tất.")
