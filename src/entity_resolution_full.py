import pandas as pd
import numpy as np
import jellyfish
import string
import re
import time
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_score, recall_score, f1_score
import os

# ==========================================
# CẤU HÌNH & HẰNG SỐ
# ==========================================
MODEL_NAME = 'all-MiniLM-L6-v2'
WEIGHT_JARO = 0.4
WEIGHT_COSINE = 0.6
THRESHOLD_AUTO_MERGE = 0.85 # Hạ xuống một chút so với 0.92 để bắt được nhiều cặp hơn dựa trên F1 test
THRESHOLD_REVIEW = 0.80
STOPWORDS_MED = ['acute', 'chronic', 'severe', 'mild', 'history of']

# Từ điển ép kiểu (Quy tắc ưu tiên cao nhất để sửa lỗi của AI)
MANUAL_DICTIONARY = {
    "tylenol": "acetaminophen", "lovenox": "enoxaparin", "coumadin": "warfarin",
    "dilantin": "phenytoin", "percocet": "oxycodone/acetaminophen",
    "augmentin": "amoxicillin/clavulanate", "micardis": "telmisartan",
    "cardizem": "diltiazem", "bumex": "bumetanide", "imuran": "azathioprine",
    "atrovent": "ipratropium", "nexium": "esomeprazole", "pulmicort": "budesonide",
    "zonegran": "zonisamide", "maxzide": "triamterene/hydrochlorothiazide",
    "novolin": "insulin isophane", "levaquin": "levofloxacin", "colace": "docusate",
    "gerd": "gastroesophageal reflux disease", "shortness of breath": "dyspnea",
    "renal problems": "kidney disease", "convulsive seizure": "seizure",
    "ptosis loss of vision": "vision loss"
}

# Các từ rác bị AI nhận nhầm
JUNK_WORDS = {"p.o", "po", "q.d", "qd", "40mgkgday", "throat", "palate", "perineum", "time3", "ble", "gu", "smoke"}

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    for word in STOPWORDS_MED:
        text = re.sub(rf'\b{word}\b', '', text).strip()
    return re.sub(r'\s+', ' ', text)

def get_block_key(text):
    if not text or len(text) < 3: return "MISC"
    return f"{text[:3]}_{jellyfish.metaphone(text.split()[0])}"

def calculate_jaro(text1, text2):
    return jellyfish.jaro_winkler_similarity(text1, text2)

print(f"[INFO] Đang tải mô hình Sentence-Transformer: {MODEL_NAME}...")
embedder = SentenceTransformer(MODEL_NAME)

def calculate_similarity(text1, text2, emb1, emb2):
    jaro_score = calculate_jaro(text1, text2)
    cosine_score = util.cos_sim(emb1, emb2).item()
    cosine_score = max(0.0, min(1.0, cosine_score))
    return (WEIGHT_JARO * jaro_score) + (WEIGHT_COSINE * cosine_score)

# ==========================================
# CHẠY ER TRÊN TOÀN BỘ DỮ LIỆU
# ==========================================
def run_full_pipeline(mentions_path='output/mentions.parquet'):
    print("\n[INFO] === BẮT ĐẦU CHẠY ER TRÊN TOÀN BỘ MENTIONS ===")
    df = pd.read_parquet(mentions_path)
    
    # Lấy danh sách các entity unique
    unique_entities = df['entity_text'].dropna().unique()
    print(f"[INFO] Tổng số unique entities cần xử lý: {len(unique_entities)}")

    # 1. Tiền xử lý & Lọc rác
    valid_entities = []
    for ent in unique_entities:
        clean_ent = clean_text(ent)
        if clean_ent and clean_ent not in JUNK_WORDS and len(clean_ent) > 2:
            valid_entities.append({'raw': ent, 'clean': clean_ent})
            
    df_valid = pd.DataFrame(valid_entities)
    print(f"[INFO] Còn lại {len(df_valid)} entities sau khi lọc rác.")

    # 2. Tạo Blocking
    df_valid['block_key'] = df_valid['clean'].apply(get_block_key)
    blocks = df_valid.groupby('block_key')
    print(f"[INFO] Tạo được {len(blocks)} blocks (giảm tải tính toán đáng kể).")

    # 3. Tính toán Matching trong từng block
    print("[INFO] Đang embedding toàn bộ danh sách...")
    all_embeddings = embedder.encode(df_valid['clean'].tolist(), convert_to_tensor=True)
    df_valid['embedding_idx'] = range(len(df_valid))
    
    edges = []
    print("[INFO] Đang ghép cặp (Matching) bên trong các blocks...")
    
    for name, group in blocks:
        idxs = group['embedding_idx'].tolist()
        cleans = group['clean'].tolist()
        
        # So sánh chéo các phần tử trong cùng 1 block
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                idx1, idx2 = idxs[i], idxs[j]
                text1, text2 = cleans[i], cleans[j]
                
                # Bỏ qua nếu là chuỗi con của nhau (ví dụ: "pain" vs "back pain") thì thường không nên merge
                if text1 in text2 or text2 in text1:
                    continue
                    
                score = calculate_similarity(text1, text2, all_embeddings[idx1], all_embeddings[idx2])
                if score >= THRESHOLD_AUTO_MERGE:
                    edges.append((text1, text2, score))

    print(f"[INFO] Tìm thấy {len(edges)} cặp có độ tương đồng >= {THRESHOLD_AUTO_MERGE}")

    # 4. Gộp cụm (Clustering) bằng NetworkX
    G = nx.Graph()
    G.add_nodes_from(df_valid['clean'].tolist())
    G.add_weighted_edges_from(edges)

    # Lấy các thành phần liên thông (mỗi cụm là các từ đồng nghĩa)
    clusters = list(nx.connected_components(G))
    
    # Tạo mapping: mention -> canonical
    mapping_dict = {}
    
    # Map từ Điển thủ công trước
    for raw in unique_entities:
        clean = clean_text(raw)
        if clean in MANUAL_DICTIONARY:
            mapping_dict[raw] = MANUAL_DICTIONARY[clean]
            
    # Map từ AI Clusters
    for cluster in clusters:
        cluster = list(cluster)
        # Chọn từ ngắn nhất làm tên chuẩn (canonical)
        canonical = sorted(cluster, key=len)[0]
        for term in cluster:
            # Chỉ map nếu chưa được map bằng từ điển
            if term not in MANUAL_DICTIONARY:
                # Gán tất cả các từ trong cluster về raw entity gốc
                # Ở đây ta có term (clean). Phải map về các raw tương ứng
                pass # Sẽ xử lý ở vòng lặp dưới
                
    # Gán vào dictionary hoàn chỉnh
    clean_to_canonical = {}
    for cluster in clusters:
        cluster = list(cluster)
        canonical = sorted(cluster, key=len)[0]
        for term in cluster:
            clean_to_canonical[term] = canonical

    for raw, clean in zip(df_valid['raw'], df_valid['clean']):
        if raw not in mapping_dict: # Ưu tiên từ điển
            if clean in clean_to_canonical:
                mapping_dict[raw] = clean_to_canonical[clean]
            else:
                mapping_dict[raw] = clean # Tự map về chính nó nếu không thuộc cụm nào

    # Các từ rác thì map về "SKIP"
    for junk in JUNK_WORDS:
        mapping_dict[junk] = "SKIP"

    # Xuất ra file
    df_mapping = pd.DataFrame(list(mapping_dict.items()), columns=['entity_raw', 'canonical_entity'])
    os.makedirs('output', exist_ok=True)
    df_mapping.to_csv('output/er_mapping.csv', index=False)
    
    print("\n======================================")
    print(f"✅ Đã tạo bảng Mapping ER thành công!")
    print(f"✅ Đã lưu tại: output/er_mapping.csv ({len(df_mapping)} dòng)")
    print("======================================")
    print("Bước tiếp theo (Tuần 6): Gộp toàn bộ vào Neo4j (build_kg.py)")

if __name__ == "__main__":
    start_time = time.time()
    
    # Bỏ comment phần này nếu muốn chạy lại đánh giá
    # evaluate_gold_set('output/gold_set_annotated.csv')
    
    # Chạy trên toàn bộ dữ liệu
    run_full_pipeline('output/mentions.parquet')
    
    print(f"\n[INFO] Hoàn thành trong {time.time() - start_time:.2f} giây.")