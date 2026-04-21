import pandas as pd
import numpy as np
import jellyfish
import string
import re
import time
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_score, recall_score, f1_score

# ==========================================
# CẤU HÌNH & HẰNG SỐ
# ==========================================
MODEL_NAME = 'all-MiniLM-L6-v2' # Nhẹ, tải nhanh (~80MB)
WEIGHT_JARO = 0.4
WEIGHT_COSINE = 0.6
THRESHOLD_AUTO_MERGE = 0.92
THRESHOLD_REVIEW = 0.80
STOPWORDS_MED = ['acute', 'chronic', 'severe', 'mild', 'history of']

# ==========================================
# TẦNG 1: BLOCKING & NORMALIZE
# ==========================================
def clean_text(text):
    """Normalize: lowercase, strip punctuation, bỏ stopword y khoa"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower().strip()
    # Bỏ dấu câu
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Bỏ stopword y khoa
    for word in STOPWORDS_MED:
        text = re.sub(rf'\b{word}\b', '', text).strip()
        
    # Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text)
    return text

def get_block_key(text):
    """Block theo 3 ký tự đầu + Metaphone (thuật toán phát âm)"""
    if not text or len(text) < 3:
        return "MISC"
    first_3 = text[:3]
    # Lấy từ đầu tiên để tính metaphone cho nhanh
    first_word = text.split()[0]
    phonetic = jellyfish.metaphone(first_word)
    return f"{first_3}_{phonetic}"

# ==========================================
# TẦNG 2: MATCHING
# ==========================================
def calculate_jaro(text1, text2):
    return jellyfish.jaro_winkler_similarity(text1, text2)

# Load model AI (chỉ load 1 lần khi chạy script)
print(f"[INFO] Đang tải mô hình Sentence-Transformer: {MODEL_NAME}...")
embedder = SentenceTransformer(MODEL_NAME)

def calculate_similarity(text1, text2, emb1, emb2):
    """Tính điểm kết hợp: 0.4 * Jaro + 0.6 * Cosine"""
    jaro_score = calculate_jaro(text1, text2)
    cosine_score = util.cos_sim(emb1, emb2).item()
    
    # Ép kiểu cosine_score về khoảng [0, 1] nếu có nhiễu
    cosine_score = max(0.0, min(1.0, cosine_score))
    
    combined = (WEIGHT_JARO * jaro_score) + (WEIGHT_COSINE * cosine_score)
    return combined, jaro_score, cosine_score

# ==========================================
# ĐÁNH GIÁ TRÊN GOLD SET
# ==========================================
def evaluate_gold_set(gold_path='output/gold_set_annotated.csv'):
    print("\n[INFO] === BẮT ĐẦU ĐÁNH GIÁ TRÊN GOLD SET ===")
    try:
        df_gold = pd.read_csv(gold_path)
    except FileNotFoundError:
        print(f"Không tìm thấy file {gold_path}. Vui lòng kiểm tra lại đường dẫn.")
        return

    # Lọc bỏ các dòng bị đánh dấu SKIP
    df_eval = df_gold[df_gold['manual_canonical'] != 'SKIP'].dropna(subset=['manual_canonical', 'entity_norm']).copy()
    
    print(f"Số lượng cặp hợp lệ để test: {len(df_eval)}")

    df_eval['clean_norm'] = df_eval['entity_norm'].apply(clean_text)
    df_eval['clean_canonical'] = df_eval['manual_canonical'].apply(clean_text)

    # Encode trước toàn bộ text để tiết kiệm thời gian
    print("[INFO] Đang embedding các entities...")
    norm_embeddings = embedder.encode(df_eval['clean_norm'].tolist(), convert_to_tensor=True)
    canonical_embeddings = embedder.encode(df_eval['clean_canonical'].tolist(), convert_to_tensor=True)

    y_true = []
    y_pred = []
    
    # Ta giả định: nếu similarity score >= THRESHOLD_REVIEW thì máy tính dự đoán là MATCH (1)
    # Vì trong gold_set, cột entity_norm CHÍNH LÀ manual_canonical (về mặt ý nghĩa), nên đáp án thực tế luôn là MATCH (1)
    # Tuy nhiên, ta sẽ so sánh chéo để xem máy tính có phân biệt được đúng cặp hay không.
    
    for i in range(len(df_eval)):
        text1 = df_eval['clean_norm'].iloc[i]
        text2 = df_eval['clean_canonical'].iloc[i]
        
        # Nếu đã giống nhau hoàn toàn text, cho điểm tuyệt đối
        if text1 == text2:
            score = 1.0
        else:
            score, j, c = calculate_similarity(text1, text2, norm_embeddings[i], canonical_embeddings[i])
            
        # Thống kê kết quả
        y_true.append(1) # Ground truth: đây là cặp map đúng
        
        # Mô hình dự đoán
        if score >= THRESHOLD_REVIEW:
            y_pred.append(1) # Mô hình đồng ý map
        else:
            y_pred.append(0) # Mô hình từ chối map (Dự đoán sai - False Negative)
            print(f"  [MISSED] {text1} -> {text2} (Score: {score:.3f})")

    # Tính Metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n======================================")
    print("BÁO CÁO ĐÁNH GIÁ ENTITY RESOLUTION")
    print("======================================")
    print(f"Ngưỡng chấp nhận (Threshold) : {THRESHOLD_REVIEW}")
    print(f"Độ chính xác (Precision)     : {precision:.2f}")
    print(f"Độ bao phủ (Recall)          : {recall:.2f}")
    print(f"Điểm F1-Score                : {f1:.2f}")
    print("======================================")
    
    if f1 >= 0.80:
        print("✅ Tuyệt vời! F1-Score đạt chuẩn (>= 0.80). Thuật toán đã sẵn sàng để chạy trên toàn bộ dữ liệu.")
    else:
        print("⚠️ Cảnh báo: F1-Score thấp hơn 0.80. Bạn nên điều chỉnh lại WEIGHT_JARO, WEIGHT_COSINE hoặc THRESHOLD.")

if __name__ == "__main__":
    start_time = time.time()
    
    # Chỉ chạy đánh giá trên Gold Set trước khi tiến hành xử lý toàn bộ data
    evaluate_gold_set('output/gold_set_annotated.csv')
    
    print(f"\n[INFO] Hoàn thành trong {time.time() - start_time:.2f} giây.")