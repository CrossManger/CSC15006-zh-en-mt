# prepare_vecalign.py
import json
import config
import preprocess
import embedding

def load_data_mock():
    # TODO: Thay hàm này bằng code đọc file json thật của bạn
    return [
        {
            "id": "REC_001",
            "zh_text": "这是一个测试。a. 第一步。b. 第二步。",
            "en_text": "This is a test. a. Step 1. b. Step 2."
        },
        {
            "id": "REC_002",
            "zh_text": "你好世界。",
            "en_text": "Hello World."
        }
    ] * 5 # Nhân bản lên demo

def main():
    # 1. Load Data
    records = load_data_mock() # Thay bằng hàm load file thật
    print(f"Total records to process: {len(records)}")

    # Lists tổng chứa toàn bộ corpus đã stitch
    all_src_sents = []
    all_tgt_sents = []
    
    # List lưu map để sau này biết câu thứ i thuộc record nào
    # Format: (sentence_index, record_id, is_barrier)
    src_map = [] 
    
    # 2. Preprocess & Stitching (Khâu dữ liệu & Chèn Rào chắn)
    print("--- Preprocessing & Segmenting ---")
    
    for rec in records:
        rec_id = rec['id']
        
        # Tách câu
        src_seg = preprocess.segment_text(rec['zh_text'], lang='zh')
        tgt_seg = preprocess.segment_text(rec['en_text'], lang='en')
        
        # Add vào list tổng
        for s in src_seg:
            all_src_sents.append(s)
            src_map.append(f"{rec_id}") # Lưu ID
            
        for s in tgt_seg:
            all_tgt_sents.append(s)
            
        # --- CRITICAL: CHÈN BARRIER ---
        # Chèn vào cả 2 bên để vecalign neo lại tại đây
        all_src_sents.append(config.BARRIER_TOKEN)
        all_tgt_sents.append(config.BARRIER_TOKEN)
        src_map.append("BARRIER") # Đánh dấu đây là rào chắn

    # 3. Export Text Files (Vecalign cần text để tham chiếu dòng)
    print("--- Saving Text Files ---")
    with open(config.SRC_TEXT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_src_sents))
        
    with open(config.TGT_TEXT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_tgt_sents))
        
    with open(config.MAP_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(src_map))

    # 4. Generate Embeddings
    print("--- Generating Embeddings ---")
    model = embedding.get_model()
    
    src_emb = embedding.create_embeddings(model, all_src_sents)
    tgt_emb = embedding.create_embeddings(model, all_tgt_sents)
    
    # 5. Save Binary Files
    embedding.save_binary(src_emb, config.SRC_EMB_FILE)
    embedding.save_binary(tgt_emb, config.TGT_EMB_FILE)
    
    print("\n[DONE] Data prepared successfully!")
    print(f"Dimensions: {src_emb.shape}")
    print("You can now run 'sh run_vecalign.sh'")

if __name__ == "__main__":
    main()