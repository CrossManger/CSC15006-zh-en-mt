# preprocess.py
import re
import pysbd

# Khởi tạo Segmenter (load 1 lần dùng mãi)
# Tiếng Trung có thể dùng logic riêng, nhưng pysbd vẫn hỗ trợ cơ bản hoặc dùng split
en_segmenter = pysbd.Segmenter(language="en", clean=False)

def clean_bullets(text):
    """
    Chèn xuống dòng trước các bullet points bị dính.
    VD: "Task A.b. Task B" -> "Task A.\nb. Task B"
    """
    if not text: return ""
    # Xử lý: Dấu chấm + (chữ/số) + dấu chấm (VD: .a. hoặc .1.)
    text = re.sub(r'(?<=\.)(?=[a-z0-9]+\.)', '\n', text)
    # Xử lý tiếng Trung (VD: 1. 2. sau dấu 。)
    text = re.sub(r'(?<=[。！？])(?=[0-9]+\.)', '\n', text)
    return text

def segment_text(text, lang='en'):
    """
    Input: Đoạn văn (str)
    Output: List các câu (List[str])
    """
    if not text or not isinstance(text, str):
        return []
    
    # 1. Clean format
    clean_text = clean_bullets(text)
    
    # 2. Tách câu
    # Split \n trước để tránh pysbd nối lại các dòng mình vừa tách
    raw_lines = clean_text.split('\n')
    final_sents = []
    
    for line in raw_lines:
        if not line.strip(): continue
        
        if lang == 'en':
            sents = en_segmenter.segment(line)
        else:
            # Với tiếng Trung, nếu pysbd ko chuẩn, dùng regex đơn giản
            # Ở đây tạm dùng regex split theo dấu câu lớn
            sents = re.split(r'(?<=[。！？])\s*', line)
            
        final_sents.extend([s.strip() for s in sents if s.strip()])
        
    return final_sents