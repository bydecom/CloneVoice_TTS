# run_tts_final.py
# Kịch bản hoàn chỉnh, sử dụng repo mới và cú pháp hiện đại.

import torch
import os
from TTS.api import TTS

# ---- PHẦN CÀI ĐẶT ----
TEXT_TO_SPEAK = "His remarks came as federal judges, including Judge Esther Salas, called for political leaders to tone down rhetoric due to an increase in intimidation and death threats against the judiciary."
REFERENCE_VOICE_PATH = "giong_mau.wav"
OUTPUT_FILE_PATH = "ket_qua.wav"

def main():
    print("--- BẮT ĐẦU QUÁ TRÌNH KHỞI TẠO CỖ MÁY TTS HIỆN ĐẠI ---")

    if not os.path.exists(REFERENCE_VOICE_PATH):
        print(f"LỖI: Không tìm thấy file giọng mẫu tại '{REFERENCE_VOICE_PATH}'!")
        return

    # Bước 1: Xác định thiết bị (cách làm chuẩn PyTorch)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sẽ sử dụng thiết bị: {device}")

    print("Đang tải mô hình XTTS-v2...")
    try:
        # Bước 2: Khởi tạo model và chuyển nó lên thiết bị đã chọn
        # Lưu ý: Không còn tham số 'gpu' nữa, thay bằng .to(device)
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print("Mô hình đã được tải và nạp thành công!")
    except Exception as e:
        print(f"Đã xảy ra lỗi nghiêm trọng khi tải mô hình: {e}")
        return

    print(f"\nBắt đầu quá trình clone giọng nói từ '{REFERENCE_VOICE_PATH}'...")
    tts.tts_to_file(
        text=TEXT_TO_SPEAK,
        file_path=OUTPUT_FILE_PATH,
        speaker_wav=REFERENCE_VOICE_PATH,
        language="en"
    )
    print("\n--- HOÀN THÀNH! ---")
    print(f"File âm thanh '{OUTPUT_FILE_PATH}' đã được tạo ra thành công.")


if __name__ == "__main__":
    main()