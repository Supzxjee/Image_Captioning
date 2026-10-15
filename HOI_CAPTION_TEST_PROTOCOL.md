# Q-Former + UPT: independent test 5.000 ảnh

Mở [notebook đầy đủ](HOI_CAPTION_TEST_FULL.ipynb) trên Kaggle hoặc dán toàn bộ [cell Python](kaggle_hoi_caption_test_full.py). **Người dùng tự bấm chạy**; trợ lý chưa khởi chạy test.

Settings: **GPU T4 x2**, Internet ON. Add Input cùng năm dataset của lượt validation:

- `vuthetam/mscoco-2014`
- `ducanh2403/qformer-epoch10`
- `ducanh2403/prompt-cache`
- `ducanh2403/visual-cache`
- `ducanh2403/objectdetectionecache`

Đường dẫn ghi rõ đầu cell, không tìm tự động trong toàn bộ Input. Dùng Q-Former gốc epoch10 và prompt cache gốc; không thay bằng E3 hay prompt relation V2.1. Không cần gắn output validation; quyết định chọn λ đã lưu trong GitHub.

Core pin `1c48b50b18111a33628670957df3dfdd2d0eb68a`. Test dùng đúng split `test` và DataFrame test, cùng beam5 cho ba nhánh; hai worker GPU độc lập. CLIP scoring chạy trên một GPU sau beam generation, như validation. UPT weights (~220MB), CLIP và reference code được tải nếu chưa có cache.

Giữ decoder/CLIP/OCC=.3/.4/.3 và **λ=.1** từ [validation](HOI_CAPTION_VALIDATION_RESULT.md). Kiểm tra SHA256 file quyết định, checkpoint, prompt cache, dataset JSON, detector cache và UPT weights trước inference. Không dò grid hay chọn mô hình theo test. Reference captions chỉ dùng chấm điểm. Giữ ảnh không có bằng chứng UPT với feature trung tính.

Xuất BLEU1–4, METEOR, ROUGE-L, CIDEr, CHAIRs/i, Recall và số hallucination sửa/tạo thêm của cả ba nhánh. Gửi lại **`/kaggle/working/qformer_upt_test5000_locked_review.zip`** khi summary có `TEST_COMPLETE`, count5000, split test, retuned_on_test=false. ZIP không chứa checkpoint/HDF5.

Kiểm tra local: 8 tests PASS cho parser/contrast, baseline λ=0, merge/journal/reuse và khóa test. Integration test đổi reference và metric trả về vẫn giữ nguyên caption ba nhánh; gọi tính CIDEr grid trên test sẽ làm test thất bại. Chưa chạy full GPU/test tại thời điểm bàn giao.
