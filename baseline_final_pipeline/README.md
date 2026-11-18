# Baseline và pipeline cuối

Mở folder này bằng VS Code. Các file Python được lấy theo phiên bản notebook đã khóa; mã nhúng trong notebook đã được xuất thành file riêng. SOURCE_MANIFEST.json ghi nguồn và SHA256 của từng file.

| Folder | Nội dung |
|---|---|
| baseline/train | Baseline CLIP ViT-B/16 + Transformer Decoder, lượt XE10 V3 của checkpoint199. |
| baseline/test | Sinh caption và đánh giá TEST5000 của checkpoint199. |
| final_pipeline/xe | Bank + prompt đối tượng–quan hệ, hai luồng memory và decoder. Runner FP32 mã hóa từng caption, H5 trực tiếp, seed43. |
| final_pipeline/scst | Học tiếp SCST thuần đến update1500 và đánh giá/rerank trên VALID. Snapshot lượt đã báo cáo. |
| final_pipeline/test_rerank | Sinh caption, tính điểm CLIP/nhất quán đối tượng, rerank và đánh giá TEST của checkpoint SCST1500. |
| preparation | File hỗ trợ tạo bank và cache prompt, tham chiếu dữ liệu và công cụ trong repo gốc. |
| notebooks | Notebook của các giai đoạn, seed43/44 và hai bản resume API. |

Đọc baseline/train/patch_model.py để hiểu baseline. Sau đó đọc các file theo thứ tự dưới đây.

| Thứ tự | File | Phần cần đọc |
|---|---|---|
| 1 | final_pipeline/xe/csa_patch_model.py | CSAPatchCaptioner.streams(), encode(), decode(), forward(). Đây là điểm ghép pipeline. |
| 2 | final_pipeline/xe/bank_layers.py | SemanticAggregation, Alignment, Fusion, CSACaptioner.encode() và DecoderLayer. M1 cung cấp bank cố định; aggregation và M2 được học trong mạng. File được xuất từ experiments/csa_paper/model.py. |
| 3 | final_pipeline/xe/semantic_patch_model.py | PromptMemory, gồm Slot MLP, masked attention và residual của M3. |
| 4 | final_pipeline/xe/semantic_data.py | Prompt/cache đầu vào, box → trọng số trên lưới patch, SemanticContext. |
| 5 | final_pipeline/xe/csa_full_runtime.py | make_model() nạp bank và khởi tạo mô hình. |
| 6 | final_pipeline/xe/full_data.py và csa_full_data.py | Đọc H5, batch caption, lịch học và sampler theo seed. |
| 7 | final_pipeline/xe/run_csa_full.py | Train XE10, chọn best bằng VALID CIDEr, checkpoint và resume. |
| 8 | final_pipeline/scst/run_csa_per_caption_scst_pilot.py | Sampling, greedy baseline, thưởng CIDEr và loss SCST của lượt1500. |
| 9 | final_pipeline/test_rerank/patch_beam.py | Beam search. |
| 10 | final_pipeline/test_rerank/run_csa_per_caption_rerank.py và captioning/multiscore_reranking.py | Chấm điểm và chọn caption cuối. |

Luồng tổng thể là patch ảnh + bank + prompt riêng ảnh → căn chỉnh bank → căn chỉnh prompt → decoder → beam search → rerank. XE và SCST là các giai đoạn huấn luyện.

Các file cùng tên trong từng folder phản ánh snapshot của giai đoạn đó. Ví dụ run_csa_full.py trong final_pipeline/xe là bản dành cho seed43. Nguồn huấn luyện gốc của checkpoint dùng cho SCST được khóa trong notebook tương ứng. Seed43/44 hiện bổ sung kết quả XE, chưa có kết quả SCST/TEST của hai seed này.

Để chạy thực nghiệm, dùng notebook trên Kaggle với dữ liệu, checkpoint và Secrets tương ứng. Notebook giữ cơ chế tải nguồn đã cấu hình. Bản tập hợp đã được kiểm tra SHA256 và cú pháp Python; việc sao chép folder chưa được kiểm chứng bằng một lượt chạy mô hình local.
