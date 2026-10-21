# UPT hidden-pair caption pilot — chuẩn bị để người dùng chạy

Theo hướng Astra chủ trì: dùng đặc trưng tương tác bên trong UPT để bổ sung memory của caption decoder, học bằng caption tự nhiên. **Chưa chạy; chưa có kết quả hoặc novelty được chứng minh.**

1. Chạy [notebook cache11k, hai GPU](experiments/upt_hidden_pair/UPT_HIDDEN_CACHE_11K_2GPU.ipynb) với MSCOCO2014 JSON+ảnh. Internet ON, T4×2, Save&RunAll.
2. **Dùng API token:** chạy [notebook tạo Private Dataset qua API](experiments/upt_hidden_pair/SAVE_UPT_HIDDEN_CACHE_KAGGLE_API.ipynb), CPU + Internet ON, gắn Output cache đã hoàn tất, bật secret `KAGGLE_API_TOKEN`, sửa OWNER đúng tài khoản token. Kiểm tra hash trước upload sáu file cache nguyên byte; không chạy lại UPT. Cache upload nằm ngay gốc dataset, không có folder con `upt_hidden_pairs_11k`; A/H sửa `HIDDEN_CACHE` theo mount thực tế. Chỉ tạo dataset mới, không tự cập nhật dataset trùng slug.

   **Cách thủ công:** lưu **toàn bộ** folder `/kaggle/working/upt_hidden_pairs_11k` thành dataset. Có thể dùng [notebook đóng gói dataset, CPU](experiments/upt_hidden_pair/SAVE_UPT_HIDDEN_CACHE_AS_DATASET.ipynb): gắn Output cache đã hoàn tất, kiểm tra hash và sao chép sáu file payload vào folder gọn; không chạy lại UPT/API token. Sau đó tạo dataset từ Saved Output. Gửi `upt_hidden_pairs_11k_review.zip` để kiểm tra. ZIP review không chứa các HDF5 cần cho bước sau.
3. Thêm cache mới và các input baseline vào [A — đối chứng, hai GPU](experiments/upt_hidden_pair/UPT_HIDDEN_CAPTION_A_2GPU.ipynb) và [H — hidden-pair memory, hai GPU](experiments/upt_hidden_pair/UPT_HIDDEN_CAPTION_H_2GPU.ipynb). Sửa `HIDDEN_CACHE` đúng đường dẫn thực tế. Hai notebook có thể chạy song song trên hai tài khoản, mỗi notebook T4×2.
4. Gửi hai ZIP review A/H; giữ riêng hai checkpoint.bin. Không dùng adapter crop/phrase, không thay baseline theo kết quả một nhánh.

Input training: checkpoint qformer-scst-object-w05-legacy, prompt-cache legacy, visual-cache, dataset_coco.json, objectdetectionecache và cache hidden-pair mới. Chỉ extraction cần ảnh gốc; không cần UPT prompt JSON/full-score cache.

[Protocol, dataset paths, cache schema, kiểm tra và giới hạn](experiments/upt_hidden_pair/README.md). [Gói cả ba notebook](experiments/upt_hidden_pair/UPT_HIDDEN_3_NOTEBOOKS.zip).

Cùng train10k/val1000 đã dùng trước;1250updates,batch8,CE200 rồi CE+.05SCST(CIDErD+.5object_count_v2). UPT frozen chạy toàn bộ trước, chọn tối đa8 cặp có cả hai detection scores>0.2. H chỉ thêm Linear1024→512+LayerNorm, không gate/text/LoRA mới. Val là development, không dùng test chọn thiết kế.
