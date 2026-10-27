# Chạy pilot căn chỉnh probability gain C/H/T

Hai notebook: `GROUNDED_GAIN_CH_2GPU.ipynb` và `GROUNDED_GAIN_T_2GPU.ipynb`.
Mỗi nhánh500update trên cùng2.000ảnh. C/H/T dùng cùng khởi tạo; H không tiếp tục từ C.
Đã điền25giờ/tài khoản theo thông tin bạn cung cấp. Chưa chạy mô hình trên máy local.

## 1. Chuẩn bị chung một lần bằng notebook C/H

Giữ `MODE='prepare'`, chọn **CPU**, bật Internet, rồi Save & Run All.
Gắn các dataset có các đường dẫn sau:

| Đầu vào | File/thư mục |
| --- | --- |
| Checkpoint A | `/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin` |
| Prompt gốc đã mã hóa | `/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt` |
| CLIP cache | `/kaggle/input/datasets/ducanh2403/visual-cache` |
| COCO JSON | `/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json` |
| Detector cache cho thưởng đối tượng cũ | `/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json` |

Không cần thư mục ảnh, UPT mới, prompt-text JSON hoặc API token.
Prepare chỉ chép3.000bộ đặc trưng có sẵn (2kTRAIN+1kDEV), tạo nhãn thưởng và lưu khởi tạo chung; không sinh caption hay huấn luyện.

Khi có `PREPARATION COMPLETE`, lưu **nguyên thư mục**
`/kaggle/working/grounded_gain_preparation` thành dataset private. Giữ `inputs.h5`,
`object_targets.json`, `gain_metadata.json.gz`, `initial_head.bin`, `receipt.json`.
ZIP review nhỏ không thay dataset này được.

## 2. Huấn luyện trên hai tài khoản

Chia sẻ dataset chuẩn bị cho Ngân và gắn cùng dataset đó vào cả hai notebook.
Hai bên đều cần checkpoint A và COCO JSON. Không cần gắn lại prompt-cache/visual-cache/detector-cache sau khi đã có bộ chuẩn bị.

- Notebook của bạn: C/H, đổi thành `MODE='train'`, chọn **T4×2**.
- Notebook của Ngân: T, giữ `MODE='train'`, chọn **T4×2**.
- Hai bên dùng cùng container Kaggle, bật Internet, Save & Run All; có thể chạy song song.

`PREPARED_ROOT=None` tự tìm đúng receipt. Nếu báo không tìm thấy, điền đường dẫn thư mục chứa `receipt.json`; không trỏ vào một file HDF5 hoặc review ZIP.
Các kiểm tra baseline, gradient, tích phân và hai GPU sẽ chạy trước train. Nếu thất bại, giữ log/ZIP và gửi lại; không tự hạ tolerance.

## 3. Gửi kết quả

Gửi đủ `grounded_gain_C_review.zip`, `grounded_gain_H_review.zip`, `grounded_gain_T_review.zip`.
Giữ riêng `checkpoint.bin` của từng nhánh và nguyên dataset chuẩn bị.
Checkpoint mới chỉ chứa nhánh bổ sung/optimizer, cần đi cùng đúng A khi dùng lại.

Mục tiêu là T tốt hơn A/C/H về caption, không chỉ loss căn chỉnh đẹp hơn.
Mức trần đã khóa: chuẩn bị30phút; preflight10phút/nhánh; train+end45phút/nhánh.
Đây là trần dừng, không phải dự báo thời gian. Lượt dừng vì tính toán chưa hoàn tất không được coi là kết quả chất lượng âm.

Gói này chưa có labelbank vùng ảnh cho held-out: phần kiểm tra grounding độc lập được ghi **UNAVAILABLE**.
Các chỉ số caption và attribution train chưa đủ chứng minh tính mới hoặc căn chỉnh đúng về mặt ngữ nghĩa.
