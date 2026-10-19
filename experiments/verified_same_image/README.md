# F — học căn chỉnh bằng cặp xác nhận trên cùng ảnh

Chạy **VERIFIED_SAME_IMAGE_F_2GPU.ipynb**: một notebook, một nhánh F, **T4 ×2**; bật Internet rồi **Save & Run All**. Notebook chưa được chạy thay bạn. Không cần chạy lại A/E lúc này.

Bản sửa giới hạn Kaggle 1 MB: mã và nhãn được nén trong notebook rồi giải nén vào `/tmp` khi chạy, có kiểm tra SHA-256. Không đổi dữ liệu, loss hay cấu hình; không cần thêm Input. Dùng notebook mới thay bản cũ đã vượt giới hạn.

## Thay đổi gì?

Giữ module E: embedding cụm từ truy vấn các CLIP patch trong vùng người–vật; token bằng chứng thị giác được đưa vào decoder. Thay loss đối chiếu giả nhãn khác ảnh bằng cặp đúng/sai có nhãn V-COCO trên **cùng ảnh**:

- Sai quan hệ: giữ người/vật, thay hành động mà người đó có nhãn hành động bằng 0.
- Sai người: giữ vật/hành động, thay người có nhãn hành động bằng 0 trong ảnh.
- Không dùng thiếu role/thiếu annotation làm nhãn âm. Không tự tạo âm sai vật thể.

Sáu quan hệ: holding, riding, sitting on, carrying, eating, talking on phone. Cặp sai quan hệ cân bằng hai chiều hành động trong từng loại vật; tương thích vật–hành động chỉ lấy từ fit. Với sai người, chọn người âm có IoU với vật lớn nhất trong các âm đã xác nhận để giảm đối chứng quá dễ. Nhãn không được suy ra từ IoU.

`L = CE + w × SCST(CIDEr-D + 0.5 × object_count_v2) + 0.1 × max(0, 0.2 − s_đúng + s_sai)`

`w=0` trong 200 bước đầu, sau đó `w=0.05`. Không cộng thêm loss giả nhãn E cũ. CLIP text đóng băng; base caption và module bằng chứng cùng học. Đây là đổi giám sát, **chưa phải kiến trúc mới được chứng minh**.

## Dữ liệu đã khóa

| Phần | Cặp | Ảnh | Sai quan hệ | Sai người | Sai vật |
|---|---:|---:|---:|---:|---:|
| Fit | 503 | 310 | 80 | 423 | 0 |
| Calibration | 140 | 79 | 24 | 116 | 0 |
| Probe cũ | 62 | 59 | 12 | 38 | 12 |

Ba nhóm không trùng ảnh. Fit/calibration lấy từ các nhóm V-COCO đã khóa trước, loại toàn bộ ảnh probe. Tất cả ảnh phụ thuộc caption train/restval; không lấy caption val/test vào fit. Các ảnh từng được backbone học caption và các bộ kiểm tra từng được nghiên cứu trước: **không gọi đây là test hoàn toàn mới**.

Mỗi lượt ảnh caption ghép một cặp fit phụ. Tổng 10.000 lượt cặp: 5.000 lượt sai quan hệ, 5.000 lượt sai người; lần lượt lặp khoảng 62–63 và 11–12 lần/cặp. Rủi ro overfit đáng kể. Số lượt caption và cohort giống A/E nhưng F có **thêm nhãn, ảnh phụ và tính toán**, không phải cùng tổng ngân sách giám sát.

## Input Kaggle

Giữ sáu Input của A/E; nhãn và mã mới đã nhúng trong notebook:

1. `qformer-scst-object-w05-legacy`: checkpoint nền SHA `5f8eead3...`.
2. `prompt-cache`: legacy `prompt_clip_tokens_cache.pt`.
3. `upt-prompts-full-all-actions-v2`: `prompt_cache.json` SHA `24e7861a...`.
4. `visual-cache`: CLIP visual HDF5.
5. `mscoco-2014`: `dataset_coco.json`.
6. `objectdetectionecache`: `objectdetectioncache.json`.

Đường dẫn đầy đủ trong cell đầu, sửa mount nếu chạy tài khoản Ngân; giữ nguyên nội dung tệp. **Không dùng checkpoint E/bin** làm warm-start F. Không cần raw UPT cache hoặc ảnh COCO để chạy nhánh này.

## Đánh giá và quyết định

- Caption: cùng train10k/val1000, 1.250 cập nhật, seed42, batch8, LR nền5e-6/nhánh1e-4 như E. Giữ checkpoint cuối; greedy, không rerank. So với A đã chạy: CIDEr **1.2055125991**, CHAIRs **52/1000**, recall **45.804%**; đồng thời kiểm tra lại caption nguồn `origin` có khớp phiên trước.
- Matcher: trước/sau học trên calibration/probe; đối chứng toàn ảnh, xóa đặc trưng ảnh, chỉ dùng IoU/khoảng cách box. Bộ probe sai vật chỉ kiểm tra chuyển giao.
- Muốn nâng quy mô: trước hết F phải tăng CIDEr mà không tăng CHAIRs so với A; một seed trên val tái sử dụng chỉ là tín hiệu sàng lọc. Không tuyên bố cải thiện đáng tin cậy chỉ từ con số nhỏ.
- Muốn nói căn chỉnh tốt hơn: cần đánh giá matcher trên ảnh tách fit và đối chứng hình học; loss giảm hoặc caption đổi chưa đủ. Nhãn GT box khi học khác UPT box khi suy luận vẫn là hạn chế.

Gửi lại `semantic_evidence_verified_pairs_F_review.zip`. Giữ riêng `checkpoint.pth` trong Saved Output; review ZIP không chứa checkpoint/HDF5. Checkpoint dùng legacy serialization để tránh Kaggle tự bung thành hàng trăm tệp.

## Kiểm tra đã thực hiện

Xem `local_validation.json`: kiểm tra chính sách nhãn, tách ảnh, lịch lấy mẫu, cấu trúc notebook và hash mã nhúng. Máy local không có PyTorch: chưa chạy model/DDP thật. Notebook bắt buộc kiểm tra synthetic CE/SCST/gradient, loss cặp và đồng bộ hai GPU trước huấn luyện. Không có kết quả F lúc chuẩn bị.
