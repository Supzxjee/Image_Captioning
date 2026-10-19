# Hai notebook thử cơ chế bằng chứng ngữ nghĩa–thị giác

**Đã có [kết quả A/E](experiments/semantic_evidence_pilot/AE_RESULT.md): E chưa vượt đối chứng A; chưa khuyến nghị chạy full phiên bản này.** Các notebook dưới đây giữ nguyên để tái lập thí nghiệm đã hoàn thành.

Chạy đúng hai nhánh, mỗi notebook dùng cả hai GPU T4:

| Notebook | Nội dung |
| --- | --- |
| [A — đối chứng](experiments/semantic_evidence_pilot/SEMANTIC_EVIDENCE_A_2GPU.ipynb) | Huấn luyện tiếp checkpoint tốt nhất với CE + SCST + thưởng đối tượng |
| [E — cơ chế mới](experiments/semantic_evidence_pilot/SEMANTIC_EVIDENCE_E_2GPU.ipynb) | Cụm ngữ nghĩa truy vấn patch ảnh trong bbox; thêm loss đối chiếu cặp khớp/tráo |

**Import notebook vào Kaggle, chọn GPU T4 x2, Internet ON, Save & Run All.** Không chạy nối tiếp từ checkpoint pilot A/B/C/D. Cả hai bắt đầu từ `qformer_scst_object_w05_legacy.pth` đã bảo toàn.

Dùng lại sáu dataset hiện có: checkpoint `qformer-scst-object-w05-legacy`, legacy `prompt-cache`, UPT `upt-prompts-full-all-actions-v2`, `visual-cache`, MSCOCO JSON và `objectdetectionecache`. Đường dẫn nằm ở cell đầu, sửa đúng mount nếu tài khoản Ngân khác. Không cần ảnh gốc, raw UPT hoặc cache mới.

Cùng train10k / val1000 và 1250 cập nhật. E dùng `CE + scheduled(.05)*SCST + .1*L_align`, margin .2, từ đầu; đối chứng không có nhánh/loss mới. Loss căn chỉnh dùng giả nhãn UPT và cặp tráo có thể nhiễu, không phải nhãn quan hệ chuẩn. R query dùng toàn bộ cụm quan hệ, không chỉ động từ.

Gửi lại `semantic_evidence_pilot_A_review.zip` và `semantic_evidence_pilot_E_review.zip`; giữ checkpoint riêng trong Saved Output. Đây mới là mã thử nghiệm, chưa có metric mới. Mình không khởi chạy notebook thay bạn.

[Chi tiết module, loss, giới hạn và cách đánh giá](experiments/semantic_evidence_pilot/README.md).
