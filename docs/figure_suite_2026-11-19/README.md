# Bộ sáu hình

BỘ 6 HÌNH PIPELINE

PNG/: ảnh để chèn báo cáo.
SVG/: vector để phóng to hoặc chỉnh sửa.
DRAWIO/: file chỉnh sửa riêng từng hình.
Tat_ca_6_hinh.drawio: cả 6 hình trong 6 trang của cùng một file.
XEM_TRUOC.html: xem nhanh và mở từng hình.

01: M1 — Xây dựng semantic bank.
02: M2 — Căn chỉnh bank với thị giác.
03: M3 — Căn chỉnh prompt với thị giác.
04: Baseline CLIP ViT-B/16 + Transformer Decoder.
05: Pipeline đầy đủ, mở cơ chế M1/M2/M3.
06: Pipeline đầy đủ, dùng ô M1/M2/M3 thay phần bên trong ba module.

M3 riêng nhận prompt embeddings/ROI links và memory từ M2; quá trình tạo các đầu vào này nằm trong pipeline đầy đủ.
Baseline có 4 decoder layers và LayerNorm sau projection/cuối decoder; pipeline cuối có 3 layers với hai nhánh cross-attention. Hình ghi đúng hai kiến trúc hiện có.
Các ví dụ chữ trong bank/prompt chỉ minh họa cách biểu diễn, không phải đầu ra mới của mô hình cho ảnh này.
Không chạy mô hình hay chấm điểm, không sửa báo cáo Word trong tác vụ tạo hình này.


Nguồn và phạm vi tại DESIGN.md; metadata/SHA tại METADATA_CHECKS.json. Artifacts trong figures/.
