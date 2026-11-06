# Báo cáo nghiên cứu 06/11/2026

Bản Word 3 trang: pipeline đầy đủ có bank từ và prompt UPT–ROI; XE/SCST+CE/rerank; kết quả TEST5000; can thiệp zero/shuffle và giới hạn diễn giải. Số liệu lấy từ các RESULT.json đã kiểm tra trong experiments/csa_paper, không chấm lại mô hình.

- `Bao_cao_nghien_cuu_06_11_2026.docx`: bản giao người dùng.
- `build_report.py`: mã dựng Word và sơ đồ `pipeline.png` từ số liệu đã lưu.
- QA: dựng Word thành công; ba bảng và các mốc dữ liệu kiểm tra lại. render_docx.py không chạy được do máy không có LibreOffice; xuất PDF bằng Microsoft Word COM, chuyển thành PNG bằng Poppler và xem đủ cả 3 trang. Không tràn bảng, không trang trắng; sơ đồ và chú thích đọc được. PDF/PNG render chỉ là tệp QA local.
- Không chạy notebook Kaggle, mô hình hay metric scorer. Không thay đổi kết quả hoặc tạo thêm hàng Sheet cho việc viết báo cáo.
