# Kế hoạch vẽ pipeline chi tiết — 06/11/2026

Phạm vi đã được người dùng yêu cầu: vẽ cấu hình PROMPT + CSA + SCST/CE + rerank hiện tại, không đề xuất hoặc chạy thêm mô hình. Xuất PNG độ phân giải cao và SVG chỉnh sửa được. Sơ đồ mở rộng phải giữ hai memory Vo/Vr, object1 → object2 → relation, Q/K/V và mask của hai adapter, hai cross-attention decoder; bank từ vẫn hiện diện. Tách luồng suy luận khỏi loss chỉ dùng TRAIN. Nguồn đối chiếu: model.py, prompt_adapter_model.py, prompt_scratch_model.py, prompt_scst_rerank.py, captioning/multiscore_reranking.py, các kết quả đã kiểm tra. Không gọi Q-Former/BLIP trong sơ đồ vì cấu hình hiện tại không dùng chúng.

Hoàn tất: PNG3000×2480 và SVG vector, có mã dựng lại. Đã so trực tiếp thứ tự CSA/adapter/decoder với mã nguồn; tự kiểm tra SVG XML, kích thước PNG, và xem bản render cuối để sửa glyph/đảm bảo chữ nằm trong hộp. Ghi nguồn/loss/rerank theo cấu hình hiện tại. Không chạy mô hình. Bản ảnh riêng chi tiết, chưa thay sơ đồ tóm tắt trong Word3trang.

Yêu cầu sửa hình theo phong cách bài báo: tạo hình vector dạng module và mũi tên, không nhét đoạn giải thích vào box. Ba panel: (a) toàn bộ luồng suy luận, vẫn có bank/CSA/prompt/decoder/rerank; (b) cấu trúc adapter với Q/K/V, mask và residual; (c) loss TRAIN. Chú thích ngoài hình nêu giới hạn. Giữ nguyên cơ chế đã chạy.
