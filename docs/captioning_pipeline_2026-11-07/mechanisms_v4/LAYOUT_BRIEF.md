# V4 — mở cơ chế theo brief người dùng

07/11/2026, Asia/Bangkok. Phạm vi: vẽ lại theo USER_BRIEF.txt mới, không đổi phương pháp hoặc chạy mô hình. Giữ V3.

- Architecture, static, `fit`, faithful, nhãn English; HTML nguồn tự chứa và xuất SVG/PDF/PNG.
- Luồng tổng thể trái→phải; khung chi tiết nét đứt A–F liên hệ bằng nhãn và guide. A bank/aggregation; B fusion chung; C hai adapter; D decoder; E reranking; F training riêng.
- Master lớn để chứa đủ cơ chế; xuất từng panel riêng cho khả năng đọc trong bài báo. Không tuyên bố master đầy đủ chi tiết sẽ đọc được khi thu vào một cột báo.
- Nền trắng, Arial/Liberation Sans, pastel thị giác/decoder #EDF3F8, object #FCEBDD, relation #E2F1F5. Module bo góc; data vuông. Tuyết xanh/lửa cam vector. Các yêu cầu người dùng về palette/font, mật độ kỹ thuật và góc vuông ưu tiên hơn mặc định skill.
- Mọi connector chỉ ngang/dọc, góc vuông; ưu tiên một đoạn. Cổng tensor cùng tên là cùng dữ liệu, dùng để tránh bus dài cắt nhau. Zoom guide nét đứt không có arrowhead; không phải data flow.
- MLP inputs dùng dấu chấm phẩy/Concat, không biểu diễn cộng vector. Slot builder dùng Vo gốc. Hai memory vào decoder riêng; adapter không thêm gate.
- Decoder giữ e thô cho gate, sqrt(512)e + PE cho input; FFN residual là từ gated sum, không tự thêm H vào sum. D/C min–max trên 5; beam sum logp, không length normalization. Training XE rồi XE+.05SCST; không object reward.
- Kiểm tra XML/nhãn, tuyến, giao/chồng/transit, đo text/icon Chromium, nhìn render, PDF vector; chỉ static/graphics.

Nguồn đối chiếu: experiments/csa_paper/model.py, prompt_adapter_model.py, prompt_adapter_data.py, prepare_real_assets.py; captioning/multiscore_reranking.py; experiments/csa_paper/scst_pilot_model.py và run_prompt_scst.py. Không import các module neural.
