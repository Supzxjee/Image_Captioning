# Hình kiến trúc UPT–CSA bằng diagram-design

06/11/2026, UTC+7. Bản vẽ theo toàn bộ cấu hình trong [yêu cầu gốc](USER_BRIEF.txt), dùng skill `diagram-design` v2.6.142. Không thay thế hình cũ trong báo cáo Word; không sửa mô hình hoặc notebook.

## Tệp bàn giao

- [SVG chỉnh sửa được](upt-csa-architecture.svg): chữ vẫn là SVG text, module/đường nối có ID; mở bằng Inkscape, Illustrator hoặc Figma.
- [PDF vector](upt-csa-architecture.pdf): một trang, giữ chữ và hình vector, không nhúng raster.
- [PNG 4800×3680](upt-csa-architecture.png).
- [HTML độc lập](upt-csa-architecture.html): xem sơ đồ và caption tiếng Việt; dùng system fonts, không gọi Google Fonts.
- Panel riêng để dàn trang dễ đọc: [a: overall](panel-a-overall.svg), [b: adapters](panel-b-adapters.svg), [c: training](panel-c-training.svg). Mỗi panel đều có HTML, SVG, PDF và PNG cùng tên.

SVG cho phép chọn/sửa từng đối tượng, không phải trình kéo thả chuyên dụng. Để sửa mà tái tạo đồng bộ mọi định dạng, thay tọa độ/nhãn trong `build_figure.py`, rồi chạy lại hai script dựng/xuất. Nên đặt hình (a) ở toàn chiều rộng hai cột; nếu bản in làm chi tiết decoder khó đọc, dùng các panel rời, không thu toàn bộ hình vào một cột.

## Caption tiếng Việt

**Kiến trúc mô hình chú thích ảnh với CSA và prompt gắn vùng.** (a) CLIP và box YOLO cung cấp đặc trưng ROI; bank danh từ/động từ toàn cục đi qua hai tầng object fusion rồi relation fusion. Song song, prompt từ dự đoán UPT được mã hóa bằng CLIP và nối với ROI hợp lệ. Hai adapter cập nhật riêng memory đối tượng và quan hệ trước Transformer Decoder. Beam search sinh năm ứng viên, chọn caption bằng tổng có trọng số của điểm decoder, CLIPScore và độ nhất quán đối tượng. (b) Các adapter đọc slot qua mask và cập nhật residual; slot quan hệ dùng `Vo` gốc và cặp `(s,o)` có thứ tự. (c) Huấn luyện XE rồi XE + 0,05 SCST; advantage CIDEr dùng reference TRAIN. Điểm đối tượng `O` chỉ dùng khi reranking; CHAIR là metric hậu kiểm.

F: thành phần đóng băng; L: thành phần học được; dấu cộng tròn: residual sum. `Vo/Vr` và `Vo′/Vr′` là bốn tensor phân biệt. `Vr` được hướng dẫn bằng bank động từ, không biểu thị một scene graph đã xác minh. UPT/YOLO là các dự đoán có thể sai.

## Quy ước và chỗ giản lược

1. **Các cổng lặp tên biểu thị cùng dữ liệu**, để tránh dây dài hoặc giao cắt: `Pv` lấy từ global prototypes; `YOLO boxes` từ nhánh YOLO; `O-links` từ `Linked prompts`; `I + YOLO labels` lấy từ ảnh/YOLO ban đầu. Đây không phải encoder hoặc detector bổ sung. `R-links` đi thẳng từ `Linked prompts` vào relation adapter.
2. Ba Fusion block được gói theo giao diện vào/ra. Không vẽ inset nội bộ cosine alignment (temperature 0,1), visual SA/FFN, semantic CA và sigmoid gate; không thay đổi các phép đó. Selected noun prototypes ký hiệu `T1` vẫn đi từ Object Fusion 1 sang Object Fusion 2. `CA` là cross-attention, `SA` là self-attention.
3. Ba decoder layers dùng một khối lặp `×3`; hai CA song song, Q/K/V và gated fusion vẫn được ghi. Norm, dropout và chi tiết vocabulary head không khai triển. `e` là word embedding; `D=Σlog p` là raw decoder log-probability cộng dồn trên caption.
4. Mask trong panel (b) ghi phần điều kiện liên kết ROI; invalid/padding masks và thao tác bảo vệ softmax ở hàng rỗng được gói trong `Masked MHA`. Nhãn `No valid slot ⇒ ΔV=0` giữ hành vi fallback. Geometry object 4 chiều, relation 8 chiều; cả hai `Wout` zero-init.
5. Tiền xử lý ảnh/cache, chuẩn hóa lemma, lọc prompt và vocabulary tokenization được gộp thành module. UPT box fields gồm subject/object boxes; actions và relation scores được ghi ngắn thành `Actions · scores`.
6. SCST được ghi bằng công thức batch mean và tổng log-prob theo token. Luồng tính loss tách khỏi inference. Không vẽ object reward, CHAIR selection, retrieval, BLIP hoặc Q-Former.

Màu trắng/pastel và các loại thành phần do người dùng yêu cầu thay thế skin mặc định của skill. Ba panel và bản panel rời giữ mức chi tiết đã yêu cầu, thay cho giới hạn mặc định 9 nodes của một sơ đồ đơn giản. Font là Arial nếu có, fallback Liberation Sans; môi trường render dùng Liberation Sans và font ký hiệu hệ thống. Chữ tên module vẫn đen, màu không phải tín hiệu duy nhất.

## Nguồn và phạm vi kiểm tra

Nguồn chính là brief người dùng mô tả mô hình đã triển khai, không phải đề xuất phương pháp mới. Sau khi đồng bộ repo đến `9bcd820`, đã đối chiếu tĩnh `prompt_adapter_model.py`, `prompt_scst_rerank.py`, tài liệu adapter và caption hình đã có: hai adapter đều dùng `Vo` gốc để tạo slot, output projection zero-init, hai memory cấp riêng, selector cố định 0,3/0,4/0,3 và tie-break theo beam rank. Không coi việc đọc các file này là kiểm chứng toàn bộ triển khai hoặc chạy lại kết quả thực nghiệm.

`source_trace.json` ghi hash nguồn. Không chạy Torch/autograd/model, scorer hoặc Kaggle; không thay số liệu và không tạo hàng experiment registry cho việc vẽ.

## QA và tái tạo

- [QA hình học/PDF](figure_audit.json): 57 tuyến kết nối, 53 đối tượng node/sum; không phát hiện giao cắt, đoạn chồng, transit qua module khác, node đè nhau hoặc text tràn bounds.
- [Đo chữ trong Chromium](render_audit.json): cả hình tổng và ba panel không có text overflow.
- Skill `self_check.py` PASS cho 4 HTML; `verify-geometry.py` kiểm label masks PASS. Đã xem PNG tổng và PDF render; PDF có 220 vector drawings, 0 raster images, text chọn được.
- Xuất SVG bằng `scripts/export_svg.py --system-fonts`; PNG bằng `scripts/export_png.py` của skill, với adapter chọn Chromium hệ thống và local HTTP vì máy chưa cài browser cache của Playwright và trình duyệt quản lý không mở URL `file:`. Không sửa helper của skill.

```bash
python build_figure.py
python render_figure.py
python audit_figure.py
```

`render_figure.py` cần Playwright, Chromium hệ thống và skill ở đường dẫn ghi trong script; `audit_figure.py` dùng PyMuPDF. `build_figure.py` chỉ cần Python stdlib. Có thể chỉnh đường dẫn `SKILL` nếu cài ở máy khác. Báo cáo QA áp dụng cho artifact đã xuất, không bảo đảm mọi trình sửa SVG giữ nguyên font/layout sau khi người dùng sửa.
