# Pipeline với tên mô tả chức năng

07/11/2026, UTC+7. Vẽ bằng diagram-design v2.6.142 theo [brief mới](USER_BRIEF.txt). Việc đổi tên chỉ phục vụ trình bày; không sửa mô hình hoặc hàm ý mọi thành phần là đóng góp mới.

## V4 mở cơ chế — bản mới nhất theo brief chi tiết

[Master và từng panel](mechanisms_v4/README.md): mở bank/aggregation, fusion chung, UPT/matching, hai adapter, decoder, ba điểm reranking và training riêng. SVG/PDF vector, PNG7200×7552 và PDF8trang panel. Các bản trước giữ nguyên.

## Bản bố cục gọn

[Bản compact theo điều chỉnh đã duyệt](compact/README.md): bỏ overview lặp, nối trực tiếp hai memory sau adapter vào decoder, thu gọn nhánh thị giác và gom caption selection ở cuối. SVG/PDF/PNG/HTML cùng mã nguồn/QA được giữ riêng; các bản dưới không bị ghi đè.

## Bản mới có icon tham số

[Bản theo brief bổ sung](parameter_icons/README.md): ảnh gốc không bbox, icon frozen/trainable vector, ba màu chỉ định, query/residual riêng và ba điểm rerank hội tụ. SVG/PDF/PNG/HTML có ở thư mục này; bản ban đầu dưới đây vẫn giữ nguyên.

## Bàn giao

- [SVG chỉnh sửa được](captioning-pipeline.svg): chữ, module và mũi tên giữ dạng vector; chỉ ảnh minh họa là raster nhúng.
- [PDF vector](captioning-pipeline.pdf): một trang, chữ chọn được; các đường và khối là vector.
- [PNG 5120×2688](captioning-pipeline.png).
- [HTML độc lập](captioning-pipeline.html): cùng sơ đồ và caption tiếng Việt bên ngoài hình, không gọi font qua mạng.

SVG mở được bằng Inkscape/Illustrator/Figma. Nguồn dựng là `build_pipeline.py`; đổi nhãn/tọa độ trong nguồn rồi xuất lại để đồng bộ mọi định dạng. Khi sửa trực tiếp SVG, các mũi tên không tự bám theo module. Bản mới bổ sung, không ghi đè các phiên bản trước hoặc thay báo cáo Word.

## Caption tiếng Việt

**Pipeline chú thích ảnh kết hợp ngữ nghĩa toàn cục và prompt riêng ảnh.** CLIP và box YOLO tạo region tokens. Bank danh từ/động từ từ TRAIN được tổng hợp thành prototypes để hợp nhất tuần tự với các vùng ảnh và tạo hai memory riêng. Prompt từ UPT được mã hóa bằng CLIP, ghép với ROI hợp lệ rồi cập nhật memory qua hai adapter. Relation adapter lấy đặc trưng subject/object từ `Vo` gốc. Transformer đọc riêng `Vo′/Vr′`, sinh năm ứng viên và chọn caption qua ba thành phần reranking. Khung nét đứt là phần phóng to của fusion và conditioning; cổng memory lặp tên biểu thị cùng tensor. Ảnh và box là minh họa, không phải đầu ra detector đã xác minh.

## Chi tiết giản lược

1. Các tên viết tắt phương pháp của bản trước được thay bằng tên chức năng. Không thêm, bỏ hoặc thay cơ chế vì việc đổi nhãn.
2. Semantic aggregation, prototype selection và ba fusion stage vẫn có mặt; residual/norm/FFN, temperature và các thao tác bên trong fusion được gói vào block. `T1` là noun prototypes đã chọn từ object fusion 1 để đưa sang object fusion 2.
3. Hai nguồn semantic được phân biệt bằng hai khung: bank toàn cục từ TRAIN và prompt phụ thuộc ảnh từ UPT. Cổng `Image I` và `YOLO boxes` nhắc lại dữ liệu từ nhánh trái, không phải ảnh/detector bổ sung. Chỉ có một ảnh minh họa.
4. Mỗi adapter gói phép ghép text–region–geometry thành `Linked slots`; giữ `Masked attention`, bypass memory và ký hiệu cộng residual. Gói output projection vào phép cập nhật; không ghi Q/K/V, dimensions, heads hoặc zero-init. Dòng giữa hai adapter giữ rõ đặc trưng vùng lấy từ `Vo` gốc; relation endpoints có thứ tự. Prompt không đi thẳng vào decoder.
5. Khối Transformer lặp ba layer; parallel cross-attention nhận hai memory riêng, embedding/head/beam nằm ngoài phần lặp. Không khai triển gate inputs hay normalization ở hình tổng quan.
6. Reranking chỉ nêu decoder likelihood, image–text alignment và object consistency. Chuẩn hóa, trọng số, tie-break, tokenization và công thức loss chuyển ra tài liệu triển khai; đổi cách vẽ không đổi selector.
7. Huấn luyện chỉ còn một dòng dưới hình, tách khỏi suy luận. Không chèn loss thành module captioning và không vẽ object reward/CHAIR selection.

Hai nét đứt từ các block trên xuống khung dưới là **zoom guides**, không phải luồng tensor. Các token memory lặp tên/màu giúp giữ riêng `Vo/Vr` và `Vo′/Vr′` mà không kéo dây xuyên khung. Tất cả box xử lý có tối đa hai dòng chữ; hướng tổng quan đọc trái → phải, nhánh trích xuất và decoder được xếp dọc để giữ hình ngang gọn.

## Kiểm tra và nguồn

- [QA hình học](figure_audit.json): 57 tuyến và 53 đối tượng, không phát hiện giao cắt/chồng đoạn, node đè nhau hoặc transit qua module khác. Các góc connector được bo trong SVG.
- [Đo chữ trong Chromium](render_audit.json): không có text overflow. Đã xem PNG và PDF render. PDF một trang có 214 vector drawings, 1 ảnh raster minh họa và chữ chọn được.
- Skill self-check và mask geometry PASS. Kiểm text của SVG không có các tên bị cấm, Q/K/V, kích thước, heads hoặc zero-init; một ảnh nhúng duy nhất. Nhãn training là chú thích, không có mũi tên vào inference.
- [Source trace](source_trace.json) ghi hash brief và ảnh có sẵn trong repo. Box được đặt bằng tay chỉ để minh họa. Không chạy model/Torch/autograd/scorer/Kaggle; không đổi số liệu hoặc tạo experiment registry row.

Nền trắng, pastel nhẹ và tên module theo brief thay thế skin mặc định của skill. Dùng font hệ thống Arial/Liberation Sans; glyph đặc biệt dùng fallback hệ thống. Ưu tiên toàn chiều rộng trang hoặc trang ngang; không ép hình nhiều nhánh này vào một cột hẹp. Bản SVG cho phép điều chỉnh typography theo template tạp chí.

```bash
python build_pipeline.py
python render_figure.py
python audit_figure.py
```

Primitives được dùng lại từ `../upt_csa_architecture_2026-11-06/build_figure.py` mà không thực thi bước dựng hình cũ. Export dùng helper của skill và Chromium hệ thống qua local HTTP; cần chỉnh đường dẫn skill/browser nếu tái tạo trên máy khác.
