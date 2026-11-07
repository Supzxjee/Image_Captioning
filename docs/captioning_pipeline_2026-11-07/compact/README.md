# Bố cục pipeline gọn, một luồng kỹ thuật liên tục

07/11/2026 (UTC+7). Người dùng duyệt thực hiện bốn điều chỉnh sau khi so cách trình bày với hình mẫu: bỏ overview lặp, đưa cơ chế vào luồng chính, thu nhánh thị giác, đặt decoder sát memory và gom beam/reranking ở cuối. [Ghi nhận phạm vi](LAYOUT_BRIEF.md). Dùng diagram-design v2.6.142, giữ kiến trúc và các quy ước đã chốt.

## Bàn giao

[SVG chỉnh sửa được](captioning-pipeline-compact.svg) · [PDF vector](captioning-pipeline-compact.pdf) · [PNG 4800×2240](captioning-pipeline-compact.png) · [HTML](captioning-pipeline-compact.html).

Các bản trước giữ nguyên. Canvas mới 2400×1120, so với 2560×1472 của bản icon trước: chiều cao giảm khoảng 24%, diện tích canvas giảm khoảng 29%. Đây là so sánh bố cục, không phải tốc độ hoặc chất lượng mô hình.

## Caption tiếng Việt

**Kiến trúc mô hình chú thích ảnh kết hợp bank ngữ nghĩa và prompt gắn vùng.** Region features đi trực tiếp qua hai object fusion và một relation fusion. Hai adapter kết hợp prompt UPT với các vùng hợp lệ, cập nhật riêng object/relation memory trước decoder. Decoder đọc trực tiếp hai memory sau adapter, sinh năm caption rồi chọn bằng điểm kết hợp. Các bank toàn cục vẫn khác với prompt riêng ảnh; relation slot dùng `Vo` gốc. Ảnh và token chỉ minh họa, không phải kết quả chạy mới. Reranking dùng **S = 0.3D + 0.4C + 0.3O**, với **D/C được min–max trên cùng 5 ứng viên**.

## Những điểm đổi về trình bày

- Bỏ hai box tổng quan `Cascaded semantic fusion` / `Region-linked prompt conditioning` và các zoom guides. Hai khung cơ chế hiện là thành phần của luồng kỹ thuật, không phải bản giải thích lặp lại.
- Region token `V` đi thẳng vào object fusion 1. `Vo` và `Vr` được tạo và nối trực tiếp vào conditioning; mỗi memory có một điểm tách query/residual rõ ràng.
- `Vo′` và `Vr′` mỗi loại chỉ có một token stack trong toàn hình; từ đó nối thẳng hoặc gấp vuông ngắn vào hai cổng riêng của parallel cross-attention. Không còn token memory lặp ở gần decoder.
- Thu gọn trích xuất thị giác và đặt các khối chuẩn bị bank ở dưới các fusion. Cụm caption ở cuối gồm vocabulary, beam, năm ứng viên, ba điểm reranking hội tụ và final caption.
- Tuyết xanh `#1677B8`, lửa cam viền `#D95F1A` / fill `#F6A23A`. Dữ liệu vuông góc, module bo góc. Tất cả connector thẳng ngang/dọc, đổi hướng vuông 90°; không dùng đường cong hoặc xiên.

## Ba chỉnh sửa cuối và mũi tên thẳng

- Hai box bo góc được đổi thành **Slot MLP**, thể hiện bước xử lý học được. Nhãn đầu vào: object `tₒ + Vo[i] + box`; relation `tᵣ + Vo[s], Vo[o] + boxes`. Cả hai vẫn dùng đặc trưng vùng gốc, không chỉ text/index.
- Công thức reranking và min–max D/C đã thêm trực tiếp vào caption của HTML và đoạn caption phía trên. Weighted scoring and selection giữ nguyên chức năng.
- Hai icon lửa của semantic aggregation được dịch sang góc phải xa hơn, không còn chạm `150×512`. Đo bounding boxes trong Chromium xác nhận không giao nhau.
- Căn cổng để mũi tên ảnh→YOLO, patches→ROI, UPT→phrases, text→bank, metadata→Slot MLP và ba điểm→weighted scoring là một đoạn thẳng. Ba thành phần điểm được đặt ngang để đi thẳng xuống selection. Các đoạn bus/phân nhánh còn lại vẫn chỉ ngang/dọc, góc vuông.

## Chi tiết được gói và quy ước

Text embeddings và valid region links hội tụ tại chấm nối rồi tách sang hai Slot MLP. Đây là thao tác tập hợp metadata đã có, không thêm một bộ xử lý học được. Các chấm còn lại là điểm tách memory sang query và residual; không phải attention mới. Object/relation text vẫn do cùng frozen CLIP text encoder xử lý; matching vẫn dùng UPT/YOLO boxes, class và IoU ≥0,5. Prompt không ghép ROI được bị bỏ khỏi slots.

Các adapter được ghi ngắn là `Object adapter` / `Relation adapter`; chúng vẫn là hai region-linked prompt adapters cũ. Slot relation dùng `Vo[s]/Vo[o]` gốc theo thứ tự; không dùng `Vo′`. Slot MLP biểu diễn linked slot projection. Output projection, geometry chi tiết, padding safety và zero-init được gói trong block; không thay cách tính. Các residual/norm/FFN nội bộ fusion cũng được gói, `T1` vẫn truyền giữa hai object stages.

Khung ×3 chỉ bao ba thao tác của decoder layer; embedding và head/beam ở ngoài. Lửa tại box embedding chỉ cho word embedding, không cho fixed positional encoding; lửa tại vocabulary chỉ cho projection, không cho softmax. Bông tuyết ở C biểu thị frozen CLIP image–text alignment. Khi inference, mọi trọng số đều giữ nguyên.

Rerank vẫn là `S=0.3D+0.4C+0.3O`, D/C min–max trên năm ứng viên, O normalized, hòa điểm ưu tiên beam rank trước. Các đường đi vào weighted scoring có thể đổi thứ tự cổng cho dễ đọc, không đổi hệ số. Ảnh/YOLO labels là đầu vào phụ; reference và CHAIR không chọn caption. Training giữ một dòng riêng, loss hiện tại `LCE+0.05LSCST` với reward CIDEr và reference TRAIN, không thêm object reward.

Không đổi nguồn ảnh, cache, kiến trúc, mô hình, notebook, metric, Word hoặc Sheet. Không chạy Torch/model/scorer/Kaggle. Font Arial/Liberation Sans; màu pastel như các bản trước. SVG có chữ chọn được và một ảnh gốc nhúng; ảnh không có bbox overlay.

## Nguồn và QA

[Audit](figure_audit.json), [đo chữ](render_audit.json), [source trace](source_trace.json). 60 tuyến / 51 đối tượng: không phát hiện giao cắt, chồng đoạn, transit qua node, node đè nhau hoặc chữ tràn. Kiểm riêng nhãn Slot MLP/input, caption có S/min–max, icon không đè kích thước và các nhánh vừa căn có đúng hai đầu mút; không còn overview/zoom lặp, mỗi updated memory xuất hiện đúng một lần, query/residual, icon colors/badge map và tên bị cấm. Đã xem PNG và PDF render. PDF một trang có 262 vector drawings và một ảnh gốc, không phải ảnh chụp toàn bộ sơ đồ.

`build_pipeline.py` dùng lại primitives và icon vectors của bản `parameter_icons`, không thực thi bước dựng bản cũ. `render_figure.py` dùng exporter của skill và Chromium/local HTTP. `audit_figure.py` dùng PyMuPDF và stdlib. Có thể sửa tọa độ/nhãn rồi xuất lại đồng bộ:

```bash
python build_pipeline.py
python render_figure.py
python audit_figure.py
```

Sửa trực tiếp SVG bằng Inkscape/Illustrator/Figma cũng được; các mũi tên không tự bám theo module khi kéo.
